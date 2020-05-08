# -*- coding: utf-8 -*-
from __future__ import absolute_import
import logging

import torch
from torch.utils.data import Dataset
import numpy as np
import tensorflow as tf
from ..nn.modules.hooks import MoveToHook

# LOGGER = logging.getLogger(__name__)


class TFDataset(Dataset):
    def __init__(self, session, dataset, num_samples):
        super(TFDataset, self).__init__()
        self.session = session
        self.dataset = dataset
        self.num_samples = num_samples
        self.next_element = None

        self.reset()

    def reset(self):
        dataset = self.dataset
        iterator = dataset.make_one_shot_iterator()
        self.next_element = iterator.get_next()
        return self

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        session = self.session if self.session is not None else tf.Session()
        try:
            example, label = session.run(self.next_element)
        except tf.errors.OutOfRangeError:
            self.reset()
            raise StopIteration

        return example, label

    def scan(self, samples=1000000, with_tensors=False, is_batch=False, device=None, half=False):
        shapes, counts, tensors = [], [], []
        labels = []
        min_list, max_list = [], []
        is_255 = False
        for i in range(min(self.num_samples, samples)):
            try:
                example, label = self.__getitem__(i)
                if i == 0 and np.mean(example) > 1:
                    is_255 = True
            except tf.errors.OutOfRangeError:
                break
            except StopIteration:
                break

            shape = example.shape
            count = np.sum(label, axis=None if not is_batch else -1)
            labels.append(label)

            shapes.append(shape)
            counts.append(count)
            min_list.append(np.min(example))
            max_list.append(np.max(example))

            if with_tensors:
                example = torch.Tensor(example)
                label = torch.Tensor(label)

                example.data = example.data.to(device=device)
                if half and example.is_floating_point():
                    example.data = example.data.half()

                label.data = label.data.to(device=device)
                if half and label.is_floating_point():
                    label.data = label.data.half()

                tensors.append([example, label])

        shapes = np.array(shapes)
        counts = np.array(counts) if not is_batch else np.concatenate(counts)

        labels = np.array(labels) if not is_batch else np.concatenate(labels)
        num_samples = labels.shape[0]
        labels = np.sum(labels, axis=0)
        zero_count = sum(labels == 0)

        pos_weights = (num_samples - labels + 10) / (labels + 10)
        info = {
            'count': len(counts),
            'is_multiclass': counts.max() > 1.0,
            'is_video': int(np.median(shapes, axis=0)[0]) > 1,
            'example': {
                'shape': [int(v) for v in np.median(shapes, axis=0)],
                'shape_avg': [int(v) for v in np.average(shapes, axis=0)],
                'value': {'min': min(min_list), 'max': max(max_list)},
                'is_255': is_255
            },
            'label': {
                'min': counts.min(),
                'max': counts.max(),
                'average': counts.mean(),
                'median': np.median(counts),
                'zero_count': zero_count,
                'pos_weights': pos_weights
            },

        }

        if with_tensors:
            return info, tensors
        return info


class TransformDataset(Dataset):
    def __init__(self, dataset, transform=None, index=None):
        self.dataset = dataset
        self.transform = transform
        self.index = index

    def __getitem__(self, index):
        tensors = self.dataset[index]
        tensors = list(tensors)

        if self.transform is not None:
            if self.index is None:
                tensors = self.transform(*tensors)
            else:
                tensors[self.index] = self.transform(tensors[self.index])

        return tuple(tensors)

    def __len__(self):
        return len(self.dataset)


def prefetch_dataset(dataset, num_workers=4, batch_size=32, device=None, half=False):
    if isinstance(dataset, list) and isinstance(dataset[0], torch.Tensor):
        tensors = dataset
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False, drop_last=False,
            num_workers=num_workers, pin_memory=False
        )
        tensors = [t for t in dataloader]
        tensors = [torch.cat(t, dim=0) for t in zip(*tensors)]

    if device is not None:
        tensors = [t.to(device=device) for t in tensors]
    if half:
        tensors = [t.half() if t.is_floating_point() else t for t in tensors]

    return torch.utils.data.TensorDataset(*tensors)
