import tensorflow as tf
import torch
import pdb
import inspect
import numpy as np

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, DALIGenericIterator

class ExternalInputIterator(object):
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size

    def __iter__(self):
        self.i = 0
        self.n = len(self.data)
        return self

    def __next__(self):
        batch = []
        labels = []

        if self.i >= self.n:
            raise StopIteration
        
        for _ in range(self.batch_size):
            batch.append(self.data[self.i][0])
            labels.append(self.data[self.i][1])
            self.i = (self.i + 1) % self.n
        return (batch, labels)

    @property
    def size(self,):
        return len(self.data)

    next = __next__

class ExternalInputIterator2(object):
    def __init__(self, session, element, num_samples, batch_size):
        self.session = session
        self.element = element
        self.batch_size = batch_size
        self.num_samples = num_samples

    def __iter__(self):
        self.i = 0
        self.n = self.num_samples
        return self

    def __next__(self):
        batch = []
        labels = []

        if self.i >= self.n:
            raise StopIteration
        
        for _ in range(self.batch_size):
            example, label = self.session.run(self.element)
            batch.append(example)
            labels.append(label)
            self.i = (self.i + 1) % self.n
        return (batch, labels)


    @property
    def size(self,):
        return self.num_samples

    next = __next__

class ExternalInputIterator3(object):
    def __init__(self, session, dataset, num_samples, batch_size, fill_last_batch=True):
        self.session = session
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.fill_last_batch = fill_last_batch
        self.steps = self.num_samples // self.batch_size
        if self.num_samples % self.batch_size != 0:
            self.steps += 1
        self.total = self.batch_size * self.steps

    def __iter__(self):
        self.i = 0
        self.n = self.num_samples
        return self

    def __next__(self):
        batch = []
        labels = []
    
        for batch_idx in range(self.batch_size):
            if not self.fill_last_batch and self.i >= self.num_samples:
                example, label = np.zeros(self.example_size, dtype=np.float32), np.zeros(self.label_size, dtype=np.float32)
            else:
                try:
                    example, label = self.session.run(self.next_element)

                    if self.i == 0:
                        self.example_size = example.shape
                        self.label_size = label.shape


                except tf.errors.OutOfRangeError:
                    self.reset()
                    raise StopIteration

            self.i  = (self.i + 1) % self.total

            batch.append(example)
            labels.append(label)

        return (batch, labels)

    def reset(self):
        dataset = self.dataset
        iterator = dataset.make_one_shot_iterator()
        self.next_element = iterator.get_next()
        return self


    @property
    def size(self,):
        return self.num_samples

    next = __next__

class ExternalSourcePipeline(Pipeline):
    def __init__(self, session, dataset, batch_size, num_threads, is_random_flip=True, num_samples=1000000, device_id=0, preprocess=None, fill_last_batch=True):
        super(ExternalSourcePipeline, self).__init__(batch_size,
                                      num_threads,
                                      device_id,
                                      seed=12)
        self.session = session
        self.num_samples = num_samples
        self.dataset = dataset
        self.is_random_flip = is_random_flip
        self.preprocess = preprocess

        if self.preprocess is not None:
            crop = (preprocess['width'], preprocess['height'])

            self.res = ops.Resize(resize_x=preprocess['width'], resize_y=preprocess['height'])

        self.flip = ops.Flip()

        self.coin = ops.CoinFlip(probability=0.5)
        self.coin2 = ops.CoinFlip(probability=0.5)
        
        self.iterator = iter(ExternalInputIterator3(self.session, self.dataset, self.num_samples, batch_size, fill_last_batch))
        self.iterator.reset()
        
        self.input = ops.ExternalSource()
        self.input_label = ops.ExternalSource()

    def reset(self):
        dataset = self.dataset
        iterator = dataset.make_one_shot_iterator()
        self.next_element = iterator.get_next()
        return self

    def define_graph(self):
        rng = self.coin()
        rng2 = self.coin2()

        self.x = self.input()
        output = self.x
        
        self.labels = self.input_label()
        return [output, self.labels]

    def iter_setup(self):
        (x, labels) = self.iterator.next()
        self.feed_input(self.x, x)
        self.feed_input(self.labels, labels)
        

class DALILoader(object):
    def gen_loader(loader, steps):
        for i, data in enumerate(loader):
            input = data[0]['data'].cuda()
            target = data[0]["label"].cuda()
            yield input, target

    def __init__(self, session, dataset, num_samples, batch_size, steps=None, num_threads=0, fill_last_batch=True, is_random_flip=True, preprocess=None):
        self.steps = steps
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_samples = num_samples

        pipe = ExternalSourcePipeline(session, self.dataset, batch_size=batch_size, num_threads=num_threads, num_samples=self.num_samples, device_id=0, is_random_flip=is_random_flip, preprocess=preprocess, fill_last_batch=fill_last_batch)
        pipe.build()
        self.dataloader = DALIClassificationIterator(pipe, self.num_samples, auto_reset=True, fill_last_batch=fill_last_batch, last_batch_padded=True)

    def __len__(self):
        if self.steps is None:
            steps = self.num_samples // self.batch_size
            if self.num_samples % self.batch_size != 0:
                steps += 1
            return steps
            
        return self.steps

    def __iter__(self):
        return DALILoader.gen_loader(self.dataloader, self.steps)

    def reset(self):
        self.dataloader.reset()

