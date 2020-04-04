# -*- coding: utf-8 -*-
from __future__ import absolute_import
import os
import logging
import hashlib
import random

import numpy as np
import torch


# LOGGER = logging.getLogger(__name__)


class Identity:
    def __call__(self, image):
        return image


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        return (image - self.mean) / self.std

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class RandomFlip:
    def __init__(self, p=0.5, dims=[-1]):
        self.p = p
        self.dims = dims

    def __call__(self, tensor):
        if random.random() < self.p:
            tensor = torch.flip(tensor, dims=self.dims)
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class Crop:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, image):
        h, w = image.shape[-2:]

        y = np.random.randint(h - self.height)
        x = np.random.randint(w - self.width)

        return image[:, y:y+self.height, x:x+self.width]


class Cutout:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, image):
        if self.height > 0 or self.width > 0:
            if isinstance(image, torch.Tensor):
                mask = torch.ones_like(image)
            elif isinstance(image, np.ndarray):
                mask = np.ones_like(image)
            else:
                raise NotImplementedError('support only tensor or numpy array')

            h, w = image.shape[-2:]

            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.height // 2, 0, h)
            y2 = np.clip(y + self.height // 2, 0, h)
            x1 = np.clip(x - self.width // 2, 0, w)
            x2 = np.clip(x + self.width // 2, 0, w)

            if len(mask.shape) == 3:
                mask[:, y1: y2, x1: x2] = 0.
            else:
                mask[:, :, y1: y2, x1: x2] = 0.
            image *= mask
        return image

    def __repr__(self):
        return self.__class__.__name__ + '(height={0}, width={1})'.format(self.height, self.width)


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            img = np.flip(img, axis=-1).copy()
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class Writer:
    def __init__(self, path, format='jpg'):
        self.path = path
        self.format = format
        os.makedirs(self.path, exist_ok=True)

    def __call__(self, image):
        filename = hashlib.md5(image.tobytes()).hexdigest()
        path = self.path + '/' + filename + '.' + self.format
        image.save(path)
        return image

    def __repr__(self):
        return self.__class__.__name__ + '(path={0}, format={1})'.format(self.path, self.format)
