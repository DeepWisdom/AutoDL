# -*- coding: utf-8 -*-
# pylint: disable=arguments-differ
from __future__ import absolute_import
import logging
from functools import wraps
from collections import OrderedDict

import torch
import numpy as np

LOGGER = logging.getLogger(__name__)


class ToDevice(torch.nn.Module):
    def __init__(self):
        super(ToDevice, self).__init__()
        self.register_buffer('buf', torch.zeros(1, dtype=torch.float32))

    def forward(self, *xs):
        if len(xs) == 1 and isinstance(xs[0], (tuple, list)):
            xs = xs[0]

        device = self.buf.device
        out = []
        for x in xs:
            if x is not None and x.device != device:
                out.append(x.to(device=device))
            else:
                out.append(x)
        return out[0] if len(xs) == 1 else tuple(out)


class CopyChannels(torch.nn.Module):
    def __init__(self, multiple=3, dim=1):
        super(CopyChannels, self).__init__()
        self.multiple = multiple
        self.dim = dim

    def forward(self, x):
        return torch.cat([x for _ in range(self.multiple)], dim=self.dim)


class Normalize(torch.nn.Module):
    def __init__(self, mean, std, inplace=False):
        super(Normalize, self).__init__()
        if isinstance(mean, list):
            self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32)[None, :, None, None])
            self.register_buffer('std', torch.tensor(std, dtype=torch.float32)[None, :, None, None])
        else:
            self.register_buffer('mean', torch.tensor([mean], dtype=torch.float32)[None, :, None, None])
            self.register_buffer('std', torch.tensor([std], dtype=torch.float32)[None, :, None, None])
        self.inplace = inplace

    def forward(self, x):
        if not self.inplace:
            x = x.clone()

        x.sub_(self.mean).div_(self.std)
        return x


class Reshape(torch.nn.Module):
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        batch = x.shape[0]
        return x.view([batch, -1])


class SplitTime(torch.nn.Module):
    def __init__(self, times):
        super(SplitTime, self).__init__()
        self.times = times

    def forward(self, x):
        batch, channels, height, width = x.shape
        return x.view(-1, self.times, channels, height, width)


class Permute(torch.nn.Module):
    def __init__(self, *dims):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class Cutout(torch.nn.Module):
    def __init__(self, ratio=0.0):
        super(Cutout, self).__init__()
        self.ratio = ratio

    def forward(self, input):
        batch, channel, height, width = input.shape
        w = int(width * self.ratio)
        h = int(height * self.ratio)

        if self.training and w > 0 and h > 0:
            x = np.random.randint(width, size=(batch,))
            y = np.random.randint(height, size=(batch,))

            x1s = np.clip(x - w // 2, 0, width)
            x2s = np.clip(x + w // 2, 0, width)
            y1s = np.clip(y - h // 2, 0, height)
            y2s = np.clip(y + h // 2, 0, height)

            mask = torch.ones_like(input)
            for idx, (x1, x2, y1, y2) in enumerate(zip(x1s, x2s, y1s, y2s)):
                mask[idx, :, y1:y2, x1:x2] = 0.

            input = input * mask
        return input


class Mul(torch.nn.Module):
    def __init__(self, weight):
        super(Mul, self).__init__()
        self.weight = weight

    def forward(self, x):
        return x * self.weight


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def decorator_tuple_to_args(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        args = list(args)
        if len(args) == 2 and isinstance(args[1], (tuple, list)):
            args[1:] = list(args[1])
        return func(*args, **kwargs)

    return wrapper


class Concat(torch.nn.Module):
    def __init__(self, dim=1):
        super(Concat, self).__init__()
        self.dim = dim

    @decorator_tuple_to_args
    def forward(self, *xs):
        return torch.cat(xs, dim=self.dim)


class MergeSum(torch.nn.Module):
    @decorator_tuple_to_args
    def forward(self, *xs):
        return torch.sum(torch.stack(xs), dim=0)


class MergeProd(torch.nn.Module):
    @decorator_tuple_to_args
    def forward(self, *xs):
        return xs[0] * xs[1]


class Choice(torch.nn.Module):
    def __init__(self, idx=0):
        super(Choice, self).__init__()
        self.idx = idx

    @decorator_tuple_to_args
    def forward(self, *xs):
        return xs[self.idx]


class Toggle(torch.nn.Module):
    def __init__(self, module):
        super(Toggle, self).__init__()
        self.module = module
        self.on = True

    def forward(self, x):
        return self.module(x) if self.on else x


class Split(torch.nn.Module):
    def __init__(self, *modules):
        super(Split, self).__init__()
        if len(modules) == 1 and isinstance(modules[0], OrderedDict):
            for key, module in modules[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(modules):
                self.add_module(str(idx), module)

    def forward(self, x):
        return tuple([m(x) for m in self._modules.values()])


class DropPath(torch.nn.Module):
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self._half = False

    def forward(self, x):
        if self.training and self.drop_prob > 0.:
            shape = list(x.shape[:1]) + [1 for _ in x.shape[1:]]
            keep_prob = 1. - self.drop_prob
            mask = torch.cuda.FloatTensor(*shape).bernoulli_(keep_prob)
            if self._half:
                mask = mask.half()
            x.div_(keep_prob)
            x.mul_(mask)
        return x

    def half(self):
        self._half = True

    def float(self):
        self._half = False


class DelayedPass(torch.nn.Module):
    def __init__(self):
        super(DelayedPass, self).__init__()
        self.register_buffer('keep', None)

    def forward(self, x):
        rv = self.keep  # pylint: disable=access-member-before-definition
        self.keep = x
        return rv


class Reader(torch.nn.Module):
    def __init__(self, x=None):
        super(Reader, self).__init__()
        self.x = x

    def forward(self, x):  # pylint: disable=unused-argument
        return self.x


class KeepByPass(torch.nn.Module):
    def __init__(self):
        super(KeepByPass, self).__init__()
        self._reader = Reader()
        self.info = {}

    @property
    def x(self):
        return self._reader.x

    def forward(self, x):
        self._reader.x = x
        return x

    def reader(self):
        return self._reader
