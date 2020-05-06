# -*- coding: utf-8 -*-
# pylint: disable=arguments-differ, abstract-method
from __future__ import absolute_import
import logging

import torch
from torch import nn


LOGGER = logging.getLogger(__name__)


class MoveToHook(nn.Module):
    @staticmethod
    def to(tensors, device, half=False):
        for t in tensors:
            if isinstance(t, (tuple, list)):
                MoveToHook.to(t, device, half)
            if not isinstance(t, torch.Tensor):
                continue
            t.data = t.data.to(device=device)
            if half:
                if t.is_floating_point():
                    t.data = t.data.half()

    @staticmethod
    def get_forward_pre_hook(device, half=False):
        def hook(module, inputs):
            _ = module
            MoveToHook.to(inputs, device, half)
        return hook

