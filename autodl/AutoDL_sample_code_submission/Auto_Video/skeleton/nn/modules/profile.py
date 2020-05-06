# -*- coding: utf-8 -*-
# pylint: disable=arguments-differ, abstract-method
from __future__ import absolute_import
import logging

import numpy as np
import torch
from torch import nn


# LOGGER = logging.getLogger(__name__)


class Profile:
    def __init__(self, module):
        self.module = module

    def params(self, name_filter=lambda name: True):
        return np.sum(params.numel() for name, params in self.module.named_parameters() if name_filter(name))

    def flops(self, *inputs, name_filter=lambda name: 'skeleton' not in name and 'loss' not in name):
        operation_flops = []

        def get_hook(name):
            def counting(module, inp, outp):
                class_name = module.__class__.__name__

                flops = 0
                module_type = type(module)
                if not name_filter(str(module_type)):
                    pass
                elif module_type in COUNT_FN_MAP:
                    fn = COUNT_FN_MAP[module_type]
                    flops = fn(module, inp, outp) if fn is not None else 0
                else:
                    pass

                data = {
                    'name': name,
                    'class_name': class_name,
                    'flops': flops,
                }
                operation_flops.append(data)
            return counting

        handles = []
        for name, module in self.module.named_modules():
            if len(list(module.children())) > 0:  # pylint: disable=len-as-condition
                continue
            handle = module.register_forward_hook(get_hook(name))
            handles.append(handle)

        _ = self.module(*inputs)

        # remove hook
        _ = [h.remove() for h in handles]

        return np.sum([data['flops'] for data in operation_flops if name_filter(data['name'])])


COUNT_OP_MULTIPLY_ADD = 1


def count_conv2d(m, x, y):
    # TODO: add support for pad and dilation
    x = x[0]

    cin = m.in_channels
    cout = m.out_channels
    kh, kw = m.kernel_size
    batch_size = x.size()[0]

    out_w = y.size(2) // m.stride[0]
    out_h = y.size(3) // m.stride[1]


    kernel_ops = COUNT_OP_MULTIPLY_ADD * kh * kw * cin // m.groups
    bias_ops = 1 if m.bias is not None else 0
    ops_per_element = kernel_ops + bias_ops


    output_elements = batch_size * out_w * out_h * cout
    total_ops = output_elements * ops_per_element

    return int(total_ops)


def count_bn2d(m, x, y):
    x = x[0]

    nelements = x.numel()
    total_sub = nelements
    total_div = nelements
    total_ops = total_sub + total_div

    return int(total_ops)


def count_relu(m, x, y):
    x = x[0]

    nelements = x.numel()
    total_ops = nelements

    return int(total_ops)


def count_softmax(m, x, y):
    x = x[0]

    batch_size, nfeatures = x.size()

    total_exp = nfeatures
    total_add = nfeatures - 1
    total_div = nfeatures
    total_ops = batch_size * (total_exp + total_add + total_div)

    return int(total_ops)


def count_maxpool(m, x, y):
    kernel_ops = torch.prod(torch.Tensor([m.kernel_size])) - 1
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    return int(total_ops)


def count_avgpool(m, x, y):
    total_add = torch.prod(torch.Tensor([m.kernel_size])) - 1
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    return int(total_ops)


def count_global_avgpool(m, x, y):
    x = x[0]

    w, h = x.size(2), x.size(1)
    total_add = w * h - 1
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    return int(total_ops)


def count_linear(m, x, y):
    total_mul = m.in_features
    total_add = m.in_features - 1
    num_elements = y.numel()
    total_ops = (total_mul + total_add) * num_elements

    return int(total_ops)


COUNT_FN_MAP = {
    torch.nn.Conv2d: count_conv2d,
    torch.nn.BatchNorm2d: count_bn2d,
    torch.nn.ReLU: count_relu,
    torch.nn.ReLU6: count_relu,
    torch.nn.LeakyReLU: count_relu,
    torch.nn.MaxPool1d: count_maxpool,
    torch.nn.MaxPool2d: count_maxpool,
    torch.nn.MaxPool3d: count_maxpool,
    torch.nn.AvgPool1d: count_avgpool,
    torch.nn.AvgPool2d: count_avgpool,
    torch.nn.AvgPool3d: count_avgpool,
    torch.nn.AdaptiveAvgPool1d: count_global_avgpool,
    torch.nn.AdaptiveAvgPool2d: count_global_avgpool,
    torch.nn.AdaptiveAvgPool3d: count_global_avgpool,
    torch.nn.Linear: count_linear,
    torch.nn.Dropout: None,
    # torch.nn.Identity: None,
}
