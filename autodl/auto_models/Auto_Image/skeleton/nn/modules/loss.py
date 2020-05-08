# -*- coding: utf-8 -*-
# pylint: disable=arguments-differ, abstract-method
from __future__ import absolute_import
import logging

import torch


LOGGER = logging.getLogger(__name__)


class CrossEntropyLabelSmooth(torch.nn.Module):
    def __init__(self, num_classes, epsilon=0.1, sparse_target=True, reduction='avg'):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.sparse_target = sparse_target
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.reduction = reduction

    def forward(self, input, target):  # pylint: disable=redefined-builtin
        log_probs = self.logsoftmax(input)
        if self.sparse_target:
            targets = torch.zeros_like(log_probs).scatter_(1, target.unsqueeze(1), 1)
        else:
            targets = target
        targets = (1 - self.epsilon) * targets + (self.epsilon / self.num_classes)
        loss = (-targets * log_probs)
        if self.reduction == 'avg':
            loss = loss.mean(0).sum()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


class BinaryCrossEntropyLabelSmooth(torch.nn.BCEWithLogitsLoss):
    def __init__(self, num_classes, epsilon=0.1, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None):
        super(BinaryCrossEntropyLabelSmooth, self).__init__(weight, size_average, reduce, reduction, pos_weight)
        self.num_classes = num_classes
        self.epsilon = epsilon

    def forward(self, input, target):  # pylint: disable=redefined-builtin
        target = (1 - self.epsilon) * target + self.epsilon
        return super(BinaryCrossEntropyLabelSmooth, self).forward(input, target)
