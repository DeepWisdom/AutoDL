# -*- coding: utf-8 -*-
from __future__ import absolute_import
import math
import logging


# LOGGER = logging.getLogger(__name__)


def gradual_warm_up(scheduler, warm_up_epoch, multiplier):
    def schedule(e, **kwargs):
        lr = scheduler(e, **kwargs)
        lr = lr * ((multiplier - 1.0) * min(e, warm_up_epoch) / warm_up_epoch + 1)
        return lr
    return schedule


def get_discrete_epoch(scheduler):
    def schedule(e, **kwargs):
        return scheduler(int(e), **kwargs)
    return schedule


def get_change_scale(scheduler, init_scale=1.0):
    def schedule(e, scale=None, **kwargs):
        lr = scheduler(e, **kwargs)
        return lr * (scale if scale is not None else init_scale)
    return schedule


def get_step_scheduler(init_lr, step_size, gamma=0.1):
    def schedule(e, **kwargs):
        lr = init_lr * gamma ** (e // step_size)
        return lr
    return schedule


def get_cosine_scheduler(init_lr, maximum_epoch, eta_min=0):
    def schedule(e, **kwargs):
        maximum = kwargs['maximum_epoch'] if 'maximum_epoch' in kwargs else maximum_epoch
        lr = eta_min + (init_lr - eta_min) * (1 + math.cos(math.pi * e / maximum)) / 2
        return lr
    return schedule


class PlateauScheduler:
    def __init__(self, init_lr, factor=0.1, patience=10, threshold=1e-4):
        self.init_lr = init_lr
        self.factor = factor
        self.patience = patience
        self.threshold = threshold

        self.curr_lr = init_lr
        self.best_loss = 10000
        self.prev_epoch = 0
        self.num_bad_epochs = 0

    def __call__(self, epoch, loss=None, **kwargs):
        if loss is None:
            loss = self.best_loss

        if self.best_loss - self.threshold > loss:
            self.num_bad_epochs = 0
            self.best_loss = loss
        else:
            self.num_bad_epochs += epoch - self.prev_epoch

        if self.num_bad_epochs >= self.patience:
            self.num_bad_epochs = 0
            self.curr_lr *= self.factor

        self.prev_epoch = epoch
        return self.curr_lr


def get_reduce_on_plateau_scheduler(init_lr, factor=0.1, patience=10, threshold=1e-4, min_lr=0, metric_name='metric'):
    class Schedule:
        def __init__(self):
            self.num_bad_epochs = 0
            self.lr = init_lr
            self.best = None
            self.metric_name = metric_name

        def __call__(self, e, **kwargs):
            if self.metric_name not in kwargs:
                return self.lr
            metric = kwargs[self.metric_name]


            if self.best is None or self.best > metric:
                self.best = metric - threshold
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1

            if self.num_bad_epochs > patience:
                self.num_bad_epochs = 0
                lr = max(min_lr, self.lr * factor)
                self.lr = lr
            return self.lr
    return Schedule()
