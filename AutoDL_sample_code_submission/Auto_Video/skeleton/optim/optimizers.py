# -*- coding: utf-8 -*-
from __future__ import absolute_import
import logging

import torch


# LOGGER = logging.getLogger(__name__)


class ScheduledOptimizer:
    def __init__(self, parameters, optimizer, steps_per_epoch=1, clip_grad_max_norm=None, tag=None, **opt_params):
        self.epoch = 0.0
        self.tag = tag
        self._parameters = parameters
        self.steps_per_epoch = steps_per_epoch
        self.clip_grad_max_norm = clip_grad_max_norm
        self._opt_params = opt_params

        self._optimizer = optimizer(parameters, **self.update_params(0))

    def update_params(self, epoch=None, **kwargs):
        return {
            k: v(self.epoch if epoch is None else epoch, **kwargs) if callable(v) else v
            for k, v in self._opt_params.items()
        }

    def update(self, epoch=None, **kwargs):
        opt_pararms = self.update_params(epoch, **kwargs)
        self._optimizer.param_groups[0].update(**opt_pararms)

        for key, value in opt_pararms.items():
            tag = self.tag if self.tag is not None else 'train'
            if not isinstance(value, (float, int)):
                continue
        return self

    def step(self, epoch=None):
        self.epoch = self.epoch + (1.0 / self.steps_per_epoch) if epoch is None else epoch
        if self.clip_grad_max_norm is not None and self.clip_grad_max_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(self._parameters, self.clip_grad_max_norm, norm_type=1)
        self._optimizer.step()

    def state_dict(self):
        state_dict = self._optimizer.state_dict()
        state_dict.update({'epoch': self.epoch})
        return state_dict

    def load_state_dict(self, state_dict):
        self.epoch = state_dict.pop('epoch')
        return self._optimizer.load_state_dict(state_dict)

    def zero_grad(self):
        return self._optimizer.zero_grad()

    def get_learning_rate(self):
        return self._optimizer.param_groups[0]['lr']

    def __getattr__(self, item):
        return getattr(self._optimizer, item)
