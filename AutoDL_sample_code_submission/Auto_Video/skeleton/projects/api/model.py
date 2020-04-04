# -*- coding: utf-8 -*-
from __future__ import absolute_import


class Model:
    def __init__(self, metadata):
        """
        :param metadata: an AutoDLMetadata object. Its definition can be found in
            https://github.com/zhengying-liu/autodl_starting_kit_stable/blob/master/AutoDL_ingestion_program/dataset.py#L41
        """
        self.metadata = metadata

        self.done_training = False
        # the loop of calling 'train' and 'test' will only run if self.done_training = False
        # otherwinse, the looop will go until the time budge in used up set self.done_training = True
        # when you think the model is converged or when is not enough time for next round of traning

    def train(self, dataset, remaining_time_budget=None):
        """
        :param dataset: a `tf.data.Dataset` object. Each of its examples is of the form (example, labels)
            where `example` is a dense 4-D Tensor of shape  (sequence_size, row_count, col_count, num_channels)
            and `labels` is a 1-D Tensor of shape (output_dim,)
            Here `output_dim` represents number of classes of this multilabel classification task.
        :param remaining_time_budget: a float, time remaining to execute train()
        :return: None
        """
        raise NotImplementedError

    def test(self, dataset, remaining_time_budget=None):
        """
        :param: Same as that of `train` method, except that the labes will be empty (all zeros)
        :return: predictions: A `numpy.ndarray` matrix of shape (sample_count, output_dim)
            The values should be binary or in the interval [0,1].
        """
        raise NotImplementedError
