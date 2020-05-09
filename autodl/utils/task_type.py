#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc    : task type binary/multiclass/regression judge

import numpy as np


def is_one_hot_vector(x, axis=None, keepdims=False):
    """Check if a vector 'x' is one-hot (i.e. one entry is 1 and others 0)."""
    norm_1 = np.linalg.norm(x, ord=1, axis=axis, keepdims=keepdims)
    norm_inf = np.linalg.norm(x, ord=np.inf, axis=axis, keepdims=keepdims)
    return np.logical_and(norm_1 == 1, norm_inf == 1)


def is_multiclass(solution):
    """Return if a task is a multi-class classification task, i.e.  each example
    only has one label and thus each binary vector in `solution` only has
    one '1' and all the rest components are '0'.

    This function is useful when we want to compute metrics (e.g. accuracy) that
    are only applicable for multi-class task (and not for multi-label task).

    Args:
      solution: a numpy.ndarray object of shape [num_examples, num_classes].
    """
    return all(is_one_hot_vector(solution, axis=1))
