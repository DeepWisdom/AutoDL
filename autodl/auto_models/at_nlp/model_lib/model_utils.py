# -*- coding: utf-8 -*-
# @Date    : 2020/3/3 11:22


def _get_last_layer_units_and_activation(num_classes, only_use_softmax=True):
    """Gets the # units and activation function for the last network layer.

    Args:
        num_classes: Number of classes.
        only_use_softmax: if only use softmax
    Returns:
        units, activation values.
    """
    if num_classes == 2 and not only_use_softmax:
        activation = 'sigmoid'
        units = 1
    else:
        activation = 'softmax'
        units = num_classes
    return units, activation
