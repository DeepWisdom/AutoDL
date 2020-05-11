#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc    : base dynamic plot model metric

import matplotlib.pyplot as plt


class PlotBaseMetric(object):

    def __init__(self):
        pass

    def save_figure(self):
        raise NotImplementedError()
