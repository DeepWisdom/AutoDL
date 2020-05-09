#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc    : pre-defined exception


class TimeoutException(Exception):
    pass


class ModelAttrLackException(Exception):
    pass


class BadPredShapeException(Exception):
    pass


class IngestionException(Exception):
    pass


class ScoringException(Exception):
    pass
