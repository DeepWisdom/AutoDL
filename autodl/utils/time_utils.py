#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc    : time utils

from contextlib import contextmanager
import signal
import math
import time

from autodl.utils.exception import TimeoutException
from autodl.utils.log_utils import logger


class Timer(object):

    def __init__(self):
        self.duration = 0
        self.total = None
        self.remain = None
        self.exec = None

    def set(self, time_budget):
        self.total = time_budget
        self.remain = time_budget
        self.exec = 0

    @contextmanager
    def time_limit(self, pname):
        def signal_handler(signum, frame):
            raise TimeoutException("Timed out!")

        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(int(math.ceil(self.remain)))
        start_time = time.time()

        try:
            yield
        finally:
            exec_time = time.time() - start_time
            signal.alarm(0)
            self.exec += exec_time
            self.duration += exec_time
            self.remain = self.total - self.exec

            logger.info("{} success, time spent so far {} sec".format(pname, self.exec))

            if self.remain <= 0:
                raise TimeoutException("Timed out for the process: {}!".format(pname))
