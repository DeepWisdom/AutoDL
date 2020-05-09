#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc    :

import os
import psutil
import numpy as np

from autodl.utils.logger import logger


def is_process_alive(pid):
    """Check if a process is alive according to its PID."""
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True


def terminate_process(pid):
    """Kill a process according to its PID."""
    process = psutil.Process(pid)
    process.terminate()
    logger.debug("Terminated process with pid={} in scoring.".format(pid))


def transform_time(t, T, t0=None):
    if t0 is None:
        t0 = T
    return np.log(1 + t / t0) / np.log(1 + T / t0)


def get_fig_name(task_name):
    """Helper function for getting learning curve figure name."""
    fig_name = "learning-curve-" + task_name + ".png"
    return fig_name
