#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc    :

import os
import psutil
import numpy as np

from autodl.utils.log_utils import logger
from autodl.auto_scoring.libscores import ls, read_array


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


def get_solution(solution_dir):
    """Get solution as NumPy array from `self.solution_dir`."""
    solution_names = sorted(ls(os.path.join(solution_dir, "*.solution")))
    if len(solution_names) != 1:  # Assert only one file is found
        logger.warning("{} solution files found: {}! ".format(len(solution_names), solution_names) +
                       "Return `None` as solution.")
        solution = None
    else:
        solution_file = solution_names[0]
        solution = read_array(solution_file)

    logger.debug("Successfully loaded solution from solution_dir={}".format(solution_dir))
    return solution
