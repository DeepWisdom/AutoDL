import logging
import os
import sys
import json
import time
from typing import Any
import multiprocessing
from collections import OrderedDict
import psutil

import functools
nesting_level = 0
is_start = None
NCPU = multiprocessing.cpu_count()


def log(entry: Any):
    global nesting_level
    space = "-" * (4 * nesting_level)
    logger.info("{}{}".format(space, entry))


def get_logger(verbosity_level, use_error_log=False, log_path=None):

    logger = logging.getLogger(__file__)
    logging_level = getattr(logging, verbosity_level)
    logger.setLevel(logging_level)

    if log_path is None:
        log_dir = os.path.join("..", "log")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_path = os.path.join(log_dir, "log.txt")
    else:
        log_path = os.path.join(log_path, "log.txt")

    formatter = logging.Formatter(
        fmt='%(asctime)s %(levelname)s %(filename)s: %(funcName)s: %(lineno)d: %(message)s')

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging_level)
    stdout_handler.setFormatter(formatter)

    logger.addHandler(stdout_handler)

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging_level)
    fh.setFormatter(formatter)

    logger.addHandler(fh)

    if use_error_log:
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.WARNING)
        stderr_handler.setFormatter(formatter)
        logger.addHandler(stderr_handler)
    logger.propagate = False
    return logger


logger = get_logger('INFO')

debug = logger.debug
info = logger.info
warning = logger.warning
error = logger.error


def timeit(method, start_log=None):
    @functools.wraps(method)
    def timed(*args, **kw):
        global is_start
        global nesting_level

        if not is_start:
            print()

        is_start = True
        # log(f"Start [{method.__name__}]:" + (start_log if start_log else ""))
        log("Start [{}]:" + (start_log if start_log else "").format(method.__name__))
        nesting_level += 1

        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()

        nesting_level -= 1
        log("End   [{}]. Time elapsed: {} sec.".format(method.__name__, end_time - start_time))
        is_start = False

        return result

    return timed



class ASTimer():
    def __init__(self):
        self.times = [time.time()]
        # self.accumulation = OrderedDict({})
        self.accumulation = list()
        self.total_time = 0.0
        self.step_time = 0.0
        self.counter = 0
        self.repr_update_cnt = 0
        self.train_start_t = time.time()
        self.test_start_t = time.time()

    def __call__(self, time_name):
        if time_name == "train_start":
            self.train_start_t = time.time()
            self.times.append(self.train_start_t)
            delta = self.times[-1] - self.times[-2]
        elif time_name == "train_end":
            self.times.append((time.time()))
            delta = self.times[-1] - self.train_start_t
        elif time_name == "test_start":
            self.test_start_t = time.time()
            self.times.append(self.test_start_t)
            delta = self.times[-1] - self.times[-2]
        elif time_name == "test_end":
            self.times.append((time.time()))
            delta = self.times[-1] - self.test_start_t
        else:
            self.times.append((time.time()))
            delta = self.times[-1] - self.times[-2]


        self.accumulation.append(["{}_{}".format(self.counter, time_name), delta])
        self.counter += 1

    def __repr__(self):

        timer_res = ["{}:        {}s".format(t[0], round(t[1], 3)) for t in self.accumulation[self.repr_update_cnt: self.counter]]
        self.repr_update_cnt = self.counter
        return json.dumps(timer_res, indent=4)

    def print_all(self):
        timer_res = ["{}:       {}s".format(t[0], t[1]) for t in self.accumulation]
        return json.dumps(timer_res, indent=4)


as_timer = ASTimer()

os.system("apt install wget")
here = os.path.dirname(os.path.abspath(__file__))

def get_env_monitor():
    # CPU usage
    print("=======Env Monitor: CPU Usage======")
    for i, percentage in enumerate(psutil.cpu_percent(percpu=True)):
        print("Core {}: {}%".format(i, percentage))
    print("Total CPU Usage: {}%".format(psutil.cpu_percent()))


