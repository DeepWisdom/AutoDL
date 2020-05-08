import logging
import os
import sys

import time
from typing import Any
import multiprocessing

import functools
nesting_level = 0
is_start = None
NCPU = multiprocessing.cpu_count()


def log(entry: Any):
    global nesting_level
    space = "-" * (4 * nesting_level)
    logger.info("{}{}".format(space, entry))


def get_logger(verbosity_level, use_error_log=False, log_path=None):
    """Set logging format to something like:
         2019-04-25 12:52:51,924 INFO score.py: <message>
    """
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
        log("Start [{}]:".format(method.__name__)+ (start_log if start_log else ""))
        nesting_level += 1

        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()

        nesting_level -= 1
        log("End   [{}]. Time elapsed: {} sec.".format(method.__name__, end_time - start_time))
        is_start = False

        return result

    return timed