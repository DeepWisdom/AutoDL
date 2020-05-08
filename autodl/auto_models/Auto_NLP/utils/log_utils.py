# -*- coding: utf-8 -*-
# @Date    : 2020/1/8 16:16
# @Author  : stellahong (stellahong@fuzhi.ai)
# @Desc    :
import logging
import sys
import os
verbosity_level = 'INFO'

def get_logger(verbosity_level, use_error_log=False, log_path=None):
    """Set logging format to something like:
       2019-04-25 12:52:51,924 INFO score.py: <message>
  """
    logger = logging.getLogger(__file__)
    logging_level = getattr(logging, verbosity_level)
    logger.setLevel(logging_level)

    if log_path is None:
        log_dir = os.path.join("./", "log")
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
    fh.setFormatter(formatter)
    fh.setLevel(logging_level)
    logger.addHandler(fh)

    if use_error_log:
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.WARNING)
        stderr_handler.setFormatter(formatter)
        logger.addHandler(stderr_handler)
    logger.propagate = False
    return logger


logger = get_logger(verbosity_level)