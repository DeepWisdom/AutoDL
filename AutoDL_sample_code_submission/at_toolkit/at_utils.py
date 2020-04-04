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

from at_toolkit.at_cons import SPEECH_TR34_PT_MODEL_PATH, SPEECH_TR34_PT_MODEL_DIR


nesting_level = 0
is_start = None
NCPU = multiprocessing.cpu_count()


def log(entry: Any):
    global nesting_level
    space = "-" * (4 * nesting_level)
    # logger.info(f"{space}{entry}")
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
        # log(f"Start [{method.__name__}]:" + (start_log if start_log else ""))
        log("Start [{}]:" + (start_log if start_log else "").format(method.__name__))
        nesting_level += 1

        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()

        nesting_level -= 1
        # log(f"End   [{method.__name__}]. Time elapsed: {end_time - start_time:0.2f} sec.")
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

        # self.accumulation["{}_{}".format(self.counter, time_name)] = delta

        self.accumulation.append(["{}_{}".format(self.counter, time_name), delta])
        self.counter += 1

    def __repr__(self):
        # for list
        # timer_res = ["{}:{}s".format(t[0], t[1]) for t in self.accumulation]
        # for ordered dict.
        # for n, t in self.accumulation.items():
        #     timer_res.append("{}:{}s".format(n, round(t, 3)))
        # timer_res = ["{}:        {}s".format(t[0], round(t[1], 3)) for t in self.accumulation[self.repr_update_cnt: self.counter]]
        timer_res = [[t[0], round(t[1], 3)] for t in self.accumulation[self.repr_update_cnt: self.counter]]
        self.repr_update_cnt = self.counter
        # return json.dumps(timer_res, indent=4)
        return json.dumps(timer_res)

    def print_all(self):
        timer_res = ["{}:       {}s".format(t[0], t[1]) for t in self.accumulation]
        return json.dumps(timer_res, indent=4)



def autodl_image_install_download():

    pass


def autodl_video_install_download():

    pass


def autodl_speech_install_download():

    os.system("apt install wget")

    if not os.path.isfile(SPEECH_TR34_PT_MODEL_PATH):
        print("Error: {} not file".format(SPEECH_TR34_PT_MODEL_PATH))

    os.system('pip3 install kapre==0.1.4 -i https://pypi.tuna.tsinghua.edu.cn/simple')


def autodl_nlp_install_download():

    os.system("pip install jieba_fast -i https://pypi.tuna.tsinghua.edu.cn/simple")
    os.system("pip install jieba -i https://pypi.tuna.tsinghua.edu.cn/simple")
    os.system("pip install pathos -i https://pypi.tuna.tsinghua.edu.cn/simple")
    os.system("pip install bpemb -i https://pypi.tuna.tsinghua.edu.cn/simple")
    os.system("pip install keras-radam -i https://pypi.tuna.tsinghua.edu.cn/simple")
    os.system("apt-get install wget")


def autodl_tabular_install_download():

    pass


def autodl_install_download(domain):
    if domain == "image":
        autodl_image_install_download()
    elif domain == "video":
        autodl_video_install_download()
    elif domain == "speech":
        autodl_speech_install_download()
    elif domain == "nlp":
        autodl_nlp_install_download()
    elif domain == "tabular":
        autodl_tabular_install_download()
    else:
        error("Error: domain is {}, can not install_download".format(domain))


