# -*- coding: utf-8 -*-
# @Date    : 2019/8/4 17:01
# @Desc    :
import os
import sys
import time
import logging
from typing import Any
import multiprocessing
from multiprocessing import Pool
from joblib import Parallel, delayed

nesting_level = 0
is_start = None

NCPU = multiprocessing.cpu_count() - 1

color_list = ['black', 'red', 'green', 'yellow', 'blue', 'purple', 'qinglan', 'white']
color_range = range(30, 38)
color_dict = dict(zip(color_list, color_range))

from collections import defaultdict, OrderedDict, Callable
from autodl.utils.log_utils import logger


class DefaultOrderedDict(OrderedDict):
    # Source: http://stackoverflow.com/a/6190500/562769
    def __init__(self, default_factory=None, *a, **kw):
        if (default_factory is not None and
                not isinstance(default_factory, Callable)):
            raise TypeError('first argument must be callable')
        OrderedDict.__init__(self, *a, **kw)
        self.default_factory = default_factory

    def __getitem__(self, key):
        try:
            return OrderedDict.__getitem__(self, key)
        except KeyError:
            return self.__missing__(key)

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = value = self.default_factory()
        return value

    def __reduce__(self):
        if self.default_factory is None:
            args = tuple()
        else:
            args = self.default_factory,
        return type(self), args, None, None, self.items()

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return type(self)(self.default_factory, self)

    def __deepcopy__(self, memo):
        import copy
        return type(self)(self.default_factory,
                          copy.deepcopy(self.items()))

    def __repr__(self):
        return 'OrderedDefaultDict(%s, %s)' % (self.default_factory,
                                               OrderedDict.__repr__(self))


time_records = DefaultOrderedDict(list)


class TimerD:
    def __init__(self):
        self.start = time.time()
        self.history = [self.start]

    def check(self, info):
        current = time.time()
        duration = current - self.history[-1]
        if duration < 0.05:
            pass
        else:
            # log(f"[{info}] spend {duration}sec")
            pass
        self.history.append(current)


def colored(s, color='red'):
    return '\033[1;{0}m{1}\033[0m'.format(color_dict[color], s)


def log(entry: Any):
    global nesting_level
    space = "-" * (4 * nesting_level)
    logger.info("{0}{1}".format(space, entry))


def timeit_factory(oneline=False, end_log=None):
    def timeit(method, start_log=None):
        def timed(*args, **kw):
            global is_start
            global nesting_level

            if not is_start and not oneline:
                print()

            is_start = True
            if not oneline:
                log("Start [{0}]:" + (start_log if start_log else "").format(method.__name__))
            nesting_level += 1

            start_time = time.time()
            result = method(*args, **kw)
            end_time = time.time()

            nesting_level -= 1
            duration = end_time - start_time
            duration_str = '{}'.format(duration)
            # if duration > 10:
            duration_str = colored(duration_str, color='purple')
            time_records[nesting_level].append((method.__name__, duration))
            log("End   [{0}]. Time elapsed: {1} sec.{2}".format(method.__name__,
                                                                duration_str,
                                                                end_log))
            is_start = False

            return result

        return timed

    return timeit


def NCPUP(*args, **kwargs):
    return Parallel(n_jobs=NCPU, verbose=1)(*args, **kwargs)


def NCPUDICT(fn, *args, **kwargs):
    dict = {}
    res = []
    p = Pool(NCPU)
    for i in range(NCPU):
        res.append(p.apply_async(fn, args=(i, kwargs)))
    p.close()
    p.join()
    for i in res:
        dict.update(i.get())
    for k, v in dict.items():
        print(k, v)
    return dict


timeit_endl = timeit_factory(oneline=False, end_log='\n')
timeit = timeit_factory(oneline=False)
timeit_ol = timeit_factory(oneline=True)
