"""
Modify: 2022.03.08
Author: SG.SUH
Python: 3.7
PyTorch: 1.8
"""

import time
import datetime
import functools
import numpy as np
import torch

from collections import defaultdict
from collections import deque

class TimeMeter(object):

    def __init__(self, max_iter):
        self.iter = 0
        self.max_iter = max_iter
        self.st = time.time()
        self.global_st = self.st
        self.curr = self.st

    def update(self):
        self.iter += 1

    def get(self):
        self.curr = time.time()
        interv = self.curr - self.st
        global_interv = self.curr - self.global_st
        eta = int((self.max_iter-self.iter) * (global_interv / (self.iter+1)))
        eta = str(datetime.timedelta(seconds=eta))
        self.st = self.curr
        return interv, eta


class AvgMeter(object):

    def __init__(self, name):
        self.name = name
        self.seq = []
        self.global_seq = []

    def update(self, val):
        self.seq.append(val)
        self.global_seq.append(val)

    def get(self):
        avg = sum(self.seq) / len(self.seq)
        global_avg = sum(self.global_seq) / len(self.global_seq)
        self.seq = []
        return avg, global_avg

class AverageMeter:
    def __init__(self, window_size = 50):
        self.deque = deque(maxlen = window_size)
        self._total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.count += 1
        self._total += value

    @property
    def median(self):
        d = np.array(list(self.deque))

        return np.median(d)

    @property
    def avg(self):
        d = np.array(list(self.deque))

        return d.mean()

    @property
    def global_avg(self):
        return self._total / max(self.count, 1e-5)

    @property
    def latest(self):
        return self.deque[-1] if len(self.deque) > 0 else None

    @property
    def total(self):
        return self._total

    def reset(self):
        self.deque.clear()
        self._total = 0.0
        self.count = 0

    def clear(self):
        self.deque.clear()

class MeterBuffer(defaultdict):
    def __init__(self, window_size = 20):
        factory = functools.partial(AverageMeter, window_size = window_size)
        
        super().__init__(factory)

    def reset(self):
        for v in self.values():
            v.reset()

    def get_filtered_meter(self, filter_key = 'time'):
        return {k: v for k, v in self.items() if filter_key in k}

    def update(self, values = None, **kwargs):
        if values is None:
            values = {}

        values.update(kwargs)

        for k, v in values.items():
            if isinstance(v, torch.Tensor):
                v = v.detach()

            self[k].update(v)

    def clear_meters(self):
        for v in self.values():
            v.clear()

def gpu_mem_usage():
    mem_usage_bytes = torch.cuda.max_memory_allocated()

    return mem_usage_bytes / (1024 * 1024)
