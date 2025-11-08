"""
Create: 2022.03.07
Author: SG.SUH
Python: 3.7
PyTorch: 1.8
"""

import os
import functools

from torch import distributed as dist
from contextlib import contextmanager

LOCAL_PROCESS_GROUP = None

def get_num_devices():
    gpu_list = os.getenv('CUDA_VISIBLE_DEVICES', None)

    if gpu_list is not None:
        return len((gpu_list.split(',')))
    else:
        devices_list_info = os.popen('nvidia-smi -L')
        devices_list_info = devices_list_info.read().strip().split('\n')

        return len(devices_list_info)

def synchronize():
    if not dist.is_available():
        return

    if not dist.is_initialized():
        return

    world_size = dist.get_world_size()

    if world_size == 1:
        return

    dist.barrier()

def get_world_size() -> int:
    if not dist.is_available():
        return 1

    if not dist.is_initialized():
        return 1

    return dist.get_world_size()

def get_rank() -> int:
    if not dist.is_available():
        return 0

    if not dist.is_initialized():
        return 0

    return dist.get_rank()

def is_main_process() -> bool:
    return get_rank() == 0

def get_local_rank() -> int:
    if not dist.is_available():
        return 0

    if not dist.is_initialized():
        return 0

    assert LOCAL_PROCESS_GROUP is not None

    return dist.get_rank(group = LOCAL_PROCESS_GROUP)

@contextmanager
def wait_for_the_master(local_rank: int):
    if local_rank > 0:
        dist.barrier()

    yield

    if local_rank == 0:
        if not dist.is_available():
            return

        if not dist.is_initialized():
            return
        else:
            dist.barrier()

@functools.lru_cache()
def get_global_gloo_group():
    if dist.get_backend() == 'nccl':
        return dist.new_group(backend = 'gloo')
    else:
        return dist.group.WORLD