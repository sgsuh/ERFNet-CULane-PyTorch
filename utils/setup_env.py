"""
Create: 2022.03.07
Author: SG.SUH
Python: 3.7
PyTorch: 1.8
"""

import os
import subprocess

from loguru import logger

from utils.dist import get_world_size
from utils.dist import is_main_process

def configure_nccl():
    os.environ['NCCL_LAUNCH_MODE'] = 'PARALLEL'
    os.environ['NCCL_IB_HCA'] = subprocess.getoutput(
        "pushd /sys/class/infiniband/ > /dev/null; for i in mlx5_*; "
        "do cat $i/ports/1/gid_attrs/types/* 2>/dev/null "
        "| grep v >/dev/null && echo $i ; done; popd > /dev/null"
    )
    os.environ['NCCL_IB_GID_INDEX'] = '3'
    os.environ['NCCL_IB_TC'] = '106'

def configure_omp(num_threads = 1):
    if 'OMP_NUM_THREADS' not in os.environ and get_world_size() > 1:
        os.environ['OMP_NUM_THREADS'] = str(num_threads)

        if is_main_process():
            logger.info('we set OMP_NUM_THREADS for each process to {} to speed up.\nplease further tune the variable for optimal performance\n'.format(os.environ['OMP_NUM_THREADS']))