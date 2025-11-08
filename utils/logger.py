#!/usr/bin/python
# -*- encoding: utf-8 -*-


import os.path as osp
import time
import logging

import torch.distributed as dist

import os
import sys
import inspect

from loguru import logger

def setup_logger(name, logpth):
    logfile = '{}-{}.log'.format(name, time.strftime('%Y-%m-%d-%H-%M-%S'))
    logfile = osp.join(logpth, logfile)
    FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
    log_level = logging.INFO
    if dist.is_initialized() and dist.get_rank() != 0:
        log_level = logging.WARNING
    try:
        logging.basicConfig(level=log_level, format=FORMAT, filename=logfile, force=True)
    except Exception:
        logging.basicConfig(level=log_level, format=FORMAT, filename=logfile)
    logging.root.addHandler(logging.StreamHandler())


def print_log_msg(it, max_iter, lr, time_meter, loss_meter, loss_pre_meter,
        loss_aux_meters):
    t_intv, eta = time_meter.get()
    loss_avg, _ = loss_meter.get()
    loss_pre_avg, _ = loss_pre_meter.get()
    loss_aux_avg = ', '.join(['{}: {:.4f}'.format(el.name, el.get()[0]) for el in loss_aux_meters])
    msg = ', '.join([
        'iter: {it}/{max_it}',
        'lr: {lr:4f}',
        'eta: {eta}',
        'time: {time:.2f}',
        'loss: {loss:.4f}',
        'loss_pre: {loss_pre:.4f}',
    ]).format(
        it=it+1,
        max_it=max_iter,
        lr=lr,
        time=t_intv,
        eta=eta,
        loss=loss_avg,
        loss_pre=loss_pre_avg,
        )
    msg += ', ' + loss_aux_avg
    logger = logging.getLogger()
    logger.info(msg)

def get_caller_name(depth = 0):
    frame = inspect.currentframe().f_back

    for _ in range(depth):
        frame = frame.f_back

    return frame.f_globals['__name__']

class StreamToLoguru:
    def __init__(self, level = 'INFO', caller_names = ('apex', 'pycocotools')):
        self.level = level
        self.linebuf = ''
        self.caller_names = caller_names

    def write(self, buf):
        full_name = get_caller_name(depth = 1)
        module_name = full_name.rsplit('.', maxsplit = -1)[0]

        if module_name in self.caller_names:
            for line in buf.rstrip().splitlines():
                logger.opt(depth = 2).log(self.level, line.rstrip())
        else:
            sys.__stdout__.write(buf)

    def flush(self):
        pass

def redirect_sys_output(log_level = 'INFO'):
    redirect_logger = StreamToLoguru(log_level)
    sys.stderr = redirect_logger
    sys.stdout = redirect_logger

def set_logger(save_dir, distributed_rank = 0, filename = 'log.txt', mode = 'a'):
    loguru_format = ('<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>')

    logger.remove()

    save_file = osp.join(save_dir, filename)

    if mode == 'o' and osp.exists(save_file):
        os.remove(save_file)

    if distributed_rank == 0:
        logger.add(sys.stderr, format = loguru_format, level = 'INFO', enqueue = True)
        logger.add(save_file)

    redirect_sys_output('INFO')

    