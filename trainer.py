"""
Create: 2022.07.18
Author: SG.SUH
Python: 3.7
PyTorch: 1.8
"""

import torch
import os
import torchvision
import time
import datetime
import numpy as np

from loguru import logger
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import transforms as tf
from utils.dist import get_local_rank
from utils.dist import get_world_size
from utils.dist import get_rank
from utils.dist import synchronize
from utils.lr_scheduler import WarmupPolyLrScheduler
from utils.meters import MeterBuffer, gpu_mem_usage
from utils.logger import set_logger
from utils.sampler import InfiniteSampler
from utils.data_prefetcher import DataPrefetcher

from models.erfnet import ERFNet

from dataset.lane_est import LaneEstDataset
from dataset.lane_seg import LaneSegDataset

class Trainer:
    def __init__(self, args):
        self.args = args
        self.local_rank = get_local_rank()
        self.device = 'cuda:{}'.format(self.local_rank)
        self.is_distributed = get_world_size() > 1
        self.start_epoch = 0
        self.max_epoch = args.epochs
        self.data_type = torch.float32
        self.scaler = torch.cuda.amp.GradScaler(enabled = False)
        self.print_interval = 10
        self.est_meter = MeterBuffer(window_size = self.print_interval)
        self.seg_meter = MeterBuffer(window_size = self.print_interval)
        self.input_size = (args.img_width, args.img_height)
        self.rank = get_rank()
        self.file_name = os.path.join('trained', 'erfnet')
        self.max_miou = 0
        self.best_epoch = 0

        if self.rank == 0:
            os.makedirs(self.file_name, exist_ok = True)

        set_logger(self.file_name, distributed_rank = self.rank, filename = 'train_log.txt', mode = 'a')

    def set_model(self):
        est_net = ERFNet(5)
        seg_net = ERFNet(4)

        try:
            est_net.encoder.load_state_dict(torch.load(self.args.encoder_path, map_location = self.device), strict = False)
            est_net.decoder.load_state_dict(torch.load(self.args.est_decoder_path, map_location = self.device), strict = False)
        except RuntimeError as e:
            logger.warning('Ignoring {}'.format(e))

        try:
            seg_net.encoder.load_state_dict(torch.load(self.args.encoder_path, map_location = self.device), strict = False)
            seg_net.decoder.load_state_dict(torch.load(self.args.seg_decoder_path, map_location = self.device), strict = False)
        except RuntimeError as e:
            logger.warning('Ignoring {}'.format(e))

        weights = [1.0 for _ in range(5)]
        weights[0] = 0.4
        class_weights = torch.FloatTensor(weights).to(self.device)
        criteria = torch.nn.NLLLoss(ignore_index = 255, weight = class_weights).to(self.device)

        return est_net, seg_net, criteria

    def set_optimizer(self):
        est_total_optim = torch.optim.SGD(self.est_net.parameters(), lr = 0.02, momentum = 0.9, weight_decay = 1e-4)
        est_decoder_optim = torch.optim.SGD(self.est_net.decoder.parameters(), lr = 0.02, momentum = 0.9, weight_decay = 1e-4)
        seg_total_optim = torch.optim.SGD(self.seg_net.parameters(), lr = 0.02, momentum = 0.9, weight_decay = 1e-4)
        seg_decoder_optim = torch.optim.SGD(self.seg_net.decoder.parameters(), lr = 0.02, momentum = 0.9, weight_decay = 1e-4)

        return est_total_optim, est_decoder_optim, seg_total_optim, seg_decoder_optim

    def get_data_loader(self, mode = 'train'):
        if mode == 'train':
            est_transform = torchvision.transforms.Compose([tf.GroupNormalize(mean = (self.est_net.input_mean, (0,)), std = (self.est_net.input_std, (1,)))])
            seg_transform = torchvision.transforms.Compose([tf.GroupRandomHorizontalFlip(), tf.GroupNormalize(mean = (self.seg_net.input_mean, (0,)), std = (self.seg_net.input_std, (1,)))])
            shuffle = True
            drop_last = True
            list_name = self.args.train_list
        else:
            est_transform = torchvision.transforms.Compose([tf.GroupNormalize(mean = (self.est_net.input_mean, (0,)), std = (self.est_net.input_std, (1,)))])
            seg_transform = torchvision.transforms.Compose([tf.GroupNormalize(mean = (self.seg_net.input_mean, (0,)), std = (self.seg_net.input_std, (1,)))])
            shuffle = False
            drop_last = False
            list_name = self.args.val_list

        est_dataset = LaneEstDataset(self.args.est_root, list_name, self.args.img_width, self.args.img_height, transform = est_transform)
        seg_dataset = LaneSegDataset(self.args.seg_root, list_name, self.args.img_width, self.args.img_height, transform = seg_transform)

        if self.is_distributed:
            if mode == 'train':
                est_sampler = InfiniteSampler(len(est_dataset), shuffle = shuffle, seed = 0)
                seg_sampler = InfiniteSampler(len(seg_dataset), shuffle = shuffle, seed = 0)
            else:
                est_sampler = torch.utils.data.distributed.DistributedSampler(est_dataset, shuffle = shuffle)
                seg_sampler = torch.utils.data.distributed.DistributedSampler(seg_dataset, shuffle = shuffle)
            
            est_batchsampler = torch.utils.data.sampler.BatchSampler(est_sampler, self.args.batch_size, drop_last = drop_last)
            seg_batchsampler = torch.utils.data.sampler.BatchSampler(seg_sampler, self.args.batch_size, drop_last = drop_last)

            est_dataloader = DataLoader(est_dataset, batch_sampler = est_batchsampler, num_workers = 4, pin_memory = True)
            seg_dataloader = DataLoader(seg_dataset, batch_sampler = seg_batchsampler, num_workers = 4, pin_memory = True)
        else:
            est_dataloader = DataLoader(est_dataset, batch_size = self.args.batch_size, shuffle = shuffle, drop_last = drop_last, num_workers = 4, pin_memory = True)
            seg_dataloader = DataLoader(seg_dataset, batch_size = self.args.batch_size, shuffle = shuffle, drop_last = drop_last, num_workers = 4, pin_memory = True)

        return est_dataloader, seg_dataloader

    def before_train(self):
        torch.cuda.set_device(self.local_rank)

        self.est_net, self.seg_net, self.criteria = self.set_model()

        self.est_net.to(self.device)
        self.seg_net.to(self.device)

        self.est_total_optim, self.est_decoder_optim, self.seg_total_optim, self.seg_decoder_optim = self.set_optimizer()

        self.est_train_loader, self.seg_train_loader = self.get_data_loader(mode = 'train')
        self.est_val_loader, self.seg_val_loader = self.get_data_loader(mode = 'val')

        logger.info('init prefetcher, this might take one minutes or less...')

        self.est_prefetcher = DataPrefetcher(self.est_train_loader)
        self.seg_prefetcher = DataPrefetcher(self.seg_train_loader)

        self.est_max_iter = len(self.est_train_loader)
        self.seg_max_iter = len(self.seg_train_loader)

        est_total_iter = self.max_epoch * self.est_max_iter
        seg_total_iter = self.max_epoch * self.seg_max_iter

        self.est_total_scheduler = WarmupPolyLrScheduler(self.est_total_optim, power = 0.9, max_iter = est_total_iter // 2, warmup_iter = 1000, warmup_ratio = 0.1, warmup = 'exp', last_epoch = -1)
        self.est_decoder_scheduler = WarmupPolyLrScheduler(self.est_decoder_optim, power = 0.9, max_iter = est_total_iter // 2, warmup_iter = 1000, warmup_ratio = 0.1, warmup = 'exp', last_epoch = -1)

        self.seg_total_scheduler = WarmupPolyLrScheduler(self.seg_total_optim, power = 0.9, max_iter = seg_total_iter // 2, warmup_iter = 1000, warmup_ratio = 0.1, warmup = 'exp', last_epoch = -1)
        self.seg_decoder_scheduler = WarmupPolyLrScheduler(self.seg_decoder_optim, power = 0.9, max_iter = seg_total_iter // 2, warmup_iter = 1000, warmup_ratio = 0.1, warmup = 'exp', last_epoch = -1)

        if self.is_distributed:
            self.est_net = DDP(self.est_net, device_ids = [self.local_rank], broadcast_buffers = False)
            self.seg_net = DDP(self.seg_net, device_ids = [self.local_rank], broadcast_buffers = False)

        self.est_net.train()
        self.seg_net.train()

        logger.info('training start...')

    def before_epoch(self):
        logger.info('--> start train epoch{}'.format(self.epoch + 1))

    def before_iter(self):
        pass

    def train_est_iter(self, optim, scheduler):
        iter_start_time = time.time()

        inps, targets = self.est_prefetcher.next()

        inps = inps.to(self.data_type)
        targets.requires_grad = False
        targets = torch.squeeze(targets, 1)

        data_end_time = time.time()

        with torch.cuda.amp.autocast(enabled = False):
            outputs = self.est_net(inps)
            loss = self.criteria(outputs, targets)

        optim.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(optim)
        self.scaler.update()

        scheduler.step()

        lr = scheduler.get_lr()

        iter_end_time = time.time()

        self.est_meter.update(iter_time = iter_end_time - iter_start_time, data_time = data_end_time - iter_start_time, lr = lr[0], loss = loss)

    def train_seg_iter(self, optim, scheduler):
        iter_start_time = time.time()

        inps, targets = self.seg_prefetcher.next()

        inps = inps.to(self.data_type)
        targets.requires_grad = False
        targets = torch.squeeze(targets, 1)

        data_end_time = time.time()

        with torch.cuda.amp.autocast(enabled = False):
            outputs = self.seg_net(inps)
            loss = self.criteria(outputs, targets)

        optim.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(optim)
        self.scaler.update()

        scheduler.step()

        lr = scheduler.get_lr()

        iter_end_time = time.time()

        self.seg_meter.update(iter_time = iter_end_time - iter_start_time, data_time = data_end_time - iter_start_time, lr = lr[0], loss = loss)
    
    @property
    def progress_in_est_iter(self):
        return self.epoch * self.est_max_iter + self.est_iter

    def after_est_iter(self):
        if (self.est_iter + 1) % self.print_interval == 0:
            left_iters = self.est_max_iter * self.max_epoch - (self.progress_in_est_iter + 1)
            eta_seconds = self.est_meter['iter_time'].global_avg * left_iters
            eta_str = 'ETA: {}'.format(datetime.timedelta(seconds = int(eta_seconds)))

            progress_str = 'epoch: {}/{}, est iter: {}/{}'.format(self.epoch + 1, self.max_epoch, self.est_iter + 1, self.est_max_iter)

            loss_meter = self.est_meter.get_filtered_meter('loss')
            loss_str = ', '.join(['{}: {:.1f}'.format(k, v.latest) for k, v in loss_meter.items()])

            time_meter = self.est_meter.get_filtered_meter('time')
            time_str = ', '.join(['{}: {:.3f}s'.format(k, v.avg) for k, v in time_meter.items()])

            logger.info('{}, mem: {:.0f}Mb, {}, {}, lr: {:.3e}'.format(progress_str, gpu_mem_usage(), time_str, loss_str, self.est_meter['lr'].latest) + (', size: {:d}, {}'.format(self.input_size[0], eta_str)))

            self.est_meter.clear_meters()

    @property
    def progress_in_seg_iter(self):
        return self.epoch * self.seg_max_iter + self.seg_iter

    def after_seg_iter(self):
        if (self.seg_iter + 1) % self.print_interval == 0:
            left_iters = self.seg_max_iter * self.max_epoch - (self.progress_in_seg_iter + 1)
            eta_seconds = self.seg_meter['iter_time'].global_avg * left_iters
            eta_str = 'ETA: {}'.format(datetime.timedelta(seconds = int(eta_seconds)))

            progress_str = 'epoch: {}/{}, seg iter: {}/{}'.format(self.epoch + 1, self.max_epoch, self.seg_iter + 1, self.seg_max_iter)

            loss_meter = self.seg_meter.get_filtered_meter('loss')
            loss_str = ', '.join(['{}: {:.1f}'.format(k, v.latest) for k, v in loss_meter.items()])

            time_meter = self.seg_meter.get_filtered_meter('time')
            time_str = ', '.join(['{}: {:.3f}s'.format(k, v.avg) for k, v in time_meter.items()])

            logger.info('{}, mem: {:.0f}Mb, {}, {}, lr: {:.3e}'.format(progress_str, gpu_mem_usage(), time_str, loss_str, self.seg_meter['lr'].latest) + (', size: {:d}, {}'.format(self.input_size[0], eta_str)))

            self.seg_meter.clear_meters()

    def train_odd_iter(self):
        for self.est_iter in range(self.est_max_iter):
            self.before_iter()
            self.train_est_iter(self.est_total_optim, self.est_total_scheduler)
            self.after_est_iter()

        self.seg_net.module.encoder = self.est_net.module.encoder

        for self.seg_iter in range(self.seg_max_iter):
            self.before_iter()
            self.train_seg_iter(self.seg_decoder_optim, self.seg_decoder_scheduler)
            self.after_seg_iter()

    def train_even_iter(self):
        for self.seg_iter in range(self.seg_max_iter):
            self.before_iter()
            self.train_seg_iter(self.seg_total_optim, self.seg_total_scheduler)
            self.after_seg_iter()

        self.est_net.module.encoder = self.seg_net.module.encoder

        for self.est_iter in range(self.est_max_iter):
            self.before_iter()
            self.train_est_iter(self.est_decoder_optim, self.est_decoder_scheduler)
            self.after_est_iter()

    def save_ckpt(self, ckpt_name, best_ckpt = False):
        if self.rank == 0:
            logger.info('save weights to {}'.format(self.file))
        
            if not os.path.exists(self.file_name):
                os.makedirs(self.file_name)

            if best_ckpt:
                encoder_path = os.path.join(self.file_name, ckpt_name + '_encoder_best.pth')
                est_decoder_path = os.path.join(self.file_name, ckpt_name + '_est_decoder_best.pth')
                seg_decoder_path = os.path.join(self.file_name, ckpt_name + '_seg_decoder_best.pth')
            else:
                encoder_path = os.path.join(self.file_name, ckpt_name + '_encoder.pth')
                est_decoder_path = os.path.join(self.file_name, ckpt_name + '_est_decoder.pth')
                seg_decoder_path = os.path.join(self.file_name, ckpt_name + '_seg_decoder.pth')

            torch.save(self.est_net.module.encoder.state_dict(), encoder_path)
            torch.save(self.est_net.module.decoder.state_dict(), est_decoder_path)
            torch.save(self.seg_net.module.decoder.state_dict(), seg_decoder_path)

    def after_epoch(self):
        save_name = '{:03d}_epoch'.format(self.epoch + 1)

        est_val_iter = len(self.est_val_loader)
        seg_val_iter = len(self.seg_val_loader)

        est_prefetcher = DataPrefetcher(self.est_val_loader)
        seg_prefetcher = DataPrefetcher(self.seg_val_loader)

        self.est_net.eval()
        self.seg_net.eval()

        est_hist = torch.zeros(5, 5).cuda().detach()

        for i in range(est_val_iter):
            inps, targets = est_prefetcher.next()
            inps.to(self.data_type)
            targets.requires_grad = False
            targets = torch.squeeze(targets, 1)
            N, H, W = targets.shape
            probs = torch.zeros((N, 5, H, W), dtype = torch.float32).cuda().detach()

            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled = False):
                    outputs = self.est_net(inps)
                    loss = self.criteria(outputs, targets)
                    probs += torch.softmax(outputs, dim = 1)
                    preds = torch.argmax(probs, dim = 1)
                    keep = targets != 255

            est_hist += torch.bincount(targets[keep] * 5 + preds[keep], minlength = 5 ** 2).view(5, 5)
        
            if i % self.print_interval == 0:
                progress_str = 'est evaluation, iter: {}/{}'.format(i + 1, est_val_iter)
                loss_str = 'loss: {:.4f}'.format(float(loss.cpu()))

                logger.info('{}, {}'.format(progress_str, loss_str))

        ious = est_hist.diag() / (est_hist.sum(dim = 0) + est_hist.sum(dim = 1) - est_hist.diag())
        est_miou = np.nanmean(ious.detach().cpu().numpy())

        seg_hist = torch.zeros(4, 4).cuda().detach()

        for i in range(seg_val_iter):
            inps, targets = seg_prefetcher.next()
            inps.to(self.data_type)
            targets.requires_grad = False
            targets = torch.squeeze(targets, 1)
            N, H, W = targets.shape
            probs = torch.zeros((N, 4, H, W), dtype = torch.float32).cuda().detach()

            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled = False):
                    outputs = self.seg_net(inps)
                    loss = self.criteria(outputs, targets)
                    probs += torch.softmax(outputs, dim = 1)
                    preds = torch.argmax(probs, dim = 1)
                    keep = targets != 255

            seg_hist += torch.bincount(targets[keep] * 4 + preds[keep], minlength = 4 ** 2).view(4, 4)
        
            if i % self.print_interval == 0:
                progress_str = 'est evaluation, iter: {}/{}'.format(i + 1, seg_val_iter)
                loss_str = 'loss: {:.4f}'.format(float(loss.cpu()))

                logger.info('{}, {}'.format(progress_str, loss_str))

        ious = seg_hist.diag() / (seg_hist.sum(dim = 0) + seg_hist.sum(dim = 1) - seg_hist.diag())
        seg_miou = np.nanmean(ious.detach().cpu().numpy())

        self.est_net.train()
        self.seg_net.train()

        synchronize()        
    
        logger.info('{} epoch, est miou: {}, seg miou: {}'.format(self.epoch + 1, est_miou, seg_miou))

        total_miou = (est_miou + seg_miou) / 2

        if total_miou > self.max_miou:
            self.save_ckpt(ckpt_name = save_name, best_ckpt = True)
            self.max_miou = total_miou
            self.best_epoch = self.epoch + 1
        else:
            self.save_ckpt(ckpt_name = save_name)

        logger.info('best epoch: {}, best miou: {}'.format(self.best_epoch, self.max_miou))

    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()

            if ((self.epoch + 1) // 10) % 2 == 0:
                self.train_even_iter()
            else:
                self.train_odd_iter()

            self.after_epoch()

    def after_train(self):
        pass

    def train(self):
        self.before_train()

        try:
            self.train_in_epoch()
        except Exception:
            raise
        finally:
            self.after_train()