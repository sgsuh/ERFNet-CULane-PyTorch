"""
Create: 2021.09.10
Author: SG.SUH
Python: 3.7
PyTorch: 1.8
"""

import os
import torch
import torch.backends.cudnn as cudnn
import torchvision
import numpy as np
import time
import shutil

import utils.transforms as tf

from options.options import parser
from models.erfnet import ERFNet
from models.erfnet import Encoder
from dataset.lane_seg import LaneSegDataset

class EvalSegmentation(object):
    def __init__(self, num_class, ignore_label = None):
        self.num_class = num_class
        self.ignore_label = ignore_label

    def __call__(self, pred, gt):
        assert pred.shape == gt.shape

        gt = gt.flatten().astype(int)
        pred = pred.flatten().astype(int)
        locs = (gt != self.ignore_label)
        sumim = gt + pred * self.num_class
        hs = np.bincount(sumim[locs], minlength = self.num_class ** 2).reshape(self.num_class, self.num_class)
        
        return hs

class AverageMeter(object):
    # Computes and Stores the Average and Current Value
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def update(self, val, n = 1):
        if self.val is None:
            self.val = val
            self.sum = val * n
            self.count = n
            self.avg = self.sum / self.count
        else:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

def adjust_learning_rate(args, optimizer, epoch, lr_steps):
    # Sets the Learning Rate to the Initial LR Decayed by 10 every 30 Epochs
    decay = ((1 - float(epoch) / args.epochs) ** 0.9)
    lr = args.lr * decay
    decay = args.weight_decay

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['weight_decay'] = decay

def train(args, train_loader, model, criterion, criterion_exist, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # Switch to Train Mode
    model.train()

    end = time.time()

    for i, (input, target, target_exist) in enumerate(train_loader):
        # Measure Data Loading Time
        data_time.update(time.time() - end)

        target = target.cuda()
        target_exist = target_exist.float().cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        target_exist_var = torch.autograd.Variable(target_exist)

        # Compute Output
        output, output_exist = model(input_var)     # output_mid
        loss = criterion(torch.nn.functional.log_softmax(output, dim = 1), target_var)

        # Measure Accuracy and Record Loss
        losses.update(loss.data.item(), input.size(0))

        # Compute Gradient and Do SGD Step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure Elapsed Time
        batch_time.update(time.time() - end)
        
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}], LR: {lr:.7f}\t' 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' 'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' 'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader), batch_time = batch_time, data_time = data_time, loss = losses, lr = optimizer.param_groups[-1]['lr']))

            batch_time.reset()
            data_time.reset()
            losses.reset()

    return model

def validate(args, val_loader, model, criterion, iter, evaluator, best_mIoU, logger = None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    IoU = AverageMeter()

    # Switch to Evaluate Mode
    model.eval()

    end = time.time()

    for i, (input, target, target_exist) in enumerate(val_loader):
        with torch.no_grad():
            input = input.cuda()
            target = target.cuda()
            
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # Compute Output
            output, _ = model(input_var)

        loss = criterion(torch.nn.functional.log_softmax(output, dim = 1), target_var)

        # Measure Accuracy and Record Loss
        pred = output.data.cpu().numpy().transpose(0, 2, 3, 1)
        pred = np.argmax(pred, axis = 3).astype(np.uint8)
        
        IoU.update(evaluator(pred, target.cpu().numpy()))
        losses.update(loss.data.item(), input.size(0))

        # Measure Elapsed Time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            acc = np.sum(np.diag(IoU.sum)) / float(np.sum(IoU.sum))
            mIoU = np.diag(IoU.sum) / (1e-20 + IoU.sum.sum(1) + IoU.sum.sum(0) - np.diag(IoU.sum))
            mIoU = np.sum(mIoU) / len(mIoU)

            print('Test: [{0}/{1}]\t' 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' 'Loss {loss.val:.4f} ({loss.avg:.4f})\t' 'Pixels Acc {acc:.3f}\t' 'mIoU {mIoU:.3f}'.format(i, len(val_loader), batch_time = batch_time, loss = losses, acc = acc, mIoU = mIoU))
    
    acc = np.sum(np.diag(IoU.sum)) / float(np.sum(IoU.sum))
    mIoU = np.diag(IoU.sum) / (1e-20 + IoU.sum.sum(1) + IoU.sum.sum(0) - np.diag(IoU.sum))
    mIoU = np.sum(mIoU) / len(mIoU)
    val_loss = losses.avg

    print('Testing Results: Pixels Acc {acc:.3f}\tmIoU {mIoU:.3f} ({bestmIoU:.4f})\tLoss {loss.avg:.5f}'.format(acc = acc, mIoU = mIoU, bestmIoU = max(mIoU, best_mIoU), loss = losses))
    
    return mIoU, val_loss

def save_checkpoint(args, state, is_best, filename = ''):
    if not os.path.exists('trained'):
        os.makedirs('trained')

    epoch = state['epoch']

    save_name = os.path.join('trained', ''.join(('{:04d}'.format(epoch), '_', args.method.lower(), '_', filename, '.pth')))

    torch.save(state, save_name)

    if is_best:
        best_name = os.path.join('trained', ''.join(('{:04d}'.format(epoch), '_', args.method.lower(), '_', filename, '_best.pth')))

        shutil.copyfile(save_name, best_name)

def main():
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(gpu) for gpu in args.gpus)
    args.gpus = len(args.gpus)

    if args.dataset == 'SKT_HD':
        num_class = 4
        ignore_label = 255
    else:
        raise ValueError('Unknown Dataset ' + args.dataset)

    encoder = Encoder(num_class + 1)
    encoder.load_state_dict(torch.load(args.encoder)['state_dict'])

    model = ERFNet(num_class, encoder = encoder)

    model.decoder.load_state_dict(torch.load(args.seg_decoder)['state_dict'], strict = False)

    input_mean = model.input_mean
    input_std = model.input_std
    model = torch.nn.DataParallel(model, device_ids = range(args.gpus)).cuda()

    cudnn.benchmark = True
    cudnn.fastest = True

    # Data Loading Code
    data_path = args.data_path

    train_loader = torch.utils.data.DataLoader(LaneSegDataset(data_path, 
                                                              args.train_list, 
                                                              args.img_width, 
                                                              args.img_height, 
                                                              transform = torchvision.transforms.Compose([tf.GroupRandomHorizontalFlip(), 
                                                                                                          tf.GroupNormalize(mean = (input_mean, (0, )), 
                                                                                                                            std = (input_std, (1, )))])), 
                                                batch_size = args.batch_size, 
                                                shuffle = True, 
                                                num_workers = args.workers, 
                                                pin_memory = False, 
                                                drop_last = True)

    val_loader = torch.utils.data.DataLoader(LaneSegDataset(data_path, 
                                                            args.val_list, 
                                                            args.img_width, 
                                                            args.img_height, 
                                                            transform = torchvision.transforms.Compose([tf.GroupNormalize(mean = (input_mean, (0, )), 
                                                                                                                          std = (input_std, (1, )))])), 
                                            batch_size = args.batch_size, 
                                            shuffle = False, 
                                            num_workers = args.workers, 
                                            pin_memory = False, 
                                            drop_last = True)

    # Define Loss Function (Criterion) Optimizer and Evaluator
    weights = [1.0 for _ in range(num_class)]
    weights[0] = 0.4
    class_weights = torch.FloatTensor(weights).cuda()
    criterion = torch.nn.NLLLoss(ignore_index = ignore_label, weight = class_weights).cuda()
    criterion_exist = torch.nn.BCEWithLogitsLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum = args.momentum, weight_decay = args.weight_decay)
    evaluator = EvalSegmentation(num_class, ignore_label)

    best_mIoU = 0

    args.evaluate = False

    if args.evaluate:
        _ = validate(args, val_loader, model, criterion, 0, evaluator, 0)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 10, verbose = True)

    for epoch in range(args.epochs):
        # Train for One Epoch
        model = train(args, train_loader, model, criterion, criterion_exist, optimizer, epoch)

        # Evaluate on Validation Set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            mIoU, val_loss = validate(args, val_loader, model, criterion, (epoch + 1) * len(train_loader), evaluator, best_mIoU)

            # Remember best mIoU and Save Checkpoint
            is_best = mIoU > best_mIoU
            best_mIoU = max(mIoU, best_mIoU)
            decoder = model.module.decoder

            save_checkpoint(args, {'epoch': epoch + 1, 'arch': args.arch, 'state_dict': decoder.state_dict(), 'best_mIoU': best_mIoU}, is_best, 'seg_decoder')

            scheduler.step(val_loss)

if __name__ == '__main__':
    main()