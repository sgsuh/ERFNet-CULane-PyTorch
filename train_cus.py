"""
Create: 2022.07.18
Author: SG.SUH
Python: 3.7
PyTorch: 1.8
"""

import argparse
import torch.backends.cudnn as cudnn

from loguru import logger

from utils.setup_env import configure_nccl
from utils.setup_env import configure_omp
from utils.dist import get_num_devices
from utils.launch import launch

from trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_height', default = 270, type = int)
    parser.add_argument('--img_width', default = 960, type = int)
    parser.add_argument('--epochs', default = 200, type = int)
    parser.add_argument('--batch_size', default = 8, type = int)
    parser.add_argument('--encoder_path', default = 'weight/210927_erfnet_encoder.pth', type = str)
    parser.add_argument('--est_decoder_path', default = 'weight/210927_erfnet_est_decoder.pth', type = str)
    parser.add_argument('--seg_decoder_path', default = 'weight/210914_erfnet_seg_decoder.pth', type = str)

    parser.add_argument('--est_root', default = '/disk2/ld_est/list', type = str)
    parser.add_argument('--seg_root', default = '/disk2/ld_seg/list', type = str)
    parser.add_argument('--train_list', default = 'train_gt', type = str)
    parser.add_argument('--val_list', default = 'val_gt', type = str)

    return parser.parse_args()

@logger.catch
def main(args):
    configure_nccl()
    configure_omp()

    cudnn.benchmark = True

    trainer = Trainer(args)
    trainer.train()

if __name__ == '__main__':
    args = parse_args()

    num_gpu = get_num_devices()

    launch(main, num_gpu, 1, 0, backend = 'nccl', dist_url = 'auto', args = (args,))