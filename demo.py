"""
Create: 2021.09.07
Author: SG.SUH
Python: 3.7
PyTorch: 1.8
"""

import torch
import numpy as np
import cv2
import torch.nn.functional as F
import time

import LogLoader
import argparse 

from models.erfnet import LaneNet

class_color = [(255, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255)]

def convert_state_dict(file_path):
    checkpoint = torch.load(file_path)
    module_state_dict = checkpoint['state_dict']
    new_state_dict = {}

    for key, val in module_state_dict.items():
        if 'module.' in key:
            new_state_dict[key[7:]] = val

    return module_state_dict

def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bin_file",
                        default="")
    parser.add_argument("--encoder_path",
                        default="")
    parser.add_argument("--est_decoder_path",
                        default="")
    parser.add_argument("--seg_decoder_path",
                        default="")
    
    return parser.parse_args()

def main():
    args = make_parser()

    num_class = [5, 4]
    bin_file = args.bin_file

    model = LaneNet(num_class)

    encoder_path = args.encoder_path
    est_decoder_path = args.est_decoder_path
    seg_decoder_path = args.seg_decoder_path

    encoder = model.encoder
    est_decoder = model.est_decoder
    seg_decoder = model.seg_decoder

    encoder.load_state_dict(torch.load(encoder_path))
    est_decoder.load_state_dict(torch.load(est_decoder_path))
    seg_decoder.load_state_dict(torch.load(seg_decoder_path))

    model.requires_grad_(False)
    model.eval()
    model = model.cuda()

    cap = LogLoader.LogLoader(bin_file, yuv = False)

    mean = np.array((103.939, 116.779, 123.68))
    std = np.array((1.0, 1.0, 1.0))

    cv2.namedWindow('Est', flags = cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Est', 960, 540)

    tick = 0

    while True:
        frame = cap.get_image()

        if frame is None:
            break

        im_height = frame.shape[0]
        im = frame[im_height // 2:, :, :].copy()
        im = cv2.resize(im, (960, 270))
        im_pad = cv2.copyMakeBorder(im, 0, 2, 0, 0, cv2.BORDER_CONSTANT)

        input = torch.from_numpy(im_pad).cuda().unsqueeze(0).float()
        

        start = time.time()
        est_output, seg_output = model(input)
        end = time.time()

        print(end - start)

        est_pred = est_output[0].cpu().numpy()
        seg_pred = seg_output[0].cpu().numpy()

        est_pred[est_pred < 0.5] = 0
        idx_max = np.argmax(est_pred, axis = 2).astype(np.int32)

        im[idx_max == 1] = class_color[0]
        im[idx_max == 2] = class_color[1]
        im[idx_max == 3] = class_color[2]
        im[idx_max == 4] = class_color[3]

        cv2.imshow('Est', im)

        cv2.waitKey(1)

if __name__ == '__main__':
    main()