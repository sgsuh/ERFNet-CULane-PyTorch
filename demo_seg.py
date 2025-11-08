"""
Create: 2021.09.13
Author: SG.SUH
Python: 3.7
PyTorch: 1.8
"""

import os
import glob
import torch
import numpy as np
import cv2
import torch.nn.functional as F
import argparse

from models.erfnet import ERFNet

def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--val_root_path",
                        default="")
    parser.add_argument("--seg_encoder_path",
                        default="")
    parser.add_argument("--seg_decoder_path",
                        default="")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = make_parser()

    val_root_path = args.val_root_path
    seg_encoder_path = args.seg_encoder_path
    seg_decoder_path = args.seg_decoder_path

    num_class = 4
    model = ERFNet(num_class)
    encoder = model.encoder
    decoder = model.decoder

    encoder.load_state_dict(torch.load(seg_encoder_path)['state_dict'])
    decoder.load_state_dict(torch.load(seg_decoder_path)['state_dict'])

    model.requires_grad_(False)
    model.eval()
    model = model.cuda()

    mean = np.array((103.939, 116.779, 123.68))
    std = np.array((1.0, 1.0, 1.0))

    sub_fold1_list = glob.glob(os.path.join(val_root_path, '*'))
    sub_fold1_list.sort()

    cv2.namedWindow('Seg', flags = cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Seg', 960, 540)

    for sub_fold1_path in sub_fold1_list:
        if not os.path.isdir(sub_fold1_path):
            continue

        sub_fold2_list = glob.glob(os.path.join(sub_fold1_path, '*'))
        sub_fold2_list.sort()

        for sub_fold2_path in sub_fold2_list:
            if not os.path.isdir(sub_fold2_path):
                continue

            img_file_list = glob.glob(os.path.join(sub_fold2_path, '*.jpg'))
            img_file_list.sort()

            for img_file_path in img_file_list:
                frame = cv2.imread(img_file_path)

                im_height = frame.shape[0]
                im = frame[im_height // 2:, :, :].copy()
                im = cv2.resize(im, (640, 180)).astype(float)
                im = (im - mean) / std

                input = torch.from_numpy(im).cuda()
                input = input.permute((2, 0, 1)).unsqueeze(0).float()

                seg_output, _ = model(input)

                seg_output = F.softmax(seg_output, dim = 1)
                seg_output = seg_output.data.cpu().numpy()
                seg_pred = seg_output[0].transpose((1, 2, 0))

                pred_bg = (seg_pred[:, :, 0] > 0.5).astype(np.uint8)
                pred_line1 = (seg_pred[:, :, 1] > 0.7).astype(np.uint8)
                pred_line2 = (seg_pred[:, :, 2] > 0.7).astype(np.uint8)
                pred_cross = (seg_pred[:, :, 3] > 0.5).astype(np.uint8)

                seg_line1 = (pred_line1 * 255).astype(np.uint8)
                seg_line2 = (pred_line2 * 255).astype(np.uint8)
                seg_cross = (pred_cross * 255).astype(np.uint8)
                zeros = np.zeros_like(seg_line1)

                disp_img_seg = np.stack([seg_line1, seg_cross, seg_line2], axis = 2)
                disp_img_seg = cv2.resize(disp_img_seg, (640, 180))
                disp_img_seg = cv2.copyMakeBorder(disp_img_seg, 180, 0, 0, 0, cv2.BORDER_CONSTANT, 0)
                disp_img_seg = cv2.resize(disp_img_seg, (1920, 1080))

                show_seg_img = (disp_img_seg).astype(np.uint8)

                cv2.imshow('Seg', show_seg_img)
                cv2.waitKey(0)