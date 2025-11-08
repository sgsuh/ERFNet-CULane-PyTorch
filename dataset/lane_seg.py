"""
Create: 2021.09.10
Author: SG.SUH
Python: 3.7
PyTorch: 1.8
"""

import os
import numpy as np
import cv2
import torch
import argparse

from torch.utils.data import Dataset

class LaneSegDataset(Dataset):
    def __init__(self, dataset_path, data_list, resize_width, resize_height, transform = None):
        super().__init__()

        with open(os.path.join(dataset_path, data_list + '.txt')) as f:
            self.img_list = []
            self.img = []
            self.label_list = []
            self.exist_list = []

            for line in f:
                self.img.append(line.strip().split(' ')[0])
                self.img_list.append(dataset_path.replace('/list', '') + line.strip().split(' ')[0])
                self.label_list.append(dataset_path.replace('/list', '') + line.strip().split(' ')[1])
                self.exist_list.append(np.array([int(line.strip().split(' ')[2]), int(line.strip().split(' ')[3]), int(line.strip().split(' ')[4])]))

        self.img_path = dataset_path
        self.gt_path = dataset_path
        self.transform = transform
        self.is_testing = data_list == 'test_img'

        self.resize_width = resize_width
        self.resize_height = resize_height

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.img_path, self.img_list[idx]))
        label = cv2.imread(os.path.join(self.gt_path, self.label_list[idx]), cv2.IMREAD_UNCHANGED)
        exist = self.exist_list[idx]

        image_height = image.shape[0]

        image = image[image_height // 2:, :, :]
        label = label[image_height // 2:, :]

        image = cv2.resize(image, (self.resize_width, self.resize_height))
        label = cv2.resize(label, (self.resize_width, self.resize_height), interpolation = cv2.INTER_NEAREST)

        image = image.astype(np.float32)
        label = label.squeeze()

        if self.transform:
            image, label = self.transform((image, label))

            image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
            label = torch.from_numpy(label).contiguous().long()

        if self.is_testing:
            return image, label, self.img[idx]
        else:
            return image, label, exist

def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path",
                        default="")
    
    return parser.parse_args()

if __name__ == '__main__':
    args = make_parser()
    dataset_path = args.dataset_path
    data_list = 'val_gt'

    dataset = LaneSegDataset(dataset_path, data_list, 960, 272)

    for i, (image, label, exist) in enumerate(dataset):
        print(os.path.join(dataset.img_path, dataset.img_list[i]))