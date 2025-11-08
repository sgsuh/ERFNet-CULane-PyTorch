"""
Create: 2021.09.14
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

class_color = [(255, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255)]

class LaneEstDataset(Dataset):
    def __init__(self, dataset_path, data_list, resize_width, resize_height, transform = None):
        super().__init__()

        with open(os.path.join(dataset_path, data_list + '.txt')) as f:
            self.img_list = []
            self.img = []
            self.label_list = []

            for line in f:
                self.img_list.append(dataset_path.replace('list', '') + line.strip().split(' ')[0])
                self.label_list.append(dataset_path.replace('list', '') + line.strip().split(' ')[1])
                
        self.img_path = dataset_path
        self.gt_path = dataset_path
        self.transform = transform

        self.resize_width = resize_width
        self.resize_height = resize_height

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.img_path, self.img_list[idx]))
        color_label = cv2.imread(os.path.join(self.gt_path, self.label_list[idx]), cv2.IMREAD_UNCHANGED)

        label = np.zeros((color_label.shape[0], color_label.shape[1]), dtype = np.uint8)

        exist = np.zeros((len(class_color)), dtype = np.int32)

        is_lane = True

        for i, color in enumerate(class_color):
            y, x = np.where(np.all(color_label == color, axis = -1))
            label[y, x] = i + 1

            left_x1 = 0
            left_y1 = 0
            left_x2 = 0
            left_y2 = 0

            right_x1 = 0
            right_y1 = 0
            right_x2 = 0
            right_y2 = 0

            if y.shape[0] > 10:
                exist[i] = 1

            if i == 1:
                if y.shape[0] > 10:
                    left_y1 = max(y)
                    j = np.argwhere(y == left_y1)
                    left_x1 = np.mean(x[j], dtype = np.int32)

                    left_y2 = min(y)
                    j = np.argwhere(y == left_y2)
                    left_x2 = np.mean(x[j], dtype = np.int32)

                    left_a = float(left_x1 - left_x2) / (left_y1 - left_y2)
                    left_b = left_x1 - (left_a * left_y1)
                else:
                    is_lane = False
            elif i == 2:
                if y.shape[0] > 10:
                    right_y1 = max(y)
                    j = np.argwhere(y == right_y1)
                    right_x1 = np.mean(x[j], dtype = np.int32)

                    right_y2 = min(y)
                    j = np.argwhere(y == right_y2)
                    right_x2 = np.mean(x[j], dtype = np.int32)

                    right_a = float(right_x1 - right_x2) / (right_y1 - right_y2)
                    right_b = right_x1 - (right_a * right_y1)
                else:
                    is_lane = False

        van_y = label.shape[0] // 2
        van_x = label.shape[1] // 2

        if is_lane:
            for y in range(label.shape[0]):
                if y == label.shape[0] - 1:
                    break

                j = label.shape[0] - 1 - y
                x1 = left_a * j + left_b
                x2 = right_a * j + right_b

                if x1 >= x2:
                    van_y = j
                    van_x = int((x1 + x2) / 2)
                    break

        # Crop ROI Image
        image_crop, label_crop = self.crop(image, label, van_x, van_y)

        # Resize Original Image
        image = image[image.shape[0] // 2:, :, :]
        label = label[label.shape[0] // 2:, :]

        image = cv2.resize(image, (self.resize_width, self.resize_height))
        label = cv2.resize(label, (self.resize_width, self.resize_height), interpolation = cv2.INTER_NEAREST)

        image = image.astype(np.float32)
        label = label.squeeze()

        image_crop = image_crop.astype(np.float32)
        label_crop = label_crop.astype(np.float32)

        if self.transform:
            image, label = self.transform((image, label))
            image_crop, label_crop = self.transform((image_crop, label_crop))

            image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
            label = torch.from_numpy(label).contiguous().long()

            image_crop = torch.from_numpy(image_crop).permute(2, 0, 1).contiguous().float()
            label_crop = torch.from_numpy(label_crop).contiguous().long()

        return image, label, image_crop, label_crop, exist

    def crop(self, image, label, van_x, van_y):
        if van_x < image.shape[1] // 2:
            left = max(0, van_x - self.resize_width // 2)
            right = left + self.resize_width
        else:
            right = min(image.shape[1] - 1, van_x + self.resize_width // 2)
            left = right - self.resize_width

        if van_y < image.shape[0] // 2:
            top = max(0, van_y - self.resize_height // 2)
            bottom = top + self.resize_height
        else:
            bottom = min(image.shape[0] - 1, van_y + self.resize_height // 2)
            top = bottom - self.resize_height

        image_crop = image[top:bottom, left:right, :].copy()
        label_crop = label[top:bottom, left:right].copy()

        return image_crop, label_crop

def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path",
                        default="")
    
    return parser.parse_args()

if __name__ == '__main__':
    args = make_parser()
    dataset_path = args.dataset_path
    data_list = 'val_gt'

    dataset = LaneEstDataset(dataset_path, data_list, 960, 270)

    for i, (image, label, crop_image, crop_label, exist) in enumerate(dataset):
        print(i)
