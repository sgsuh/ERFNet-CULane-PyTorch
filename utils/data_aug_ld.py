import cv2
import numpy as np
import torch

class ToTensor(object):
    def __call__(self, data):
        image, line_map, line_exist = data

        image = torch.from_numpy(image).float().permute((2,0,1))
        line_map = torch.from_numpy(line_map).long()
        line_exist = torch.from_numpy(line_exist).float()

        return [image, line_map ,line_exist]

class ResizeToERFNetInput(object):
    def __call__(self, data):
        image, line_map, line_exist = data

        image = cv2.resize(image, (960,270))
        image = cv2.copyMakeBorder(image, 0, 2, 0, 0, cv2.BORDER_CONSTANT)

        line_map = cv2.resize(line_map, (960,270),interpolation=cv2.INTER_NEAREST)
        line_map = cv2.copyMakeBorder(line_map, 0, 2, 0, 0, cv2.BORDER_CONSTANT)

        return [image, line_map, line_exist]

class Normalization(object):
    def __init__(self):
        self.mean = np.array((0.406, 0.456, 0.485))
        self.std = np.array((0.225, 0.224, 0.229))

    def __call__(self, data):
        image, line_map, line_exist = data

        image = image / 255.
        image = (image - self.mean) / self.std

        return [image, line_map, line_exist]

class NormalizationForERFNet(object):
    def __init__(self, mean, std, random_shift=False, shift_size=0.1):
        if not isinstance(mean, np.ndarray):
            self.mean = np.array(mean)
        else:
            self.mean = mean

        if not isinstance(std, np.ndarray):
            self.std = np.array(std)
        else:
            self.std = std

        self.random_shift = random_shift
        self.shift_size = shift_size

    def __call__(self, data):
        image, line_map, line_exist = data

        mean = self.mean
        std = self.std

        if(self.random_shift):
            if np.random.rand() > 0.5:
                mean = mean * (1.0 + np.random.rand() * (self.shift_size * 2 - self.shift_size))

        image = (image - mean) / std

        return [image, line_map, line_exist]

class CropLineROI(object):
    def __init__(self, hood_height = 0):
        self.hood_height = hood_height

    def __call__(self, data):
        image, line_map, line_exist = data
        image_height = image.shape[0]
        start_offset = image_height // 2 - self.hood_height
        end_offset = image_height - self.hood_height

        image = image[start_offset:end_offset , :, :]
        line_map = line_map[start_offset:end_offset, :]

        return [image, line_map, line_exist]

class RandomResizer(object):
    def __init__(self, min_scale = 0.5):
        self.min_scale = min_scale
        assert(min_scale >= 0 and min_scale < 1)

    def __call__(self, data):
        image, line_map, line_exist = data

        if np.random.rand() < 0.5:
            return [image, line_map, line_exist]

        image_size = np.array(image.shape[:2])

        scale = self.min_scale + np.random.rand() * (1. - self.min_scale)

        image = cv2.resize(image, (0,0), fx=scale, fy=scale)
        line_map = cv2.resize(line_map, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

        pad_size = image_size - np.array(image.shape[:2])
        image = cv2.copyMakeBorder(image, 0, pad_size[0], 0, pad_size[1], cv2.BORDER_CONSTANT)
        line_map = cv2.copyMakeBorder(line_map, 0, pad_size[0], 0, pad_size[1], cv2.BORDER_CONSTANT)

        return [image, line_map, line_exist]