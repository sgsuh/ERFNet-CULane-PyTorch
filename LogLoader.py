"""
Create: 2022.06.20
Author: SG.SUH
Python: 3.7
"""

import numpy as np
import cv2

class LogLoader():
    def __init__(self, file_path, yuv = False):
        self.width = 1920
        self.height = 1080
        self.yuv = yuv

        if self.yuv:
            self.channel = 2
        else:
            self.channel = 3

        self.img_size = self.width * self.height * self.channel
        self.f = open(file_path, 'rb')

    def get_image(self):
        try:
            data = self.f.read(self.img_size)
        except:
            self.f.close()

            return None

        im = np.frombuffer(data, dtype = np.uint8)

        if im.shape[0] != self.img_size:
            self.f.close()

            return None

        im = im.reshape(self.height, self.width, self.channel)

        if self.yuv:
            im = cv2.cvtColor(im, cv2.COLOR_YUV2BGR_UYVY)

        return im