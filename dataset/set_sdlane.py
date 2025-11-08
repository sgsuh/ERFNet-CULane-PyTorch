"""
Create: 2022.06.14
Author: SG.SUH
Python: 3.7
"""

import os
import glob
import cv2
import json
import numpy as np
import argparse 

def make_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_path",
                        default="")
    parser.add_argument("--dst_root_path",
                        default="")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = make_parse()

    est_color = [(255, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255)]

    root_path = args.root_path
    img_root_path = os.path.join(root_path, 'images')
    label_root_path = os.path.join(root_path, 'labels')
    save_root_path = os.path.join(root_path, 'save')

    dst_root_path = args.dst_root_path

    if not os.path.isdir(dst_root_path):
        os.makedirs(dst_root_path)

    dst_img_root_path = os.path.join(dst_root_path, 'images')
    dst_label_root_path = os.path.join(dst_root_path, 'labels')

    if not os.path.isdir(dst_img_root_path):
        os.makedirs(dst_img_root_path)

    if not os.path.isdir(dst_label_root_path):
        os.makedirs(dst_label_root_path)

    save_fold_list = glob.glob(os.path.join(save_root_path, '*'))
    save_fold_list.sort()

    for save_fold_path in save_fold_list:
        fold_name = os.path.basename(save_fold_path)

        dst_img_fold_path = os.path.join(dst_img_root_path, fold_name)
        dst_label_fold_path = os.path.join(dst_label_root_path, fold_name)

        if not os.path.isdir(dst_img_fold_path):
            os.makedirs(dst_img_fold_path)

        if not os.path.isdir(dst_label_fold_path):
            os.makedirs(dst_label_fold_path)

        img_fold_path = os.path.join(img_root_path, fold_name)
        label_fold_path = os.path.join(label_root_path, fold_name)

        save_file_list = glob.glob(os.path.join(save_fold_path, '*'))
        save_file_list.sort()

        for save_file_path in save_file_list:
            file_name = os.path.splitext(os.path.basename(save_file_path))[0]

            print(save_file_path)

            img_file_path = img_fold_path + '/' + file_name + '.jpg'
            label_file_path = label_fold_path + '/' + file_name + '.json'

            dst_img_file_path = dst_img_fold_path + '/' + file_name + '.png'
            dst_label_file_path = dst_label_fold_path + '/' + file_name + '.png'

            img = cv2.imread(img_file_path)
            label = np.zeros((img.shape), dtype = np.uint8)

            with open(label_file_path, 'r') as f:
                json_data = json.load(f)

            left_idx = -1

            for i, points in enumerate(json_data['geometry']):
                max_y = 0

                for point in points:
                    y = point[1]

                    if y > max_y:
                        max_y = y
                        x = point[0]

                if x > img.shape[1] // 2:
                    left_idx = i - 1

                    break

            cnt = 0

            for idx in range(left_idx - 1, left_idx + 3):
                if idx < 0:
                    cnt += 1

                    continue

                if idx == len(json_data['geometry']):
                    break

                for j, point in enumerate(json_data['geometry'][idx]):
                    if j == 0:
                        prev_x = int(point[0])
                        prev_y = int(point[1])

                        continue

                    curr_x = int(point[0])
                    curr_y = int(point[1])

                    cv2.line(label, (prev_x, prev_y), (curr_x, curr_y), est_color[cnt], 5)

                    prev_x = curr_x
                    prev_y = curr_y

                cnt += 1

            img = img[128:, :, :]
            label = label[128:, :, :]

            cv2.imwrite(dst_img_file_path, img)
            cv2.imwrite(dst_label_file_path, label)