"""
Create: 2022.06.14
Author: SG.SUH
Python: 3.7
"""

import os
import glob
import json
import cv2
import argparse

def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_path",
                        default="")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = make_parser()

    root_path = args.root_path
    img_root_path = os.path.join(root_path, 'images')
    label_root_path = os.path.join(root_path, 'labels')
    save_root_path = os.path.join(root_path, 'save')

    if not os.path.isdir(save_root_path):
        os.makedirs(save_root_path)

    label_fold_list = glob.glob(os.path.join(label_root_path, '*'))
    label_fold_list.sort()

    est_color = [(255, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255)]

    for label_fold_path in label_fold_list:
        fold_name = os.path.basename(label_fold_path)

        label_file_list = glob.glob(os.path.join(label_fold_path, '*'))
        label_file_list.sort()

        save_fold_path = os.path.join(save_root_path, fold_name)

        if not os.path.isdir(save_fold_path):
            os.makedirs(save_fold_path)

        for label_file_path in label_file_list:
            file_name = os.path.splitext(os.path.basename(label_file_path))[0]

            img_file_path = img_root_path + '/' + fold_name + '/' + file_name + '.jpg'
            save_file_path = save_fold_path + '/' + file_name + '.jpg'

            img = cv2.imread(img_file_path)

            print(img_file_path)

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

                for n, point in enumerate(json_data['geometry'][idx]):
                    if n == 0:
                        prev_x = int(point[0])
                        prev_y = int(point[1])

                        continue

                    curr_x = int(point[0])
                    curr_y = int(point[1])

                    cv2.line(img, (prev_x, prev_y), (curr_x, curr_y), est_color[cnt], 5)

                    prev_x = curr_x
                    prev_y = curr_y

                cnt += 1

            cv2.imwrite(save_file_path, img)