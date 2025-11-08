"""
Create: 2021.12.08
Author: SG.SUH
Python: 3.7
"""

import glob
import os
import cv2
import numpy as np
import json
import argparse

def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_path",
                        default="")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = make_parser()

    root_path = args.root_path
    image_fold = root_path + '/image'
    label_fold = root_path + '/label'

    image_fold_list = glob.glob(os.path.join(image_fold, '*'))
    image_fold_list.sort()

    for image_fold_path in image_fold_list:
        if not os.path.isdir(image_fold_path):
            continue

        fold_name = os.path.basename(image_fold_path)
        label_fold_path = os.path.join(label_fold, fold_name)
        image_file_list = glob.glob(os.path.join(image_fold_path, '*.png'))
        image_file_list.sort()

        for image_file_path in image_file_list:
            file_name = os.path.basename(image_file_path)
            label_file_path = os.path.join(label_fold_path, file_name)

            label = cv2.imread(label_file_path)
            img = cv2.imread(image_file_path)

            colors = [(255, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255)]
            cls_points = []

            for c_idx, color in enumerate(colors):
                y_points, x_points = np.where(np.all(label == color, axis = -1))

                points = []
                
                for i, y in enumerate(y_points):
                    if i == 0:
                        s = x_points[0]
                        cnt = 1
                        prev_y = y

                        continue

                    if y == prev_y:
                        cnt += 1
                        s += x_points[i]
                    else:
                        x = float(s) / cnt

                        points.append([float(x), float(prev_y)])

                        s = x_points[i]
                        cnt = 1
                        prev_y = y

                cls_points.append(points)

            
            json_file_path = os.path.splitext(image_file_path)[0] + '.json'

            print(json_file_path)

            infos = {}
            infos['version'] = '4.5.6'
            infos['flags'] = {}
            infos['shapes'] = []

            for c_idx, points in enumerate(cls_points):
                if len(points) == 0:
                    continue

                temp = {'label': {}, 'points': {}, 'group_id': None, 'shape_type': 'linestrip', 'flags': {}}

                if c_idx == 0:
                    temp['label'] = 'L2'
                elif c_idx == 1:
                    temp['label'] = 'L1'
                elif c_idx == 2:
                    temp['label'] = 'R1'
                else:
                    temp['label'] = 'R2'

                samp = int(len(points) / 40.0)
                json_points = []

                for i in range(0, len(points), samp):
                    json_points.append(points[i])

                temp['points'] = json_points
                infos['shapes'].append(temp)

            infos['imageData'] = None
            infos['imagePath'] = file_name
            infos['imageHeight'] = int(img.shape[0])
            infos['imageWidth'] = int(img.shape[1])

            with open(json_file_path, 'w') as f:
                json.dump(infos, f, indent = 2)