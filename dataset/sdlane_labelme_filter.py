"""
Create: 2022.07.07
Author: SG.SUH
Python: 3.7
"""

import os
import glob
import shutil
import json
import cv2
import argparse

def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src_root_path",
                        default="")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = make_parser()

    class_names = ['L2', 'L1', 'R1', 'R2']

    src_root_path = args.src_root_path
    dst_root_path = os.path.join(src_root_path, '4class')

    if not os.path.isdir(dst_root_path):
        os.makedirs(dst_root_path)

    src_sub_fold_list = glob.glob(os.path.join(src_root_path, '*'))
    src_sub_fold_list.sort()

    for src_sub_fold_path in src_sub_fold_list:
        sub_fold_name = os.path.basename(src_sub_fold_path)

        if sub_fold_name == '4class':
            continue

        dst_sub_fold_path = os.path.join(dst_root_path, sub_fold_name)

        if not os.path.isdir(dst_sub_fold_path):
            os.makedirs(dst_sub_fold_path)

        src_save_root_path = os.path.join(src_sub_fold_path, 'save')
        src_img_root_path = os.path.join(src_sub_fold_path, 'images')
        src_label_root_path = os.path.join(src_sub_fold_path, 'labels')

        dst_img_root_path = os.path.join(dst_sub_fold_path, 'images')

        if not os.path.isdir(dst_img_root_path):
            os.makedirs(dst_img_root_path)

        src_save_fold_list = glob.glob(os.path.join(src_save_root_path, '*'))
        src_save_fold_list.sort()

        for src_save_fold_path in src_save_fold_list:
            fold_name = os.path.basename(src_save_fold_path)

            src_img_fold_path = os.path.join(src_img_root_path, fold_name)
            src_label_fold_path = os.path.join(src_label_root_path, fold_name)
            dst_img_fold_path = os.path.join(dst_img_root_path, fold_name)

            if not os.path.isdir(dst_img_fold_path):
                os.makedirs(dst_img_fold_path)

            src_save_file_list = glob.glob(os.path.join(src_save_fold_path, '*'))
            src_save_file_list.sort()

            for src_save_file_path in src_save_file_list:
                img_file_name = os.path.basename(src_save_file_path)
                file_name = os.path.splitext(img_file_name)[0]

                src_img_file_path = os.path.join(src_img_fold_path, img_file_name)
                src_label_file_path = os.path.join(src_label_fold_path, file_name) + '.json'
                dst_img_file_path = os.path.join(dst_img_fold_path, img_file_name)
                dst_json_file_path = os.path.join(dst_img_fold_path, file_name) + '.json'

                print(src_img_file_path)

                img = cv2.imread(src_img_file_path)

                shutil.copy(src_img_file_path, dst_img_file_path)

                with open(src_label_file_path, 'r') as f:
                    json_data = json.load(f)

                infos = {}
                infos['version'] = '5.0.1'
                infos['flags'] = {}
                infos['shapes'] = []

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

                    temp = {'label': {}, 'points': {}, 'group_id': None, 'shape_type': 'linestrip', 'flags': {}}

                    temp['label'] = str(class_names[cnt])
                    temp['points'] = json_data['geometry'][idx]

                    infos['shapes'].append(temp)

                    cnt += 1

                infos['imagePath'] = img_file_name
                infos['imageData'] = None
                infos['imageHeight'] = 1208
                infos['imageWidth'] = 1920

                with open(dst_json_file_path, 'w') as f:
                    json.dump(infos, f, indent = 2)

