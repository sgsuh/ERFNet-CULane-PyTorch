"""
Create: 2022.07.05
Author: SG.SUH
Python: 3.7
"""

import os
import glob
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

    sub_fold_list = glob.glob(os.path.join(root_path, '*'))
    sub_fold_list.sort()

    for sub_fold_path in sub_fold_list:
        image_root_path = os.path.join(sub_fold_path, 'images')
        label_root_path = os.path.join(sub_fold_path, 'labels')

        label_fold_list = glob.glob(os.path.join(label_root_path, '*'))
        label_fold_list.sort()

        for label_fold_path in label_fold_list:
            fold_name = os.path.basename(label_fold_path)
            image_fold_path = os.path.join(image_root_path, fold_name)

            label_file_list = glob.glob(os.path.join(label_fold_path, '*.json'))
            label_file_list.sort()

            for label_file_path in label_file_list:
                print(label_file_path)

                file_name = os.path.basename(label_file_path)
                image_file_name = os.path.splitext(os.path.basename(label_file_path))[0] + '.jpg'

                save_file_path = os.path.join(image_fold_path, file_name)

                with open(label_file_path, 'r') as f:
                    json_data = json.load(f)

                infos = {}
                infos['version'] = '5.0.1'
                infos['flags'] = {}
                infos['shapes'] = []

                for idx, lines in enumerate(json_data['geometry']):
                    temp = {'label': {}, 'points': {}, 'group_id': None, 'shape_type': 'linestrip', 'flags': {}}

                    temp['label'] = str(json_data['idx'][idx])
                    temp['points'] = lines
                    infos['shapes'].append(temp)

                infos['imagePath'] = image_file_name
                infos['imageData'] = None
                infos['imageHeight'] = int(1208)
                infos['imageWidth'] = int(1920)

                with open(save_file_path, 'w') as f:
                    json.dump(infos, f, indent = 2)