"""
Create: 2021.09.28
Author: SG.SUH
Python: 3.7
"""

import glob
import os
import argparse

def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_path",
                        default="")
    
    return parser.parse_args()

if __name__ == '__main__':
    args = make_parser()
    root_path = args.root_path
    image_root = root_path + '/image'
    label_root = root_path + '/label'

    list_fold = root_path + '/list'

    if not os.path.isdir(list_fold):
        os.makedirs(list_fold)

    train_file = open(os.path.join(list_fold, 'train_gt.txt'), 'w')
    val_file = open(os.path.join(list_fold, 'val_gt.txt'), 'w')

    class_color = [(255, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255)]

    image_fold_list = glob.glob(os.path.join(image_root, '02*'))
    image_fold_list.sort()

    train_num = int(len(image_fold_list) * 0.9)

    for i, image_fold_path in enumerate(image_fold_list):
        fold_name = os.path.basename(image_fold_path)
        label_fold_path = os.path.join(label_root, fold_name)

        image_file_list = glob.glob(os.path.join(image_fold_path, '*.png'))
        image_file_list.sort()

        for image_file_path in image_file_list:
            file_name = os.path.basename(image_file_path)
            label_file_path = os.path.join(label_fold_path, file_name)

            print(image_file_path)

            dst_image_path = 'image/' + fold_name + '/' + file_name
            dst_label_path = 'label/' + fold_name + '/' + file_name

            write_line = dst_image_path + ' ' + dst_label_path + '\n'

            if i < train_num:
                train_file.write(write_line)
            else:
                val_file.write(write_line)

    image_fold_list = glob.glob(os.path.join(image_root, '03*'))
    image_fold_list.sort()

    train_num = int(len(image_fold_list) * 0.9)

    for i, image_fold_path in enumerate(image_fold_list):
        fold_name = os.path.basename(image_fold_path)
        label_fold_path = os.path.join(label_root, fold_name)

        image_file_list = glob.glob(os.path.join(image_fold_path, '*.png'))
        image_file_list.sort()

        for image_file_path in image_file_list:
            file_name = os.path.basename(image_file_path)
            label_file_path = os.path.join(label_fold_path, file_name)

            print(image_file_path)

            dst_image_path = 'image/' + fold_name + '/' + file_name
            dst_label_path = 'label/' + fold_name + '/' + file_name

            write_line = dst_image_path + ' ' + dst_label_path + '\n'

            if i < train_num:
                train_file.write(write_line)
            else:
                val_file.write(write_line)

    train_file.close()
    val_file.close()
