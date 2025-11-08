"""
Create: 2021.09.10
Author: SG.SUH
Python: 3.7
PyTorch: 1.8
"""

import os
import cv2
import numpy as np
import argparse

"""
(B, G, R)
Class0 - (255, 255, 0)
Class1 - (255, 0, 0)
Class2 - (0, 0, 255)
Class3 - (0, 255, 255)
"""

def parser_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_path",
                        default="")
    parser.add_argument("--culane_list_fold",
                        default="")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parser_args()

    root_path = args.root_path

    list_fold = os.path.join(root_path, 'list')

    if not os.path.isdir(list_fold):
        os.makedirs(list_fold)

    train_file = open(os.path.join(list_fold, 'train_gt.txt'), 'w')
    val_file = open(os.path.join(list_fold, 'val_gt.txt'), 'w')

    class_color = [(255, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255)]

    """
    CULANE
    """
    culane_name = 'culane/new'
    culane_fold = os.path.join(root_path, culane_name)

    culane_list_fold = args.culane_list_fold

    with open(os.path.join(culane_list_fold, 'train_gt.txt'), 'r') as f:
        for read_line in f:
            src_img_path = read_line.strip().split(' ')[0]
            src_label_path = read_line.strip().split(' ')[1]

            print(src_img_path)

            dst_img_path = culane_name + '/resize_images' + src_img_path
            dst_label_path = culane_name + src_label_path.replace('laneseg_label_w16', 'laneseg_label_w5_color')

            write_line = dst_img_path + ' ' + dst_label_path + '\n'

            train_file.write(write_line)

    with open(os.path.join(culane_list_fold, 'val_gt.txt'), 'r') as f:
        for read_line in f:
            src_img_path = read_line.strip().split(' ')[0]
            src_label_path = read_line.strip().split(' ')[1]

            print(src_img_path)

            dst_img_path = culane_name + '/resize_images' + src_img_path
            dst_label_path = culane_name + src_label_path.replace('laneseg_label_w16', 'laneseg_label_w5_color')

            write_line = dst_img_path + ' ' + dst_label_path + '\n'

            val_file.write(write_line)


    """
    EXTRA_LANE
    """
    extra_lane_name = 'extra_lane'
    extra_lane_fold = os.path.join(root_path, extra_lane_name)

    with open(os.path.join(extra_lane_fold, 'convert_extra_lane_list.txt'), 'r') as f:
        num_frame = len(f.readlines())

    with open(os.path.join(extra_lane_fold, 'convert_extra_lane_list.txt'), 'r') as f:
        num_train = int(num_frame * 0.9)
        num_val = num_frame - num_train

        for _ in range(num_train):
            read_line = f.readline()

            src_img_path = read_line.strip().split(' ')[0]
            src_label_path = read_line.strip().split(' ')[1]

            print(src_img_path)

            dst_img_path = os.path.join(extra_lane_name, src_img_path)
            dst_label_path = os.path.join(extra_lane_name, src_label_path)

            

            write_line = dst_img_path + ' ' + dst_label_path + '\n'

            train_file.write(write_line)

        for _ in range(num_val):
            read_line = f.readline()

            src_img_path = read_line.strip().split(' ')[0]
            src_label_path = read_line.strip().split(' ')[1]

            print(src_img_path)

            dst_img_path = os.path.join(extra_lane_name, src_img_path)
            dst_label_path = os.path.join(extra_lane_name, src_label_path)

            

            write_line = dst_img_path + ' ' + dst_label_path + '\n'

            val_file.write(write_line)

    """
    TUSIMPLE
    """
    tusimple_name = 'lane_tusimple'
    tusimple_fold = os.path.join(root_path, tusimple_name)

    with open(os.path.join(tusimple_fold, 'lane_train.txt'), 'r') as f:
        num_frame = len(f.readlines())

    with open(os.path.join(tusimple_fold, 'lane_train.txt'), 'r') as f:
        num_train = int(num_frame * 0.9)
        num_val = num_frame - num_train

        for _ in range(num_train):
            read_line = f.readline()

            src_img_path = read_line.strip().split(' ')[0]
            src_label_path = read_line.strip().split(' ')[1]
            src_label_path = os.path.splitext(src_label_path)[0] + '_convert.png'

            print(src_img_path)

            dst_img_path = os.path.join(tusimple_name, src_img_path)
            dst_label_path = os.path.join(tusimple_name, src_label_path)

            

            write_line = dst_img_path + ' ' + dst_label_path + '\n'

            train_file.write(write_line)

        for _ in range(num_val):
            read_line = f.readline()

            src_img_path = read_line.strip().split(' ')[0]
            src_label_path = read_line.strip().split(' ')[1]
            src_label_path = os.path.splitext(src_label_path)[0] + '_convert.png'

            print(src_img_path)

            dst_img_path = os.path.join(tusimple_name, src_img_path)
            dst_label_path = os.path.join(tusimple_name, src_label_path)

            

            write_line = dst_img_path + ' ' + dst_label_path + '\n'

            val_file.write(write_line)

    """
    VIL100
    """
    vil100_name = 'VIL100'
    vil100_fold = os.path.join(root_path, vil100_name)

    with open(os.path.join(vil100_fold, 'normal_5_color.txt'), 'r') as f:
        num_frame = len(f.readlines())

    with open(os.path.join(vil100_fold, 'normal_5_color.txt'), 'r') as f:
        num_train = int(num_frame * 0.9)
        num_val = num_frame - num_train

        black_lists = ['0_Road029', '0_Road030', '0_Road031', '0_Road036', '1_Road031', '1_Road034', '2_Road036', '7_Road003', '8_Road033']

        for _ in range(num_train):
            read_line = f.readline()

            src_img_path = read_line.strip().split(' ')[0]
            src_label_path = read_line.strip().split(' ')[1]

            is_false = False

            for black_list in black_lists:
                if black_list in src_img_path:
                    is_false = True

                    break

            if is_false:
                continue

            print(src_img_path)

            dst_img_path = vil100_name + src_img_path
            dst_label_path = vil100_name + src_label_path

            full_img_path = os.path.join(root_path, dst_img_path)
            img = cv2.imread(full_img_path)
            if img.shape[0] < 270 or img.shape[1] < 960:
                continue

            write_line = dst_img_path + ' ' + dst_label_path + '\n'

            train_file.write(write_line)

        for _ in range(num_val):
            read_line = f.readline()

            src_img_path = read_line.strip().split(' ')[0]
            src_label_path = read_line.strip().split(' ')[1]

            is_false = False

            for black_list in black_lists:
                if black_list in src_img_path:
                    is_false = True

                    break

            if is_false:
                continue

            print(src_img_path)

            dst_img_path = vil100_name + src_img_path
            dst_label_path = vil100_name + src_label_path

            full_img_path = os.path.join(root_path, dst_img_path)
            img = cv2.imread(full_img_path)
            if img.shape[0] < 270 or img.shape[1] < 960:
                continue

            write_line = dst_img_path + ' ' + dst_label_path + '\n'

            val_file.write(write_line)

    """
    JIQING Expressway
    """
    jiqing_name = 'jiqing_expressway'
    jiqing_fold = os.path.join(root_path, jiqing_name)
    jiqing_list_fold = os.path.join(jiqing_fold, 'list')

    with open(os.path.join(jiqing_list_fold, 'train_gt.txt'), 'r') as f:
        for read_line in f:
            src_img_path = read_line.strip().split(' ')[0]
            src_label_path = read_line.strip().split(' ')[1]

            print(src_img_path)

            dst_img_path = os.path.join(jiqing_name, src_img_path)
            dst_label_path = os.path.join(jiqing_name, src_label_path)

            write_line = dst_img_path + ' ' + dst_label_path + '\n'

            train_file.write(write_line)

    with open(os.path.join(jiqing_list_fold, 'val_gt.txt'), 'r') as f:
        for read_line in f:
            src_img_path = read_line.strip().split(' ')[0]
            src_label_path = read_line.strip().split(' ')[1]

            print(src_img_path)

            dst_img_path = os.path.join(jiqing_name, src_img_path)
            dst_label_path = os.path.join(jiqing_name, src_label_path)

            write_line = dst_img_path + ' ' + dst_label_path + '\n'

            val_file.write(write_line)

    """
    SDLANE
    """
    sdlane_name = 'sdlane'
    sdlane_fold = os.path.join(root_path, sdlane_name)
    sdlane_list_fold = os.path.join(sdlane_fold, 'list')

    with open(os.path.join(sdlane_list_fold, 'train_gt.txt'), 'r') as f:
        for read_line in f:
            src_img_path = read_line.strip().split(' ')[0]
            src_label_path = read_line.strip().split(' ')[1]

            print(src_img_path)

            dst_img_path = os.path.join(sdlane_name, src_img_path)
            dst_label_path = os.path.join(sdlane_name, src_label_path)

            write_line = dst_img_path + ' ' + dst_label_path + '\n'

            train_file.write(write_line)

    with open(os.path.join(sdlane_list_fold, 'val_gt.txt'), 'r') as f:
        for read_line in f:
            src_img_path = read_line.strip().split(' ')[0]
            src_label_path = read_line.strip().split(' ')[1]

            print(src_img_path)

            dst_img_path = os.path.join(sdlane_name, src_img_path)
            dst_label_path = os.path.join(sdlane_name, src_label_path)

            write_line = dst_img_path + ' ' + dst_label_path + '\n'

            val_file.write(write_line)


    train_file.close()
    val_file.close()