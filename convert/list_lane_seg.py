"""
Create: 2021.09.27
Author: SG.SUH
Python: 3.7
PyTorch: 1.8
"""

import os
import argparse

def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_path",
                        default="")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = make_parser()

    root_path = args.root_path

    list_fold = os.path.join(root_path, 'list')

    if not os.path.isdir(list_fold):
        os.makedirs(list_fold)

    train_file = open(os.path.join(list_fold, 'train_gt.txt'), 'w')
    val_file = open(os.path.join(list_fold, 'val_gt.txt'), 'w')

    """
    APOLLOSCAPE
    """
    apollo_name = 'apolloscape'
    apollo_fold = os.path.join(root_path, apollo_name)

    with open(os.path.join(apollo_fold, 'list/train_gt.txt'), 'r') as f:
        for read_line in f:
            src_img_path = read_line.strip().split(' ')[0]
            src_label_path = read_line.strip().split(' ')[1]
            exist = read_line.strip().split(' ')[2:]

            print(src_img_path)

            dst_img_path = '/' + apollo_name + src_img_path
            dst_label_path = '/' + apollo_name + src_label_path

            write_line = dst_img_path + ' ' + dst_label_path + ' ' + exist[0] + ' ' + exist[1] + ' ' + exist[2] + '\n'

            train_file.write(write_line)

    with open(os.path.join(apollo_fold, 'list/val_gt.txt'), 'r') as f:
        for read_line in f:
            src_img_path = read_line.strip().split(' ')[0]
            src_label_path = read_line.strip().split(' ')[1]
            exist = read_line.strip().split(' ')[2:]

            print(src_img_path)

            dst_img_path = '/' + apollo_name + src_img_path
            dst_label_path = '/' + apollo_name + src_label_path

            write_line = dst_img_path + ' ' + dst_label_path + ' ' + exist[0] + ' ' + exist[1] + ' ' + exist[2] + '\n'

            val_file.write(write_line)

    train_file.close()
    val_file.close()