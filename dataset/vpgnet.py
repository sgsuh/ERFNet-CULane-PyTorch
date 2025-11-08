"""
Create: 2022.02.08
Author: SG.SUH
Python: 3.7
"""

import glob
import os
import cv2
import numpy as np
import argparse

from scipy import io

"""
### Lane and road markings (4th channel) ###
0	background
1	lane_solid_white
2	lane_broken_white
3	lane_double_white
4	lane_solid_yellow
5	lane_broken_yellow
6	lane_double_yellow
7	lane_broken_blue
8	lane_slow
9	stop_line
10	arrow_left
11	arrow_right
12	arrow_go_straight
13	arrow_u_turn
14	speed_bump
15	crossWalk
16	safety_zone
17	other_road_markings
"""

def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root",
                        default="")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = make_parser()

    data_root = args.data_root

    scene_list = glob.glob(data_root + '/*')
    scene_list.sort()

    for scene_path in scene_list:
        if not os.path.isdir(scene_path):
            continue

        fold_list = glob.glob(scene_path + '/*')
        fold_list.sort()

        for fold_path in fold_list:
            file_list = glob.glob(fold_path + '/*.mat')
            file_list.sort()

            for file_path in file_list:
                mat_file = io.loadmat(file_path)

                mat_data = mat_file['rgb_seg_vp']
                im = mat_data[:, :, :3]
                annot = mat_data[:, :, 3]

                new_annot = np.zeros_like(annot, dtype = np.uint8)

                # Line
                for idx in [1, 2, 3, 4, 5, 6, 7, 8]:
                    new_annot[annot == idx] = 1

                # Stop Line
                new_annot[annot == 9] = 2

                # Crosswalk
                new_annot[annot == 15] = 3

                im[new_annot == 1, 0] = 255
                im[new_annot == 1, 1] = 0
                im[new_annot == 1, 2] = 0

                im[new_annot == 2, 0] = 0
                im[new_annot == 2, 1] = 255
                im[new_annot == 2, 2] = 0

                im[new_annot == 3, 0] = 0
                im[new_annot == 3, 1] = 0
                im[new_annot == 3, 2] = 255

                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

                cv2.imshow('im', im)
                cv2.waitKey(0)

                
