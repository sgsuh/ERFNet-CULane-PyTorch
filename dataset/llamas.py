"""
Create: 2022.03.24
Author: SG.SUH
Python: 3.7
PyTorch: 1.8
"""

import glob
import os
import cv2
import json
import numpy as np
import pdb
import argparse

DCOLORS = [(110, 30, 30), (75, 25, 230), (75, 180, 60), (200, 130, 0), (48, 130, 245), (180, 30, 145), (0, 0, 255), (24, 140, 34), (255, 0, 0), (0, 255, 255), (40, 110, 170), (200, 250, 255), (255, 190, 230), (0, 0, 128), (195, 255, 170), (0, 128, 128), (195, 255, 170), (75, 25, 230)]
LANE_NAMES = ['l7', 'l6', 'l5', 'l4', 'l3', 'l2', 'l1', 'l0', 'r0', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8']
DICT_COLORS = dict(zip(LANE_NAMES, DCOLORS))

def filter_lanes_by_size(label, min_height = 40):
    filtered_lanes = []

    for lane in label['lanes']:
        lane_start = min([int(marker['pixel_start']['y']) for marker in lane['markers']])
        lane_end = max([int(marker['pixel_start']['y']) for marker in lane['markers']])

        if (lane_end - lane_start) < min_height:
            continue

        filtered_lanes.append(lane)

    label['lanes'] = filtered_lanes

def filter_few_markers(label, min_markers = 2):
    filtered_lanes = []

    for lane in label['lanes']:
        if len(lane['markers']) >= min_markers:
            filtered_lanes.append(lane)

    label['lanes'] = filtered_lanes

def fix_lane_names(label):
    l_counter = 0
    r_counter = 0
    mapping = {}
    lane_ids = [lane['lane_id'] for lane in label['lanes']]

    for key in sorted(lane_ids):
        if key[0] == 'l':
            mapping[key] = 'l' + str(l_counter)
            l_counter += 1

        if key[0] == 'r':
            mapping[key] = 'r' + str(r_counter)
            r_counter += 1

    for lane in label['lanes']:
        lane['lane_id'] = mapping[lane['lane_id']]

def read_json(json_path, min_lane_height = 20):
    with open(json_path, 'r') as jf:
        label_content = json.load(jf)

    filter_lanes_by_size(label_content, min_height = min_lane_height)
    filter_few_markers(label_content, min_markers = 2)
    fix_lane_names(label_content)

    content = {'projection_matrix': label_content['projection_matrix'], 'lanes': label_content['lanes']}

    for lane in content['lanes']:
        for marker in lane['markers']:
            for pixel_key in marker['pixel_start'].keys():
                marker['pixel_start'][pixel_key] = int(marker['pixel_start'][pixel_key])

            for pixel_key in marker['pixel_end'].keys():
                marker['pixel_end'][pixel_key] = int(marker['pixel_end'][pixel_key])

            for pixel_key in marker['world_start'].keys():
                marker['world_start'][pixel_key] = float(marker['world_start'][pixel_key])

            for pixel_key in marker['world_end'].keys():
                marker['world_end'][pixel_key] = float(marker['world_end'][pixel_key])

    return content

def project_point(point, projection_matrix):
    point = np.asarray(point)
    projection_matrix = np.asarray(projection_matrix)

    point_projected = projection_matrix.dot(point)
    point_projected /= point_projected[2]

    return point_projected

def project_lane_marker(p1, p2, width, projection_matrix, color, img):
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)

    p1_projected = project_point(p1, projection_matrix)
    p2_projected = project_point(p2, projection_matrix)

    points = np.zeros((4, 2), dtype = np.float32)
    shift = 0

    shift_multiplier = 1

    projection_matrix = np.asarray(projection_matrix)
    projected_half_width1 = projection_matrix[0, 0] * width / p1[2] / 2.0
    points[0, 0] = (p1_projected[0] - projected_half_width1) * shift_multiplier
    points[0, 1] = p1_projected[1] * shift_multiplier
    points[1, 0] = (p1_projected[0] + projected_half_width1) * shift_multiplier
    points[1, 1] = p1_projected[1] * shift_multiplier

    projected_half_width2 = projection_matrix[0, 0] * width / p2[2] / 2.0
    points[2, 0] = (p2_projected[0] + projected_half_width2) * shift_multiplier
    points[2, 1] = p2_projected[1] * shift_multiplier
    points[3, 0] = (p2_projected[0] - projected_half_width2) * shift_multiplier
    points[3, 1] = p2_projected[1] * shift_multiplier

    points = np.round(points).astype(np.int32)

    if not points[0, 1] == points[3, 1]:
        try:
            aliasing = cv2.LINE_AA
        except AttributeError:
            aliasing = cv2.CV_AA

        cv2.fillConvexPoly(img, points, color, aliasing, shift)

def ir(some_value):
    return int(round(some_value))

def extend_lane(lane, projection_matrix):
    filtered_markers = filter(lambda x: (x['pixel_start']['y'] != x['pixel_end']['y'] and x['pixel_start']['x'] != x['pixel_end']['x']), lane['markers'])

    closest_marker = min(filtered_markers, key = lambda x: x['world_start']['z'])

    if closest_marker['world_start']['z'] < 0:
        return lane

    x_gradient = (closest_marker['world_end']['x'] - closest_marker['world_start']['x']) / (closest_marker['world_end']['z'] - closest_marker['world_start']['z'])
    y_gradient = (closest_marker['world_end']['y'] - closest_marker['world_start']['y']) / (closest_marker['world_end']['z'] - closest_marker['world_start']['z'])

    zero_x = closest_marker['world_start']['x'] - (closest_marker['world_start']['z'] - 1) * x_gradient
    zero_y = closest_marker['world_start']['y'] - (closest_marker['world_start']['z'] - 1) * y_gradient

    pixel_x_gradient = (closest_marker['pixel_end']['x'] - closest_marker['pixel_start']['x']) / (closest_marker['pixel_end']['y'] - closest_marker['pixel_start']['y'])
    pixel_y_gradient = (closest_marker['pixel_end']['y'] - closest_marker['pixel_start']['y']) / (closest_marker['pixel_end']['x'] - closest_marker['pixel_start']['x'])

    pixel_zero_x = closest_marker['pixel_start']['x'] + (716 - closest_marker['pixel_start']['y']) * pixel_x_gradient

    if pixel_zero_x < 0:
        left_y = closest_marker['pixel_start']['y'] - closest_marker['pixel_start']['x'] * pixel_y_gradient
        new_pixel_point = (0, left_y)
    elif pixel_zero_x > 1276:
        right_y = closest_marker['pixel_start']['y'] + (1276 - closest_marker['pixel_start']['x']) * pixel_y_gradient
        new_pixel_point = (1276, right_y)
    else:
        new_pixel_point = (pixel_zero_x, 716)

    new_marker = {'lane_marker_id': 'FAKE', 'world_end': {'x': closest_marker['world_start']['x'], 'y': closest_marker['world_start']['y'], 'z': closest_marker['world_start']['z']}, 
                    'world_start': {'x': zero_x, 'y': zero_y, 'z': 1}, 'pixel_end': {'x': closest_marker['pixel_start']['x'], 'y': closest_marker['pixel_start']['y']}, 
                    'pixel_start': {'x': ir(new_pixel_point[0]), 'y': ir(new_pixel_point[1])}}

    lane['markers'].insert(0, new_marker)

    return lane

class SplineCreator():
    def __init__(self, json_path):
        self.json_path = json_path
        self.json_content = read_json(json_path)
        self.lanes = self.json_content['lanes']
        self.lane_marker_points = {}
        self.sampled_points = {}

    def sample_points(self, lane, ypp = 5, between_markers = True):
        x_values = [[] for i in range(717)]

        for marker in lane['markers']:
            try:
                x_values[marker['pixel_start']['y']].append(marker['pixel_start']['x'])
            except:
                pdb.set_trace()

            height = marker['pixel_start']['y'] - marker['pixel_end']['y']

            if height > 2:
                slope = (marker['pixel_end']['x'] - marker['pixel_start']['x']) / height
                step_size = (marker['pixel_start']['y'] - marker['pixel_end']['y']) / float(height)

                for i in range(height + 1):
                    x = marker['pixel_start']['x'] + slope * step_size * i
                    y = marker['pixel_start']['y'] - step_size * i

                    x_values[ir(y)].append(ir(x))

        for y, xs in enumerate(x_values):
            if not xs:
                x_values[y] = -1
            else:
                x_values[y] = sum(xs) / float(len(xs))

        if not between_markers:
            return x_values

        current_y = 0

        while x_values[current_y] == -1:
            current_y += 1

        next_set_y = 0

        try:
            while current_y < 717:
                if x_values[current_y] != -1:
                    current_y += 1

                    continue

                while next_set_y <= current_y or x_values[next_set_y] == -1:
                    next_set_y += 1

                    if next_set_y >= 717:
                        raise StopIteration

                x_values[current_y] = x_values[current_y - 1] + (x_values[next_set_y] - x_values[current_y - 1]) / (next_set_y - current_y + 1)
                current_y += 1
        except StopIteration:
            pass

        return x_values

    def lane_points_fit(self, lane):
        lane = extend_lane(lane, self.json_content['projection_matrix'])
        sampled_points = self.sample_points(lane, ypp = 1)

        self.sampled_points[lane['lane_id']] = sampled_points

        return sampled_points

    def create_all_points(self):
        for lane in self.lanes:
            self.lane_points_fit(lane)
        
def draw_points(debug_image, x_coordinates, color):
    for y, x in enumerate(x_coordinates):
        if x != -1:
            cv2.circle(debug_image, (int(round(x)), y), 5, color, -1)

def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_path",
                        default="")
    
    return parser.parse_args()

if __name__ == '__main__':
    args = make_parser()
    
    root_path = args.root_path
    label_root_path = root_path + '/labels'
    img_root_path = root_path + '/color_images'

    label_task_list = glob.glob(label_root_path + '/*')
    label_task_list.sort()

    for label_task_path in label_task_list:
        task_name = os.path.basename(label_task_path)

        img_task_path = img_root_path + '/' + task_name

        label_fold_list = glob.glob(label_task_path + '/*')
        label_fold_list.sort()

        for label_fold_path in label_fold_list:
            fold_name = os.path.basename(label_fold_path)

            img_fold_path = img_task_path + '/' + fold_name

            label_file_list = glob.glob(label_fold_path + '/*.json')
            label_file_list.sort()

            for label_file_path in label_file_list:
                file_name = os.path.splitext(os.path.basename(label_file_path))[0]

                img_file_path = img_fold_path + '/' + file_name + '_color_rect.png'

                print(img_file_path)

                label = read_json(label_file_path)
                image = cv2.imread(img_file_path)

                for lane in label['lanes']:
                    lane_id = lane['lane_id']

                    if lane_id != 'l1' and lane_id != 'l0' and lane_id != 'r0' and lane_id != 'r1':
                        continue

                    for marker in lane['markers']:
                        x1 = marker['pixel_start']['x']
                        y1 = marker['pixel_start']['y']

                        x2 = marker['pixel_end']['x']
                        y2 = marker['pixel_end']['y']

                        cv2.line(image, (x1, y1), (x2, y2), DICT_COLORS[lane_id], 5)

                cv2.imshow('im', image)
                cv2.waitKey(0)