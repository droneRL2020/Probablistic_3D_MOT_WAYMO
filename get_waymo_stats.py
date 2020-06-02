'''
This code is based on Probabilistic Tracking applying Waymo data instead of Nuscene one.
'''
from os import listdir
from os.path import isfile, join

import imp
import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools
import matplotlib.pyplot as plt

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
from collections import defaultdict

from loaders import load_prediction, load_gt, create_tracks
from sklearn.utils.linear_assignment_ import linear_assignment
tf.enable_eager_execution()

WAYMO_TRACKING_NAMES = [
  1, #'TYPE_VEHICLE',
  2, #'TYPE_PEDESTRIAN',
  4, #'TYPE_CYCLIST'
]
#__type_list = ['UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST']

def gt_vehicle_to_world(trans_v_w, bbox):
    """
    trasform bounding box of objects w.r.t to vehicle to the world
    :param trans_v_w: transform matrix vehicle to world.
    :param bbox: bounding box of objects.
    :return: pose of objects w.r.t to world
    """
    trans_mat = np.reshape(trans_v_w,[4,4])
    pose_v = np.array([[bbox['center_x']],[bbox['center_y']],[bbox['center_z']],[1]])
    pose_w = np.matmul(trans_mat, pose_v)
    theta_v_w = np.arctan2(trans_mat[1][0], trans_mat[0][0])
    theta_v = bbox['rotation']
    theta_w = theta_v_w + theta_v
    if theta_w > np.pi:
        theta_w = theta_w - np.pi
    elif theta_w <= -np.pi:
        theta_w = theta_w + np.pi
    return pose_w, theta_w

def pred_vehicle_to_world(trans_v_w, bbox):
    """
    trasform bounding box of objects w.r.t to vehicle to the world
    :param trans_v_w: transform matrix vehicle to world.
    :param bbox: bounding box of objects.
    :return: pose of objects w.r.t to world
    """
    trans_mat = np.reshape(trans_v_w,[4,4])
    pose_v = np.array([[bbox['translation'][0]],[bbox['translation'][1]],[bbox['translation'][2]],[1]])
    pose_w = np.matmul(trans_mat, pose_v)
    
    theta_v_w = np.arctan2(trans_mat[1][0], trans_mat[0][0])
    theta_v = bbox['rotation']
    theta_w = theta_v_w + theta_v
    if theta_w > np.pi:
        theta_w = theta_w - np.pi
    elif theta_w <= -np.pi:
        theta_w = theta_w + np.pi
    return pose_w, theta_w


def get_mean(tracks):
    gt_trajectory_map = {tracking_name: {scene_token: {} for scene_token in tracks.keys()} for tracking_name in WAYMO_TRACKING_NAMES}
    gt_box_data = {tracking_name: [] for tracking_name in WAYMO_TRACKING_NAMES}
    for scene_token in tracks.keys():
        for t_idx in range(len(tracks[scene_token].keys())):
            t = sorted(tracks[scene_token].keys())[t_idx]
            # trans_v_w = transforms[t]
            for box_id in range(len(tracks[scene_token][t])):
                bbox = tracks[scene_token][t][box_id]
                if bbox['tracking_name'] not in WAYMO_TRACKING_NAMES:
                    continue
                trans_v_w = bbox['transform']
                pose_w, theta_w = gt_vehicle_to_world(trans_v_w, bbox)
                # [h, w, l, x, y, z, ry,
                #  x_t - x_{t-1}, ...,  for [x,y,z,ry]
                #  (x_t - x_{t-1}) - (x_{t-1} - x_{t-2}), ..., for [x,y,z,ry]
                box_data = np.array([bbox['height'], bbox['width'], bbox['length'],
                                     pose_w[0][0], pose_w[1][0], pose_w[2][0],
                                     theta_w,  
                                     0, 0, 0, 0,
                                     0, 0, 0, 0])
                if bbox['tracking_id'] not in gt_trajectory_map[bbox['tracking_name']][scene_token]:
                    gt_trajectory_map[bbox['tracking_name']][scene_token][bbox['tracking_id']] = {t_idx: box_data}
                else:
                    gt_trajectory_map[bbox['tracking_name']][scene_token][bbox['tracking_id']][t_idx] = box_data
                # if we can find the same object in the previous frame, get the velocity
                if bbox['tracking_id'] in gt_trajectory_map[bbox['tracking_name']][scene_token] and t_idx-1 in gt_trajectory_map[bbox['tracking_name']][scene_token][bbox['tracking_id']]:
                    residual_vel = box_data[3:7] - gt_trajectory_map[bbox['tracking_name']][scene_token][bbox['tracking_id']][t_idx-1][3:7]
                    box_data[7:11] = residual_vel
                    gt_trajectory_map[bbox['tracking_name']][scene_token][bbox['tracking_id']][t_idx] = box_data
                    if gt_trajectory_map[bbox['tracking_name']][scene_token][bbox['tracking_id']][t_idx-1][7] == 0:
                        gt_trajectory_map[bbox['tracking_name']][scene_token][bbox['tracking_id']][t_idx-1][7:11] = residual_vel
                    # if we can find the same object in the previous two frames, get the acceleration
                    if bbox['tracking_id'] in gt_trajectory_map[bbox['tracking_name']][scene_token] and t_idx-2 in gt_trajectory_map[bbox['tracking_name']][scene_token][bbox['tracking_id']]:
                        residual_a = residual_vel - (gt_trajectory_map[bbox['tracking_name']][scene_token][bbox['tracking_id']][t_idx-1][3:7] - gt_trajectory_map[bbox['tracking_name']][scene_token][bbox['tracking_id']][t_idx-2][3:7])
                        box_data[11:15] = residual_a
                        gt_trajectory_map[bbox['tracking_name']][scene_token][bbox['tracking_id']][t_idx] = box_data
                        # back fill
                        if gt_trajectory_map[bbox['tracking_name']][scene_token][bbox['tracking_id']][t_idx-1][11] == 0:
                            gt_trajectory_map[bbox['tracking_name']][scene_token][bbox['tracking_id']][t_idx-1][11:15] = residual_a
                        if gt_trajectory_map[bbox['tracking_name']][scene_token][bbox['tracking_id']][t_idx-2][11] == 0:
                            gt_trajectory_map[bbox['tracking_name']][scene_token][bbox['tracking_id']][t_idx-2][11:15] = residual_a

                gt_box_data[bbox['tracking_name']].append(box_data)

    gt_box_data = {tracking_name: np.stack(gt_box_data[tracking_name], axis=0) for tracking_name in WAYMO_TRACKING_NAMES}
    var = {tracking_name: np.var(gt_box_data[tracking_name], axis=0) for tracking_name in WAYMO_TRACKING_NAMES}

    return var

def matching_and_get_diff_stats(pred_boxes, gt_boxes, tracks):
    diff = {tracking_name: [] for tracking_name in WAYMO_TRACKING_NAMES} # [h, w, l, x, y, z, a]
    diff_vel = {tracking_name: [] for tracking_name in WAYMO_TRACKING_NAMES} # [x_dot, y_dot, z_dot, a_dot]  
    reorder = [3, 4, 5, 6, 2, 1, 0]
    reorder_back = [6, 5, 4, 0, 1, 2, 3]
    for scene_token in tracks.keys():
        match_diff_t_map = {tracking_name: {} for tracking_name in WAYMO_TRACKING_NAMES}
        for t_idx in range(len(tracks[scene_token].keys())):
            t = sorted(tracks[scene_token].keys())[t_idx]
            if len(tracks[scene_token][t]) == 0:
                continue
            bbox = tracks[scene_token][t][0]
            trans_v_w = bbox['transform']
            sample_time_token = t
            for tracking_name in WAYMO_TRACKING_NAMES:
                gt_all = [[gt_vehicle_to_world(trans_v_w, bbox), bbox] for bbox in gt_boxes[t] if bbox['tracking_name'] == tracking_name]
                if len(gt_all) == 0:
                    continue
                gts = np.stack([np.array([
                                bbox[1]['length'], bbox[1]['height'], bbox[1]['width'],
                                bbox[0][0][0][0], bbox[0][0][1][0], bbox[0][0][2][0],       #center_x, center_y, center_z
                                bbox[0][1]   #heading
                                ]) for bbox in gt_all], axis=0)
                gts_ids = [bbox[1]['tracking_id'] for bbox in gt_all]

                det_all = [[pred_vehicle_to_world(trans_v_w, bbox), bbox] for bbox in pred_boxes[t] if bbox['detection_name'] == tracking_name]
                if len(det_all) == 0:
                    continue
                dets = np.stack([np.array([
                                 bbox[1]['size'][2], bbox[1]['size'][0], bbox[1]['size'][1],
                                 bbox[0][0][0][0], bbox[0][0][1][0], bbox[0][0][2][0],
                                 bbox[0][1]
                ]) for bbox in det_all], axis=0)
                dets = dets[:, reorder]
                gts = gts[:, reorder]

                # matching distance is 2d_centor or 3d_iou. I tried 2d_center first
                distance_matrix = np.zeros((dets.shape[0], gts.shape[0]),dtype=np.float32)
                for d in range(dets.shape[0]):
                    for g in range(gts.shape[0]):
                        #그냥 x축끼리 y축끼리 거리 구해서 거리 매트릭스를 만듬
                        distance_matrix[d][g] = np.sqrt((dets[d][0] - gts[g][0])**2 + (dets[d][1] - gts[g][1])**2) 
                threshold = 2  # 2 meters stated in paper pg5
                matched_indices = linear_assignment(distance_matrix)
                dets = dets[:, reorder_back]
                gts = gts[:, reorder_back]
                for pair_id in range(matched_indices.shape[0]):
                    if distance_matrix[matched_indices[pair_id][0]][matched_indices[pair_id][1]] < threshold:
                        diff_value = dets[matched_indices[pair_id][0]] - gts[matched_indices[pair_id][1]]
                        diff[tracking_name].append(diff_value)
                        gt_track_id = gts_ids[matched_indices[pair_id][1]]
                        if t_idx not in match_diff_t_map[tracking_name]:
                            match_diff_t_map[tracking_name][t_idx] = {gt_track_id: diff_value}
                        else:
                            match_diff_t_map[tracking_name][t_idx][gt_track_id] = diff_value
                        # check if we have previous time_step's matching pair for current gt object
                        #print('t: ', t)
                        #print('len(match_diff_t_map): ', len(match_diff_t_map))
                        # 이 부분은 초기 Covcariance 구하기 위함이지(prev_covar)
                        if t_idx > 0 and t_idx-1 in match_diff_t_map[tracking_name] and gt_track_id in match_diff_t_map[tracking_name][t_idx-1]:
                            diff_vel_value = diff_value - match_diff_t_map[tracking_name][t_idx-1][gt_track_id]
                            diff_vel[tracking_name].append(diff_vel_value)

    diff = {tracking_name: np.stack(diff[tracking_name], axis=0) for tracking_name in WAYMO_TRACKING_NAMES}
    mean = {tracking_name: np.mean(diff[tracking_name], axis=0) for tracking_name in WAYMO_TRACKING_NAMES}
    std = {tracking_name: np.std(diff[tracking_name], axis=0) for tracking_name in WAYMO_TRACKING_NAMES}
    var = {tracking_name: np.var(diff[tracking_name], axis=0) for tracking_name in WAYMO_TRACKING_NAMES}
    # 이건 initial pose 를 위해서?
    diff_vel = {tracking_name: np.stack(diff_vel[tracking_name], axis=0) for tracking_name in WAYMO_TRACKING_NAMES}
    mean_vel = {tracking_name: np.mean(diff_vel[tracking_name], axis=0) for tracking_name in WAYMO_TRACKING_NAMES}
    std_vel = {tracking_name: np.std(diff_vel[tracking_name], axis=0) for tracking_name in WAYMO_TRACKING_NAMES}
    var_vel = {tracking_name: np.var(diff_vel[tracking_name], axis=0) for tracking_name in WAYMO_TRACKING_NAMES}

    return var, var_vel


if __name__ == '__main__':
    gt_boxes, scenes, gt_samples, _ = load_gt()
    tracks = create_tracks(gt_boxes, scenes)
    var = get_mean(tracks)
    print("get_mean_var", var)
    # set_gt = set(gt_samples)
    # set_pred = set(pred_samples)

    pred_boxes, pred_samples = load_prediction()
    var_r, var_vel_r = matching_and_get_diff_stats(pred_boxes, gt_boxes, tracks)
    print("var_r", var_r)
    print("var_vel_r", var_vel_r)





