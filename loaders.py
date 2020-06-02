import math
from collections import defaultdict
from waymo_open_dataset.protos import metrics_pb2

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

from bisect import bisect

tf.enable_eager_execution()

WAYMO_TRACKING_NAMES = [
  1, #'TYPE_VEHICLE',
  2, #'TYPE_PEDESTRIAN',
  4, #'TYPE_CYCLIST'
]

#__type_list = ['UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST']


def load_prediction():
    objects = metrics_pb2.Objects()
    sample_time_tokens = []
    pred_boxes = defaultdict(list)
    #f = open("/home/gowithrobo/4_semester/tracking/waymo/detection/detection_3d_cyclist_detection_train.bin", 'rb')
    #objects.ParseFromString(f.read())
    #f.close()
    Path = "/scratch/shk642/waymo/detection/"
    filelist = listdir(Path)
    for i in filelist:
        if i.endswith(".bin"):
            with open(Path + i, 'rb') as f:
                objects.ParseFromString(f.read())
                for object in objects.objects:
                    sample_time_tokens.append(object.frame_timestamp_micros)
                    pred_boxes[object.frame_timestamp_micros].append(
            {'sample_time_token':object.frame_timestamp_micros,
            'translation':(object.object.box.center_x, object.object.box.center_y, object.object.box.center_z),
            'size':(object.object.box.height, object.object.box.width, object.object.box.length),
            'rotation':object.object.box.heading,
            'detection_name':object.object.type})
    return pred_boxes, sample_time_tokens

def load_gt():
    mypath = '/scratch/shk642/waymo/tracking/training/waymo_v_1_2_0/'
    gt_boxes = defaultdict(list)
    scenes = {}
    transforms = {}
    sample_time_tokens = []
    filelist = listdir(mypath)
    for i in filelist:
        dataset = tf.data.TFRecordDataset(mypath+i, compression_type='')        
        for data in dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            box_datas = list(frame.laser_labels)
            samples = []
            for box_data in box_datas:
                sample = {}
                center_x = np.array(box_data.box.center_x)
                center_y = np.array(box_data.box.center_y)
                center_z = np.array(box_data.box.center_z)
                width = np.array(box_data.box.width)
                length = np.array(box_data.box.length)
                height = np.array(box_data.box.height)
                center_z = np.array(box_data.box.height)
                heading = np.array(box_data.box.heading)
                transform = np.array(frame.pose.transform)
                # center_x = center_x.astype('float64')
                # center_y = center_y.astype('float64')
                # center_z = center_z.astype('float64')
                # width = width.astype('float64')
                # length = length.astype('float64')
                # height = height.astype('float64')
                # heading = heading.astype('float64')
                sample['sample_token'] = frame.timestamp_micros
                sample['center_x'] = center_x
                sample['center_y'] = center_y
                sample['center_z'] = center_z
                sample['width'] = width
                sample['length'] = length
                sample['height'] = height
                sample['rotation'] = heading
                sample['tracking_id'] = box_data.id
                sample['tracking_name'] = box_data.type
                sample['transform'] = transform
                samples.append(sample)
            transforms[frame.timestamp_micros] = transform            
            gt_boxes[frame.timestamp_micros] = samples  
            scenes[frame.timestamp_micros] = frame.context.name
            sample_time_tokens.append(frame.timestamp_micros)
    return gt_boxes, scenes, sample_time_tokens, transforms

def interpolate_tracking_boxes(left_box, right_box, right_ratio):
    def interp_list(left, right, rratio):
        return tuple(
            (1.0 - rratio) * np.array(left, dtype=float)
            + rratio * np.array(right, dtype=float)
        )

    def interp_float(left, right, rratio):
        return (1.0 - rratio) * float(left) + rratio * float(right)

    # Interpolate
    rotation = interp_float(left_box['rotation'], right_box['rotation'], right_ratio)
    center_x = interp_float(left_box['center_x'], right_box['center_x'], right_ratio)
    center_y = interp_float(left_box['center_y'], right_box['center_y'], right_ratio)
    center_z = interp_float(left_box['center_z'], right_box['center_z'], right_ratio)
    height = interp_float(left_box['height'], right_box['height'], right_ratio)
    length = interp_float(left_box['length'], right_box['length'], right_ratio)
    width  = interp_float(left_box['width'], right_box['width'], right_ratio)

    interp_box = right_box
    interp_box['rotation'] = rotation
    interp_box['center_x'] = center_x
    interp_box['center_y'] = center_y
    interp_box['center_z'] = center_z
    interp_box['height']   = height
    interp_box['width']    = width
    interp_box['length']   = length

    return interp_box



def interpolate_tracks(tracks_by_timestamp):
    # Group tracks by id.
    tracks_by_id = defaultdict(list)
    track_timestamps_by_id = defaultdict(list)
    for timestamp, tracking_boxes in tracks_by_timestamp.items():
        for tracking_box in tracking_boxes:
            tracks_by_id[tracking_box['tracking_id']].append(tracking_box)
            track_timestamps_by_id[tracking_box['tracking_id']].append(timestamp)
    # Interpolate missing timestamps for each track.
    timestamps = tracks_by_timestamp.keys()
    interpolate_count = 0
    for timestamp in timestamps:
        for tracking_id, track in tracks_by_id.items():
            if track_timestamps_by_id[tracking_id][0] <= timestamp <= track_timestamps_by_id[tracking_id][-1] and timestamp not in track_timestamps_by_id[tracking_id]:
                # Find the closest boxes before and after this timestamp.
                right_ind = bisect(track_timestamps_by_id[tracking_id], timestamp)
                left_ind = right_ind - 1
                right_timestamp = track_timestamps_by_id[tracking_id][right_ind]
                left_timestamp = track_timestamps_by_id[tracking_id][left_ind]
                right_tracking_box = tracks_by_id[tracking_id][right_ind]
                left_tracking_box = tracks_by_id[tracking_id][left_ind]
                right_ratio = float(right_timestamp - timestamp) / (right_timestamp - left_timestamp)

                # Interpolate.
                tracking_box = interpolate_tracking_boxes(left_tracking_box, right_tracking_box, right_ratio)
                interpolate_count += 1
                tracks_by_timestamp[timestamp].append(tracking_box)

    return tracks_by_timestamp

def create_tracks(gt_boxes, scenes):
    # Tracks are stored as dict {scene_token: {timestamp: List[TrackingBox]}}.
    tracks = defaultdict(lambda: defaultdict(list))
    # Init all scenes and timestamps to guarantee completeness.
    for sample_time_token in gt_boxes.keys():
        scene_token = scenes[sample_time_token]  # scene_token
        tracks[scene_token][sample_time_token] = []

    # Group annotations wrt scene and timestamp.
    for sample_time_token in gt_boxes.keys():
        scene_token = scenes[sample_time_token]  # scene_token
        tracks[scene_token][sample_time_token] = gt_boxes[sample_time_token]

    # Interpolate GT and predicted tracks.
    for scene_token in tracks.keys():
        tracks[scene_token] = interpolate_tracks(tracks[scene_token])

    return tracks


# if __name__ == "__main__":
    #pred_boxes, pred_samples = load_prediction()
    # gt_boxes, scenes, gt_samples, _ =load_gt()
    # tracks = create_tracks(gt_boxes, scenes)
