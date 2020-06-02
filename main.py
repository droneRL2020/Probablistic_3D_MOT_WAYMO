from __future__ import print_function

import sys

import os.path, copy, numpy as np, time, sys
from numba import jit
from sklearn.utils.linear_assignment_ import linear_assignment
from filterpy.kalman import KalmanFilter
#from utils import load_list_from_folder, fileparts, mkdir_if_missing
from scipy.spatial import ConvexHull
from covariance_temp import Covariance
import json
from pyquaternion import Quaternion
from tqdm import tqdm

from loaders import load_prediction, load_gt, create_tracks
from get_waymo_stats import pred_vehicle_to_world
from create_prediction import create_result
from waymo_open_dataset import dataset_pb2 as open_dataset


WAYMO_TRACKING_NAMES = [
  1, #'TYPE_VEHICLE',
  2, #'TYPE_PEDESTRIAN',
  4, #'TYPE_CYCLIST'
]

def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])

def convert_3dbox_to_8corner(bbox3d_input, nuscenes_to_kitti=False):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
        Note: the output of this function will be passed to the funciton iou3d
            for calculating the 3D-IOU. But the function iou3d was written for 
            kitti, so the caller needs to set nuscenes_to_kitti to True if 
            the input bbox3d_input is in nuscenes format.
    '''
    # compute rotational matrix around yaw axis
    bbox3d = copy.copy(bbox3d_input)
    R = roty(bbox3d[3])    

    # 3d bounding box dimensions
    l = bbox3d[4]
    w = bbox3d[5]
    h = bbox3d[6]
    
    # 3d bounding box corners
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
    y_corners = [0,0,0,0,-h,-h,-h,-h];
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];
    
    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d[0,:] = corners_3d[0,:] + bbox3d[0]
    corners_3d[1,:] = corners_3d[1,:] + bbox3d[1]
    corners_3d[2,:] = corners_3d[2,:] + bbox3d[2]
 
    return np.transpose(corners_3d)

class KalmanBoxTracker(object):
  """
  This class represents the internel state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self, bbox3D, covariance_id=0, tracking_name='car', use_angular_velocity=False):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    if not use_angular_velocity:
      self.kf = KalmanFilter(dim_x=10, dim_z=7)       
      self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0],      # state transition matrix
                            [0,1,0,0,0,0,0,0,1,0],
                            [0,0,1,0,0,0,0,0,0,1],
                            [0,0,0,1,0,0,0,0,0,0],  
                            [0,0,0,0,1,0,0,0,0,0],
                            [0,0,0,0,0,1,0,0,0,0],
                            [0,0,0,0,0,0,1,0,0,0],
                            [0,0,0,0,0,0,0,1,0,0],
                            [0,0,0,0,0,0,0,0,1,0],
                            [0,0,0,0,0,0,0,0,0,1]])     
    
      self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0],      # measurement function,
                            [0,1,0,0,0,0,0,0,0,0],
                            [0,0,1,0,0,0,0,0,0,0],
                            [0,0,0,1,0,0,0,0,0,0],
                            [0,0,0,0,1,0,0,0,0,0],
                            [0,0,0,0,0,1,0,0,0,0],
                            [0,0,0,0,0,0,1,0,0,0]])
    else:
      # with angular velocity
      self.kf = KalmanFilter(dim_x=11, dim_z=7)       
      self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0,0],      # state transition matrix A
                            [0,1,0,0,0,0,0,0,1,0,0],
                            [0,0,1,0,0,0,0,0,0,1,0],
                            [0,0,0,1,0,0,0,0,0,0,1],  
                            [0,0,0,0,1,0,0,0,0,0,0],
                            [0,0,0,0,0,1,0,0,0,0,0],
                            [0,0,0,0,0,0,1,0,0,0,0],
                            [0,0,0,0,0,0,0,1,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0],
                            [0,0,0,0,0,0,0,0,0,1,0],
                            [0,0,0,0,0,0,0,0,0,0,1]])     
     
      self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0,0],      # measurement function, H
                            [0,1,0,0,0,0,0,0,0,0,0],
                            [0,0,1,0,0,0,0,0,0,0,0],
                            [0,0,0,1,0,0,0,0,0,0,0],
                            [0,0,0,0,1,0,0,0,0,0,0],
                            [0,0,0,0,0,1,0,0,0,0,0],
                            [0,0,0,0,0,0,1,0,0,0,0]])

    # Initialize the covariance matrix, see covariance.py for more details
    covariance = Covariance(covariance_id)
    self.kf.P = covariance.P[tracking_name]  ## Covar_prev
    self.kf.Q = covariance.Q[tracking_name]
    self.kf.R = covariance.R[tracking_name]
    if not use_angular_velocity:
      self.kf.P = self.kf.P[:-1,:-1]
      self.kf.Q = self.kf.Q[:-1,:-1]

    self.kf.x[:7] = bbox3D.reshape((7, 1))

    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 1           # number of total hits including the first detection
    self.hit_streak = 1     # number of continuing hit considering the first detection
    self.first_continuing_hit = 1
    self.still_first = True
    self.age = 0
    # self.info = info        # other info
    # self.track_score = track_score
    self.tracking_name = tracking_name
    self.use_angular_velocity = use_angular_velocity

  def update(self, bbox3D): 
    """ 
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1          # number of continuing hit
    if self.still_first:
      self.first_continuing_hit += 1      # number of continuing hit in the fist time
    
    ######################### orientation correction
    if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the range
    if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

    new_theta = bbox3D[3]
    if new_theta >= np.pi: new_theta -= np.pi * 2    # make the theta still in the range
    if new_theta < -np.pi: new_theta += np.pi * 2
    bbox3D[3] = new_theta

    predicted_theta = self.kf.x[3]
    if abs(new_theta - predicted_theta) > np.pi / 2.0 and abs(new_theta - predicted_theta) < np.pi * 3 / 2.0:     # if the angle of two theta is not acute angle
      self.kf.x[3] += np.pi       
      if self.kf.x[3] > np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the range
      if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2
      
    # now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
    if abs(new_theta - self.kf.x[3]) >= np.pi * 3 / 2.0:
      if new_theta > 0: self.kf.x[3] += np.pi * 2
      else: self.kf.x[3] -= np.pi * 2
    
    ######################### 

    self.kf.update(bbox3D)

    if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the range
    if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2
    # self.info = info

  def predict(self):       
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    self.kf.predict()      
    if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2
    if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
      self.still_first = False
    self.time_since_update += 1
    self.history.append(self.kf.x)
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return self.kf.x[:7].reshape((7, ))



def angle_in_range(angle):
  '''
  Input angle: -2pi ~ 2pi
  Output angle: -pi ~ pi
  '''
  if angle > np.pi:
    angle -= 2 * np.pi
  if angle < -np.pi:
    angle += 2 * np.pi
  return angle

def diff_orientation_correction(det, trk):
  '''
  return the angle diff = det - trk
  if angle diff > 90 or < -90, rotate trk and update the angle diff
  '''
  diff = det - trk
  diff = angle_in_range(diff)
  if diff > np.pi / 2:
    diff -= np.pi
  if diff < -np.pi / 2:
    diff += np.pi
  diff = angle_in_range(diff)
  return diff

def greedy_match(distance_matrix):
  '''
  Find the one-to-one matching using greedy allgorithm choosing small distance
  distance_matrix: (num_detections, num_tracks)
  '''
  matched_indices = []

  num_detections, num_tracks = distance_matrix.shape
  distance_1d = distance_matrix.reshape(-1)
  index_1d = np.argsort(distance_1d)
  index_2d = np.stack([index_1d // num_tracks, index_1d % num_tracks], axis=1)
  detection_id_matches_to_tracking_id = [-1] * num_detections
  tracking_id_matches_to_detection_id = [-1] * num_tracks
  for sort_i in range(index_2d.shape[0]):
    detection_id = int(index_2d[sort_i][0])
    tracking_id = int(index_2d[sort_i][1])
    if tracking_id_matches_to_detection_id[tracking_id] == -1 and detection_id_matches_to_tracking_id[detection_id] == -1:
      tracking_id_matches_to_detection_id[tracking_id] = detection_id
      detection_id_matches_to_tracking_id[detection_id] = tracking_id
      matched_indices.append([detection_id, tracking_id])

  matched_indices = np.array(matched_indices)
  return matched_indices

def associate_detections_to_trackers(detections,trackers,iou_threshold=0.1, 
  use_mahalanobis=False, dets=None, trks=None, trks_S=None, mahalanobis_threshold=0.1, print_debug=False, match_algorithm='greedy'):
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  detections:  N x 8 x 3
  trackers:    M x 8 x 3

  dets: N x 7
  trks: M x 7
  trks_S: N x 7 x 7

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,8,3),dtype=int)    
  distance_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)

  if use_mahalanobis:
    assert(dets is not None)
    assert(trks is not None)
    assert(trks_S is not None)

  if use_mahalanobis and print_debug:
    print('dets.shape: ', dets.shape)
    print('dets: ', dets)
    print('trks.shape: ', trks.shape)
    print('trks: ', trks)
    print('trks_S.shape: ', trks_S.shape)
    print('trks_S: ', trks_S)
    S_inv = [np.linalg.inv(S_tmp) for S_tmp in trks_S]  # 7 x 7
    S_inv_diag = [S_inv_tmp.diagonal() for S_inv_tmp in S_inv]# 7
    print('S_inv_diag: ', S_inv_diag)

  for d,det in enumerate(detections):
    for t,trk in enumerate(trackers):
      S_inv = np.linalg.inv(trks_S[t]) # 7 x 7
      diff = np.expand_dims(dets[d] - trks[t], axis=1) # 7 x 1
      # manual reversed angle by 180 when diff > 90 or < -90 degree
      corrected_angle_diff = diff_orientation_correction(dets[d][3], trks[t][3])
      diff[3] = corrected_angle_diff
      distance_matrix[d, t] = np.sqrt(np.matmul(np.matmul(diff.T, S_inv), diff)[0][0])  ## make distance matrix

  matched_indices = greedy_match(distance_matrix)      ## Pairs

  if print_debug:
    print('distance_matrix.shape: ', distance_matrix.shape)
    print('distance_matrix: ', distance_matrix)
    print('matched_indices: ', matched_indices)

  unmatched_detections = []
  for d,det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t,trk in enumerate(trackers):
    if len(matched_indices) == 0 or (t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:                           ## matched_indices are pairs
    match = True
    if use_mahalanobis:
      if distance_matrix[m[0],m[1]] > mahalanobis_threshold:
        match = False
    if not match:
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  if print_debug:
    print('matches: ', matches)
    print('unmatched_detections: ', unmatched_detections)
    print('unmatched_trackers: ', unmatched_trackers)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class AB3DMOT(object):
  def __init__(self,covariance_id=0, max_age=2,min_hits=3, tracking_name='car', use_angular_velocity=False, tracking_waymo=False):
    """              
    observation: 
      before reorder: [h, w, l, x, y, z, rot_y]
      after reorder:  [x, y, z, rot_y, l, w, h]
    state:
      [x, y, z, rot_y, l, w, h, x_dot, y_dot, z_dot]
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.trackers = []
    self.frame_count = 0
    self.reorder = [3, 4, 5, 6, 2, 1, 0]
    self.reorder_back = [6, 5, 4, 0, 1, 2, 3]
    self.covariance_id = covariance_id
    self.tracking_name = tracking_name
    self.use_angular_velocity = use_angular_velocity
    self.tracking_waymo = tracking_waymo
             
  #mot_trackers[tracking_name].update(dets_all[tracking_name], match_distance, match_threshold, match_algorithm, scene_token)
  def update(self,dets_all, match_distance, match_threshold, match_algorithm, seq_name):
    """
    Params:
      dets_all: dict
        dets - a numpy array of detections in the format [[x,y,z,theta,l,w,h],[x,y,z,theta,l,w,h],...]
        info: a array of other info for each det
    Requires: this method must be called once for each frame even with empty detections.
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    dets = dets_all['dets']         # dets: N x 7, float numpy array
    #print('dets.shape: ', dets.shape)
    #print('info.shape: ', info.shape)
    dets = dets[:, self.reorder]


    self.frame_count += 1

    print_debug = False
    # if False and seq_name == '2f56eb47c64f43df8902d9f88aa8a019' and self.frame_count >= 25 and self.frame_count <= 30:
    #   print_debug = True
    #   print('self.frame_count: ', self.frame_count)
    # if print_debug:
    #   for trk_tmp in self.trackers:
    #     print('trk_tmp.id: ', trk_tmp.id)

    trks = np.zeros((len(self.trackers),7))         # N x 7 , #get predicted locations from existing trackers.
    to_del = []
    ret = []
    for t,trk in enumerate(trks):
      pos = self.trackers[t].predict().reshape((-1, 1))
      trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6]]       
      if(np.any(np.isnan(pos))):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))  ## delete empty space  
    for t in reversed(to_del):
      self.trackers.pop(t)

    if print_debug:
      for trk_tmp in self.trackers:
        print('trk_tmp.id: ', trk_tmp.id)

    dets_8corner = [convert_3dbox_to_8corner(det_tmp) for det_tmp in dets]
    if len(dets_8corner) > 0: dets_8corner = np.stack(dets_8corner, axis=0)
    else: dets_8corner = []

    trks_8corner = [convert_3dbox_to_8corner(trk_tmp) for trk_tmp in trks]
    trks_S = [np.matmul(np.matmul(tracker.kf.H, tracker.kf.P), tracker.kf.H.T) + tracker.kf.R for tracker in self.trackers]  ## eq(5) explicitly define to use for mahalanobis distance

    if len(trks_8corner) > 0: 
      trks_8corner = np.stack(trks_8corner, axis=0)
      trks_S = np.stack(trks_S, axis=0)
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets_8corner, trks_8corner, use_mahalanobis=True, dets=dets, trks=trks, trks_S=trks_S, mahalanobis_threshold=match_threshold, print_debug=print_debug, match_algorithm=match_algorithm)
   
    #update matched trackers with assigned detections
    for t,trk in enumerate(self.trackers):
      if t not in unmatched_trks:
        d = matched[np.where(matched[:,1]==t)[0],0]     # a list of index
        trk.update(dets[d,:][0])  ## 밑에서 만든 self.trackers 에서 꺼낸거기에 KalmanBoxTracker의 update임.
        # detection_score = info[d, :][0][-1]
        # trk.track_score = detection_score

    #create and initialise new trackers for unmatched detections
    for i in unmatched_dets:        # a scalar of index
        # detection_score = info[i][-1]
        # track_score = detection_score
        trk = KalmanBoxTracker(dets[i,:], self.covariance_id, self.tracking_name, use_angular_velocity) 
        self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()      # bbox location
        d = d[self.reorder_back]

        if((trk.time_since_update < self.max_age) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits)):      
          ret.append(np.concatenate((d, [trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
        i -= 1
        #remove dead tracklet
        if(trk.time_since_update >= self.max_age):   ## birth and death part
          self.trackers.pop(i)
    if(len(ret)>0):
      return np.concatenate(ret)      # x, y, z, theta, l, w, h, ID, other info, confidence
    return np.empty((0,15 + 7))    




def track_waymo(covariance_id, match_distance, match_threshold, match_algorithm, save_root, use_angular_velocity):
  '''
  submission {
    "meta": {
        "use_camera":   <bool>  -- Whether this submission uses camera data as an input.
        "use_lidar":    <bool>  -- Whether this submission uses lidar data as an input.
        "use_radar":    <bool>  -- Whether this submission uses radar data as an input.
        "use_map":      <bool>  -- Whether this submission uses map data as an input.
        "use_external": <bool>  -- Whether this submission uses external data as an input.
    },
    "results": {
        sample_token <str>: List[sample_result] -- Maps each sample_token to a list of sample_results.
    }
  }
  '''
  # save_dir = os.path.join(save_root, data_split); mkdir_if_missing(save_dir)
  pred_boxes, pred_samples = load_prediction()
  gt_boxes, scenes, _, transforms = load_gt() # scenes is need to match sample(timestamps in Waymo) with scenes, gt_boxes is just needed for input of create_tracks
  tracks = create_tracks(gt_boxes, scenes)  # tracks has connected data for scene_token and timestamp
  #t = sorted(tracks[scene_token].keys())[t_idx]

  ## get inverse trans_mat
  inv_trans_mats = {}
  for sampe_time, trans_v_w in transforms.items():
    trans_mat = np.reshape(trans_v_w,[4,4])
    determinant = np.linalg.det(trans_mat)
    if determinant != 0:
      inv_trans_mats[sampe_time] = np.linalg.inv(trans_mat)  # inverse exists
    else:
      inv_trans_mats[sample_time] = np.linalg.pinv(trans_mat) # when inverse doesn't exist do pseudo inverse

  ## Have to check if all transform matrices have inverse
  # checks = []
  # for _, trans_v_w in transforms.items():
  #   trans_mat = np.reshape(trans_v_w,[4,4])
  #   a = np.linalg.inv(trans_mat)
  #   b = np.linalg.pinv(trans_mat)
  #   #determinant = np.linalg.det(trans_mat)
  #   if np.array_equal(a,b) != False:
  #     checks.append(trans_mat)

  results = {}
  total_time = 0.0
  total_frames = 0
  processed_scene_tokens = set()
  for sample_token_idx in tqdm(range(len(pred_samples))):
    sample_time_token = pred_samples[sample_token_idx]
    scene_token = scenes[sample_time_token]
    if scene_token in processed_scene_tokens:
      continue

    sample_tokens = sorted(tracks[scene_token].keys())
    # first_sample_token = sample_tokens[0]
    # current_sample_token = first_sample_token
    ## 수정필요
    mot_trackers = {tracking_name: AB3DMOT(covariance_id, tracking_name=tracking_name, use_angular_velocity=use_angular_velocity, tracking_waymo=True) for tracking_name in WAYMO_TRACKING_NAMES}

    # while current_sample_token != '':
    for current_sample_token in sample_tokens:
      results[current_sample_token] = []
      dets = {tracking_name: [] for tracking_name in WAYMO_TRACKING_NAMES}
      # info = {tracking_name: [] for tracking_name in WAYMO_TRACKING_NAMES}
      for box in pred_boxes[current_sample_token]:
        trans_v_w = transforms[current_sample_token]
        pose_w, theta_w = pred_vehicle_to_world(trans_v_w, box)
        if box['detection_name'] not in WAYMO_TRACKING_NAMES:
          continue
        #q = Quaternion(box.rotation)
        #angle = q.angle if q.axis[2] > 0 else -q.angle   ## check
        angle = theta_w
        #print('box.rotation,  angle, axis: ', box.rotation, q.angle, q.axis)
        #print('box.rotation,  angle, axis: ', q.angle, q.axis)
        #[h, w, l, x, y, z, rot_y]
        detection = np.array([
          box['size'][2], box['size'][0], box['size'][1], 
          pose_w[0],  pose_w[1], pose_w[2],
          angle], dtype=np.float64)
        #print('detection: ', detection)
        #information = np.array([box.detection_score])
        dets[box['detection_name']].append(detection)
        #info[box.detection_name].append(information)
        
      dets_all = {tracking_name: {'dets': np.array(dets[tracking_name])} for tracking_name in WAYMO_TRACKING_NAMES}

      total_frames += 1
      start_time = time.time()
      for tracking_name in WAYMO_TRACKING_NAMES:
        if dets_all[tracking_name]['dets'].shape[0] > 0:
          trackers = mot_trackers[tracking_name].update(dets_all[tracking_name], match_distance, match_threshold, match_algorithm, scene_token)
          # (N, 9)
          # (h, w, l, x, y, z, rot_y), tracking_id, tracking_score 
          # print('trackers: ', trackers)
          for i in range(trackers.shape[0]):
            #sample_result = format_sample_result(current_sample_token, tracking_name, trackers[i])
            #results[current_sample_token].append(sample_result)
            
            ## transform pose from world to vehicle
            inv_trans_mat = inv_trans_mats[current_sample_token]
            pose_w = np.array([[trackers[i][3]],[trackers[i][4]],[trackers[i][5]],[1]])
            pose_v = np.matmul(inv_trans_mat, pose_w)

            ## transform angle from world to vehicle
            trans_mat = np.reshape(transforms[current_sample_token], [4,4])
            theta_v_w = np.arctan2(trans_mat[1][0], trans_mat[0][0])
            theta_w = trackers[i][6]
            theta_v = theta_w - theta_v_w
            if theta_v > np.pi:
                theta_v = theta_v - np.pi
            elif theta_v <= -np.pi:
                theta_v = theta_v + np.pi
            #tracking_id = tracks[scene_token][current_sample_token]
            create_result(pose_v, theta_v, trackers[i], tracking_name, scene_token, current_sample_token)

      cycle_time = time.time() - start_time
      total_time += cycle_time

      # get next frame and continue the while loop
      # time_idx += 1
      # current_sample_token = sorted(tracks[scene_token].keys())[time_idx]

    # left while loop and mark this scene as processed
    processed_scene_tokens.add(scene_token)

  # # finished tracking all scenes, write output data
  # output_data = {'meta': meta, 'results': results}
  # with open(output_path, 'w') as outfile:
  #   json.dump(output_data, outfile)

  print("Total Tracking took: %.3f for %d frames or %.1f FPS"%(total_time,total_frames,total_frames/total_time))

        


if __name__ == '__main__':
  covariance_id = 2
  match_distance = 'm'
  match_threshold = 11
  match_algorithm = 'greedy'
  use_angular_velocity = True
  save_root = os.path.join('./' + 'results/')
  track_waymo(covariance_id, match_distance, match_threshold, match_algorithm, save_root, use_angular_velocity)
