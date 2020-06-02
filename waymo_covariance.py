import numpy as np


class Covariance(object):
  '''
  Define different Kalman Filter covariance matrix
  Kalman Filter states:
  [x, y, z, rot_y, l, w, h, x_dot, y_dot, z_dot]
  '''
  def __init__(self, covariance_id):
    if covariance_id == 2:
      self.num_states = 11 # with angular velocity
    else:
      self.num_states = 10
    self.num_observations = 3
    self.P = np.eye(self.num_states)
    self.Q = np.eye(self.num_states)
    self.R = np.eye(self.num_observations)

    # NUSCENES_TRACKING_NAMES = [
    #   'bicycle',
    #   'bus',
    #   'car',
    #   'motorcycle',
    #   'pedestrian',
    #   'trailer',
    #   'truck'
    # ]

    WAYMO_TRACKING_NAMES = [
      1, #'TYPE_VEHICLE',
      2, #'TYPE_PEDESTRIAN',
      4, #'TYPE_CYCLIST'
    ]

      # nuscenes
      # see get_nuscenes_stats.py for the details on  how the numbers come from
      #Kalman Filter state: [x, y, z, rot_z, l, w, h, x_dot, y_dot, z_dot, rot_z_dot]
      # P is initial state  ##initial STate의 covariance임
    P = {
      4: [0.0171637 , 0.01883819,0.28209099, 1.76560084, 0.01101036, 0.01824623, 0.04796067, 0.01355756, 0.01432095,0.01589215, 1.84338848],
      1: [0.07590666, 0.07557932, 0.68856205, 1.11118027, 0.03002007, 0.03768302, 0.2333028, 0.05443894, 0.05434438, 0.01092   , 0.93042048],
      2: [0.04026198, 0.04128253, 0.4189349 , 2.21789253 , 0.01826929, 0.02660983 , 0.02790554, 3.26013160e-02, 3.35970756e-02, 1.48890165e-02, 1.79724330e+00 ]
    }

    Q = {
      4: [1.43197298e+08, 2.34724662e+08, 4.09074035e+04, 2.54027723e+00, 0, 0, 0, 1.43197298e+08, 2.34724662e+08, 4.09074035e+04, 2.54027723e+00],
      1: [1.23491718e+08, 1.73895957e+08, 8.81231574e+04, 2.69821970e+00, 0, 0, 0, 1.23491718e+08, 1.73895957e+08, 8.81231574e+04, 2.69821970e+00],
      2: [1.59148748e+08, 2.06916453e+08, 3.19063116e+04, 2.61124847e+00, 0, 0, 0, 1.59148748e+08, 2.06916453e+08, 3.19063116e+04, 2.61124847e+00]
    }

    R = {
      4: [0.0171637 , 0.01883819,0.28209099, 1.76560084, 0.01101036, 0.01824623, 0.04796067],
      1: [0.07590666, 0.07557932, 0.68856205, 1.11118027, 0.03002007, 0.03768302, 0.2333028],
      2: [0.04026198, 0.04128253, 0.4189349 , 2.21789253 , 0.01826929, 0.02660983 , 0.02790554 ]
    }

    self.P = {tracking_name: np.diag(P[tracking_name]) for tracking_name in WAYMO_TRACKING_NAMES}
    self.Q = {tracking_name: np.diag(Q[tracking_name]) for tracking_name in WAYMO_TRACKING_NAMES}
    self.R = {tracking_name: np.diag(R[tracking_name]) for tracking_name in WAYMO_TRACKING_NAMES}
   
