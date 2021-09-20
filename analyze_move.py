from CPT import *
from config import *

class AnalyzeMove():
  def __init__(self):
      # CPT-based move prediction
      self.model = CPT()
      self.data = None
      self.target = None

      self.cfg = Config()
      #                           map loc, orient, 
      self.robot_location_hist = [[(0,0),  0]]
      self.robot_location    = None
      self.robot_orientation = None

      self.move_dist = {}
      for action in self.cfg.base_actions:
        self.move_dist[action] = {}
        self.move_dist[action]["MEAN"] = [None, 0]
        self.move_dist[action]["VAR"] = [None, 0]
        self.move_dist[action]["MAX"] = self.cfg.MAX_ROTATION_RADS 
        self.move_dist[action]["MIN"] = 0
      self.move_history = []

      self.robot_state = {}
      self.robot_state["KNOWN_STATE"] = None
      for action in self.cfg.arm_actions_no_wrist:
        self.robot_state[action] = 0
      self.robot_state_history = []

      self.position_label = {}
      self.location_label = {}

      self.app_name = None
      self.app_run_num = 0
      self.app_move_num = 0
 
  def in_mappable_position(action, direction):
      if (action not in self.cfg.base_actions or 
          self.robot_state["KNOWN_STATE"] is None or 
          (self.robot_state["KNOWN_STATE"] == "PARK_ARM_RETRACTED"
           and not self.robot_state["DELTA"])):
        return False

  def update_robot_location(action, direction):
      if not in_mappable_position(action, direction):
        # don't update map / location
        # combine history with optical flow estimates
        pass

  def record_metrics(self, frame_path, data):
      # record Metrics for different rotation angles of a frame
      # record Line/KP rotation analysis
      pass

  def new_app_run(self, app_name):
      self.app_name = app_name
      self.app_run_num += 1
      self.app_move_num = 0

  def get_expected_movement(self, action):
      return self.move_dist[action]

  def train_predictions(self,move):
      self.data = self.model.add_next_data_list([move])

  def predict(self):
      if self.data is None:
        return None
      print("len data:", len(self.data))
      self.model.train(self.data)
      prediction = self.model.predict(self.data,self.data,3,1)
      return prediction

  def estimated_battery_level(self):
      # track how movements decrease over time as battery levels go down
      pass

  # 
  # known state: keep track of arm movements since the arm was 
  # parked in a known position
  # 
  def set_known_robot_state(self, new_known_state):
      self.state_history.append([self.robot_state[:]])
      self.robot_state["KNOWN_STATE"] = new_known_state
      self.robot_state["DELTA"] = False
      for action in self.arm_actions_no_wrist:
        self.robot_state[action] = 0

  def get_robot_state(self):
      return [self.robot_state]

  def add_delta_from_known_state(self, action):
      if action in self.arm_actions_no_wrist:
        self.robot_state_delta[move] += 1
      self.robot_state["DELTA"] = True

  # 
  # Position labels
  # 
  def label_robot_state(self, label):
      # based upon successful functional action
      self.robot_state_label[label] = [self.robot_state]

  def label_robot_location(self, label):
      # based upon successful functional action
      self.location_label[label] = [self.robot_location, self.robot_orientation]

  # use left/right views of map/new_map or of prev/curr_move image
  # computed and returned by map analysis
  def analyze_move(self, action, movement):
      self.app_move_num += 1
      self.move_history.append(action, movement)
      self.update_robot_location(action, movement)
      self.train_predictions(action)
      self.add_delta_from_known_state(action)

  # Gripper Analysis also tracks robot movement and objects
  # use optical flow views of map/new_map
      pass

