from CPT import *
from config import *

class AnalyzeMove():
  def __init__(self, alset_state):
      # CPT-based move prediction
      self.alset_state = alset_state
      self.model = CPT(5)
      self.data = None
      self.target = None
      self.cfg = Config()
 
  def in_mappable_position(action, direction):
      pass

  def get_expected_movement(self, action):
      self.move_dist = {}
      for action in self.cfg.base_actions:
        self.move_dist[action] = {}
        self.move_dist[action]["MEAN"] = [None, 0]
        self.move_dist[action]["VAR"] = [None, 0]
        self.move_dist[action]["MAX"] = self.cfg.MAX_ROTATION_RADS
        self.move_dist[action]["MIN"] = 0
      return self.move_dist[action]

  def train_predictions(self,move, done=False):
      self.model.add_next_move(move, done)
      # self.data = self.model.add_next_data_list(move)

  def predict(self):
      prediction = self.model.predict_move()
      return prediction

  def estimated_battery_level(self):
      # track how movements decrease over time as battery levels go down
      pass

  # 
  # Position labels
  # 

  # use left/right views of map/new_map or of prev/curr_move image
  # computed and returned by map analysis
  def analyze_move(self, action, movement, done=False):
      self.train_predictions(action, done)

  def load_state(self):
      for fr_num in range(self.alset_state.frame_num):
        move =  self.alset_state.get_action(fr_num)
        self.train_predictions(move)
