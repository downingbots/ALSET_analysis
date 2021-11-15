import cv2
from cv_analysis_tools import *
from analyze_texture import *
from dataset_utils import *
# from cube import *

class AnalyzeArm():

  def __init__(self, alset_state):
      self.stop_at_frame = 117
      self.gripper_img = None
      self.prev_frame_num = -1
      self.prev_action = ""
      self.background = None
      self.threshold = 10
      self.arm_cnt = {}
      self.arm_state = "UNKNOWN"
      self.alset_state = alset_state
      self.cvu = CVAnalysisTools()
      self.cfg = Config()

  def analyze(self, frame_num, action, prev_img_path, curr_img_path, done=False, curr_func_name=None):
      save = False 
      arm_actions = ["UPPER_ARM_UP", "UPPER_ARM_DOWN", "LOWER_ARM_UP", "LOWER_ARM_DOWN"]
      opposite_dir = {"UPPER_ARM_UP":"UPPER_ARM_DOWN", "LOWER_ARM_UP":"LOWER_ARM_DOWN",
                      "UPPER_ARM_DOWN":"UPPER_ARM_UP", "LOWER_ARM_DOWN":"LOWER_ARM_UP"}
      init = True
      for k in self.arm_cnt:
        if self.arm_cnt[k] != 0:
          init = False
      opt_flw = self.cvu.optflow(prev_img_path, curr_img_path, thresh = 0.05) 
      if curr_func_name == "PARK_ARM_RETRACTED" and done:
        self.arm_state = "PARKED_RETRACTED"
        for act in arm_actions:
          self.arm_cnt[act] = 0 
      elif frame_num == self.prev_frame_num + 1 and action.startswith("UPPER_ARM"): 
        print("1 prev_frame, frame, prev_action, action:", self.prev_frame_num, frame_num, self.prev_action, action)

        # look for opt flow and upper arm or lower arm not moving in consistent direction
        unknown_action = "UNKNOWN_" + action
        lower_arm_done = False
        upper_arm_done = False
        if not opt_flw and self.arm_state.startswith("UNKNOWN") and lower_arm_done and upper_arm_done:
          self.arm_state = "PARKED_RETRACTED"
          for act in arm_actions:
            self.arm_cnt[act] = 0 
        elif not opt_flw and self.arm_state in ["UNKNOWN","PARK_RETRACTED"] and not lower_arm_done and action.startswith("LOWER_ARM"):
          lower_arm_done == True
        elif opt_flw and self.arm_state in ["UNKNOWN"] and lower_arm_done and action.startswith("LOWER_ARM"):
          lower_arm_done == False
        elif not opt_flw and self.arm_state.startswith("UNKNOWN") and not upper_arm_done and action.startswith("UPPER_ARM"):
          upper_arm_done == True
        elif opt_flw and self.arm_state.startswith("UNKNOWN") and upper_arm_done and action.startswith("UPPER_ARM"):
          upper_arm_done == False
        elif self.arm_state == "UNKNOWN":
          pass
        elif not opt_flw and self.arm_state == "PARK_RETRACTED":
          pass
        elif not opt_flw and self.arm_state == "PARK_RETRACTED_DELTA":
          if action.startswith("UPPER_ARM"):
            upper_arm_done == True
          elif action.startswith("LOWER_ARM"):
            lower_arm_done == True
          if upper_arm_done and lower_arm_done:
            self.arm_state = "PARKED_RETRACTED"
            self.arm_cnt[act] = 0 
            self.arm_cnt[opposite_dir[act]] = 0 
          else:
            self.arm_state = "PARKED_RETRACTED_DELTA"
        elif opt_flw and self.arm_state in ["PARK_RETRACTED","PARK_RETRACTED_DELTA"]:
          self.arm_state = "PARKED_RETRACTED_DELTA"
          if action.startswith("UPPER_ARM"):
            upper_arm_done == False
          elif action.startswith("LOWER_ARM"):
            lower_arm_done == False
          self.arm_cnt[act] += 1 
          self.arm_cnt[opposite_dir[act]] -= 1 

      # self.prev_frame_num = frame_num
      # self.prev_action = action
      self.alset_state.record_arm_state(self.arm_state, self.arm_cnt)
      # note: safe_ground_info recorded by self.get_drivable_ground()
      return self.arm_state, self.arm_cnt
