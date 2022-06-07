import cv2
from cv_analysis_tools import *
from analyze_texture import *
from dataset_utils import *
from arm_nav import *
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
      self.lower_arm_done = False
      self.upper_arm_done = False
      self.alset_state = alset_state
      self.cvu = CVAnalysisTools(self.alset_state)
      self.cfg = Config()
      self.arm_nav = ArmNavigation()
      self.pix_moved = 0

  def analyze(self, action, arm_movement, prev_img_path, curr_img_path, prev_stego_img, curr_stego_img, done=False, curr_func_name=None):
      save = False 
      frame_num = self.alset_state.get_frame_num()
      self.pix_moved, angle_moved, robot_loc = arm_movement
      arm_actions = ["UPPER_ARM_UP", "UPPER_ARM_DOWN", "LOWER_ARM_UP", "LOWER_ARM_DOWN"]
      opposite_dir = {"UPPER_ARM_UP":"UPPER_ARM_DOWN", "LOWER_ARM_UP":"LOWER_ARM_DOWN",
                      "UPPER_ARM_DOWN":"UPPER_ARM_UP", "LOWER_ARM_DOWN":"LOWER_ARM_UP"}
      # opt_flw = self.cvu.optflow(prev_img_path, curr_img_path, thresh = 0.05) 
      opt_flw = self.cvu.optflow(prev_img_path, curr_img_path, thresh=self.cfg.OPTFLOWTHRESH/11)
      # currently only supported known position is PARK_ARM_RETRACTED where
      # UPPER_ARM_UP and LOWER_ARM_DOWN are to their maximum settings.
      print(frame_num, "Arm Analyze: func, done: ", curr_func_name, done)
      if curr_func_name == "PARK_ARM_RETRACTED" and done:
        print("func PARK_ARM_RETRACTED done; set state to ARM_RETRACTED; pix_moved=", pix_moved)
        self.arm_state = "ARM_RETRACTED"
        for act in arm_actions:
          self.arm_cnt[act] = 0 
      elif frame_num == self.prev_frame_num + 1 and (action.startswith("UPPER_ARM") or action.startswith("LOWER_ARM")): 
        print("1 prev_frame, frame, prev_action, action:", self.prev_frame_num, frame_num, self.prev_action, action)

        # look for opt flow and upper arm or lower arm not moving in consistent direction
        unknown_action = "UNKNOWN_" + action
        if not opt_flw and self.arm_state.startswith("UNKNOWN") and self.lower_arm_done and self.upper_arm_done and action in ["UPPER_ARM_UP", "LOWER_ARM_DOWN"]:
          self.arm_state = "ARM_RETRACTED"
          for act in arm_actions:
            self.arm_cnt[act] = 0 
        elif not opt_flw and self.arm_state in ["UNKNOWN","ARM_RETRACTED"] and not self.lower_arm_done and action == "LOWER_ARM_DOWN":
          self.lower_arm_done = True
        elif opt_flw and self.arm_state in ["UNKNOWN"] and self.lower_arm_done and action == "LOWER_ARM_DOWN":
          self.lower_arm_done = False
        elif not opt_flw and self.arm_state.startswith("UNKNOWN") and not self.upper_arm_done and action == "UPPER_ARM_UP":
          self.upper_arm_done = True
        elif opt_flw and self.arm_state.startswith("UNKNOWN") and self.upper_arm_done and action == "UPPER_ARM_UP":
          self.upper_arm_done = False
        elif self.arm_state == "UNKNOWN":
          pass
        elif not opt_flw and self.arm_state == "ARM_RETRACTED" and action in ["UPPER_ARM_UP", "LOWER_ARM_DOWN"]:
          # self.arm_state = "ARM_RETRACTED_DELTA"
          self.arm_state = "ARM_RETRACTED"
        elif self.arm_state == "ARM_RETRACTED" and action in ["UPPER_ARM_UP", "LOWER_ARM_DOWN"]:
          # self.arm_state = "ARM_RETRACTED_DELTA"
          self.arm_state = "ARM_RETRACTED"
        elif not opt_flw and self.arm_state == "ARM_RETRACTED_DELTA":
          if action == "UPPER_ARM_UP":
            self.upper_arm_done = True
          elif action == "LOWER_ARM_DOWN":
            self.lower_arm_done = True
          if self.upper_arm_done and self.lower_arm_done:
            self.arm_state = "ARM_RETRACTED"
            self.arm_cnt[action] = 0 
            self.arm_cnt[opposite_dir[action]] = 0 
          else:
            self.arm_state = "ARM_RETRACTED_DELTA"
        elif opt_flw and self.arm_state in ["ARM_RETRACTED","ARM_RETRACTED_DELTA"]:
          self.arm_state = "ARM_RETRACTED_DELTA"
          if action.startswith("UPPER_ARM"):
            self.upper_arm_done = False
          elif action.startswith("LOWER_ARM"):
            self.lower_arm_done = False
          self.arm_cnt[action] += 1 
          # self.arm_cnt[opposite_dir[action]] -= 1 
        self.alset_state.set_known_robot_state(frame_num, self.arm_state, self.pix_moved)
      if self.arm_state in ["ARM_RETRACTED","ARM_RETRACTED_DELTA"]:
        print("ARM_NAV set_current_position:", self.arm_cnt)
        self.arm_nav.set_current_position(self.arm_cnt, update_plot=True, action=action, pixels_moved=self.pix_moved)
      self.prev_frame_num = frame_num
      self.prev_action = action
      print("arm opt_flw, uad, lad:", opt_flw, self.upper_arm_done, self.lower_arm_done)
      blackboard = [self.upper_arm_done, self.lower_arm_done]
      self.alset_state.record_arm_state(self.arm_state, self.arm_cnt, blackboard, frame_num)
      # note: safe_ground_info recorded by self.get_drivable_ground()
      return self.arm_state, self.arm_cnt

  def load_state(self):
      try:
        self.arm_state = self.alset_state.arm_state["ARM_POSITION"]
        self.arm_cnt = self.alset_state.arm_state["ARM_POSITION_COUNT"].copy()
      except:
        self.arm_state = "UNKNOWN"
        self.arm_cnt = {}
      blbrd = self.alset_state.get_blackboard("ARM")
      if blbrd is not None:
        [self.upper_arm_done, self.lower_arm_done] = blbrd
      else:
        self.upper_arm_done = False 
        self.lower_arm_done = False
