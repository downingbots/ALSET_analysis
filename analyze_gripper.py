import cv2
from cv_analysis_tools import *

class AnalyzeGripper():

  def __init__(self):
      self.gripper_img = None
      self.prev_frame_num = -1
      self.prev_action = ""
      self.background = None
      self.threshold = 10
      self.gripper_img = None
      self.gripper_history = []
      self.gripper_open_cnt = 0
      self.gripper_close_cnt = 0
      self.gripper_state = "UNKNOWN"
      self.cvu = CVAnalysisTools()

  def analyze(self, frame_num, action, prev_img, curr_img, done=False):
      save = False 
      if self.gripper_open_cnt == 0 and self.gripper_close_cnt == 0:
        init = True
      else:
        init = False
      if frame_num == self.prev_frame_num + 1 and action == "GRIPPER_OPEN" and action == self.prev_action:
        print("1 prev_frame, frame, prev_action, action:", self.prev_frame_num, frame_num, self.prev_action, action)
        # self.gripper_img = self.cvu.moved_pixels(prev_img, curr_img, init)
        if self.cvu.optflow(prev_img, curr_img):
          self.gripper_img = self.cvu.moved_pixels(prev_img, curr_img, init)
        if done:
          self.gripper_state = "FULLY_OPEN"
        elif not self.cvu.optflow(prev_img, curr_img) and self.gripper_state != "FULLY_CLOSE":
          self.gripper_state = "FULLY_OPEN"
        else:
          # do OPT check if unchanged, then done
          self.gripper_open_cnt += 1
          self.gripper_close_cnt = 0
          self.gripper_state = "PARTIALLY_OPEN"
        print("gripper state: ", self.prev_frame_num, self.gripper_open_cnt, self.gripper_close_cnt, self.gripper_state)
      elif frame_num == self.prev_frame_num + 1 and action == "GRIPPER_CLOSE" and action == self.prev_action:
        print("2 prev_frame, frame, prev_action, action:", self.prev_frame_num, frame_num, self.prev_action, action)
        if self.cvu.optflow(prev_img, curr_img):
          # self.gripper_img = self.cvu.moved_pixels_over_time(prev_img, curr_img, init)
          self.gripper_img = self.cvu.moved_pixels(prev_img, curr_img, init)
        if done:
          self.gripper_state = "FULLY_CLOSE"
        elif not self.cvu.optflow(prev_img, curr_img) and self.gripper_state != "FULLY_OPEN":
          self.gripper_state = "FULLY_CLOSE"
        else:
          self.gripper_close_cnt += 1
          self.gripper_open_cnt = 0
          self.gripper_state = "PARTIALLY_CLOSE"
        print("gripper state: ", self.prev_frame_num, self.gripper_open_cnt, self.gripper_close_cnt, self.gripper_state)
      else:
        print("3 prev_frame, frame, prev_action, action:", self.prev_frame_num, frame_num, self.prev_action, action)
        save = True 
        # self.gripper_open_cnt = 0
        # self.gripper_close_cnt = 0

      if save or done:
        self.gripper_history.append([self.prev_frame_num, self.gripper_img, self.gripper_open_cnt, self.gripper_close_cnt, self.gripper_state])
        print("gripper state: ", self.prev_frame_num, self.gripper_open_cnt, self.gripper_close_cnt, self.gripper_state)
      self.prev_frame_num = frame_num
      self.prev_action = action
         

  def check_cube_in_gripper(self, frame_num, action, prev_img, curr_img, done):
      pass

