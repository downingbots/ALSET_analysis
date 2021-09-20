import cv2
from cv_analysis_tools import *

class AnalyzeGripper():

  def __init__(self):
      self.stop_at_frame = 117
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
      self.add_edges = False   # didn't seem to add much, if anything
      self.cvu = CVAnalysisTools()

  def detect_robot_body(self, image_path):
      # store non-gripper image fragments of the robot body
      # todo
      pass

  def detect_grabbed_object(self, image_path):
      # register non-gripper image fragments of the grabbed object
      # with other classes that collect images of interest.
      # todo
      pass

  # bound the images based upon movement
  # then get gripper contours from the area of movement
  def gripper_edges(self, image_path):
      image = cv2.imread(image_path)
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      # Threshold the image
      ret, thresh = cv2.threshold(gray, 20, 255, 0)
      edges = cv2.Canny(gray, 50, 200, None, 3)
      cv2.imshow("gripper_edges", edges)
      cv2.waitKey(0)

  # then get gripper contours from the area of movement
  def gripper_contours(self, image_path):
      image = cv2.imread(image_path)
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      # Threshold the image
      ret, thresh = cv2.threshold(gray, 20, 255, 0)
      # Find contours 
      contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
      # print("contours:",contours)
      # print("hierarchy:",hierarchy)
      # Iterate through each contour and compute the approx contour
      for c in contours:
          # Calculate accuracy as a percent of the contour perimeter
          accuracy = 0.03 * cv2.arcLength(c, True)
          approx = cv2.approxPolyDP(c, accuracy, True)
          # cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)
      # cv2.imshow("gripper_contours", image)
      # cv2.waitKey(0)

  def analyze(self, frame_num, action, prev_img_path, curr_img_path, done=False):
      save = False 
      if self.gripper_open_cnt == 0 and self.gripper_close_cnt == 0:
        init = True
      else:
        init = False
      if frame_num == self.prev_frame_num + 1 and action == "GRIPPER_OPEN" and action == self.prev_action:
        print("1 prev_frame, frame, prev_action, action:", self.prev_frame_num, frame_num, self.prev_action, action)
        # self.gripper_img = self.cvu.moved_pixels(prev_img_path, curr_img_path, init)
        if self.cvu.optflow(prev_img_path, curr_img_path, add_edges=self.add_edges):
          self.gripper_img = self.cvu.moved_pixels(prev_img_path, curr_img_path, init)
        if done:
          self.gripper_state = "FULLY_OPEN"
        elif not self.cvu.optflow(prev_img_path, curr_img_path, add_edges=self.add_edges) and self.gripper_state != "FULLY_CLOSE":
          self.gripper_state = "FULLY_OPEN"
        else:
          # do OPT check if unchanged, then done
          self.gripper_open_cnt += 1
          self.gripper_close_cnt = 0
          self.gripper_state = "PARTIALLY_OPEN"
        print("gripper state: ", self.prev_frame_num, self.gripper_open_cnt, self.gripper_close_cnt, self.gripper_state)
        if frame_num >= self.stop_at_frame:
          cv2.imshow("Gripper FG", self.gripper_img)
          cv2.waitKey(0)

      elif frame_num == self.prev_frame_num + 1 and action == "GRIPPER_CLOSE" and action == self.prev_action:
        print("2 prev_frame, frame, prev_action, action:", self.prev_frame_num, frame_num, self.prev_action, action)
        if self.cvu.optflow(prev_img_path, curr_img_path, add_edges=self.add_edges):
          # self.gripper_img = self.cvu.moved_pixels_over_time(prev_img_path, curr_img_path, init)
          self.gripper_img = self.cvu.moved_pixels(prev_img_path, curr_img_path, init)
        if done:
          self.gripper_state = "FULLY_CLOSE"
        elif not self.cvu.optflow(prev_img_path, curr_img_path) and self.gripper_state != "FULLY_OPEN":
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
      # gripper contour and edges do worse than pixel movements (averaged/overlapped).
      # self.gripper_contours(curr_img_path)
      # self.gripper_edges(curr_img_path)
         

  def check_cube_in_gripper(self, frame_num, action, prev_img_path, curr_img_path, done):
      pass

