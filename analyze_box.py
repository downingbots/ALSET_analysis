#!/usr/bin/env python
from alset_state import *
from arm_nav import *
from matplotlib import pyplot as plt
import cv2
import STEGO.src
from alset_stego import *
import numpy as np
import copy
from math import sin, cos, pi, sqrt
from utilborders import *
from cv_analysis_tools import *
from dataset_utils import *
from PIL import Image
import imutils
from skimage.feature import peak_local_max
from scipy import ndimage
from scipy.cluster.hierarchy import ward
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN
from sklearn import metrics, linear_model
import matplotlib.image as mpimg
from sortedcontainers import SortedList, SortedSet, SortedDict

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

class AnalyzeLines(object):
  def line_intersection(self, line1, line2):
      xdiff = (line1[0][0] - line1[0][2], line2[0][0] - line2[0][2])
      ydiff = (line1[0][1] - line1[0][3], line2[0][1] - line2[0][3])
  
      def det(a, b):
          return a[0] * b[1] - a[1] * b[0]
  
      div = det(xdiff, ydiff)
      if div == 0:
         return None
         # raise Exception('lines do not intersect')
  
      # d = (det(line1[0][0:1], line1[0][2:3]), det(line2[0][0:1],line2[0][2:3]))
      d = (det(line1[0][0:2], line1[0][2:4]), det(line2[0][0:2],line2[0][2:4]))
      x = det(d, xdiff) / div
      y = det(d, ydiff) / div
      # print("intersection: ", line1, line2, [x,y])
      if ( x >= max(line1[0][0], line1[0][2]) + 2
        or x <= min(line1[0][0], line1[0][2]) - 2
        or x >= max(line2[0][0], line2[0][2]) + 2
        or x <= min(line2[0][0], line2[0][2]) - 2
        or y <= min(line1[0][1], line1[0][3]) - 2
        or y >= max(line1[0][1], line1[0][3]) + 2
        or y <= min(line2[0][1], line2[0][3]) - 2
        or y >= max(line2[0][1], line2[0][3])) + 2:
        # intersection point outside of line segments' range
        return None
         
      xdiff = (line1[0][0] - line1[0][2], line2[0][0] - line2[0][2])
      ydiff = (line1[0][1] - line1[0][3], line2[0][1] - line2[0][3])
      return [x, y]
  
  def is_same_line(self, line1, line2):
      if self.is_parallel(line1, line2) and self.line_intersection(line1, line2) is not None:
        # print("same line: line1, line2", line1, line2)
        return True
  
  def extend_line(self, line1, line2):
      def det(a, b):
          return a[0] * b[1] - a[1] * b[0]
      def get_dist(x1,y1, x2, y2):
          return sqrt((x2-x1)**2 + (y2-y1)**2)
  
      if not self.is_parallel(line1, line2):
        return None
  
      dist0 = get_dist(line1[0][0], line1[0][1], line2[0][0], line1[0][1])
      dist1 = get_dist(line1[0][0], line1[0][1], line2[0][2], line1[0][3])
      dist2 = get_dist(line1[0][2], line1[0][3], line2[0][0], line1[0][1])
      dist3 = get_dist(line1[0][2], line1[0][3], line2[0][2], line1[0][3])
  
      extended_line = None
      if dist0 == max(dist0,dist1,dist2,dist3):
        extended_line = [[line1[0][0], line1[0][1], line2[0][0], line1[0][1]]]
      elif dist1 == max(dist0,dist1,dist2,dist3):
        extended_line = [[line1[0][0], line1[0][1], line2[0][2], line1[0][3]]]
      elif dist2 == max(dist0,dist1,dist2,dist3):
        extended_line = [[line1[0][2], line1[0][3], line2[0][0], line1[0][1]]]
      elif dist3 == max(dist0,dist1,dist2,dist3):
        extended_line = [[line1[0][2], line1[0][3], line2[0][2], line1[0][3]]]
     
      if not (self.is_parallel(line1, extended_line) and self.is_parallel(line2, extended_line)):
        return None
      # print("broken line: ", line1, line2, extended_line)
      return extended_line

  def is_broken_line(self, line1, line2):
      if self.line_intersection(line1, line2) is not None:
        return None
      return self.extend_line(line1, line2)
  
  def is_parallel(self, line1, line2):
      angle1 = np.arctan2((line1[0][0]-line1[0][2]), (line1[0][1]-line1[0][3]))
      angle2 = np.arctan2((line2[0][0]-line2[0][2]), (line2[0][1]-line2[0][3]))
      allowed_delta = .1
      if abs(angle1-angle2) <= allowed_delta:
        # print("is_parallel line1, line2", line1, line2, angle1, angle2)
        return True
      if abs(np.pi-abs(angle1-angle2)) <= allowed_delta:
        # note: .01 and 3.14 should be considered parallel
        # print("is_parallel line1, line2", line1, line2, angle1, angle2)
        return True
      return False
  
  def parallel_dist(self, line1, line2, dbg=False):
      if not self.is_parallel(line1, line2):
        return None
      # line1, line2 [[151 138 223 149]] [[ 38  76 139  96]]
  
      # y = mx + c
      # pts = [(line1[0][0], line1[0][2]), (line1[0][1], line1[0][3])]
      pts = [(line1[0][0], line1[0][1]), (line1[0][2], line1[0][3])]
      x_coords, y_coords = zip(*pts)
      A = np.vstack([x_coords,np.ones(len(x_coords))]).T
      l1_m, l1_c = np.linalg.lstsq(A, y_coords)[0]
      if dbg:
        print("x,y,m,c", x_coords, y_coords, l1_m, l1_c)
  
      pts = [(line2[0][0], line2[0][1]), (line2[0][2], line2[0][3])]
      x_coords, y_coords = zip(*pts)
      A = np.vstack([x_coords,np.ones(len(x_coords))]).T
      l2_m, l2_c = np.linalg.lstsq(A, y_coords)[0]
      if dbg:
        print("x,y,m,c", x_coords, y_coords, l2_m, l2_c)
  
      # coefficients = np.polyfit(x_val, y_val, 1)
      # Goal: set vert(y) the same on both lines, compute horiz(x).
      # with a vertical line, displacement will be very hard to compute
      # unless same end-points are displaced.
      if ((line1[0][0] >= line2[0][0] >= line1[0][2]) or
          (line1[0][0] <= line2[0][0] <= line1[0][2])):
        x1 = line2[0][0]
        y1 = line2[0][1]
        y2 = y1
        x2 = (y2 - l1_c) / l1_m
        # y2 = l1_m * x1 + l1_c
        # x2 = (y1 - l2_c) / l2_m
      elif ((line1[0][0] >= line2[0][2] >= line1[0][2]) or
            (line1[0][0] <= line2[0][2] <= line1[0][2])):
        x1 = line2[0][2]
        y1 = line2[0][3]
        y2 = y1
        x2 = (y2 - l1_c) / l1_m
        # y2 = l1_m * x1 + l1_c
        # x2 = (y1 - l2_c) / l2_m
      elif ((line2[0][0] >= line1[0][0] >= line2[0][2]) or
            (line2[0][0] <= line1[0][0] <= line2[0][2])):
        x1 = line1[0][0]
        y1 = line1[0][1]
        y2 = y1
        x2 = (y2 - l2_c) / l2_m
      elif ((line2[0][0] >= line1[0][2] >= line2[0][2]) or
            (line2[0][0] <= line1[0][2] <= line2[0][2])):
        x1 = line1[0][2]
        y1 = line1[0][3]
        y2 = y1
        x2 = (y2 - l2_c) / l2_m
      elif ((line1[0][1] >= line2[0][1] >= line1[0][3]) or
            (line1[0][1] <= line2[0][1] <= line1[0][3])):
        x1 = line2[0][0]
        y1 = line2[0][1]
        y2 = y1
        x2 = (y2 - l1_c) / l1_m
      elif ((line1[0][1] >= line2[0][3] >= line1[0][3]) or
            (line1[0][1] <= line2[0][3] <= line1[0][3])):
        y1 = line2[0][3]
        x1 = line2[0][2]
        y2 = y1
        x2 = (y2 - l1_c) / l1_m
      elif ((line2[0][1] >= line1[0][1] >= line2[0][3]) or
            (line2[0][1] <= line1[0][1] <= line2[0][3])):
        y1 = line1[0][1]
        x1 = line1[0][0]
        y2 = y1
        x2 = (y2 - l2_c) / l2_m
      elif ((line2[0][1] >= line1[0][3] >= line2[0][3]) or
            (line2[0][1] <= line1[0][3] <= line2[0][3])):
        y1 = line1[0][3]
        x1 = line1[0][2]
        y2 = y1
        x2 = (y2 - l2_c) / l2_m
      else:
        return None
      # print("parallel_dist", (x1-x2),(y1-y2))
      return x1-x2, y1 - y2
  
  ##########################################################
  # Using line analysis and Stego, track various obstacles
  ##########################################################
  #       ___   w2 = 3
  #      /   \
  #     /     \   h = 3D
  #    /       \
  #    123456789   w1=9
  #
  #    w1/d1 = w2/d2  => but d1 is near zero????. depends on angle of camera.
  #                      get pi camera angle
  #                      pi FOV: 62.2deg x 48.8deg
  #    w1/w2 = d1/d2 = (d2-d1)
  #                   => use to get distance traveled
  #                   => # pixels moved?
  #
  #    If you know the end width, you can compute the distance to the end
  #
  # vanishing point 
  # [1  0  0  0   [ D       [Dx
  #  0  1  0  0  *  0 ]  =   Dy
  #  0  0  1  0]             Dz]
  #
  def track_table(self, img):
      # perspective, width/length scale
      pass
  
  def track_wall(self, img):
      pass
  
  def track_ground_barrier(self, img):
      pass
  
  def track_lines(self, img):
      pass
  
  def track_road(self, img):
      pass
  
  
  def get_hough_lines(self, img, max_line_gap = 10):
        # rho_resolution = 1
        # theta_resolution = np.pi/180
        # threshold = 155
        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        #  threshold = 15  # minimum number of votes (intersections in Hough grid cell)
        threshold = 10  # minimum number of votes (intersections in Hough grid cell)
        # min_line_length = 50  # minimum number of pixels making up a line
        min_line_length = 40  # minimum number of pixels making up a line
        # max_line_gap = 10  # maximum gap in pixels between connectable line segments
        # max_line_gap = 5  # maximum gap in pixels between connectable line segments
  
        # Output "lines" is an array containing endpoints of detected line segments
        hough_lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                      min_line_length, max_line_gap)
        return hough_lines
  
  def get_hough_lines_img(self, hough_lines, gray):
      hough_lines_image = np.zeros_like(gray)
      if hough_lines is not None:
        for line in hough_lines:
          for x1,y1,x2,y2 in line:
            cv2.line(hough_lines_image,(x1,y1),(x2,y2),(255,0,0),3)
      else:
        print("No houghlines")
      return hough_lines_image
  
  def get_box_lines(self, curr_img, gripper_img = None):
        try:
          gray = cv2.cvtColor(curr_img.copy(), cv2.COLOR_BGR2GRAY)
        except:
          gray = curr_img.copy()
          # gray = cv2.bitwise_not(gray)
        edges = cv2.Canny(gray, 50, 200, None, 3)
        edges = cv2.dilate(edges,None,iterations = 1)
        # cv2.imshow("edges1", edges)
        if gripper_img is not None: 
          edges = cv2.bitwise_and(edges, cv2.bitwise_not(gripper_img))
          # cv2.imshow("edges2", edges)
        hough_lines = self.get_hough_lines(edges)
        hough_lines_image = np.zeros_like(curr_img)
        # print("hough_lines", hough_lines)
        if hough_lines is None:
          return None
        for line in hough_lines:
          for x1,y1,x2,y2 in line:
            cv2.line(hough_lines_image,(x1,y1),(x2,y2),(255,0,0),5)
        # cv2.imshow("box lines houghlines", hough_lines_image)
        # cv2.waitKey(0)
        return hough_lines
  
  def exact_line(self, l1, l2):
      if l1[0][0]==l2[0][0] and l1[0][1]==l2[0][1] and l1[0][2]==l2[0][2] and l1[0][3]==l1[0][3]:
        return True
      return False
  
  def display_lines(self, img_label, line_lst, curr_img):
      lines_image = np.zeros_like(curr_img)
      for line in line_lst:
        print("line:", line)
        # for x1,y1,x2,y2 in line[0]:
        x1,y1,x2,y2 = line[0]
        cv2.line(lines_image,(x1,y1),(x2,y2),(255,0,0),5)
      cv2.imshow(img_label, lines_image)
  
  def display_line_pairs(self, img_label, line_pair_lst, curr_img, mode=2):
      lines_image = np.zeros_like(curr_img)
      for line0, line1, rslt in line_pair_lst:
        if mode == 0:
          # print("line0:", img_label, line0)
          pass
        elif mode == 1:
          # print("line1:", img_label, line1)
          pass
        elif mode == 2:
          pass
          # print("line0,line1:", img_label, line0, line1)
        if mode == 0 or mode == 2:
          x1,y1,x2,y2 = line0[0]
          cv2.line(lines_image,(x1,y1),(x2,y2),(255,0,0),3)
        if mode == 1 or mode == 2:
          x1,y1,x2,y2 = line1[0]
          cv2.line(lines_image,(x1,y1),(x2,y2),(130,0,0),5)
      cv2.imshow(img_label, lines_image)
  
  
  def analyze_box_lines(self, dataset, box_lines, actions, gripper_img, drop_off_img, img_paths):
      angle_sd = SortedDict()
      box_line_intersections = []
      world_lines = []
      arm_pos = []
      num_ds = len(box_lines)
      prev_lines = None
      hough_lines = None
      num_passes = 4
      # ret, gripper_img = cv2.threshold(gripper_img, 100, 255, cv2.THRESH_TOZERO)
      ret, drop_off_img = cv2.threshold(drop_off_img, 100, 255, cv2.THRESH_TOZERO)
      line_group = [[]]  # includes gripper line_group[0]
      gripper_lines = self.get_hough_lines(gripper_img)
      gripper_lines2 = self.get_hough_lines(gripper_img)
      gripper_lines_image = np.zeros_like(gripper_img)
      min_gripper_y = 100000000
      for line in gripper_lines2:
        for x1,y1,x2,y2 in line:
          cv2.line(gripper_lines_image,(x1,y1),(x2,y2),(255,0,0),5)
          for y in [y1,y2]:
            if y < min_gripper_y:
              MIN_GRIPPER_Y = y
  
      cv2.imshow("gripper lines", gripper_lines_image)
      unknown_lines = []
      same_frame_parallel = []
      diff_frame_parallel = []
      min_diff_frame_parallel = []
      broken_lines = []
      gripper_lines = []
      gripper_lines2 = []
      drop_off_gray = cv2.cvtColor(drop_off_img, cv2.COLOR_BGR2GRAY)
      drop_off_lines = self.get_hough_lines(drop_off_gray)
      drop_off_lines_image = np.zeros_like(drop_off_gray)
      for line in drop_off_lines:
        for x1,y1,x2,y2 in line:
          cv2.line(drop_off_lines_image,(x1,y1),(x2,y2),(255,0,0),5)
      cv2.imshow("drop_off lines", drop_off_lines_image)
      # cv2.waitKey(0)
  
      action_cnt = -1
      persist_line_stats = []
      persist_line = []
      persist_line_most_recent = [] # [[ds_num, img_num]]
      persist_line_min_max_ds_img = []
      persist_line_counter = []
      for ds_num, ds in enumerate(dataset):
        for pass_num in range(num_passes):
          num_imgs = len(box_lines[ds_num])
          if pass_num == 0:
            unknown_lines.append([])
            same_frame_parallel.append([])
            diff_frame_parallel.append([])
            min_diff_frame_parallel.append([])
            broken_lines.append([])
          for img_num in range(num_imgs):
            new_img_num = True
            if pass_num == 0:
              unknown_lines[ds_num].append([])
              same_frame_parallel[ds_num].append([])
              diff_frame_parallel[ds_num].append([])
              min_diff_frame_parallel[ds_num].append([])
              broken_lines[ds_num].append([])
            action_cnt += 1
            prev_lines = hough_lines
            hough_lines = box_lines[ds_num][img_num]
            if hough_lines is None:
              hough_lines = []
            if img_num > 0 and box_lines[ds_num][img_num-1] is not None:
              prev_hough_lines = box_lines[ds_num][img_num-1]
            else:
              prev_hough_lines = []
            for hl_num, h_line in enumerate(hough_lines):
              # print(pass_num, hl_num, "new h_line:", h_line)
              # Gripper lines now computed differently
              unknown_lines[ds_num][img_num].append(copy.deepcopy(h_line))
              
              # Gripper Lines should be the same from frame to frame
              if len(h_line) == 0:
                continue
              if pass_num == 0 and False:
                # Gripper lines now computed differently
                unknown_lines[ds_num].append([])
                # find gripper (line_group[0])
                found = False
                # TODO: increase strength of unmoved lines.
                # remove moved lines, esp over FWD/REV/L/R.
                for g_line in line_group[0]:
                  if len(g_line) == 0:
                    continue
                  if self.is_same_line(h_line, g_line):
                    # gripper_lines.append(g_line)
                    gripper_lines2.append(copy.deepcopy(g_line))
                    # line_group[0].append(g_line)
                    found = True
                    break
                # if found:
                #   continue
                # for g_line in gripper_lines:
                for g_line in gripper_lines2:
                  if len(g_line) == 0:
                    continue
                  if self.is_same_line(h_line, g_line):
                    # gripper_lines.append(g_line)
                    gripper_lines2.append(copy.deepcopy(g_line))
                    # line_group[0].append(g_line)
                    found = True
                    break
                # if found:
                #   continue
  
                for g_line in prev_hough_lines:
                  if len(g_line) == 0:
                    continue
                  if self.is_same_line(h_line, g_line):
                    # gripper_lines.append(g_line)
                    gripper_lines2.append(copy.deepcopy(g_line))
                    # line_group[0].append(g_line)
                    found = True
                # if found:
                #   continue
  
              if pass_num == 1:
                # for u_line in unknown_lines[ds_num][-3:-1]:
                for u_line in unknown_lines[ds_num][img_num]:
                  if len(u_line) == 0:
                    continue
                  # broken lines
                  broken_line = self.is_broken_line(u_line, h_line)
                  if broken_line is not None:
                    # check if already known broken line
                    found = False
                    for x1_line, x2_line, b_line in broken_lines[ds_num][img_num]:
                      if (self.exact_line(x1_line, h_line) and self.exact_line(x2_line, u_line)) or (self.exact_line(x1_line, u_line) and self.exact_line(x2_line, h_line)):
                        found = True
                        break
                    if not found:
                      # new broken line
                      # print("broken_line", ds_num, img_num)
                      broken_lines[ds_num][img_num].append([copy.deepcopy(u_line), copy.deepcopy(h_line), broken_line])
                min_dist = 100000000
                best_p_line = None
                for p_line in prev_hough_lines:
                    if len(p_line) == 0:
                      continue
                    dist = self.parallel_dist(p_line, h_line)
                    if dist is not None:
                      found = False
                      for x1_line, x2_line, d in diff_frame_parallel[ds_num][img_num]:
                        if self.exact_line(x1_line, h_line) and self.exact_line(x2_line, p_line):
                          found = True
                          break
                      if not found :
                        # print("diff_frame_parallel", ds_num, img_num)
                        diff_frame_parallel[ds_num][img_num].append([copy.deepcopy(p_line), copy.deepcopy(h_line), dist])
                        if min_dist > sqrt(abs(dist[0])+abs(dist[1])):
                          min_dist = sqrt(abs(dist[0])+abs(dist[1]))
                          best_p_line = copy.deepcopy(p_line)
                if best_p_line is not None:
                  disp = self.parallel_dist(best_p_line, h_line)
                  min_diff_frame_parallel[ds_num][img_num].append([copy.deepcopy(best_p_line), copy.deepcopy(h_line), disp])
                # cv2.waitKey(0)
  
              if pass_num == 2:
                # display broken lines
                self.display_line_pairs("broken lines", broken_lines[ds_num][img_num], drop_off_img)
                # note: we could generalize this analysis to be
                # based upon edge contours instead of lines.
                # display diff_frame_parallel lines
                self.display_line_pairs("best moved lines", min_diff_frame_parallel[ds_num][img_num], drop_off_img)
  
            if pass_num == 1:
              if img_num == num_imgs-1:
                print("##################")
                print("Image Sorted Lines")
                print("##################")
                for is_ds_num, is_ds in enumerate(dataset):
                  is_num_imgs = len(is_ds)
                  for is_img_num in range(is_num_imgs):
                    hough_lines = box_lines[is_ds_num][is_img_num]
                    if hough_lines is None:
                      print("No hough_lines for", is_ds_num, is_img_num)
                      continue
                    else:
                      # for hl_num, h_line in enumerate(hough_lines):
                        # print((is_ds_num, is_img_num, hl_num), h_line)
                      pass
  
            if pass_num == 2:
              # print("pass_num", pass_num)
        
              ###################
              # Angle Sorted Dict
              ###################
              hough_lines = box_lines[ds_num][img_num]
              if hough_lines is None:
                print("No hough_lines for", ds_num, img_num)
                continue
              else:
                # print("##################")
                # print("Angle Sorted Lines")
                # print("##################")
                for hl_num, h_line in enumerate(hough_lines):
                  angle = np.arctan2((h_line[0][0]-h_line[0][2]), (h_line[0][1]-h_line[0][3]))
                  if angle <= 0:
                    angle += np.pi
                  try_again = True
                  while try_again:
                    try:
                      # ensure slightly unique angle so no lines are overwritten
                      val = angle_sd[angle]
                      angle += .000001
                      try_again = True
                    except:
                      try_again = False
                  angle_sd[angle] = (ds_num, img_num, h_line)
  
    
              if img_num == num_imgs-1:
                print("##################")
                print("Angle Sorted Lines")
                print("##################")
                skv = angle_sd.items()
                for item in skv:
                  print(item)
    
              ###########
              # Intersections within same image
              if hough_lines is None:
                print("No hough_lines for", ds_num, img_num)
                # continue
              else:
                for hl_num, h_line in enumerate(hough_lines):
                  for hl_num2, h_line2 in enumerate(hough_lines):
                    if hl_num2 > hl_num:
                      xy_intersect = self.line_intersection(h_line, h_line2)
                      if xy_intersect is not None:
                        box_line_intersections.append([ds_num, img_num, xy_intersect, h_line, h_line2])
              if img_num == num_imgs-1:
                print("##################")
                print("Line Intersections")
                print("##################")
                for bli in box_line_intersections:
                   print(bli)
                # ARD: todo: find same intersections between frames
                # to determine disparity, scale
  
      ############
      # find persistent lines
      # We've now done the basic characterizations of the lines.  
      # Now go through the angle data to find persistent lines.
      ############
      num_lines = 0
      # persist_line = []
      # persist_line_most_recent = [] # [[ds_num, img_num]]
      # persist_line[lineno] = {(ds_num, img_num), [key1, key2]}
      # new persist_line[lineno][(d,i for d in range(num_ds) for i in range(num_imgs))] = []
      persist_line_3d = []  
      # 3d_persist_line[line_num] = [composite_3D_line]
      max_angle_dif = .1   # any bigger, can be purged from cache
      line_cache = []      # [(line_num, angle)] 
      akv = angle_sd.items()
      min_dist_allowed = 3
      a_cnt = 0
# ARD: No ds 1's persist lines ?!
      for angle, item in akv:
        a_cnt += 1
        if a_cnt % 50 == 0 or len(persist_line) < 7:
          print(a_cnt, "angle:", angle, len(persist_line))
        # angle, pl_angle => too big to be parallel even though same line.... 
        ds_num, img_num, a_line = item
        min_dist = 1000000000000
        best_pl_num = -1
        best_pl_line = -1
        best_pl_ds_num = -1
        best_pl_img_num = -1
        max_counter = -1
        max_counter_num = -1
        # instead of going through each persist line,
        # go to persist line with angles above/below curr angle.
        for pl_num, pl in enumerate(persist_line):
          mr_pl_ds_num, mr_pl_img_num = persist_line_most_recent[pl_num]
          pl_angle_lst = persist_line[pl_num][(mr_pl_ds_num, mr_pl_img_num)]
          if len(persist_line) < 7:
            print("pl ds,img num:", mr_pl_ds_num, mr_pl_img_num)
            # print("pl_angle_lst:", pl_angle_lst)
          for pl_angle in pl_angle_lst:
            pl_item = angle_sd[pl_angle]
            pl_ds_num, pl_img_num, pl_line = pl_item
            dist = self.parallel_dist(pl_line, a_line)
            if dist is None:
              # if not parallel, skip this line
              break
            # if len(persist_line) < 7:
            if True:
              # check if close
              cont = True
              for i in range(4):
                if abs(pl_line[0][i] - a_line[0][i]) > 4:
                 cont = False
              if cont:
                dist = self.parallel_dist(pl_line, a_line, False)
                print("pl_angle, item:", pl_angle, pl_item)
                print("pl_line:", pl_line, a_line)
                print("pl_dist:", dist)
                if dist is not None:
                  print(min_dist,sqrt(dist[0]**2+dist[1]**2))
            if dist is None:
              # lines may be broken continutions of each other 
              extended_line = self.is_broken_line(pl_line, a_line)
              if extended_line is not None:
                dist = self.parallel_dist(pl_line, extended_line)
                if len(persist_line) < 7:
                  print("dist:", dist, pl_line, extended_line)
            if dist is not None and min_dist > sqrt(dist[0]**2+dist[1]**2):
              min_dist = sqrt(dist[0]**2+dist[1]**2)
              best_pl_num = pl_num
              best_pl_ds_num = mr_pl_ds_num
              best_pl_img_num = mr_pl_img_num

        # ARD TODO: Approx Same angle.  Big overlap in horiz (small angle) 
        # OR vert direction (small angle) in horiz & vert angle (middle angle)

        ############
        # add best persistent line match 
        ############
        # if ds_num == best_pl_ds_num: 
        #   print("min_dist", min_dist, abs(img_num - best_pl_img_num), best_pl_img_num)
     
        if ds_num == best_pl_ds_num and min_dist < abs(img_num - best_pl_img_num+1) * min_dist_allowed:
          # same line
          # print("p_l",persist_line[best_pl_num])
          # print("p_l2", persist_line[best_pl_num][(best_pl_ds_num, best_pl_img_num)])
          try:
            lst = persist_line[best_pl_num][(ds_num, img_num)]
            lst.append(angle)
          except:
            print("BUG: PERSIST_LINE ANGLE_LIST", angle, ds_num, img_num)
            persist_line[best_pl_num][(ds_num, img_num)] = []
            persist_line[best_pl_num][(ds_num, img_num)].append(angle)
          persist_line_most_recent[best_pl_num] = [ds_num, img_num]
        else:
          persist_line_most_recent.append([ds_num, img_num])
          if len(persist_line) < 7:
              print("best pl, ds, img:", best_pl_num, best_pl_ds_num, best_pl_img_num)
          persist_line.append({})
          persist_line[-1][(ds_num, img_num)] = []
          persist_line[-1][(ds_num, img_num)].append(angle)

      ############
      # Compute persistent line statistics 
      ############
      # persist_line_stats = []
      mean_gripper_line = []
      mean_gripper_line1 = []
      mean_gripper_line2 = []
      # persist_line_counter = []
      # persist_line_min_max_ds_img = []
      non_gripper = []
      none_disp = []
      big_count = []
      density = []
      for pl_num, pl in enumerate(persist_line):
        print("PERSISTENT LINE #", pl_num)
        persist_line_stats.append(None)
        counter = 0
        angle_list = None
        running_sum_x = 0
        running_sum_y = 0
        running_sum2_x = 0
        running_sum2_y = 0
        got_disp = False
        dispcnt = 0
        running_sum_counter = 0
        running_sum2_counter = 0
        running_sum_disp_x = 0
        running_sum_disp_y = 0
        running_sum2_disp_x = 0
        running_sum2_disp_y = 0
        running_line_length = 0
        running_line = [0,0,0,0]
        running_angle = 0
        # for ds = 0, get max/min img#
        persist_line_min_max_ds_img.append([1000000, -1])
        a_ds_num = -1
        a_img_num = -1
        a_line = []
        # Hmmm... only for single DS?
        # for ds_num, ds in enumerate(dataset):
          # for pass_num in range(num_passes):
        for pl_ds_num in range(num_ds):
          if pl_ds_num > 0:
            # TODO: num_imgs depends on ds_num; get ds_num 0 to work first.
            print("pl_ds_num > 0", pl_ds_num)
            break
          for pl_img_num in range(num_imgs):
            try:
              angle_list = pl[(pl_ds_num, pl_img_num)]
            except:
              continue
            # first and last line appearance
            if pl_img_num  < persist_line_min_max_ds_img[pl_num][0]:
              persist_line_min_max_ds_img[pl_num][0] = pl_img_num
            if pl_img_num  > persist_line_min_max_ds_img[pl_num][1]:
              persist_line_min_max_ds_img[pl_num][1] = pl_img_num
 
            for pl_angle in angle_list:
              prev_a_line = copy.deepcopy(a_line)
              prev_ds_num = a_ds_num
              prev_img_num = a_img_num
              asd_item = angle_sd[pl_angle]
              a_ds_num, a_img_num, a_line = asd_item
              if prev_ds_num != -1:
                disp = self.parallel_dist(prev_a_line, a_line)
                if disp is not None:
                  got_disp = True
                  running_sum_disp_x += disp[0]
                  running_sum_disp_y += disp[1]
                  running_sum2_disp_x += disp[0] * disp[0]
                  running_sum2_disp_y += disp[1] * disp[1]
                  dispcnt += 1
              running_sum_x += (a_line[0][0] + a_line[0][2])/2
              running_sum_y += (a_line[0][1] + a_line[0][3])/2
              for i in range(4):
                running_line[i] += a_line[0][i]
              x_dif = abs(a_line[0][0] - a_line[0][2])
              y_dif = abs(a_line[0][1] - a_line[0][3])
              running_line_length += sqrt(x_dif*x_dif + y_dif*y_dif)
              running_angle += pl_angle
              counter += 1
        if counter == 0:
          print("counter:", counter)
          print("angle_list:", angle_list)
          print("pl, ds, img:", pl_num, a_ds_num, a_img_num)
          continue
        if dispcnt == 0:
          stddev_disp_x = None
          stddev_disp_y = None
          mean_disp_x = None
          mean_disp_y = None
        else:
          stddev_disp_x = sqrt(running_sum2_disp_x / dispcnt - running_sum_disp_x * running_sum_disp_x / dispcnt / dispcnt)
          stddev_disp_y = sqrt(running_sum2_disp_y / dispcnt - running_sum_disp_y * running_sum_disp_y / dispcnt / dispcnt)
          mean_disp_x = running_sum_disp_x / dispcnt
          mean_disp_y = running_sum_disp_y / dispcnt
        mean_x = running_sum_x / counter
        mean_y = running_sum_y / counter
        mean_line_length = running_line_length / counter
        mean_angle = running_angle / counter
        mean_line = [[0,0,0,0]]
        for i in range(4):
          mean_line[0][i] = int(running_line[i]/counter)
        print("mean disp, angle, linlen:", got_disp, mean_disp_x, mean_disp_y, mean_angle, mean_line_length, mean_line, counter) 
        persist_line_stats[pl_num] = [mean_disp_x, mean_disp_y, stddev_disp_x, stddev_disp_y, mean_x, mean_y, copy.deepcopy(mean_line), mean_line_length, mean_angle, counter, copy.deepcopy(persist_line_min_max_ds_img[pl_num])]
        persist_line_counter.append(counter)
        running_sum_counter += counter
        running_sum2_counter += counter * counter
        if got_disp and (abs(mean_disp_x) + abs(mean_disp_y) < 1):
          mean_gripper_line2.append(mean_line)
          if got_disp and (abs(mean_disp_x) + abs(mean_disp_y) < .5):
            mean_gripper_line1.append(mean_line)
            if got_disp and (abs(mean_disp_x) + abs(mean_disp_y) < .001):
              mean_gripper_line.append(mean_line)
        elif got_disp:
          non_gripper.append(mean_line)
        else:
          none_disp.append(mean_line)
        
        if counter > 7:
          dens = ((persist_line_min_max_ds_img[pl_num][1] - persist_line_min_max_ds_img[pl_num][0] +1) / counter)
          if dens >= .2:
            big_count.append(mean_line)
            density.append(dens)
            if max_counter < counter:
              max_counter = counter
              max_counter_num = pl_num
      print("pl density:", density)
      if len(big_count) == 0:
        big_count.append([[0,0,0,0]])
      sum_density = 0
      for dense in density:
        sum_density += dense
      print("mean pl density:", (sum_density/len(big_count)))
      counter_cnt = len(persist_line_counter)
      mean_counter = running_sum_counter / counter_cnt
      stddev_counter = sqrt(running_sum2_counter / counter_cnt - running_sum_counter * running_sum_counter / counter_cnt / counter_cnt)
      print("counter mean, stdev", mean_counter, stddev_counter)
      # MIN_GRIPPER_Y
      for pl_num in [max_counter_num]:
        print("PERSIST LINE", pl_num)
        for pl_item in persist_line[pl_num].items():
          pl_key, angle_list = pl_item
          for a_num, pl_angle in enumerate(angle_list):
            asd_item = angle_sd[pl_angle]
            a_ds_num, a_img_num, a_line = asd_item
            print(pl_key, a_num, a_line, persist_line_min_max_ds_img[pl_num])
 
      # print("mean_gripper_line1", len(mean_gripper_line1))
      # print("mean_gripper_line2", len(mean_gripper_line2))
      # print("non_gripper_line", len(non_gripper))
      print("none_disp", len(none_disp))
      print("mean_gripper_line", len(mean_gripper_line))
      self.display_lines("Mean_Gripper_Lines", mean_gripper_line, drop_off_img)
      # cv2.waitKey(0)
      # display_lines("Mean_Gripper_Lines1", mean_gripper_line1, drop_off_img)
      # display_lines("Mean_Gripper_Lines2", mean_gripper_line2, drop_off_img)
      # display_lines("Mean_NonGripper_Line", non_gripper, drop_off_img)
      self.display_lines("Mean_BigCount", big_count, drop_off_img)
      # cv2.waitKey(0)
 
      mean_counter = running_sum_counter / counter_cnt
      stddev_counter = sqrt(running_sum2_counter / counter_cnt - running_sum_counter * running_sum_counter / counter_cnt / counter_cnt)
      print("counter mean, stdev", mean_counter, stddev_counter)
      # MIN_GRIPPER_Y
      return persist_line, persist_line_stats, angle_sd
  

  def map_to_segment(self, persist_line, gripper_img):
      # map to the segment from STEGO, make BB
      if True:
              pl_last_img_line = {}
              # ARD: eventually fix the nesting of the ds_num loops
              # for ds_num, ds in enumerate(dataset):
              #   for pass_num in range(num_passes):
              for pl_ds_num in range(num_ds):
                pl_num_imgs = len(dataset[pl_ds_num])
                for pl_img_num in range(pl_num_imgs):
                  bb = []
                  bb_maxw, bb_minw, bb_maxh, bb_minh = -1, 10000, -1, 10000
                  for pl_num in range(len(persist_line)):
                    pl_stats = persist_line_stats[pl_num]
                    if pl_stats is not None:
                      [mean_disp_x, mean_disp_y, stddev_disp_x, stddev_disp_y, mean_x, mean_y, mean_line, mean_line_length, mean_angle, counter, [pl_min_img_num, pl_max_img_num]] = pl_stats
                      print("PL", pl_num, counter, mean_line, pl_min_img_num, pl_max_img_num)
                    else:
                      continue
                    if mean_y > MIN_GRIPPER_Y:
                      continue
                    if counter < mean_counter:
                      continue
                    a_line = None
                    try:
                      print("pl angle_list:")
                      angle_list = persist_line[pl_num][(pl_ds_num, pl_img_num)]
                      # print(angle_list)
                      l_maxw, l_minw, l_maxh, l_minh = -1, 10000, -1, 10000
                      for a_num, pl_angle in enumerate(angle_list):
                        asd_item = angle_sd[pl_angle]
                        a_ds_num, a_img_num, a_line = asd_item
                        l_maxw = max(a_line[0][0], a_line[0][2], l_maxw)
                        l_minw = min(a_line[0][0], a_line[0][2], l_minw)
                        l_maxh = max(a_line[0][1], a_line[0][3], l_maxh)
                        l_minh = min(a_line[0][1], a_line[0][3], l_minh)
                        if l_maxh > MIN_GRIPPER_Y:
                          l_maxh = MIN_GRIPPER_Y
                      pl_last_img_line[pl_num] = [l_maxw, l_minw, l_maxh, l_minh]
                    except:
                      print("except pl angle_list:")
                      try:
                        [l_maxw, l_minw, l_maxh, l_minh] = pl_last_img_line[pl_num]
                      except:
                        continue
                    if l_maxw == -1 or l_maxh == -1:
                      print("skipping PL", pl_num)
                      continue
      #           bb = make_bb(bb_maxw, bb_minw, bb_maxh, bb_minh)
      #           img_path = img_paths[pl_ds_num][pl_img_num]
      #           img = cv2.imread(img_path)
      #           bb_img = get_bb_img(img, bb)
      #           print(pl_img_num, "bb", bb)
      #           cv2.imshow("bb", bb_img)
      #           # cv2.waitKey(0)

  def add_estimated_lines(self, persist_line):
      pass

  def display_persist_lines(self, dataset, persist_line, persist_line_stats, angle_sd, img):
      def get_plline(pl_stats, angle_list, angle_sd):
          [mean_disp_x, mean_disp_y, stddev_disp_x, stddev_disp_y, mean_x, mean_y, mean_line, mean_line_length, mean_angle, counter, [pl_min_img_num, pl_max_img_num]] = pl_stats
          extended_line = None
          for a_num, pl_angle in enumerate(angle_list):
            asd_item = angle_sd[pl_angle]
            a_ds_num, a_img_num, a_line = asd_item
            if extended_line is None:
              extended_line = copy.deepcopy(a_line)
            else:
              # make line longer if necessary
              extended_line = extend_line(extended_line, a_line)
          # Possibly mod line to use mean_line if appropriate
          return extended_line

      unique_color = {}
      for ds_num,ds in enumerate(dataset):
        num_imgs = len(ds)
        for img_num in range(num_imgs):
          lines_image = np.zeros_like(img)
          for pl_num in range(len(persist_line)):
            pl_stats = persist_line_stats[pl_num]
            if pl_stats is not None:
              [mean_disp_x, mean_disp_y, stddev_disp_x, stddev_disp_y, mean_x, mean_y, mean_line, mean_line_length, mean_angle, counter, [pl_min_img_num, pl_max_img_num]] = pl_stats
            else:
              continue
            # if mean_y > MIN_GRIPPER_Y:
            #   continue
            if counter <= 7:
              continue
            dens = ((pl_max_img_num - pl_min_img_num +1) / counter)
            if pl_min_img_num > img_num or pl_max_img_num < img_num:
              continue
            if dens < .2:
              continue
            print("PL", pl_num, len(persist_line), counter, mean_line, pl_min_img_num, pl_max_img_num)
            angle_list = None
            prev_angle_list = None
            next_angle_list = None
            try:
              angle_list = persist_line[pl_num][(ds_num, img_num)]
              pl_img_num = img_num
              print("PL img num", img_num, pl_stats, angle_list)
              new_line = get_plline(pl_stats, angle_list, angle_sd)
            except:
              pl_img_num = img_num
              prev_img_num = None
              while True:
                prev_img_num = None
                pl_img_num -= 1
                print(pl_num, ds_num, pl_img_num, "PL key", persist_line[pl_num].keys())
                try:
                  prev_angle_list = persist_line[pl_num][(ds_num, pl_img_num)]
                  prev_img_num = pl_img_num
                  print("Prev PL img num", pl_img_num)
                  prev_line = get_plline(pl_stats, prev_angle_list, angle_sd)
                  break
                except:
                  if pl_img_num < pl_min_img_num:
                    print("FAILURE1", pl_img_num, pl_min_img_num)
                    break
                  continue
              if prev_img_num is None:
                continue
              pl_img_num = img_num
              while True:
                next_img_num = None
                pl_img_num += 1
                try:
                  next_angle_list = persist_line[pl_num][(ds_num, pl_img_num)]
                  next_img_num = pl_img_num
                  next_line = get_plline(pl_stats, next_angle_list, angle_sd)
                  print("Next PL img num", pl_img_num)
                  break
                except:
                  if pl_img_num > pl_max_img_num:
                    print("FAILURE2", pl_img_num, pl_max_img_num)
                    break
                  continue
              if next_img_num is None:
                continue
              pct = (pl_img_num - prev_img_num) / (next_img_num - prev_img_num)
              new_line = [[0,0,0,0]]
              print("prev_line", prev_line)
              print("next_line", next_line)
              for i in range(4):
                pl = prev_line[0][i]
                nl = next_line[0][i]
                new_line[0][i] = round(pl + (pl-nl)*pct)
              x1,y1,x2,y2 = new_line[0]
              cv2.line(lines_image,(x1,y1),(x2,y2),(255,0,0),5)
          print("persist lines", ds_num, img_num) 
          cv2.imshow("persistent lines", lines_image)
          [time, app, mode, func_name, action, img_name, img_path] = self.dsu.get_dataset_info(ds[img_num],mode="FUNC") 
          ds_img = cv2.imread(img_path)
          stego_img, unique_color, stego_key = stego(ds_img, unique_color)
          cv2.imshow("stego img", stego_img)
          cv2.waitKey(30)

  def __init__(self):
      self.arm_nav = ArmNavigation()
  
      self.alset_state = AlsetState()
      self.cvu = CVAnalysisTools(self.alset_state)
      func_idx_file = "sample_code/TT_BOX.txt"
      self.dsu = DatasetUtils(app_name="TT", app_type="FUNC")
      dataset = [[]]
      self.curr_dataset = dataset[0]
      self.fwd_actions = [[]]
      self.arm_pos = []
      unmoved_pix = None
      slow_moved_pix = None
      self.delta_arm_pos = {"UPPER_ARM_UP":0,"UPPER_ARM_DOWN":0,
                            "LOWER_ARM_UP":0,"LOWER_ARM_DOWN":0}
      self.final_delta_arm_pos = []
      self.arm_nav.set_current_position(self.delta_arm_pos, update_plot=True)
      drop_off_img = []
      drop_off_img.append(None)
      num_datasets = 0
      num_images = 0
      func_name = ""
      img = None
      prev_func_name = ""
      unique_color = {}
      curr_ds_num = 0
      stego_prev_func_name = None
      stego_func_name = None
      stego_img = None
      DO_STEGO = False
      # DO_STEGO = False
      with open(func_idx_file, 'r') as file1:
        while True:
          ds_idx = file1.readline()
          if not ds_idx:
            break
          if ds_idx[-1:] == '\n':
            ds_idx = ds_idx[0:-1]
          # ./apps/FUNC/GOTO_BOX_WITH_CUBE/dataset_indexes/FUNC_GOTO_BOX_WITH_CUBE_21_05_16a.txt
          prev_func_name = func_name
          func_name = self.dsu.get_func_name_from_idx(ds_idx)
          # if func_name == "GOTO_BOX_WITH_CUBE" and prev_func_name != "GOTO_BOX_WITH_CUBE":
          if prev_func_name == "DROP_CUBE_IN_BOX" and func_name != "DROP_CUBE_IN_BOX":
            # func_name = self.dsu.dataset_idx_to_func(ds_idx)
            self.final_delta_arm_pos.append(copy.deepcopy(self.delta_arm_pos))
            if curr_ds_num == 1:
              # just debug with first 2 datasets initially
              print("BREAK")
              break
            self.delta_arm_pos = {"UPPER_ARM_UP":0,"UPPER_ARM_DOWN":0,
                                  "LOWER_ARM_UP":0,"LOWER_ARM_DOWN":0}
            self.arm_nav.set_current_position(self.delta_arm_pos, update_plot=True)
            curr_ds_num += 1
            drop_off_img.append(None)
            self.fwd_actions.append([])
            dataset.append([])
            self.curr_dataset = dataset[curr_ds_num]
          self.arm_pos = []
          with open(ds_idx, 'r') as file2:
            while True:
              img_line = file2.readline()
              # 21:24:34 ./apps/FUNC/QUICK_SEARCH_FOR_BOX_WITH_CUBE/dataset/QUICK_SEARCH_FOR_BOX_WITH_CUBE/LEFT/9e56a302-b5fe-11eb-83c4-16f63a1aa8c9.jpg
              if not img_line:
                 break
              print("img_line", img_line)
              self.curr_dataset.append(img_line)
              stego_prev_func_name = stego_func_name
              [time, app, mode, stego_func_name, action, img_name, img_path] = self.dsu.get_dataset_info(img_line,mode="FUNC") 
              func_name = stego_func_name
              self.fwd_actions[curr_ds_num].append(action)
              stego_img = cv2.imread(img_path)
              print("DO_STEGO BOX", stego_prev_func_name, func_name)
              if ((stego_prev_func_name is not None and stego_prev_func_name == "GOTO_BOX_WITH_CUBE" and stego_func_name == "DROP_CUBE_IN_BOX") or
                 (stego_prev_func_name is not None and stego_prev_func_name == "QUICK_SEARCH_FOR_BOX_WITH_CUBE" and stego_func_name == "GOTO_BOX_WITH_CUBE")):
                 DO_STEGO = True
              DO_STEGO = True
              if DO_STEGO:
                if stego_img is not None:
                  stego_img, unique_color, stego_plot_img = stego(stego_img, unique_color)
                  print("BOX STRAIGHT AHEAD")
                  cv2.imshow("BOX STRAIGHT AHEAD", stego_img)
                  self.find_polygon(stego_img, unique_color)
                  # cv2.waitKey(30)
                else:
                  print("bug with BOX STRAIGHT AHEAD")
                  cv2.waitKey(30)
              if action == "GRIPPER_OPEN":
                img = cv2.imread(img_path)
                drop_off_img[curr_ds_num] = img
                print("drop_off_img", curr_ds_num, len(drop_off_img))
              if action in ["UPPER_ARM_UP", "UPPER_ARM_DOWN", "LOWER_ARM_UP", "LOWER_ARM_DOWN"]:
                self.delta_arm_pos[action] += 1
                print(self.delta_arm_pos.items())
                img = cv2.imread(img_path)
                self.arm_nav.set_current_position(self.delta_arm_pos, update_plot=True, img=img)
              self.arm_pos.append(self.delta_arm_pos.copy())
  
      # print("dataset",dataset)
      for doi in drop_off_img:
        if doi is None:
          print("drop_off_img None, cnt", len(drop_off_img))
          print("actions", self.fwd_actions)
          
      img = None
      img_paths = []
      prev_img = None
      num_passes = 4
      box_lines = []
      actions = []
      # img_copy = []
      edge_copy = []
      rl_copy = []
      delta_arm_pos_copy = self.delta_arm_pos.copy()
      compute_gripper = True
      for pass_num in range(num_passes):
        # if pass_num > 0:
        #   self.delta_arm_pos = delta_arm_pos_copy.copy()
        for ds_num, ds in enumerate(dataset):
          # img_copy_num = len(ds) // 24
          ################################
          # REMOVE to handle more ds_nu
          ################################
          self.delta_arm_pos = copy.deepcopy(self.final_delta_arm_pos[ds_num])
          self.arm_nav.set_current_position(self.delta_arm_pos, update_plot=True)
          if pass_num == 0:
            box_lines.append([])
            actions.append([])
            img_paths.append([])
          for img_num, img_line in enumerate(reversed(ds)):
            # note: img_num is really the reversed img num!
            # img_num = len(ds) - rev_img_num
            [time, app, mode, func_name, action, img_name, img_path] = self.dsu.get_dataset_info(img_line,mode="FUNC") 
            prev_action = action
            img_paths[ds_num].append(img_path)
            actions[ds_num].append(action)
            if img is not None:
              prev_img = img
            img = cv2.imread(img_path)
            # print("img_path", img_path)
            if action in ["UPPER_ARM_UP", "UPPER_ARM_DOWN", "LOWER_ARM_UP", "LOWER_ARM_DOWN"]:
              self.delta_arm_pos[action] -= 1   # going in reversed(ds) order
              self.arm_nav.set_current_position(self.delta_arm_pos, update_plot=True, img=img)
            if pass_num == 0:
              adj_img,mean_diff,rl = self.cvu.adjust_light(img_path)
              if rl is not None:
                rl_img = img.copy()
                mask = rl["LABEL"]==rl["LIGHT"]
                mask = mask.reshape((rl_img.shape[:2]))
                # print("mask",mask)
                rl_img[mask==rl["LIGHT"]] = [0,0,0]
                # adj_img = rl_img
                center = np.uint8(rl["CENTER"].copy())
                rl_copy = rl["LABEL"].copy()
                res    = center[rl_copy.flatten()]
                rl_img2  = res.reshape((img.shape[:2]))
              if prev_img is not None:
                gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 200, None, 3)
                edges = cv2.dilate(edges,None,iterations = 1)
                d_edges = cv2.dilate(edges,None,iterations = 3)
                edge_copy.append(d_edges.copy())
              box_lines[ds_num].append([])
  
            ##################################
            #  Compute Gripper
            ##################################
            elif pass_num == 1:
              if compute_gripper:
                compute_gripper = False
                # num_ors = int(np.sqrt(len(edge_copy)))
                num_ors = min(6, len(edge_copy))
                # img_gap = len(edge_copy) // num_ors
                # 3, 12, 13
                print("num edge_copy", num_ors, len(edge_copy), len(ds))
                running_ors = None
                running_ands = None
                do_or = 0
                for i2 in range(2, len(edge_copy)-2):
                  if do_or == 0:
                    running_ors = cv2.bitwise_or(edge_copy[i2-1], edge_copy[i2])
                    do_or = 2
                  else:
                    running_ors = cv2.bitwise_or(running_ors, edge_copy[i2])
                    do_or += 1
                  if do_or == num_ors:
                    do_or = 0
                    if running_ands is None:
                      running_ands = running_ors
                    else:
                      running_ands = cv2.bitwise_and(running_ands, running_ors)
                gripper = running_ands
                cv2.imshow("running_gripper", running_ands)
                # cv2.waitKey(0)
              # overwrite previous attempt at getting gripper.
              print("box_lines sz", len(box_lines), len(box_lines[ds_num]), img_num)
              # 1,0,0
              box_lines[ds_num][img_num] = self.get_box_lines(img, gripper)
              box_lines_image = self.get_hough_lines_img(box_lines[ds_num][img_num], img)
              cv2.imshow("box hough lines", box_lines_image)
            elif pass_num == 2:
              persist_line,persist_line_stats, angle_sd = self.analyze_box_lines(dataset, box_lines, actions, gripper, drop_off_img[ds_num], img_paths)
              self.display_persist_lines(dataset, persist_line, persist_line_stats, angle_sd, img)
            elif False and pass_num == 3:
              stego_img, unique_color, stego_key = stego(img, unique_color)
              # cv2.imshow("stego orig input img", img)
              cv2.imshow("stego img", stego_img)
              stego_gray = cv2.cvtColor(stego_img, cv2.COLOR_BGR2GRAY)
              # do again but with each color isolated?
              stego_hough_lines = self.get_box_lines(stego_img)
              stego_hough_lines_image = self.get_hough_lines_img(stego_hough_lines, stego_gray)
              cv2.imshow("stego hough lines", stego_hough_lines_image)
              # cv2.imshow("stego gray", stego_gray)
  
              # cv2.imshow("stego adj input img", adj_img)
              # cv2.imshow("stego input img", rl_img)
              # cv2.imshow("stego rl img", rl_img2)
              cv2.waitKey(0)
  
  
  # allow detection of tiny squares
  def find_polygon(self, img, unique_color):
        # already done by find_cube: # convert the stitched image to grayscale and threshold it
        # such that all pixels greater than zero are set to 255
        # (foreground) while all others remain 0 (background)
        for obj_key, obj_color in unique_color.items():
          obj_img = img.copy()
          print("obj_color", obj_color)
          print("obj_img[0][0]", obj_img[0][0])
          print("l0,l00:",len(obj_img), len(obj_img[0]), len(obj_img[0][0]))
          tot_num_pix_in_obj = 0
          for i in range(len(obj_img)):
            for j in range(len(obj_img[0])):
              if (obj_img[i][j][0] != obj_color[0] or
                  obj_img[i][j][1] != obj_color[1] or
                  obj_img[i][j][2] != obj_color[2]):
                obj_img[i][j] = [255,255,255]
              else:
                obj_img[i][j] = [0,0,0]
                tot_num_pix_in_obj += 1
          cv2.imshow("stego obj_img", obj_img)

          shape, approximations = None, None
          squares = []
          # find all external contours in the threshold image then find
          # the *largest* contour which will be the contour/outline of
          # the stitched image
          try:
            sqimg = cv2.cvtColor(obj_img, cv2.COLOR_BGR2GRAY)
          except:
            sqimg = obj_img
          sqimg = cv2.bitwise_not(sqimg)
          imagecontours, hierarchy = cv2.findContours(sqimg,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
          # for each of the contours detected, the shape of the contours is approximated 
          # using approxPolyDP() function and the contours are drawn in the image using 
          # drawContours() function
          # For our border case, there may be a few dots or small contours that won't
          # be considered part of the border.
          # print("real_map_border count:", len(imagecontours))
          if len(imagecontours) > 1:
            print("hierarchy:", hierarchy)
            for i, c  in enumerate(imagecontours):
              area = cv2.contourArea(c)
              M = cv2.moments(c)
              print(i, "area, moment:", area, M, len(c))
              print(i, "area:", area, len(c))
              if M["m00"] != 0:
                centroid_x = int(M["m10"] / M["m00"])
                centroid_y = int(M["m01"] / M["m00"])
                print(i, "centroid:", centroid_x, centroid_y)
          for count in imagecontours:
            area = cv2.contourArea(count)
            if area < 16:
              continue
            # epsilon = 0.01 * cv2.arcLength(count, True)
            epsilon = 0.01 * cv2.arcLength(count, True)
            approximations = cv2.approxPolyDP(count, epsilon, True)
            # e.g. [[[224 224]] [[252 372]] [[420 372]] [[447 224]]]
            #the name of the detected shapes are written on the image
            #
            # sqimg2 = sqimg.copy()
            sqimg2 = np.zeros_like(sqimg)
            cv2.drawContours(sqimg2, count, -1, 255, cv2.FILLED)
            # cv2.drawContours(sqimg2, count, -1, 255, 5)
            # sqimg2 = np.ones_like(sqimg)*255
            cv2.imshow("contours", sqimg2)
            print(cv2.contourArea(count), cv2.arcLength(count,True))
            if (cv2.contourArea(count) < cv2.arcLength(count,True)):
              print("closed contour")
            bb_x,bb_y,bb_w,bb_h = cv2.boundingRect(count)
            num_pix_in_obj = 0
            for w in range(bb_w):
              for h in range(bb_h):
                x = int(bb_x+w)
                y = int(bb_y+h)
                if obj_img[y][x][0] == 0:
                  num_pix_in_obj += 1
            # num_pix_in_obj = cv2.countNonZero(obj_img)  
            sq_area = bb_w*bb_h
            # cv2.rectangle(sqimg2,(bb_x,bb_y),(bb_x+bb_w,bb_y+bb_h),(0,255,0),2)
            #####
            print("bounding box", num_pix_in_obj, (bb_x, bb_y), (bb_w, bb_h), num_pix_in_obj/sq_area)
            rect = cv2.minAreaRect(count)
            rot_rect_area = rect[1][0] * rect[1][1]
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # cv2.drawContours(img,[box],0,(0,0,255),2)
            print("rotated bounding box", num_pix_in_obj, rect, num_pix_in_obj/rot_rect_area)
            #####
            (circ_x,circ_y),radius = cv2.minEnclosingCircle(count)
            center = (int(circ_x),int(circ_y))
            radius = int(radius)
            circ_area = np.pi * radius * radius
            print("bounding circle", center, radius, num_pix_in_obj/circ_area)
            print("area", tot_num_pix_in_obj, num_pix_in_obj, sq_area, rot_rect_area, circ_area)
            # cv2.circle(img,center,radius,(0,255,0),2)
            cv2.waitKey(30)
            # hough_lines_image = np.ones_like(sqimg2)*255
            hough_image = np.zeros_like(sqimg2)
            hough_lines = self.get_hough_lines(sqimg2)
            print("hough_lines", hough_lines)
            if hough_lines is not None:
              print("new houghline")
              for line in hough_lines:
                # print("line", line)
                for x1,y1,x2,y2 in line:
                  print( x1,y1,x2,y2)
                  cv2.line(hough_image,(x1,y1),(x2,y2),255,5)
                  # cv2.line(hough_image,(x1,y1),(x2,y2),0,5)
              cv2.imshow("stego lines obj_img", hough_image)
            if True:
              hough_image = np.zeros_like(sqimg2)
              circles = cv2.HoughCircles(sqimg2,cv2.HOUGH_GRADIENT,1,20,
                             param1=50,param2=30,minRadius=0,maxRadius=0)
              if circles is not None:
                circles = np.uint16(np.around(circles))
                print("circles", circles)
                for i in circles[0,:]:
                  # draw the outer circle
                  cv2.circle(hough_image,(i[0],i[1]),i[2],255,2)
                cv2.imshow("stego circles obj_img", hough_image)
              else:
                print("circles None")
            if cv2.isContourConvex(count):
              print("Convex Contour")
            if False:
              hough_image = np.zeros_like(sqimg2)
              convhull = cv2.convexHull(count)
              # cv2.drawContours( hough_image, convhull, (int)0, 255 )
              cv2.drawContours( hough_image, convhull, color=255 )
              cv2.imshow("stego hull obj_img", hough_image)

            cv2.waitKey(0)
            i, j = approximations[0][0] 
            if len(approximations) == 3:
              shape = "Triangle"
            elif len(approximations) == 4:
              shape = "Trapezoid"
              area = cv2.contourArea(approximations)
              # sqimg2 = sqimg.copy()
              # cv2.drawContours(sqimg2, count, -1, (0,255,0), 3)
              # cv2.imshow("contours", sqimg2)
              # cv2.waitKey()
              # if area > 100 &  cv2.isContourConvex(approximations):
              if cv2.isContourConvex(approximations):
                maxCosine = -100000000000000
                for j in range(2, 5):
                  cosine = abs(middle_angle(approximations[j%4][0], approximations[j-2][0], approximations[j-1][0]))
                  print("cosine ", cosine, ":", approximations[j%4][0], approximations[j-2][0], approximations[j-1][0])
                  maxCosine = max(maxCosine, cosine)
                # if cosines of all angles are small
                # (all angles are ~90 degree) then write quandrange
                # vertices to resultant sequence
                if maxCosine < 0.3 and maxCosine >= 0:
                  shape = "Square"
                  squares.append(approximations)
                  print("square found:", approximations)
                else:
                  print("maxCos:", maxCosine)
              else:
                print("non convex contour", approximations)
            elif len(approximations) == 5:
              shape = "Pentagon"
            elif 6 < len(approximations) < 15:
              shape = "Ellipse"
            else:
              shape = "Circle"
            # if len(imagecontours) > 1:
            #   cv2.putText(thresh,shape,(i,j),cv2.FONT_HERSHEY_COMPLEX,1,0,2)
            #   cv2.waitKey(0)
            # print("map shape:", shape, approximations)
            #displaying the resulting image as the output on the screen
            # imageread = mapimg.copy()
            # print("contour:", count)
            # print("approx contour:", approximations)
            # return shape, approximations
            print("shape:", shape, squares)
          # return squares
        return squares

  # Map lines to stego_number_list
  # map stego_number_list to role list:
  #   - Top of box
  #     - arm pos on top of box
  #     - camera angle
  #     - bounding box
  #   - box boundary (line around top)
  #   - table
  #     - front/side of full box
  #   - off table
  #   - gripper: 
  #     - gripper with cube 
  #     - gripper without cube 
  #   - front/side of box
  #     - Size when arm is parked
  #     - in front of top
  #     - bounding box
  #
  #   - outside table
  #   - unknown
  #
  # Arm Estimated Position
  #   - estimated camera angle
  #
  # Map stego categories from beginning:
  #   - likely-cube
  #   - confirmed-cube
  #   - rectangular-table
  #   - off table
  #   - likely-box
  #   - confirmed-box
  
  
  ###########################################
  # pickup -> bounding box
  # pickup -> work backwards
  #
  ###########################################
  # estimate distance to pick-up object
  # estimate location in rectangle
  # 
  ###########################################
  # Circle around, find rectangular bounds
  # Find little thing within boundary
  # Find Big think within boundary
  # Put little thing on/in big thing
  # obstacle avoidance
  
if __name__ == '__main__':
  box = AnalyzeLines()

# https://livecodestream.dev/post/object-tracking-with-opencv/
#
# Different Tracking Algorithms include
# * Multiple Hypothesis Tracking (MHT) 
# * Joint Probabilistic Data Association Filter (JPDAF) 
# * Simple online and realtime tracking (SORT) => Python implementation in opencv
#
# Parameters include:
# * # of observations until added
# * # of observations until deleted
# * We're doing post-run analysis for training purposes.
#   vs. real-time tracking.

#
# Most of the time, it is the arm that is moving in only a single dimention.
# Movements are not linear, constant velocity but up/down based on UA/LA actions.
# We're doing lines, not bounding boxes here.
# 
# Object detection of moving objects might be able to use BB overlap, velocity,
# and the Kalman Filter SORT methods. Arm would be parked.

# Evaluation can be carried out according to the following metrics:
# • Multi-object tracking accuracy (MOTA): Summary of over-
# all tracking accuracy in terms of false positives, false nega-
# tives and identity switches [23].
# • Multi-object tracking precision (MOTP): Summary of over-
# all tracking precision in terms of bounding box overlap be-
# tween ground-truth and reported location [23].
# • Mostly tracked (MT): Percentage of ground-truth tracks
# that have the same label for at least 80% of their life span.
# • Mostly lost(ML): Percentage of ground-truth tracks that are
# tracked for at most 20% of their life span.
# • Identity switches (ID): Number of times the reported iden-
# tity of a ground-truth track changes.
# • Fragmentation (FM): Number of times a track is interrupted
# by a missing detection.

              # estimate drop point into box
  
              # estimate position of camera
