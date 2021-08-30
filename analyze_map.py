# import the necessary packages
import numpy as np
from numpy import asarray
from PIL import Image
import cv2
import argparse
from keypoint import *
from imutils import paths
import imutils
from matplotlib import pyplot as plt
from cv_analysis_tools import *
from stitching import *
from shapely.geometry import *
import statistics 
from operator import itemgetter, attrgetter
  

# based on: 
# https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
class AnalyzeMap():
    def __init__(self):
        self.stop_at_frame = 112
        self.curr_frame_num = 0
        # self.stop_at_frame = 117
        self.KPs = None
        self.map = None
        # real map array info
        # real map array sizes change over time
        self.map_overlay = None
        self.map_overlay_KP = None
        self.map_arr = None
        self.map_KP = None
        self.map_height = None
        self.map_width = None
        self.map_ch = None
        self.border_buffer = None   # move_img
        self.border_multiplier = 2   # move_img
        # To compute the distances,var of moves
        self.curr_move_height = None
        self.curr_move_rows = None
        self.curr_move = None
        self.curr_move_KP = None
        self.prev_move = None
        self.prev_move_KP = None
        # dist/degrees = 0, var = 1 in lists below
        self.forw_dist = []  
        self.back_dist = []
        self.left_degrees = []
        self.right_degrees = []
        # the robot starts in the middle of an empty square of VIRTUAL_MAP_SIZE pixels
        # the virtual map locations do not change.
        self.VIRTUAL_MAP_SIZE = None
        self.virtual_map_center = None
        self.robot_location = None
        self.robot_location_history = []
        self.robot_orientation = None
        self.robot_location_hist = []
        self.robot_orientation_hist = []
        self.robot_length = None
        self.parked_gripper_left = []
        self.parked_gripper_right = []
        self.gripper_height = None
        self.gripper_width = None
        self.clusters = []
        self.grabable_objects = []
        self.container_objects = []
        self.last_line = None
        self.dif_avg = []
        self.dif_var = []
        self.min_pt = []
        self.max_pt = []
        self.color_quant_num_clust = 0
        self.INFINITE = 1000000000000000000
        self.cvu = CVAnalysisTools()

    def real_to_virtual_map_coordinates(self, pt):
        self.virtual_map_center = None
        x = pt[0] - self.virtual_map_center[0] + self.VIRTUAL_MAP_SIZE/2
        y = pt[1] - self.virtual_map_center[1] + self.VIRTUAL_MAP_SIZE/2
        return x,y

    def virtual_to_real_map_coordinates(self, pts):
        x = pt[0] + self.virtual_map_center[0] - self.VIRTUAL_MAP_SIZE/2
        y = pt[1] + self.virtual_map_center[1] - self.VIRTUAL_MAP_SIZE/2
        return x,y

    ######################
    # Keypoint Transforms
    ######################
    def order_points(self, pts):
        # initialzie a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
        rect = np.zeros((4, 2), dtype = "float32")

        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # return the ordered coordinates
        return rect

    def get_birds_eye_view(self, image):
        w = image.shape[0]
        h = image.shape[1]
        new_w = int(48 * w / 32)
        bdr   = int((new_w - w) / 2)
        print("new_w, w, bdr:", 55*w/32, w, (new_w - w)/2)
        image = cv2.copyMakeBorder( image, top=0, bottom=0, left=bdr, right=bdr,
            borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])

        # src = np.float32([[0, IMAGE_H], [1207, IMAGE_H], [0, 0], [IMAGE_W, 0]])
        # dst = np.float32([[569, IMAGE_H], [711, IMAGE_H], [0, 0], [IMAGE_W, 0]])
        src = np.float32([[0, h], [new_w, h], [bdr, 0], [w, 0]])
        # need to modify new_w
        # dst = np.float32([[0, h], [new_w, h], [0, 0], [(new_w), 0]])
        # dst = np.float32([[0, h], [new_w, h], [0, 0], [int((new_w+w)/2), 0]])
        # dst = np.float32([[0, h], [new_w, h], [0, 0], [w, 0]])
        dst = np.float32([[0, h], [new_w, h], [0, 0], [(w+(bdr/2)), 0]])
        M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
        # Minv = cv2.getPerspectiveTransform(dst, src) # Inverse transformation
        birds_eye_view = cv2.warpPerspective(image, M, (new_w, h)) # Image warping

        self.gripper_height = int(8 * h / 32)
        self.gripper_width = int(11 * new_w / 32)
        print("gripper h,w:", self.gripper_height, self.gripper_width)

        self.parked_gripper_left = np.zeros((self.gripper_height, self.gripper_width, 3), dtype=np.uint8)
        self.parked_gripper_right = np.zeros((self.gripper_height, self.gripper_width, 3), dtype=np.uint8)
        black = [0, 0, 0]
        # cv2.imshow("full birds_eye_view", birds_eye_view )
        for gw in range(self.gripper_width):
          for gh in range(self.gripper_height):
            self.parked_gripper_left[self.gripper_height-gh-1,gw] = birds_eye_view[h-gh-1, gw]
            self.parked_gripper_right[self.gripper_height-gh-1,self.gripper_width-gw-1] = birds_eye_view[h-gh-1, new_w-gw-1]
            birds_eye_view[h-gh-1, gw] = black
            birds_eye_view[h-gh-1, new_w-gw-1] = black
        # cv2.imshow("b4", image )
        # cv2.imshow("birds_eye_view", birds_eye_view )
        # cv2.imshow("left gripper", self.parked_gripper_left )
        # cv2.imshow("right gripper", self.parked_gripper_right )
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return birds_eye_view

    def four_point_transform(self, image, pts):
        # obtain a consistent order of the points and unpack them
        # individually
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect
        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order


        dst = np.array([
              [0, 0], [maxWidth-1, 0], [maxWidth-1, maxHeight-1], [0, maxHeight-1]
              ], dtype = "float32")
       
        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(dst, rect)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight),  cv2.WARP_INVERSE_MAP)
        # return the warped image
        return warped

    def kp_match_stats(self, kp_matches, KP1=None, KP2=None):
        if len(kp_matches) >= 3:
            # kps are sorted by distance. Lower distances are better.
            num_kp_matches = len(kp_matches)
            avg_x_dif = 0
            avg_y_dif = 0
            x_dev = []
            y_dev = []
            min_x = self.INFINITE
            max_x = -self.INFINITE
            min_y = self.INFINITE
            max_y = -self.INFINITE
            # check if use defaults
            if KP1 is None:
              KP1=self.map_KP
            if KP2 is None:
              KP2=self.curr_move_KP
            for kpm in kp_matches:
              if len(KP1.keypoints) <= kpm[0].queryIdx or len(KP2.keypoints) <= kpm[0].trainIdx:
                print("trI, numkp1, numkp2", kpm[0].queryIdx, kpm[0].trainIdx, len(KP1.keypoints), len(KP2.keypoints))
                continue
              KP1_kp = KP1.keypoints[kpm[0].queryIdx].pt
              KP2_kp = KP2.keypoints[kpm[0].trainIdx].pt
              # print("map,new,dist:", KP1_kp, KP2_kp, dist)
              x_dev.append(KP1_kp[0] - KP2_kp[0])
              y_dev.append(KP1_kp[1] - KP2_kp[1])
              avg_x_dif += KP1_kp[0] - KP2_kp[0]
              avg_y_dif += KP1_kp[1] - KP2_kp[1]
              min_x = min(min_x, KP1_kp[0])
              max_x = max(max_x, KP1_kp[0])
              min_y = min(min_y, KP1_kp[1])
              max_y = max(max_y, KP1_kp[1])
            # ARD: todo: autotune to minimize the avg kp difs
            avg_x_dif /= num_kp_matches
            avg_y_dif /= num_kp_matches

            x_dev = [(x_dev[i] - avg_x_dif) ** 2 for i in range(len(x_dev))]
            y_dev = [(y_dev[i] - avg_y_dif) ** 2 for i in range(len(y_dev))]
            x_var = sum(x_dev) / num_kp_matches
            y_var = sum(y_dev) / num_kp_matches

            print("avg x,y dif: ", avg_x_dif, avg_y_dif, np.sqrt(x_var), np.sqrt(y_var))
            print("max/min x,y: ", [max_x, max_y],[min_x, min_y])
        else:
            print("insufficient keypoints")
            return [None, None], [None, None], [None, None], [None, None]
            # x = undefined_var
        return [avg_x_dif, avg_y_dif], [x_var, y_var], [min_x, min_y], [max_x, max_y]

    def move_state(self, action, new_above_view_img):
        self.prev_move = self.curr_move
        self.prev_move_KP = self.curr_move_KP
        self.curr_move = new_above_view_img
        self.curr_move_KP = Keypoints(self.curr_move)
        # TODO: compute distances for each action

    ############################
    # MAP KEYPOINT ANALYSIS FUNCTIONS
    ############################

    def evaluate_map_kp_offsets(self, map_KP, new_map_rot):
        shape, map_border = self.real_map_border(new_map_rot)
        new_map_rot_KP = Keypoints(new_map_rot, kp_mode=map_KP.get_kp_mode())
        good_matches,notsogood = map_KP.compare_kp(new_map_rot_KP)
        print("kp_offsets: len good matches:", len(good_matches))
        map_pts, new_map_rot_pts = map_KP.get_n_match_kps(good_matches, new_map_rot_KP, 6, return_list=False, border=map_border)
        delta_h, delta_w = 0,0
        x,y = 0,1
        for i, map_pt in enumerate(map_pts):
          delta_h += (map_pt.pt[x] - new_map_rot_pts[i].pt[x])
          delta_w += (map_pt.pt[y] - new_map_rot_pts[i].pt[y])
        if len(map_pts) > 0:
          delta_h = int(delta_h / len(map_pts))
          delta_w = int(delta_w / len(map_pts))
        print("kp_offsets hw: ", delta_h, delta_w) 
        return delta_h, delta_w, map_pts, new_map_rot_pts

    ##############
    # find_best_kp_angle():
    #   map is the full composite map
    #   new map is the new robot image to be integrated into the full map
    #   Does a binary search to find best angle
    def find_best_kp_angle(self, map_KP, new_map, start_angle, end_angle):
      x = 0
      y = 1
      best_kp_angle, best_kp_delta_h, best_kp_delta_w = None, None, None
      best_map_pts, best_new_map_pts = None, None
      min_angle = start_angle
      max_angle = end_angle
      while True:
        angle = (start_angle + end_angle)/2 
        new_map_rot = self.rotate_about_robot(new_map, angle)

        delta_h, delta_w, map_pts, new_map_rot_pts = self.evaluate_map_kp_offsets(map_KP, new_map_rot)
        if len(map_pts) > 0 and (best_kp_delta_h is None or delta_h < best_kp_delta_h):
            best_kp_angle, best_kp_delta_h, best_kp_delta_w = angle, delta_h, delta_w 
            best_map_pts, best_new_map_pts = map_pts, new_map_rot_pts
            print("new best kp angle:",best_map_pts,best_new_map_pts)
        if abs(max_angle - min_angle) <= .05 or delta_h == 0:
          break
        elif delta_h > 0:
          min_angle = angle
        elif delta_h < 0:
          max_angle = angle
      return best_kp_angle, best_kp_delta_h, best_kp_delta_w, best_map_pts, best_new_map_pts

    def show_rot(self, new_map):
      pt = [int(self.robot_location[1]), int(self.robot_location[0])]
      new_map = cv2.circle(new_map,pt,3,(255,0,0),-1)
      pt = [int(new_map.shape[1]/2), int(new_map.shape[0]/2)]
      new_map = cv2.circle(new_map,pt,3,(255,0,0),-1)
      for angle in range(0,28,7):
        new_map_rot = self.rotate_about_robot(new_map, angle)
        cv2.imshow(str(angle), new_map_rot)
        cv2.waitKey(0)

    #########################
    # Contours
    #########################
    def get_contours(self,img):
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
        # find contours in the thresholded image
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        return cnts
        
    def draw_contours(self,img,cnt,text="",def_clr=(0,255,0)):
        for i,c in enumerate(cnt):
            # compute the center of the contour
            M = cv2.moments(c)
            # cX = int((M["m10"] / M["m00"]) )
            # cY = int((M["m01"] / M["m00"]) )
            c = c.astype("int")
            # print(i,"c",c)
            itext = text + str(i)
            cv2.drawContours(img, [c], -1, def_clr, 2)
            # cv2.putText(img, itext, (cX, cY),
            #       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    #########################
    # Border manipulation
    #########################
    def add_border(self, img, bordersize):
        print("bordersize:", bordersize, img.shape[:2])
        row, col = img.shape[:2]
        bottom = img[row-2:row, 0:col]
        mean = cv2.mean(bottom)[0]
        border = cv2.copyMakeBorder(
            img,
            top=bordersize,
            bottom=bordersize,
            left=bordersize,
            right=bordersize,
            borderType=cv2.BORDER_CONSTANT,
            # value=[mean, mean, mean]
            value=[0, 0, 0]  # black
        )
        return border

    def replace_border(self, img, desired_rows, desired_cols, offset_rows, offset_cols):
        shape, border = self.real_map_border(img)
        maxh, minh, maxw, minw = self.cvu.get_min_max_borders(border)
        orig_row, orig_col = img.shape[:2]
        print("shapes:", orig_row,orig_col,minh,maxh,minw,maxw)
        extract_img_rect = img[minw:maxw, minh:maxh]
        extracted_rows, extracted_cols = extract_img_rect.shape[:2]
        border_top = int((desired_rows - extracted_rows)/2) + offset_rows 
        border_bottom = desired_rows - border_top - extracted_rows
        border_left = int((desired_cols - extracted_cols)/2) + offset_cols 
        border_right = desired_cols - border_left - extracted_cols 
        print("replace_border:",border_top, border_bottom, border_left, border_right, offset_rows, offset_cols)
        bordered_img = cv2.copyMakeBorder(
            extract_img_rect,
            top=border_top,
            bottom=border_bottom,
            left=border_left,
            right=border_right,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]  # black
        )
        return bordered_img


    # angle = int(math.atan((y1-y2)/(x2-x1))*180/math.pi)

    # image is "new map" == the transformed "curr move" with border
    # center is about 4xfinger pads -> self.robot_location
    def rotate_about_robot(self, image, angle):
        # Draw the text using cv2.putText()
        # Rotate the image using cv2.warpAffine()
        M = cv2.getRotationMatrix2D(self.robot_location, angle, 1)
        print("angle,M:", angle, M, self.robot_location, image.shape)
        out = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        # Display the results
        # cv2.imshow('rotate about robot', out)
        # cv2.waitKey(0)
        return out

    def real_map_border(self, mapimg, ret_outside=True):
        # convert the stitched image to grayscale and threshold it
        # such that all pixels greater than zero are set to 255
        # (foreground) while all others remain 0 (background)
        gray = cv2.cvtColor(mapimg, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
        # find all external contours in the threshold image then find
        # the *largest* contour which will be the contour/outline of
        # the stitched image
        if ret_outside:
          # imagecontours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
          imagecontours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
          # for each of the contours detected, the shape of the contours is approximated 
          # using approxPolyDP() function and the contours are drawn in the image using 
          # drawContours() function
          # For our border case, there may be a few dots or small contours that won't
          # be considered part of the border.
          print("real_map_border count:", len(imagecontours))
          if len(imagecontours) > 1:
            # print("hierarchy:", hierarchy)
            for i, c  in enumerate(imagecontours):
              area = cv2.contourArea(c)
              M = cv2.moments(c)
              # print(i, "area, moment:", area, M, len(c))
          for count in imagecontours:
            if len(imagecontours) > 1:
              area = cv2.contourArea(count)
              if area < 1000:
                continue
            epsilon = 0.01 * cv2.arcLength(count, True)
            approximations = cv2.approxPolyDP(count, epsilon, True)
            # e.g. [[[224 224]] [[252 372]] [[420 372]] [[447 224]]]
            #the name of the detected shapes are written on the image
            i, j = approximations[0][0] 
            if len(approximations) == 3:
              shape = "Triangle"
            elif len(approximations) == 4:
              shape = "Trapezoid"
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
            return shape, approximations

    #########################
    # Map Line Analysis 
    #########################
    def analyze_map_lines(self, map, frame_num=0):
      self.curr_frame_num = frame_num
      for K in range(3,6):
        floor_clusters = self.cvu.color_quantification(map, K)
        linesP, imglinesp = self.cvu.get_lines(floor_clusters)
        if linesP is None:
          continue
        shape, map_border = self.real_map_border(map)
        max_dist, map_line, map_dx, map_dy, map_slope, in_brdr_cnt = self.cvu.find_longest_line(linesP, map_border)
        print("#### Map Line Analysis: ", len(linesP), in_brdr_cnt) 
        print("in border line cnt: ", in_brdr_cnt)
        if map_dx is None or map_dy is None:
            print("no map line: ", K)
            continue
        self.color_quant_num_clust = K
        self.last_line = map_line
        print("longest line",map_line)
        if frame_num >= self.stop_at_frame:
          cv2.line(floor_clusters, (map_line[0], map_line[1]), (map_line[2], map_line[3]), (0,255,0), 3, cv2.LINE_AA)
          pt = [int(self.robot_location[1]), int(self.robot_location[0])]
          imglinesp = cv2.circle(floor_clusters,pt,3,(255,0,0),-1)
          # cv2.imshow("best map line", floor_clusters);
          # cv2.imshow("all map lines", imglinesp);
          # cv2.waitKey(0)
        return map_line, map_slope, max_dist
      return None, None, None

    ##############
    # find_best_line_angle()
    ##############
    #   map is the full composite map
    #   new map is the new robot image to be integrated into the full map
    #   Does a binary search to find best angle
    def find_best_line_angle(self, map, map_slope, new_map, start_angle, end_angle, frame_num=0):
      min_slope_dif_angle = None
      min_slope_dif = self.INFINITE
      best_line_angle = None
      best_line = None
      # do binary search
      angle_min = start_angle
      angle_max = end_angle
      while True:
        angle = (angle_min + angle_max) / 2
        print("min, mid, max angle:", angle_min, angle, angle_max)
        if self.color_quant_num_clust > 0:
          floor_clusters = self.cvu.color_quantification(new_map, self.color_quant_num_clust)
          new_map_rot = self.rotate_about_robot(floor_clusters, angle)
        else:
          new_map_rot = self.rotate_about_robot(new_map, angle)
        shape, map_border = self.real_map_border(new_map_rot)
        linesP, imglinesp = self.cvu.get_lines(new_map_rot)
        # print(linesP)
        in_brdr_cnt = 0
        min_slope_dif = self.INFINITE
        rot_min_slope_dif = self.INFINITE
        for [[l0,l1,l2,l3]] in linesP:
          if self.cvu.line_in_border(map_border, (l0,l1), (l2,l3)):
            in_brdr_cnt += 1
            # print("l0-4:", l0,l1,l2,l3)
            dx = l0 - l2
            dy = l1 - l3
            dist = np.sqrt(dx**2 + dy**2)
            diff_slope = map_slope - np.arctan2(dx, dy)
            print("map, cur slopes:", map_slope, np.arctan2(dx,dy))
            # print("diff slope/min:", diff_slope, min_slope_dif)
            if abs(diff_slope) < abs(rot_min_slope_dif):
              rot_min_slope_dif = diff_slope
            if abs(diff_slope) < abs(min_slope_dif):
              min_slope_dif = diff_slope
              min_slope_dif_angle = angle
              best_line = [l0,l1,l2,l3]
              print("best line so far:", best_line, angle, diff_slope, "dxy:", map_slope, (dx/dy))
        print("in border line cnt: ", in_brdr_cnt)
        if frame_num >= self.stop_at_frame:
          # pt = [int(self.robot_location[1]), int(self.robot_location[0])]
          # imglinesp = cv2.circle(imglinesp,pt,3,(255,0,0),-1)
          # cv2.imshow("rotated new map:",imglinesp)
          # cv2.waitKey(0)
          pass
        if in_brdr_cnt == 0:
          # did we rotate too far?
          angle_max = angle
          break
        elif min_slope_dif == 0 or abs(angle_max - angle_min) <= .05:
          print("angle min max: ", angle_min, angle_max, min_slope_dif)
          # angle min max:  14.970703125 15 16.365384615384613
          # how is min_slope_dif bigger than max angle?
          #  - angle is angle of rotation, min slope dif is slope of resulting line
          break
        # elif min_slope_dif == self.INFINITE or diff_slope < 0:
        elif rot_min_slope_dif > 0:
          angle_min = angle
        elif rot_min_slope_dif <= 0:
          angle_max = angle
      if in_brdr_cnt == 0 and best_line is not None:
        print("######## Using Best Slope So Far #########")
        print("angle min max: ", angle_min, angle_max)
        print("New Map best line/angle:", best_line, min_slope_dif_angle)
        final_angle = min_slope_dif_angle
      elif min_slope_dif == self.INFINITE:
        print("######## New Map Slope not found #########")
        best_line, final_angle = None, None
      else:
        print("######## New Map Slope found #########")
        print("New Map info:", best_line, min_slope_dif_angle)
        final_angle = min_slope_dif_angle
      return best_line, final_angle
    

    # compare best Map Line to best "new map img" Line
    # Two lines defined by: y = mx + b
    # Function to find the vertical distance between parallel lines
    # Adjust robot location to match
    def adjustRobotLocationByLines(self, MapPt1, MapPt2, NewPt1, NewPt2):
          x = 0
          y = 1
          # dx / dy
          MapSlope = (MapPt1[x] - MapPt2[x]) / (MapPt1[y] - MapPt2[y])
          NewSlope = (NewPt1[x] - NewPt2[x]) / (NewPt1[y] - NewPt2[y])
          # b = y - Mx
          MapB     = MapPt1[y] - MapSlope*MapPt1[x]
          MapB2    = MapPt2[y] - MapSlope*MapPt2[x]
          NewB     = NewPt1[y] - NewSlope*NewPt1[x]
          NewB2    = NewPt2[y] - NewSlope*NewPt2[x]
          print("MapB,B2; NewB,B2:", MapB, MapB2, NewB, NewB2, (MapB-NewB))
          # x = (y - b)/M
          MapXatRobot = (self.robot_location[y] - MapB) / MapSlope
          NewXatRobot = (self.robot_location[y] - NewB) / NewSlope
          NewXatRobot2= (self.robot_location[y] - MapB) / NewSlope
          print("NewX, oldX:", NewXatRobot, NewXatRobot2)
          # y = Mx + b
          MapYatRobot = MapSlope * self.robot_location[x] + MapB
          NewYatRobot = NewSlope * self.robot_location[x] + NewB
          NewYatRobot2= NewSlope * self.robot_location[x] + MapB
          # Note: same Robot Loc (x,y), diff angle
          #
          Ydif = NewYatRobot - NewYatRobot2  
          # Ydif = MapYatRobot - NewYatRobot  
          # self.robot_location[y] -= Ydif
          # print("robot loc:", self.robot_location, Ydif, MapYatRobot, NewYatRobot, MapB, NewB)
          # return Ydif
          #
          # Xdif = MapXatRobot - NewXatRobot  
          Xdif = NewXatRobot2 - NewXatRobot
          #          MapXatRobot = (self.robot_location[y] - MapB) / MapSlope
          NewXatRobot = (self.robot_location[y] - NewB) / NewSlope
          NewXatRobot2= (self.robot_location[y] - MapB) / NewSlope
          print("NewX, oldX:", NewXatRobot, NewXatRobot2)
          # y = Mx + b
          MapYatRobot = MapSlope * self.robot_location[x] + MapB
          NewYatRobot = NewSlope * self.robot_location[x] + NewB
          NewYatRobot2= NewSlope * self.robot_location[x] + MapB
          # self.robot_location[x] += Xdif
          print("Xdif Ydiff: ", Xdif, Ydif)
          # self.robot_location[y] = NewYatRobot2
          # self.robot_location_history.append([self.robot_location, "Parallel Line adjustment"])
          print("robot loc:", self.robot_location, Xdif, MapXatRobot, NewXatRobot, MapB, NewB)
          return Xdif
                 
    #########################
    # Hardcoding / Manually tuning / gathering data to figure out algorithm
    def manual_tuning(self, frame_num):
        if frame_num == 112:
          [off_h, off_w] = [0, 0] # best
        if frame_num == 113:
          # [off_h, off_w] = [15, -50]
          # [off_h, off_w] = [-25, -10] raises it way too high
          # [off_h, off_w] = [-5, -20]   much closer
          # [off_h, off_w] = [0, -20]   gap on upper // increased
          # [off_h, off_w] = [-5, -25] go lower/to left
          # [off_h, off_w] = [0, -30]  good height/ need left
          # [off_h, off_w] = [0, -40]  very close
          ############# new image format
          # [off_h, off_w] = [-20, -5]   Too high, too right
          # [off_h, off_w] = [-5, -20]   OK height, too right
          [off_h, off_w] = [-5, -70]  
        if frame_num == 114:
          # [off_h, off_w] = [0, -90] need right
          # [off_h, off_w] = [10, -85] need lower
          # [off_h, off_w] = [5, -85] need right
          # [off_h, off_w] = [5, -80] close
          # [off_h, off_w] = [5, -75] closer
          # [off_h, off_w] = [10, -80]  
          # [off_h, off_w] = [10, -75]  need higher
          # [off_h, off_w] = [5, -75]  # best so far
          # [off_h, off_w] = [3, -73]  # best so far
          ################ New format
          # [off_h, off_w] = [3, -60]  Too far right
          [off_h, off_w] = [-5, -130]  # best so far
        if frame_num == 115:
          # can focus on the cube too
          # to lower, increase x ; to right, decrease y
          # [off_h, off_w] = [0, -75] # much closer, need lower
          # [off_h, off_w] = [5, -55] # need lower, left
          # [off_h, off_w] = [10, -60] # closer. need left
          # [off_h, off_w] = [3, -73] # good, need lower
          # [off_h, off_w] = [8, -73] # good
          ################
          # [off_h, off_w] = [-5, -200] # too far left
          # [off_h, off_w] = [0, -180] # close
          [off_h, off_w] = [-1, -185] # a bit blury
        if frame_num == 116:
          # [off_h, off_w] = [12, -65] # good
          # [off_h, off_w] = [12, -65] # good
          ################
          # [off_h, off_w] = [-5, -250] # too high, too left
          # [off_h, off_w] = [5, -200] # too high, too right
          #[off_h, off_w] = [15, -220] # much closer
          [off_h, off_w] = [15, -215] # much closer
        print("manual_tuning: ", off_h, off_w)
        return off_h, off_w
    

    def get_height_dif_by_line(self, rnm, map_line, map_slope):
        floor_clusters = self.cvu.color_quantification(rnm, self.color_quant_num_clust)
        shape, new_map_rot_border = self.real_map_border(rnm)
        linesP, imglinesp = self.cvu.get_lines(rnm)
        # print(linesP)
        print("map line:", map_line)
        min_slope_dif = self.INFINITE
        for [[l0,l1,l2,l3]] in linesP:
          if self.cvu.line_in_border(new_map_rot_border, (l0,l1), (l2,l3)):
            dx = l0 - l2
            dy = l1 - l3
            dist = np.sqrt(dx**2 + dy**2)
            diff_slope = map_slope - np.arctan2(dx, dy)
            print("map, cur slopes:", map_slope, np.arctan2(dx,dy), [l0,l1,l2,l3])
            if abs(diff_slope) < abs(min_slope_dif):
              min_slope_dif = diff_slope
              best_line = [l0,l1,l2,l3]
        if (map_line[3] - map_line[1]) == 0:
            map_b = self.INFINITE
        else:
            # For the robot w location, compute the h location
            # h1 = m1*w1 + b1   ; h2 = m2*w2 + b2
            # h1 - m1*w1 = h2 - m2*w2
            # h1 - h2 = m1*w1  - m2*w2
            robot_w = self.robot_location[1]
            off_b = robot_w * (map_line[1]-map_line[3])/(map_line[0]-map_line[2]) - robot_w * (best_line[1]-best_line[3])/(best_line[0]-best_line[2])
            print("off_b", off_b)
        return round(off_b)

    def feature_offsets(self, map_overlay, rotated_new_map, map_line, map_slope, rot_line, frame_num):
        print("###############")
        print("feature offsets")
        off_h = None
        off_w = None
        map_pts = []
        rot_pts = []
        mol = map_overlay.copy()
        rnm = rotated_new_map.copy()
        shape, map_border = self.real_map_border(mol)
        shape, rot_border = self.real_map_border(rnm)
        gray_mol = cv2.cvtColor(mol, cv2.COLOR_BGR2GRAY)
        gray_rnm = cv2.cvtColor(rnm, cv2.COLOR_BGR2GRAY)

        # imageInput 8-bit or floating-point 32-bit, single-channel image.
        # cornersOutput vector of detected corners.
        # maxCornersMaximum number of corners to return. If there are more corners 
        #     than are found, the strongest of them is returned. maxCorners <= 0 implies 
        #     that no limit on the maximum is set and all detected corners are returned.
        # qualityLevelParameter characterizing the minimal accepted quality of image 
        #     corners. The parameter value is multiplied by the best corner quality measure,
        #      which is the minimal eigenvalue (see cornerMinEigenVal ) or the Harris 
        #     function response (see cornerHarris ). The corners with the quality measure 
        #     less than the product are rejected. For example, if the best corner has the 
        #     quality measure = 1500, and the qualityLevel=0.01 , then all the corners with 
        #     the quality measure less than 15 are rejected.
        # minDistanceMinimum possible Euclidean distance between the returned corners.
        # maskOptional region of interest. If the image is not empty (it needs to have 
        #     the type CV_8UC1 and the same size as image ), it specifies the region in 
        #     which the corners are detected.
        # blockSizeSize of an average block for computing a derivative covariation 
        #     matrix over each pixel neighborhood. See cornerEigenValsAndVecs .
        # useHarrisDetectorParameter indicating whether to use a Harris detector 
        #     or cornerMinEigenVal.
        # kFree parameter of the Harris detector.
        feature_params = dict( maxCorners = 100,
               # qualityLevel = 0.3,
               qualityLevel = 0.01,
               minDistance = 7,
               blockSize = 7 )
        map_features = cv2.goodFeaturesToTrack(gray_mol, mask = None, **feature_params)
        rot_features = cv2.goodFeaturesToTrack(gray_rnm, mask = None, **feature_params)
        print("num map,rot features:", len(map_features), len(rot_features))
        delta = []
        first_time_through = True
        # print("map_features:", map_features)
        print("len map_features:", len(map_features))
        for i, map_pt_lst in enumerate(map_features):
          map_pt = [int(map_pt_lst[0][0]), int(map_pt_lst[0][1])]
          # print("i map_pt:", i, map_pt)
          if not self.cvu.point_in_border(map_pt,map_border):
            # print("map pt not in border:", map_pt, map_border)
            mol=cv2.circle(mol,map_pt,3,(0,255,0),-1)
            continue
          cv2.circle(mol,map_pt,3,(255,0,0),-1)
          for j, rot_pt_lst in enumerate(rot_features):
            rot_pt = [int(rot_pt_lst[0][0]), int(rot_pt_lst[0][1])]
            if not self.cvu.point_in_border(rot_pt,rot_border):
              # print("rot pt not in border:", rot_pt, rot_border)
              rnm=cv2.circle(rnm,rot_pt,3,(0,255,0),-1)
              continue
            if first_time_through:
              rnm=cv2.circle(rnm,rot_pt,3,(255,0,0),-1)
            dist = math.sqrt((map_pt[0]-rot_pt[0])**2+(map_pt[1]-rot_pt[1])**2)
            # with y=mx+b, we only need to rank y and compute x
            ranking = map_pt[0]-rot_pt[0]
            delta.append([ranking, i, j])
        cv2.imshow("mol feature offset", mol)
        cv2.imshow("rot feature offset", rnm)
        cv2.waitKey(0)
        min_var = self.INFINITE
        # for n in range(6,len(delta)):
        n = 20
        # n = int(.1 * len(delta))
        print("Top N:", n, len(delta))
        if True:
         if len(delta) >= n:
          # check both minimize x&y dif variance
          # for j in range(2):
            delta = sorted(delta,key=itemgetter(0))
            print("delta:", delta[0:n])
            for i in range(len(delta)-n):
              # var0 = statistics.variance(delta[i:i+n][1])
              # var1 = statistics.variance(delta[i:i+n][2])
              # if var0 + var1 < min_var:
              #   min_var = var0 + var1
              #   min_i = i
              var = statistics.variance(delta[i:i+n][2])
              if var < min_var:
                min_var = var
                min_i = i
                print("min_var, i:", min_var, min_i)
         if min_var != self.INFINITE:
          sum_h = 0
          sum_w = 0
          for i in range(n):
            # get the map and rot features 
            mf = delta[min_i+i][1]
            rf = delta[min_i+i][2]
            # get the feature points 
            mfeat = map_features[mf][0]
            rfeat = rot_features[rf][0]
            map_pts.append(map_features[mf])
            rot_pts.append(rot_features[rf])
            # print("map_feat:", mfeat, rfeat)
            # compute the mean difference
            sum_h += mfeat[1] - rfeat[1]
            sum_w += mfeat[0] - rfeat[0]
          off_h = int(sum_h/n)
          off_w = int(sum_w/n)

          # x = (y - b)/M
          off_b = self.get_height_dif_by_line(rnm, map_line, map_slope)

          moff_h, moff_w = self.manual_tuning(frame_num)
          print("n off xy, manual xy", n, [off_h, off_w], [moff_h, moff_w], [off_b, off_w])
          #     # n off xy, manual xy 20 None       -8   0      -45 61 1310.0262377400063
          #     # n off xy, manual xy 20 None       5    3      -73 87 1242.2292568801622

        return off_b, off_w, map_pts, rot_pts
        # return moff_h, moff_w, map_pts, rot_pts
    

    #
    # feature_slope_offsets
    #   feature_offsets should find an approximate location for off_h, off_w.
    #   feature_slope_offsets takes this approximate location and fine tunes it based on the main line slope.
    #
    def feature_slope_offsets(self, map_overlay, rotated_new_map, map_line, map_slope, foff_h, foff_w, frame_num):
        off_h = foff_h
        off_w = foff_w
        while True:
          rnm = rotated_new_map.copy()
          rnm = self.replace_border(rnm,
                            self.map_overlay.shape[0], self.map_overlay.shape[1],
                            off_h,off_w)
# ARD: the recentering changes the calculations of B a lot...
          # the general idea is that the slope computes the height axis and
          # feature points control the width axis.
          off_b = self.get_height_dif_by_line(rnm, map_line, map_slope)
          off_h += off_b   # solves 112 slightly off with huge oscilations for unknown reasons
          print("off_b, off_h, off_w", off_b, off_h, off_w)
          # off_h and off_w are absolute offsets for replace_border 
          rnm = rotated_new_map.copy()
          rnm = self.replace_border(rnm,
                            self.map_overlay.shape[0], self.map_overlay.shape[1],
                            off_h,off_w)

          print("###############")
          print("feature slope offsets")
          map_pts = []
          rot_pts = []
          mol = map_overlay.copy()
          shape, map_border = self.real_map_border(mol)
          shape, rot_border = self.real_map_border(rnm)
          gray_mol = cv2.cvtColor(mol, cv2.COLOR_BGR2GRAY)
          gray_rnm = cv2.cvtColor(rnm, cv2.COLOR_BGR2GRAY)
  
          feature_params = dict( maxCorners = 100,
                       # qualityLevel = 0.3,
                       qualityLevel = 0.01,
                       minDistance = 7,
                       blockSize = 7 )
          map_features = cv2.goodFeaturesToTrack(gray_mol, mask = None, **feature_params)
          rot_features = cv2.goodFeaturesToTrack(gray_rnm, mask = None, **feature_params)
          print("num map,rot features:", len(map_features), len(rot_features))
          best_match = []
          first_time_through = True
          map_feature_cnt = 0
          rot_feature_cnt = 0
          i_j_best_match = []
          for i, map_pt_lst in enumerate(map_features):
            map_pt = [int(map_pt_lst[0][0]), int(map_pt_lst[0][1])]
            # only consider features that are in both rotated image and the composite map
            if (not self.cvu.point_in_border(map_pt,map_border) or
                not self.cvu.point_in_border(map_pt,rot_border)):
              mol=cv2.circle(mol,map_pt,3,(0,255,0),-1)
              continue
            cv2.circle(mol,map_pt,3,(255,0,0),-1)
            map_feature_cnt += 1
            min_dist = self.INFINITE
            for j, rot_pt_lst in enumerate(rot_features):
              rot_pt = [int(rot_pt_lst[0][0]), int(rot_pt_lst[0][1])]
              if (not self.cvu.point_in_border(rot_pt,map_border)
                 or not self.cvu.point_in_border(rot_pt,rot_border)):
                rnm=cv2.circle(rnm,rot_pt,3,(0,255,0),-1)
                continue
              if first_time_through:
                rnm=cv2.circle(rnm,rot_pt,3,(255,0,0),-1)
                rot_feature_cnt += 1
              # dist = math.sqrt((map_pt[0]-rot_pt[0])**2+(map_pt[1]-rot_pt[1])**2)
              dist = abs(map_pt[1]-rot_pt[1])
              # print("dist:", dist, min_dist)
              if dist < min_dist:
                min_dist = dist
                i_j_best_match = [dist,i,j]
            # print(i, "i_j_bm:", i_j_best_match, map_feature_cnt, rot_feature_cnt)
            # there's at most one match between every i,j
            if min_dist < self.INFINITE:
              found = False
              for k, bm in enumerate(best_match):
                if i_j_best_match[1] == bm[1]:
                  if i_j_best_match[0] < bm[0]:
                    best_match[k] = i_j_best_match
                    found = True
                    print("i_j_bm break", i, j, k)
                    break
              if not found:
                best_match.append(i_j_best_match)
          # get mean best matches
          sum_dist, sum_h_dif, sum_w_dif = 0,0,0
          # n = min(int(.1 * len(best_match)),3)
          # n = 20
          n = 10
          print("Top N:", n,  len(best_match))
          # n = len(best_match)
          best_match = sorted(best_match,key=itemgetter(0))
          print("best match:", best_match)
          for k, bm in enumerate(best_match):
            if k >= n:
              break
            # print("bm:", bm)
            # print("mf:", map_features[bm[1]])
            # print("rf:", rot_features[bm[2]])
            # map_pt = [round(map_features[bm[1]][0][1]), round(map_features[bm[1]][0][0])]
            # rot_pt = [round(rot_features[bm[2]][0][1]), round(rot_features[bm[2]][0][0])]
            map_pt = [round(map_features[bm[1]][0][0]), round(map_features[bm[1]][0][1])]
            rot_pt = [round(rot_features[bm[2]][0][0]), round(rot_features[bm[2]][0][1])]
            # print("bm: ",k, [round(bm[0]), map_pt, rot_pt])
            sum_dist += bm[0]
            sum_h_dif += map_pt[0] - rot_pt[0]
            sum_w_dif += map_pt[1] - rot_pt[1]
          if len(best_match) == 0:
            print("NO MATCHING FEATURES/SLOPE")
            return None, None, [], []
          bm_off_h = round(sum_h_dif / n)
          bm_off_w = round(sum_w_dif / n)
          # bm_off_h = round(sum_h_dif / len(best_match))
          # bm_off_w = round(sum_w_dif / len(best_match))
          #
          # the general idea is that the slope is computing one axis and
          # feature points control the other axis
          #
          # off_w is solved via equation for B above, so no need for following:
          # off_h += bm_off_h
          off_w += bm_off_w
          if sum_dist/n <= 2 or (abs(bm_off_w) <= 1 and abs(off_b) <= 1):
            print("FOUND MATCHING FEATURES_SLOPE:", frame_num, off_h, off_w, int(sum_dist/len(best_match)), off_b, bm_off_w)
            break
          else:
            print("TRY FEATURES_SLOPE AGAIN:", frame_num, off_h, off_w, int(sum_dist/n), off_b, bm_off_w)
            cv2.imshow("mol feature offset", mol)
            cv2.imshow("rot feature offset", rnm)
            cv2.waitKey(0)
          continue   # end of loop
        #### FOUND MATCHING FEATURES/SLOPE ####
        map_pts = []
        rot_pts = []
        cv2.imshow("mol feature offset", mol)
        cv2.imshow("rot feature offset", rnm)
        cv2.waitKey(0)
        for k, bm in enumerate(best_match):
          map_pts.append(map_features[bm[1]])
          rot_pts.append(rot_features[bm[2]])
        return off_h, off_w, map_pts, rot_pts

    #################################
    # Map merging
    #################################
    def merge_maps(self, map, new_map):
        map_shape, map_border = self.real_map_border(map)
        new_map_shape, new_map_border = self.real_map_border(new_map)
        # find overlap
        map_maxh, map_minh, map_maxw, map_minw = self.cvu.get_min_max_borders(map_border)
        new_map_maxh, new_map_minh, new_map_maxw, new_map_minw = self.cvu.get_min_max_borders(new_map_border)
        print("new_map/map minw:", new_map_minw, map_minw)
        final_map = map.copy()
        buffer = 3  # eliminate the black border at the merge points
        for h in range(new_map_minh, new_map_maxh):
          for w in range(new_map_minw, new_map_maxw):
            if final_map[h,w].all() == 0:
              final_map[h,w] = new_map[h,w]
            elif new_map[h,w].all() == 0:
              pass
            elif (new_map[h-buffer,w].all() == 0 or
                  new_map[h+buffer,w].all() == 0 or
                  new_map[h, w+buffer].all() == 0 or
                  new_map[h, w-buffer].all() == 0):
              pass
            elif (final_map[h-buffer,w].all() == 0 or
                  final_map[h+buffer,w].all() == 0 or
                  final_map[h, w+buffer].all() == 0 or
                  final_map[h, w-buffer].all() == 0):
              final_map[h,w] = new_map[h,w]
            else:
              final_map[h,w] = .5*final_map[h,w] + .5*new_map[h,w]
        return final_map
    
    #################################
    # CREATE MAP - main map driver routine
    #################################

    def analyze(self, frame_num, action, prev_img_pth, curr_img_pth, done):
        curr_image = cv2.imread(curr_img_pth)
        curr_image_KP = Keypoints(curr_image)
        self.curr_move = self.get_birds_eye_view(curr_image)
        self.move_state(action, self.curr_move)
        # show the original and warped images
        # curr_image_KP.drawKeypoints()
        # curr_move_KP.drawKeypoints()
        # cv2.imshow("Original", image2)
        # cv2.imshow("Warped", self.curr_move )
        # cv2.waitKey(0)
        ###########################
        # Initialize map with first frame.
        ###########################
        if self.map is None:
          self.curr_move_height,self.curr_move_width,ch = self.curr_move.shape
          # add a big border
          self.border_buffer = max(self.curr_move_height,self.curr_move_width)*self.border_multiplier
          print("border buffer:", self.border_buffer)
          self.map = self.add_border(self.curr_move, self.border_buffer)
          self.map_overlay = self.map.copy()
          self.map_height,self.map_width,self.map_ch = self.map.shape
          self.map_KP = Keypoints(self.map)
          self.map_arr = asarray(self.map)

          self.robot_length = 200
          self.robot_location = [(self.border_buffer+self.robot_length + self.curr_move_height), (self.border_buffer + self.curr_move_width/2)]
          # self.robot_location = [(self.border_buffer + self.curr_move_height/2),(self.border_buffer+self.robot_length + self.curr_move_height)]
          print("self.robot_location:",self.map_height, self.map_width, self.border_buffer, self.robot_length)

          self.robot_location_history.append([self.robot_location, "Initial Guess"])
          self.VIRTUAL_MAP_SIZE = self.border_buffer * 1 + self.map_height
          self.map_virtual_map_center = self.VIRTUAL_MAP_SIZE / 2
          kp_matches,notsogood = self.map_KP.compare_kp(self.map_KP)
          print("num matches:", len(kp_matches))
          # orient to self
          map_pts, map_pts2 = self.map_KP.get_n_match_kps(kp_matches, self.map_KP, 3)
          self.robot_orientation = 0  # starting point in degrees
          print("orientation: ", self.robot_orientation)

        ###########################
        # Analyze and Integrate Each Subsequent Frame
        ###########################
        else:
          # initialize images and map for new move
          rows,cols,ch = self.curr_move.shape
          # cv2.imshow("curr_move:", self.curr_move)
          new_map = self.add_border(self.curr_move, self.border_buffer)
          # cv2.imshow("orig new_map:", new_map)
          # test to see if Img_Stitch could handle rotation of new_map/curr_move.
          # new_map2 = new_map.copy()
          # new_map2 = self.curr_move.copy()
          # cv2.imshow("new_map_border", new_map)
          new_map_KP = Keypoints(new_map)
          self.map_KP = Keypoints(self.map)

          #####################################################################
          # MAP OVERLAY APPROACH WITH IMAGE ROTATION AROUND ROBOT LOCATION
          # Line Analysis, Keypoint Analysis
          #####################################################################
          # line detection and keypoint detection
          # note: includes black edges from border
          #################################
          # Map Line Analysis 
          # On a tabletop and roads, the edges are typically well deliniated and visible.
          # For the initial tabletop apps, the line analysis has proven more reliable
          # than the keypoint analysis, which is also used below.
          #################################
          h = 0
          w = 1
          map_line, map_slope, max_dist = self.analyze_map_lines(self.map_overlay, frame_num)
          if map_slope is not None:
            new_map_img = new_map.copy()
            # assuming all left turns
            # start_angle = self.robot_orientation - 2
            start_angle = self.robot_orientation 
            best_line, best_line_angle = self.find_best_line_angle(self.map_overlay, 
                                         map_slope, new_map_img, start_angle=start_angle,
                                         end_angle=start_angle+15, frame_num=frame_num)
            if best_line_angle is not None:
              final_angle = best_line_angle
              print("Final best angle:", final_angle)
              self.robot_orientation = final_angle
              # compare best Map Line to best "new map img" Line
              # Two lines defined by: y = mx + b
              # Function to find the vertical distance between parallel lines
              # Adjust robot location to match
              # rotated_new_map = new_map.copy()  # for resilience against bugs
              # rotated_new_map = self.rotate_about_robot(rotated_new_map, final_angle)
              # cv2.line(rotated_new_map, (best_line[0], best_line[1]), (best_line[2], best_line[3]), (0,255,0), 3, cv2.LINE_AA)
              # cv2.imshow("Before adjust, rotated around robot", rotated_new_map)
              # Xdif = self.adjustRobotLocationByLines((map_line[0], map_line[1]), 
              #           (map_line[2], map_line[3]), (best_line[0], best_line[1]), 
              #           (best_line[2], best_line[3]))
              # print("final line angle, Xdif:", final_angle, Xdif)

              rotated_new_map = new_map.copy()  # for resilience against bugs
              rotated_new_map = self.rotate_about_robot(rotated_new_map, final_angle)
              cv2.line(rotated_new_map, (best_line[0], best_line[1]), (best_line[2], best_line[3]), (0,255,0), 3, cv2.LINE_AA)
              # self.show_rot(new_map)
            else:
              ############################
              # Keypoint Analysis 
              # line analysis failed. Try to find initial rotation via keypoints
              ############################
              self.map_overlay_KP = Keypoints(self.map_overlay)
              start_angle = .5
              end_angle = start_angle+10
              new_map_img = new_map.copy()
              best_kp_angle, best_kp_delta_h, best_kp_delta_w, map_pts, new_map_pts = self.find_best_kp_angle(self.map_overlay_KP, new_map_img, start_angle, end_angle)
              print("Final Best KP angle/delta:", best_kp_angle,best_kp_delta_h, best_kp_delta_w)
              # best KP
              print("map/new_map kps:", map_pts, new_map_pts)
              rotated_new_map = new_map.copy()  # for resilience against bugs
              rotated_new_map_KP = Keypoints(new_map)  # for resilience against bugs
              rotated_new_map = self.rotate_about_robot(rotated_new_map, best_kp_angle)
              if frame_num >= self.stop_at_frame:
                  rotated_new_map_disp = rotated_new_map_KP.drawKeypoints()
                  cv2.imshow("map kp", rotated_new_map_disp)
                  rotated_new_map_disp = rotated_new_map.copy()
                  rotated_new_map_disp_KP = Keypoints(rotated_new_map_disp)
                  rotated_new_map_disp = rotated_new_map_disp_KP.drawKeypoints()
                  cv2.imshow("best kp, rotated", rotated_new_map_disp)
                  cv2.waitKey(0)

            ##############
            # Best Rotated Image found via line or KP analysis
            # Next, Merge and display images
            if best_line_angle is not None or best_kp_angle is not None:
              print("frame_num:",frame_num)
              # (self.border_buffer+3*self.robot_gripper_pad_len + self.map_width)]
              rotated_new_map = self.replace_border(rotated_new_map,
                                  self.map_overlay.shape[0], self.map_overlay.shape[1],
                                  0, 0)
              mol = self.map_overlay.copy()
              rnm = rotated_new_map.copy()
              for kp_mode in ["SIFT", "ORB"]:
                print("kp_mode: ", kp_mode)
                rnm_kp = Keypoints(rnm, kp_mode)
                mol_kp = Keypoints(mol, kp_mode)
                kp_off_h, kp_off_w, map_pts, new_map_rot_pts = self.evaluate_map_kp_offsets(mol_kp, rnm)
                rnm = rnm_kp.drawKeypoints()
                mol = mol_kp.drawKeypoints()
                #if frame_num >= self.stop_at_frame:
                #  cv2.imshow("rotate_new_mapkp", rnm)
                #  cv2.imshow("mol kp", mol)
                #  cv.waitKey(0)
                #  cv2.destroyAllWindows()
                if len(map_pts) > 0:
                  break
              print("KP match: ",(kp_off_h, kp_off_w))

              # If kp_mode didn't work, try more generic features, combined with line slope
              foff_h = 0 
              foff_w = 0
              # if len(map_pts) == 0:
              if True:
                moff_h,moff_w = self.manual_tuning(frame_num)
                # if len(map_pts) == 0:
                if True:
                  foff_h, foff_w, map_pts, new_map_rot_pts = self.feature_offsets(self.map_overlay, rotated_new_map, map_line, map_slope, best_line, frame_num)
                  print("features: ",(foff_h, foff_w),(moff_h, moff_w))
                else:
                  foff_h = kp_off_h
                  foff_w = kp_off_w
                  print("KP match: ",(foff_h, foff_w),(moff_h, moff_w))

                # compute from scratch; ignore above variance/manual/keypoint calculations.
                foff_h, foff_w = 0, 0
                off_h, off_w, map_pts, new_map_rot_pts = self.feature_slope_offsets(self.map_overlay, rotated_new_map, map_line, map_slope, foff_h, foff_w, frame_num)

              if len(map_pts) == 0:
                print("KEYPOINT, FEATURE AND LINE TUNING FAILED, FALLBACK TO MANUAL TUNING", frame_num)
                off_h,off_w = self.manual_tuning(frame_num)
                print("off_h, off_w:", off_h, off_w)

              if frame_num >= self.stop_at_frame:
                cv2.imshow("old overlay:", self.map_overlay)
                # Final rotation
                rotated_new_map = self.replace_border(rotated_new_map,
                                  self.map_overlay.shape[0], self.map_overlay.shape[1],
                                  off_h,off_w)
                rotated_new_map_disp = rotated_new_map.copy() # display rotation
                pt = [int(self.robot_location[1]), int(self.robot_location[0])]
                rotated_new_map_disp=cv2.circle(rotated_new_map_disp,pt,3,(255,0,0),-1)
                txt = "rot reborder h:" + str(off_h) + " w:" + str(off_w)
                cv2.imshow(txt, rotated_new_map_disp)
                # Note: if robot_location is correct and doing LEFT with correct angle,
                #       then just merge?  Nope, didn't work.
              # map_ovrly = cv2.addWeighted(self.map_overlay,0.5,rotated_new_map,0.5,0.5)
              self.map_overlay = self.merge_maps(self.map_overlay, rotated_new_map)
              if frame_num >= self.stop_at_frame:
                map_ovrly = self.merge_maps(self.map_overlay, rotated_new_map_disp)
                # cv2.line(map_ovrly, (map_line[0], map_line[1]), (map_line[2], map_line[3]), (0,255,0), 3, cv2.LINE_AA)
                txt = "map overlay " + str(frame_num)
                cv2.imshow(txt,map_ovrly)
                # cv2.imshow("rotated around robot", rotated_new_map)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

          #####################################################################
          # MAP STICHING APPROACH BASED UPON KEYPOINTS
          # Problem when no matching keypoints!
          #####################################################################
          if False:
            # Stitching - based on keypoints
            # OpenCV stitching came up with something asthetically pleasing, but
            # not accurate. 
            ###################
            # map_border = self.real_map_border(self.map)
            # new_map_border = self.real_map_border(new_map)
            s = Img_Stitch(self.map, new_map)
            s.leftshift()
            s.showImage('left')
            s.rightshift()
            self.map = s.leftImage
            # Img_stitch: the rotation of the curr_move to map eventually fails
            # s2 = Img_Stitch(self.map, new_map2)
            # s2.leftshift()
            # s2.showImage('left2')
            # s2.rightshift()
            # self.map = s2.leftImage
            self.map = s.leftImage
            self.map_height,self.map_width,self.map_ch = self.map.shape
            self.map_KP = Keypoints(self.map)
            self.map_arr = asarray(self.map)

          return
