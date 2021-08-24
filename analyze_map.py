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
        self.stop_at_frame = 114
        # self.stop_at_frame = 117
        self.KPs = None
        self.map = None
        # real map array info
        # real map array sizes change over time
        self.map_overlay = None
        self.map_overlay_KP = None
        self.map_arr = None
        self.map_KP = None
        self.map_rows = None
        self.map_cols = None
        self.map_ch = None
        self.border_buffer = None   # move_img
        self.border_multiplier = 2   # move_img
        # To compute the distances,var of moves
        self.move_rows = None
        self.move_cols = None
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
        self.robot_gripper_pad_len = None
        self.clusters = []
        self.grabable_objects = []
        self.container_objects = []
        self.last_line = None
        self.dif_avg = []
        self.dif_var = []
        self.min_pt = []
        self.max_pt = []
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
            INFINITE = 10000000000000000000
            min_x = INFINITE
            max_x = -INFINITE
            min_y = INFINITE
            max_y = -INFINITE
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
        print("len good matches:", len(good_matches))
        map_pts, new_map_rot_pts = map_KP.get_n_match_kps(good_matches, new_map_rot_KP, 6, return_list=False, border=map_border)
        delta_x, delta_y = 0,0
        x,y = 0,1
        for i, map_pt in enumerate(map_pts):
          # weight the deltas
          # weight = len(map_pts) - i
          # delta_x += weight*(map_pt.pt[x] - new_map_rot_pts[i].pt[x])
          # Problem: what if only 5KP are returned? Then weighted delta computed based upon
          #   6KP would be wrong.  Could store 6 deltas from prev run, and add in missing KPs
          print("map_pt", map_pt, new_map_rot_pts)
          delta_x += (map_pt.pt[x] - new_map_rot_pts[i].pt[x])
          delta_y += (map_pt.pt[y] - new_map_rot_pts[i].pt[y])
        if len(map_pts) > 0:
          delta_x = int(delta_x / len(map_pts))
          delta_y = int(delta_y / len(map_pts))
        return delta_x, delta_y, map_pts, new_map_rot_pts

    ##############
    # find_best_kp_angle():
    #   map is the full composite map
    #   new map is the new robot image to be integrated into the full map
    def find_best_kp_angle(self, map_KP, new_map, start_angle, num_angles, delta_deg):
      x = 0
      y = 1
      best_kp_angle, best_kp_delta_x, best_kp_delta_y = None, None, None
      best_map_pts, best_new_map_pts = None, None
      for i in range(num_angles):
        angle = start_angle + delta_deg*i
        new_map_rot = self.rotate_about_robot(new_map, angle)
        delta_x, delta_y, map_pts, new_map_rot_pts = self.evaluate_map_kp_offsets(map_KP, new_map_rot)
        if len(map_pts) > 0 and (best_kp_delta_x is None or delta_x < best_kp_delta_x):
            best_kp_angle, best_kp_delta_x, best_kp_delta_y = angle, delta_x, delta_y 
            best_map_pts, best_new_map_pts = map_pts, new_map_rot_pts
            print("new best kp angle:",best_map_pts,best_new_map_pts)
      return best_kp_angle, best_kp_delta_x, best_kp_delta_y, best_map_pts, best_new_map_pts

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

    def border_to_polygon(self, border):
        b = []
        for bdr in border:
          b.append(list(bdr[0]))
        # print("brdr to poly:", b)
        poly = Polygon(b)
        return poly

    def get_min_max_borders(self, border):
        b = []
        for bdr in border:
          b.append(list(bdr[0]))
        if len(b) != 4:
          return None, None, None, None
        # print("get_min_max:", b, border)
        maxx = max(b[0][0], b[1][0], b[2][0], b[3][0])
        minx = min(b[0][0], b[1][0], b[2][0], b[3][0])
        maxy = max(b[0][1], b[1][1], b[2][1], b[3][1])
        miny = min(b[0][1], b[1][1], b[2][1], b[3][1])
        return maxx, minx, maxy, miny

    def replace_border(self, img, desired_rows, desired_cols, offset_rows, offset_cols):
        shape, border = self.real_map_border(img)
        maxx, minx, maxy, miny = self.get_min_max_borders(border)
        orig_row, orig_col = img.shape[:2]
        print("shapes:", orig_row,orig_col,minx,maxx,miny,maxy)
        extract_img_rect = img[miny:maxy, minx:maxx]
        extracted_rows, extracted_cols = extract_img_rect.shape[:2]
        border_top = int((desired_rows - extracted_rows)/2) + offset_rows
        border_bottom = desired_rows - border_top - extracted_rows
        border_left = int((desired_cols - extracted_cols)/2) + offset_cols
        border_right = desired_cols - border_left - extracted_cols
        print("replace_border:",border_top, border_bottom, border_left, border_right)
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
        print("angle,M:", angle, M)
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
          imagecontours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
          # for each of the contours detected, the shape of the contours is approximated 
          # using approxPolyDP() function and the contours are drawn in the image using 
          # drawContours() function
          for count in imagecontours:
            epsilon = 0.01 * cv2.arcLength(count, True)
            approximations = cv2.approxPolyDP(count, epsilon, True)
            # e.g. [[[224 224]] [[252 372]] [[420 372]] [[447 224]]]
            #the name of the detected shapes are written on the image
            i, j = approximations[0][0] 
            if len(approximations) == 3:
              # cv2.putText(imageread, "Triangle", (i, j), cv2.FONT_HERSHEY_COMPLEX, 1, 0, 2)
              shape = "Triangle"
            elif len(approximations) == 4:
              # cv2.putText(imageread,"Rectangle",(i,j),cv2.FONT_HERSHEY_COMPLEX,1,0,2)
              shape = "Trapezoid"
            elif len(approximations) == 5:
              # cv2.putText(imageread,"Pentagon",(i,j),cv2.FONT_HERSHEY_COMPLEX,1,0,2)
              shape = "Pentagon"
            elif 6 < len(approximations) < 15:
              # cv2.putText(imageread,"Ellipse",(i,j),cv2.FONT_HERSHEY_COMPLEX,1,0,2)
              shape = "Ellipse"
            else:
              # cv2.putText(imageread,"Circle",(i,j),cv2.FONT_HERSHEY_COMPLEX,1,0,2)
              shape = "Circle"
            # print("map shape:", shape, approximations)
            #displaying the resulting image as the output on the screen
            # imageread = mapimg.copy()
            # cv2.drawContours(thresh, [approximations], 0, (0), 3)
            # cv2.imshow(shape, thresh)
            # cv2.waitKey(0)
            return shape, approximations


    def line_in_border(self,border,pt0,pt1):
      # use Poly for Trapazoids
      if False and len(border == 4):
        maxx, minx, maxy, miny = self.get_min_max_borders(border)
        if maxx is None:
           return False
        bufzone = 25
        if ((maxx-bufzone <= pt0[0] and maxx-bufzone <= pt1[0]) or 
            (maxy-bufzone <= pt0[1] and maxy-bufzone <= pt1[1]) or 
            (minx+bufzone >= pt0[0] and minx+bufzone >= pt1[0]) or 
            (miny+bufzone >= pt0[1] and miny+bufzone >= pt1[1])):
          print("not_in_border:",pt0,pt1)
          return False
        else:
          print("in_border:",pt0,pt1)
          return True
      else:
        # Create Line objects
        poly = self.border_to_polygon(border)
        line_a = LineString([pt0, pt1])
        if line_a.within(poly):
          print("Line within Poly")
          return True
        else:
          print("Line not within Poly")
          if line_a.intersects(poly):
            print("Line intersects Poly")
          if line_a.touches(poly):
            print("Line touches Poly")
          return False

    def point_in_border(self,pt,border):
       bufzone = 10
       b = []
       for bdr in border:
         b.append(list(bdr[0]))
       # must use Poly for Trapazoid!
       if False and len(b) == 4:
          if (b[0][0]+bufzone <= pt[0] and b[0][1]+bufzone <= pt[1] and
             b[1][0]-bufzone <= pt[0] and b[1][1]-bufzone >= pt[1] and
             b[2][0]-bufzone >= pt[0] and b[2][1]-bufzone >= pt[1] and
             b[3][0]+bufzone >= pt[0] and b[3][1]+bufzone <= pt[1]):
             return True
          else:
             return False
       elif True or len(b) > 4:
        # Create Point objects
        poly = self.border_to_polygon(border)
        for i in range(4):
          if i == 0:
            p0 = pt[0] - bufzone
            p1 = pt[1]
          elif i == 1:
            p0 = pt[0] + bufzone
            p1 = pt[1]
          elif i == 2:
            p0 = pt[0]
            p1 = pt[1] - bufzone
          elif i == 3:
            p0 = pt[0]
            p1 = pt[1] + bufzone
          Pt = Point(p0,p1)
          if Pt.within(poly):
            # print("Point within Poly")
            # return True
            continue
          else:
            # print("Point not within Poly")
            # if Pt.touches(poly):
            #   print("Point touches Poly")
            return False
        return True
       else:
          return False

    #########################
    # Map Line Analysis 
    #########################
    def analyze_map_lines(self, map):
      linesP, imglinesp = self.cvu.get_lines(map)
      # cv2.imshow("Map LinesP - Probabilistic Line Transform", imglinesp);
      # cv2.waitKey(0)
      # print(linesP.flatten)
      shape, map_border = self.real_map_border(map)
      max_dist = 0
      map_dx = None
      map_dy = None
      map_slope = None
      # prefer choosing same line
      if self.last_line is not None:
        [ll0, ll1, ll2, ll3] = self.last_line
        last_line_slope = (ll0 - ll2) / (ll1 - ll3)
        last_line = LineString([(ll0,ll1),(ll2,ll3)])
        min_diff = None
        # slope_diff_thresh = .25
        slope_diff_thresh = 1.25
        for [[l0,l1,l2,l3]] in linesP:
          if (l1-l3) == 0:
            INFINITE = 10000000000000000000
            map_slope = INFINITE
          else:
            map_slope = (l0 - l2) / (l1 - l3)
          dist = np.sqrt((l0-l2)**2 + (l1-l3)**2)
          print("slope diff:", abs(1 - (last_line_slope / map_slope)), map_slope)
          if abs(1 - (last_line_slope / map_slope)) < slope_diff_thresh:
            curr_line = LineString([(l0,l1),(l2,l3)])
            map_line = [l0,l1,l2,l3]
            if curr_line.intersects(last_line):
              print("######## Found Map SlopeA #########")
              print("map intercept")
              return map_line, map_slope, dist
            if curr_line.touches(last_line):
              print("######## Found Map SlopeB #########")
              print("map touches")
              return map_line, map_slope, dist
            # y = mx + b  ; b = y - mx
            llb = ll1 - last_line_slope * ll0
            mb = l1 - map_slope * l0
            if abs(1 - (llb / mb)) < .02:
              print("######## Found Map Slope1 #########")
              print("map close", mb, llb)
              return map_line, map_slope, dist
            else:
              print("potential line slope: M1,B1;M2,B2:", last_line_slope, llb, map_slope, mb)
            if min_diff is None or min_diff > abs(1 - (llb / mb)):
              min_diff = abs(1 - (llb / mb))
              best_map_slope = map_slope
              best_map_line = map_line
              best_dist = dist
              print("min slope:", min_diff, best_map_line, best_map_slope)
        if min_diff is not None and min_diff < slope_diff_thresh:
              print("######## Found Map Slope3 #########")
              print("best last slope:", best_map_line, best_map_slope)
              return best_map_line, best_map_slope, best_dist 
          # fall through
      # Find slope of longest line within the map border
      for [[l0,l1,l2,l3]] in linesP:
          if self.line_in_border(map_border, (l0,l1), (l2,l3)):
            print("in border:", l0,l1,l2,l3)
            dx = l0 - l2
            dy = l1 - l3
            dist = np.sqrt(dx**2 + dy**2)
            if max_dist < dist:
              map_line = [l0,l1,l2,l3]
              map_dx = dx
              map_dy = dy
              max_dist = dist
          else:
            print("not in border:", l0,l1,l2,l3)
      if map_dx is None or map_dy is None:
          print("no map line")
          return None, None, None
      map_slope = map_dx / map_dy
      self.last_line = map_line
      print("longest line",map_line)
      return map_line, map_slope, max_dist
    
    ##############
    # find_best_line_angle()
    ##############
    # map is the full composite map
    # new map is the new robot image to be integrated into the full map
    def find_best_line_angle(self, map, map_slope, new_map, start_angle, num_angles, delta_deg, frame_num=0):
      min_slope_dif_angle = None
      INFINITE = 10000000000000000000
      min_slope_dif = INFINITE
      best_line_angle = None
      best_line = None
      for i in range(num_angles):
        angle = start_angle + delta_deg*i
        new_map_rot = self.rotate_about_robot(new_map, angle)
        # if frame_num >= self.stop_at_frame:
        #   cv2.imshow("rotated new map:",new_map_rot)
        #   cv2.waitKey(0)
        shape, map_border = self.real_map_border(new_map_rot)
        linesP, imglinesp = self.cvu.get_lines(new_map_rot)
        # print(linesP)
        for [[l0,l1,l2,l3]] in linesP:
          if self.line_in_border(map_border, (l0,l1), (l2,l3)):
            # print("l0-4:", l0,l1,l2,l3)
            # print("map_slope:", map_slope)
            dx = l0 - l2
            dy = l1 - l3
            dist = np.sqrt(dx**2 + dy**2)
            if dy == 0:
              diff_slope = INFINITE
            else:
              diff_slope = map_slope - (dx/dy)
            # print("diff slope/min:", diff_slope, min_slope_dif)
            if abs(diff_slope) < abs(min_slope_dif):
              min_slope_dif = diff_slope
              min_slope_dif_angle = angle
              best_line = [l0,l1,l2,l3]
              print("best line so far:", best_line, angle, diff_slope, "dxy:", map_slope, (dx/dy))
      if min_slope_dif == INFINITE:
        print("######## New Map Slope not found #########")
        final_angle = None
      else:
        print("######## New Map Slope found #########")
        print("New Map info:", best_line, min_slope_dif_angle)
        final_angle = start_angle + min_slope_dif_angle
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
          # self.robot_location[x] += Xdif
          print("Xdif Ydiff: ", Xdif, Ydif)
          # self.robot_location[y] = NewYatRobot2
          self.robot_location_history.append([self.robot_location, "Parallel Line adjustment"])
          print("robot loc:", self.robot_location, Xdif, MapXatRobot, NewXatRobot, MapB, NewB)
          return Xdif
                 
    #########################
    # Hardcoding / Manually tuning / gathering data to figure out algorithm
    def manual_tuning(self, frame_num):
        if frame_num == 112:
          [off_x, off_y] = [0, 0] # best
        if frame_num == 113:
          # [off_x, off_y] = [15, -50]
          # [off_x, off_y] = [-25, -10] raises it way too high
          # [off_x, off_y] = [-5, -20]   much closer
          # [off_x, off_y] = [0, -20]   gap on upper // increased
          # [off_x, off_y] = [-5, -25] go lower/to left
          # [off_x, off_y] = [0, -30]  good height/ need left
          # [off_x, off_y] = [0, -40]  very close
          [off_x, off_y] = [0, -45]  # best
        if frame_num == 114:
          # [off_x, off_y] = [0, -90] need right
          # [off_x, off_y] = [10, -85] need lower
          # [off_x, off_y] = [5, -85] need right
          # [off_x, off_y] = [5, -80] close
          # [off_x, off_y] = [5, -75] closer
          # [off_x, off_y] = [10, -80]  
          # [off_x, off_y] = [10, -75]  need higher
          [off_x, off_y] = [5, -75]  # best so far
          [off_x, off_y] = [3, -73]  # best so far
        if frame_num == 115:
          # can focus on the cube too
          # to lower, increase x ; to right, decrease y
          # [off_x, off_y] = [0, -75] # much closer, need lower
          # [off_x, off_y] = [5, -55] # need lower, left
          # [off_x, off_y] = [10, -60] # closer. need left
          # [off_x, off_y] = [3, -73] # good, need lower
          [off_x, off_y] = [8, -73] # good
        if frame_num == 116:
          [off_x, off_y] = [12, -65] # good
        print("manual_tuning: ", off_x, off_y)
        return off_x, off_y
    
    def feature_offsets(self, map_overlay, rotated_new_map):
        print("###############")
        print("feature offsets")
        off_x = None
        off_y = None
        map_pts = []
        rot_pts = []
        mol = map_overlay.copy()
        rnm = rotated_new_map.copy()
        shape, map_border = self.real_map_border(mol)
        shape, rot_border = self.real_map_border(rnm)
        gray_mol = cv2.cvtColor(mol, cv2.COLOR_BGR2GRAY)
        gray_rnm = cv2.cvtColor(rnm, cv2.COLOR_BGR2GRAY)

        # image	Input 8-bit or floating-point 32-bit, single-channel image.
        # corners	Output vector of detected corners.
        # maxCorners	Maximum number of corners to return. If there are more corners 
        #     than are found, the strongest of them is returned. maxCorners <= 0 implies 
        #     that no limit on the maximum is set and all detected corners are returned.
        # qualityLevel	Parameter characterizing the minimal accepted quality of image 
        #     corners. The parameter value is multiplied by the best corner quality measure,
        #      which is the minimal eigenvalue (see cornerMinEigenVal ) or the Harris 
        #     function response (see cornerHarris ). The corners with the quality measure 
        #     less than the product are rejected. For example, if the best corner has the 
        #     quality measure = 1500, and the qualityLevel=0.01 , then all the corners with 
        #     the quality measure less than 15 are rejected.
        # minDistance	Minimum possible Euclidean distance between the returned corners.
        # mask	Optional region of interest. If the image is not empty (it needs to have 
        #     the type CV_8UC1 and the same size as image ), it specifies the region in 
        #     which the corners are detected.
        # blockSize	Size of an average block for computing a derivative covariation 
        #     matrix over each pixel neighborhood. See cornerEigenValsAndVecs .
        # useHarrisDetector	Parameter indicating whether to use a Harris detector 
        #     or cornerMinEigenVal.
        # k	Free parameter of the Harris detector.
        feature_params = dict( maxCorners = 100,
        	       # qualityLevel = 0.3,
        	       qualityLevel = 0.01,
        	       minDistance = 7,
        	       blockSize = 7 )
        map_features = cv2.goodFeaturesToTrack(gray_mol, mask = None, **feature_params)
        rot_features = cv2.goodFeaturesToTrack(gray_rnm, mask = None, **feature_params)
        print("num map,rot features:", len(map_features), len(rot_features))
        delta = []
        INFINITE = 10000000000000000000
        first_time_through = True
        # print("map_features:", map_features)
        print("len map_features:", len(map_features))
        for i, map_pt_lst in enumerate(map_features):
          map_pt = [int(map_pt_lst[0][0]), int(map_pt_lst[0][1])]
          # print("i map_pt:", i, map_pt)
          if not self.point_in_border(map_pt,map_border):
            # print("map pt not in border:", map_pt, map_border)
            cv2.circle(mol,map_pt,3,(0,255,0),-1)
            continue
          cv2.circle(mol,map_pt,3,(255,0,0),-1)
          for j, rot_pt_lst in enumerate(rot_features):
            rot_pt = [int(rot_pt_lst[0][0]), int(rot_pt_lst[0][1])]
            if not self.point_in_border(rot_pt,rot_border):
              # print("rot pt not in border:", rot_pt, rot_border)
              cv2.circle(rnm,rot_pt,3,(0,255,0),-1)
              continue
            if first_time_through:
              cv2.circle(rnm,rot_pt,3,(255,0,0),-1)
            dist = math.sqrt((map_pt[0]-rot_pt[0])**2+(map_pt[1]-rot_pt[1])**2)
            delta.append([dist, i, j])
        cv2.imshow("mol feature offset", mol)
        cv2.imshow("rot feature offset", rnm)
        cv2.waitKey(0)
        min_var = INFINITE
        if len(delta) > 6:
          delta = sorted(delta,key=itemgetter(0))
          print("delta:", delta[0:6])
          n = 6   # find top-6 matches for consistent offsets
          for i in range(len(delta)-n):
            var = statistics.variance(delta[i:i+n][0])
            if var < min_var:
              min_var = var
              min_i = i
          print("min_var:",i,min_var, delta[min_i:min_i+n])
        if min_var != INFINITE:
          sum_x = 0
          sum_y = 0
          for i in range(n):
            mf = delta[min_i+i][1]
            rf = delta[min_i+i][2]
            mfeat = map_features[mf][0]
            rfeat = rot_features[rf][0]
            map_pts.append(map_features[mf])
            rot_pts.append(rot_features[rf])
            print("map_feat:", mfeat, rfeat)
            sum_x += mfeat[0] - rfeat[0]
            sum_y += mfeat[1] - rfeat[1]
          off_x = int(sum_x/n)
          off_y = int(sum_y/n)
        return off_x, off_y, map_pts, rot_pts
    
    #################################
    # Map merging
    #################################
    def merge_maps(self, map, new_map):
        map_shape, map_border = self.real_map_border(map)
        new_map_shape, new_map_border = self.real_map_border(new_map)
        # find overlap
        map_maxx, map_minx, map_maxy, map_miny = self.get_min_max_borders(map_border)
        new_map_maxx, new_map_minx, new_map_maxy, new_map_miny = self.get_min_max_borders(new_map_border)
        final_map = map.copy()
        for x in range(new_map_minx, new_map_maxx):
          for y in range(new_map_miny, new_map_maxy):
            if final_map[y,x].all() == 0:
              final_map[y,x] = new_map[y,x]
            elif new_map[y,x].all() == 0:
              pass
            else:
              final_map[y,x] = .5*final_map[y,x] + .5*new_map[y,x]
        return final_map
    
    #################################
    # CREATE MAP - main map driver routine
    #################################

    def analyze(self, frame_num, action, prev_img_pth, curr_img_pth, done):
        curr_image = cv2.imread(curr_img_pth)
        curr_image_KP = Keypoints(curr_image)
        w = curr_image.shape[0]
        h = curr_image.shape[1]

        # avg (x,y) dif, var:  (2.37,1.62) (1715,423)
        pts = np.array([(0,0),(w,0),(w*28/32,h*21/32),(w*4/32,h*21/32)], dtype = "float32")
        self.robot_gripper_pad_len = (h * 11/32)
        # print("w,h:", w,h)
        print("pts:", pts)
        # apply the four point tranform to obtain a "birds eye view" of the image
        above_view = self.four_point_transform(curr_image, pts)
        self.move_state(action, above_view)
        # show the original and warped images
        # curr_image_KP.drawKeypoints()
        # curr_move_KP.drawKeypoints()
        # cv2.imshow("Original", image2)
        # cv2.imshow("Warped", above_view2 )
        # cv2.waitKey(0)
        ###########################
        # Initialize map with first frame.
        ###########################
        if self.map is None:
          self.move_rows,self.move_cols,ch = self.curr_move.shape
          # add a big border
          self.border_buffer = max(self.move_rows,self.move_cols)*self.border_multiplier
          print("border buffer:", self.border_buffer)
          self.map = self.add_border(self.curr_move, self.border_buffer)
          self.map_overlay = self.map.copy()
          self.map_rows,self.map_cols,self.map_ch = self.map.shape
          self.map_KP = Keypoints(self.map)
          self.map_arr = asarray(self.map)

          # center of rotation relative to "new map"
          self.robot_location = [((self.map_rows / 2)+self.border_buffer), (self.border_buffer+3*self.robot_gripper_pad_len + self.map_cols)]

          self.robot_location_history.append([self.robot_location, "Initial Guess"])
          self.VIRTUAL_MAP_SIZE = self.border_buffer * 1 + self.map_rows
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
          x = 0
          y = 1
          map_line, map_slope, max_dist = self.analyze_map_lines(self.map_overlay)
          if map_slope is not None:
            new_map_img = new_map.copy()
            # assuming all left turns
            start_angle = self.robot_orientation - 2
            best_line, best_line_angle = self.find_best_line_angle(self.map_overlay, 
                                         map_slope, new_map_img, start_angle=start_angle,
                                         num_angles=20, delta_deg=.5, frame_num=frame_num)
            print("intermediate best line angle:", best_line_angle)
            # after narrowing down to +-.5, fine tune angle to closest .1degree
            if best_line_angle is not None:
              best_line, best_line_angle = self.find_best_line_angle(self.map_overlay, 
                              map_slope, new_map_img, start_angle=(best_line_angle - .5), 
                              num_angles=9, delta_deg=.1, frame_num=frame_num)
              print("final best line angle:", best_line_angle)
            if best_line_angle is not None:
              # compare best Map Line to best "new map img" Line
              # Two lines defined by: y = mx + b
              # Function to find the vertical distance between parallel lines
              # Adjust robot location to match
              rotated_new_map = new_map.copy()  # for resilience against bugs
              rotated_new_map = self.rotate_about_robot(rotated_new_map, best_line_angle)
              cv2.line(rotated_new_map, (best_line[0], best_line[1]), (best_line[2], best_line[3]), (0,255,0), 3, cv2.LINE_AA)
              # cv2.imshow("Before adjust, rotated around robot", rotated_new_map)
              Xdif = self.adjustRobotLocationByLines((map_line[0], map_line[1]), 
                        (map_line[2], map_line[3]), (best_line[0], best_line[1]), 
                        (best_line[2], best_line[3]))
              self.robot_orientation = best_line_angle
              print("best_line_angle, Xdif:", best_line_angle, Xdif)

              rotated_new_map = new_map.copy()  # for resilience against bugs
              rotated_new_map = self.rotate_about_robot(rotated_new_map, best_line_angle)
              rotated_new_map_disp = rotated_new_map.copy()
              cv2.line(rotated_new_map_disp, (best_line[0], best_line[1]), (best_line[2], best_line[3]), (0,255,0), 3, cv2.LINE_AA)
            else:
              ############################
              # Keypoint Analysis 
              # line analysis failed. Try to find initial rotation via keypoints
              ############################
              self.map_overlay_KP = Keypoints(self.map_overlay)
              start_angle = .5
              num_angles = 20
              new_map_img = new_map.copy()
              delta_deg = .5
              best_kp_angle, best_kp_delta_x, best_kp_delta_y, map_pts, new_map_pts = self.find_best_kp_angle(self.map_overlay_KP, new_map_img, start_angle, num_angles, delta_deg)
              if best_kp_angle is not None:
                print(".5 delta Best KP angle/delta:", best_kp_angle,best_kp_delta_x,best_kp_delta_y)
                start_angle = best_kp_angle - .5
                num_angles = 9
                delta_deg = .1
                best_kp_angle, best_kp_delta, best_kp_delta_y, map_pts, new_map_pts = self.find_best_kp_angle(self.map_overlay_KP, new_map_img, start_angle, num_angles, delta_deg)
                print("Final Best KP angle/delta:", best_kp_angle,best_kp_delta, best_kp_delta_y)
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

            ##############
            # Best Rotated Image found via line or KP analysis
            # Next, Merge and display images
            if best_line_angle is not None or best_kp_angle is not None:
              print("frame_num:",frame_num)
              # (self.border_buffer+3*self.robot_gripper_pad_len + self.map_cols)]
              rotated_new_map = self.replace_border(rotated_new_map,
                                  self.map_overlay.shape[0], self.map_overlay.shape[1],
                                  0, 0)
              mol = self.map_overlay.copy()
              rnm = rotated_new_map.copy()
              for kp_mode in ["SIFT", "ORB"]:
                print("kp_mode: ", kp_mode)
                rnm_kp = Keypoints(rnm, kp_mode)
                mol_kp = Keypoints(mol, kp_mode)
                off_x, off_y, map_pts, new_map_rot_pts = self.evaluate_map_kp_offsets(mol_kp, rotated_new_map)
                rnm = rnm_kp.drawKeypoints()
                mol = mol_kp.drawKeypoints()
                if frame_num >= self.stop_at_frame:
                  cv2.imshow("rotate_new_mapkp", rnm)
                  cv2.imshow("mol kp", mol)
                  cv.waitKey(0)
                  cv2.destroyAllWindows()
                if len(map_pts) > 0:
                  break

              # If kp_mode didn't work, try more generic features
              if len(map_pts) == 0:
                off_x, off_y, map_pts, new_map_rot_pts = self.feature_offsets(self.map_overlay, rotated_new_map)
       
                moff_x,moff_y = self.manual_tuning(frame_num)
                print("features: ",(off_x, off_y),(moff_x, moff_y))

              if len(map_pts) == 0:
                print("keypoint tuning failed, fallback to manual tuning", frame_num)
                off_x,off_y = self.manual_tuning(frame_num)

              if frame_num >= self.stop_at_frame:
                cv2.imshow("old overlay:", self.map_overlay)
                rotated_new_map_disp = self.replace_border(rotated_new_map_disp,
                                  self.map_overlay.shape[0], self.map_overlay.shape[1],
                                  off_x,off_y)
                cv2.imshow("after rotate and rebordering", rotated_new_map_disp)
              rotated_new_map = self.replace_border(rotated_new_map,
                                  self.map_overlay.shape[0], self.map_overlay.shape[1],
                                  off_x,off_y)
              # map_ovrly = cv2.addWeighted(self.map_overlay,0.5,rotated_new_map,0.5,0.5)
              self.map_overlay = self.merge_maps(self.map_overlay, rotated_new_map)
              if frame_num >= self.stop_at_frame:
                map_ovrly = self.merge_maps(self.map_overlay, rotated_new_map_disp)
                # cv2.line(map_ovrly, (map_line[0], map_line[1]), (map_line[2], map_line[3]), (0,255,0), 3, cv2.LINE_AA)
                cv2.imshow("map overlay",map_ovrly)
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
            self.map_rows,self.map_cols,self.map_ch = self.map.shape
            self.map_KP = Keypoints(self.map)
            self.map_arr = asarray(self.map)

          return
