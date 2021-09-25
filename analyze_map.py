# import the necessary packages
import numpy as np
from numpy import asarray
from PIL import Image
import cv2
import argparse
from analyze_keypoints import *
from analyze_color import *
from analyze_lines import *
from imutils import paths
import imutils
from matplotlib import pyplot as plt
from cv_analysis_tools import *
from shapely.geometry import *
from utilradians import *
from utilborders import *
import statistics 
from operator import itemgetter, attrgetter
from skimage.metrics import structural_similarity as ssim  
import random

# based on: 
# https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
class AnalyzeMap():
    def __init__(self):
        # self.stop_at_frame = 114
        self.curr_frame_num = 0
        self.stop_at_frame = 119
        # Analysis Objects
        self.KPs = None
        self.map = None
        # real map array info
        # real map array sizes change over time
        self.map_overlay = None
        self.map_height = None
        self.map_width= None
        #
        self.border_buffer = None   # move_img
        self.border_multiplier = 2   # move_img
        # To compute the distances,var of moves
        self.curr_move_height = None
        self.curr_move_width = None
        self.curr_move = None
        self.prev_move = None

        # the robot starts in the middle of an empty square of VIRTUAL_MAP_SIZE pixels
        # the virtual map locations do not change.
        self.virtual_map_center = None
        self.robot_length = None
        self.parked_gripper_left = []
        self.parked_gripper_right = []
        self.gripper_height = None
        self.gripper_width = None
        self.grabable_objects = []
        self.container_objects = []

        self.frame_num = 0
        self.color_quant_num_clust = 0
        self.cvu = CVAnalysisTools()
        self.cfg = Config()
        self.robot_orientation = 0  # starting point in radians
        self.robot_location = [0,0]  # starting point 
        print("1orientation:", self.robot_orientation)
        self.lna = LineAnalysis()

    def real_to_virtual_map_coordinates(self, pt):
        self.virtual_map_center = None
        x = pt[0] - self.virtual_map_center[0] + self.cfg.VIRTUAL_MAP_SIZE/2
        y = pt[1] - self.virtual_map_center[1] + self.cfg.VIRTUAL_MAP_SIZE/2
        return x,y

    def virtual_to_real_map_coordinates(self, pts):
        x = pt[0] + self.virtual_map_center[0] - self.cfg.VIRTUAL_MAP_SIZE/2
        y = pt[1] + self.virtual_map_center[1] - self.cfg.VIRTUAL_MAP_SIZE/2
        return x,y

    ######################
    # Transforms
    ######################
    def get_birds_eye_view(self, image):
        w = image.shape[0]
        h = image.shape[1]
        new_w = int(48 * w / 32)
        bdr   = int((new_w - w) / 2)
        print("new_w, w, bdr:", 55*w/32, w, (new_w - w)/2)
        image = cv2.copyMakeBorder( image, top=0, bottom=0, left=bdr, right=bdr,
            borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])

        src = np.float32([[0, h], [new_w, h], [bdr, 0], [w, 0]])
        # need to modify new_w
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

    #########################
    # Util Func
    #########################
    def move_state(self, action, new_above_view_img):
        if self.curr_move is not None:
          self.prev_move = self.curr_move.copy()
        self.curr_move = new_above_view_img.copy()

    def show_rot(self, new_map):
      pt = [int(self.robot_location[1]), int(self.robot_location[0])] # circle uses [w,h]
      new_map = cv2.circle(new_map,pt,3,(255,0,0),-1)
      pt = [int(new_map.shape[1]/2), int(new_map.shape[0]/2)]
      new_map = cv2.circle(new_map,pt,3,(255,0,0),-1)
      for angle in range(0,np.pi/6, 0.1):
        new_map_rot = rotate_about_robot(new_map, angle, self.robot_location)
        if self.frame_num >= self.stop_at_frame:
          cv2.imshow(str(angle), new_map_rot)
          cv2.waitKey(0)

    #############################
    # Lines
    # Pre-rotation Input1: prev_new_map, curr_new_map
    # Post-rotation Input2: Map, Rotated_new_map
    #############################
    def analyze_move_by_line(self, prev_map_frame, curr_map_frame, action, frame_num=0):
        pm = prev_map_frame.copy()
        cm = curr_map_frame.copy()
        # Compute Color Quant
        pos = self.lna.best_line_pos(pm, cm, action, self.robot_location, frame_num=0)
        if action == "LEFT":
          angle = pos
          return angle
        elif action == "RIGHT":
          # angle computed with rad_interval to find slope diff, always pos
          angle = -pos
          print("RIGHT line angle:", angle)
          return angle    
          # return angle  
        elif action == "FORWARD":
          pixels_moved = pos
          return pixels_moved
        elif action == "REVERSE":
          pixels_moved = pos
          return -pixels_moved
        else:
          print("expected L/R/Fwd/Rev")
          return None

    #########################
    # KeyPoints
    # Input: Map, Intersected Rotated New Map, Phase 
    # Handles SIFT, ORB, FEATURES using the same algorithm.
    #########################
    def score_keypoints(self, map, rotated_new_map):
        mol = new_map.copy()
        rnm = rotated_new_map.copy()
        kp_results["score"] = kp_score
        kp_results["recommendation"] = None
        return kpe_score

    def analyze_move_by_kp(self, prev_map_frame, curr_map_frame, action, frame_num, kp_mode="BEST"):
        # look for prev/curr KPs that are same distance to robot_location
        # find angles between these KPs.
        # find same angles. 
        pm = prev_map_frame.copy()
        cm = curr_map_frame.copy()
        print("pm,cm shape", (pm.shape), (cm.shape), self.robot_location, self.robot_location_no_border)
        if kp_mode == "BEST":
          kp_mode_list = ["SIFT", "ORB", "FEATURE"]
        else:
          kp_mode_list = [kp_mode]
        for mode in kp_mode_list:
          pm_kp = Keypoints(pm,kp_mode=mode)
          cm_kp = Keypoints(cm,kp_mode=mode)
          move = pm_kp.best_keypoint_move(cm_kp, action, self.robot_location_no_border, frame_num=0)
          if move is None:
            continue
        if move is None:
          return None
        if action == "LEFT":
          angle = move
          return angle
        elif action == "RIGHT":
          angle = move
          print("RIGHT kp angle:", angle)
          return angle  # already negative
        elif action == "FORWARD":
          offset_h = move
          return offset_h 
        elif action == "REVERSE":
          offset_h = move
          return offset_h 
        else:
          print("expected L/R/Fwd/Rev")
          return None


    #########################
    # Metrics: mse, lbp, ssim
    # Input: Map, Intersected Rotated New Map, Phase (ignored)
    # Output: Score for current New Map. No estimated Rotation or Offsets
    #########################
    def score_metrics(self, action, map, rotated_new_map, show_intersect=False):
        mol = map.copy()
        rnm = rotated_new_map.copy()
        gray_mol = cv2.cvtColor(mol, cv2.COLOR_BGR2GRAY)
        gray_rnm = cv2.cvtColor(rnm, cv2.COLOR_BGR2GRAY)
        shape, mol_border = real_map_border(mol)
        shape, rnm_border = real_map_border(rnm)
        mol_maxh, mol_minh, mol_maxw, mol_minw = get_min_max_borders(mol_border)
        rnm_maxh, rnm_minh, rnm_maxw, rnm_minw = get_min_max_borders(rnm_border)
        pct_w = 4 * abs(rnm_maxw - mol_maxw) / rnm_maxw
        pct_h = 4 * abs(rnm_maxh - mol_maxh) / rnm_maxh
        # print("mol_border:", mol_border)
        # print("rnm_border:", rnm_border)
        intersect_border = intersect_borders(mol_border, rnm_border)
        if len(intersect_border) == 0:
          cv2.imshow("nonintersecting mol", mol)
          cv2.imshow("nonintersecting rnm", rnm)
          cv2.waitKey(0)
        
        intersect_mol = image_in_border(intersect_border, gray_mol)
        intersect_rnm = image_in_border(intersect_border, gray_rnm)
        if intersect_mol is None or intersect_rnm is None:
          return self.cfg.INFINITE, self.cfg.INFINITE, self.cfg.INFINITE
        if action == "LEFT" or action == "RIGHT":
          try:
            shape, intersect_mol_border = real_map_border(intersect_mol)
            shape, intersect_rnm_border = real_map_border(intersect_rnm)
            # the intersect borders should be the same before cropping
            # wants a linear ring
            mol_maxh, mol_minh, mol_maxw, mol_minw = get_min_max_borders(intersect_mol_border)
            rnm_maxh, rnm_minh, rnm_maxw, rnm_minw = get_min_max_borders(intersect_rnm_border)
            print("intersect min/max", mol_maxh, mol_minh, mol_maxw, mol_minw)
            # TODO: base % on full mol_border (not intersected border)
            # remove ~20% of left; ~5% of top for RNM 
            print("pct w h:",pct_w, pct_h)
            rnm_minw = round(rnm_minw + pct_w*(rnm_maxw-rnm_minw))
            rnm_minh = round(rnm_minh + pct_h*(rnm_maxh-rnm_minh))
            # border in format of a linear ring
            intersect_rnm_border = [[[rnm_maxh,rnm_minw]],[[rnm_maxh, rnm_maxw]], [[rnm_minh, rnm_maxw]], [[rnm_minh, rnm_minw]], [[rnm_maxh,rnm_minw]]]
            # remove ~20% of right; ~5% of bottom for MOL 
            mol_maxw = round(mol_maxw - pct_w*(mol_maxw-mol_minw))
            mol_maxh = round(mol_maxh - pct_h*(mol_maxh-mol_minh))
            # border in format of a linear ring
            intersect_mol_border = [[[mol_maxh,mol_minw]],[[mol_maxh, mol_maxw]], [[mol_minh, mol_maxw]], [[mol_minh, mol_minw]], [[mol_maxh,mol_minw]]]
            # print("mol_border", mol_border)
            # print("rnm_border", intersect_rnm_border)
            # print("intersect_mol h,w", intersect_mol.shape[0], intersect_mol.shape[1])
            # print("intersect_rnm h,w", intersect_rnm.shape[0], intersect_rnm.shape[1])
  
            intersect_mol = image_in_border(intersect_mol_border, intersect_mol)
            intersect_rnm = image_in_border(intersect_rnm_border, intersect_rnm)
            # print("final intersect_mol h,w", intersect_mol.shape[0], intersect_mol.shape[1])
          except:
           intersect_mol = None 
           intersect_rnm = None

        if intersect_mol is None or intersect_rnm is None:
          return self.cfg.INFINITE, self.cfg.INFINITE, self.cfg.INFINITE
        radius = min(intersect_mol.shape[0], intersect_mol.shape[1])
        num_radius = max(int((radius-1) / 6), 2)
        mse_score,ssim_score = self.compare_images(intersect_mol, intersect_rnm, "compare")
        i, tot_lbp_score = 0, 0
        if show_intersect:
          cv2.imshow("mol", intersect_mol)
          cv2.imshow("rnm", intersect_rnm)
          cv2.imshow("full mol", mol)
          cv2.imshow("full rnm", rnm)
          cv2.waitKey(0)
        # print("radius", radius, num_radius)
        for rad in range(1,(radius-1), num_radius):
          lbp = LocalBinaryPattern(intersect_mol, rad)
          lbp_score = lbp.get_score(intersect_rnm)
          tot_lbp_score += lbp_score
          i += 1
          # print("lbp: ", i, lbp_score, rad, radius)
          lbp_score = tot_lbp_score / (i)
        # return mse_results, ssim_results, lbp_results
        return mse_score, ssim_score, lbp_score

    ##############
    # find_lmse()
    ##############
    def find_lmse(self, action, mode, prev_frame, curr_frame, start_pos, end_pos, frame_num=0, start_N=0):
        N = 17
        dbg_compare = False
        best_mse_score, best_ssim_score = self.cfg.INFINITE, self.cfg.INFINITE
        best_lbp_score, best_metric_score = self.cfg.INFINITE, self.cfg.INFINITE
        best_mse_pos, best_ssim_pos, best_lbp_pos = None, None, None
        best_metric_pos, best_score, best_pos     = None, None, None
        pos_lower, pos_upper = None, None
        mse_pos, ssim_pos   = 0, 0
        lbp_pos, metric_pos = 0, 0
        mse, ssim, lbp, metric, pos = [], [], [], [], []
        prev_move = add_border(prev_frame.copy(), self.border_buffer)
        for k in range(start_N,N):
          k_interval = rad_interval(end_pos,start_pos)*k*.125
          if k_interval <= self.cfg.RADIAN_THRESH:
            k_interval = self.cfg.RADIAN_THRESH
          # measure input
          if k < 7:  
            # gather some data to get basic MSE curve
            # from 7 interior points followed by the two endpoints 
            mse_pos = rad_sum2(action, start_pos, k_interval)
            print(action, "mse_pos:", mse_pos, start_pos, end_pos)
          elif k == 7:  
              mse_pos = start_pos
          elif k == 8:  
              mse_pos = rad_dif(end_pos, 0)
          elif k < 16:  
            # drill down to 7 finer resolution values within the min range
            if pos_lower is None:
              print("mse,pos", len(mse), len(pos))
              mse_z = np.polyfit(mse, pos, 2)
              mse_f = np.poly1d(mse_z)
              # predict new value
              prev_mse_score  = mse_score
              prev_ssim_score = ssim_score
              prev_lbp_score  = lbp_score
  
              # get local minima for MSE
              mse_crit = mse_f.deriv().r
              mse_r_crit = mse_crit[mse_crit.imag==0].real
              mse_test = mse_f.deriv(2)(mse_r_crit)
              mse_x_min = mse_r_crit[mse_test>0]
              mse_y_min = mse_f(mse_x_min)
              print("mse_x_min, mse_y_min:", mse_x_min, mse_y_min)

              # Hopefully, stats can narrow this down in the future
              # [0.18868234 0.04772681]
              if len(mse_x_min )== 0:
                mse_y_min = min(mse) 
                min_index = mse.index(mse_y_min)
                mse_x_min = pos[min_index]
                if min_index == 0:
                  increment = rad_interval(end_pos,start_pos)*.125/2
                  pos_lower = start_pos
                  pos_upper = rad_sum2(action, pos_lower, increment)
                  print(action, "pos_low/up3:", pos_lower, pos_upper)
                elif min_index == len(mse)-1:
                  increment = rad_interval(end_pos,start_pos)*.125/2
                  pos_upper = rad_dif2(action, pos_lower, increment)
                  print(action, "pos_low/up1:", pos_lower, pos_upper)
                elif mode == "ROTATE":
                  increment = rad_interval(end_pos,start_pos)*.125/2
                  pos_lower = rad_dif2(action, mse_x_min, increment)
                  pos_upper = rad_sum2(action, mse_x_min, increment)
                  print(action, "pos_low/up2:", pos_lower, pos_upper)
                else:
                  increment = (end_pos-start_pos)*.125/2
                  pos_lower = mse_x_min - increment/2
                  pos_upper = mse_x_min + increment/2
              else:
                for x in range(8):
                  if mode == "ROTATE":
                    pos_lower = rad_sum2(action, start_pos, rad_interval(end_pos,start_pos)*x*.125)
                    pos_upper = rad_sum2(action, start_pos, rad_interval(end_pos,start_pos)*(x+1)*.125)
                    print(action, "pos lower, upper, mse_x_min:", pos_lower, pos_upper, mse_x_min)
                    if len(mse_x_min) > 1:
                      increment = rad_interval(end_pos,start_pos)*.125
                      mse_pos = rad_sum2(action, rad_sum2(start_pos, increment/2), increment*x)
                    elif pos_lower < mse_x_min and mse_x_min < pos_upper:
                      break
                    elif pos_lower == mse_x_min:
                      mid_range = rad_interval(pos_upper,pos_lower)*x*.125
                      pos_upper = rad_sum2(action, pos_lower, mid_range)
                      pos_lower = rad_dif2(action, pos_lower, mid_range)
                      break
                    elif mse_x_min == pos_upper:
                      mid_range = rad_interval(pos_lower,pos_upper)*x*.125
                      pos_upper = rad_sum2(action, pos_lower, mid_range)
                      pos_lower = rad_dif2(action, pos_lower, mid_range)
                      if pos_upper > end_pos and action == "LEFT":
                          pos_upper = end_pos
                      elif pos_upper < end_pos and action == "RIGHT":
                          pos_upper = end_pos
                      print(action, "pos_low/up5:", pos_lower, pos_upper)
                      break
                  elif mode == "OFFSET_H" or mode == "OFFSET_W":
                    pos_lower = start_pos + (end_pos-start_pos)*x*.125
                    pos_upper = start_pos + (end_pos-start_pos)*(x+1)*.125
                    print("pos lower, upper, mse_x_min:", pos, pos, mse_x_min)
                    if len(mse_x_min) > 1:
                      increment = (end_pos-start_pos)*.125
                      mse_pos = start_pos + increment/2 + increment*x
                    elif pos_lower < mse_x_min and mse_x_min < pos_upper:
                      break
                    elif pos_lower == mse_x_min:
                      mid_range = (pos_upper-pos_upper) * 0.125 / 2
                      pos_upper = pos_lower + mid_range
                      pos_lower = pos_lower - mid_range
                      break
                    elif mse_x_min == pos_upper:
                      mid_range = (pos_upper-pos_lower) * 0.125 / 2
                      pos_lower = pos_upper - pos_range
                      pos_upper = pos_upper + pos_range
                      if pos_upper > end_pos:
                        pos_upper = end_pos
                      break

            # now compute the finer resolution pos values 
            x = 17 - k
            if mode == "ROTATE":
              mse_pos = rad_sum2(action, pos_lower, rad_interval(pos_upper,pos_lower)*x*.125)
              print(action, "low, dif:", mse_pos, rad_interval(pos_upper,pos_lower))
            elif mode == "OFFSET_H" or mode == "OFFSET_W":
              mse_pos = pos_lower + (pos_upper-pos_lower)*x*.125

            print(k, "position:", mse_pos)
          else:
            # fit a curve to MSE
            # Note about other metrics:
            # - ssim is correlated to MSE
            # - lbp fluctuates a lot as the curve is moved. It is a measure
            #   of image comparison, but not a good fit for this MSE algorithm.
            print("mse polyfit", len(mse))
            print("pos polyfit", len(pos))
            mse_z = np.polyfit(mse, pos, 2)
            mse_f = np.poly1d(mse_z)                            
            # predict new value
            prev_mse_score = mse_score
            prev_ssim_score = ssim_score
            prev_lbp_score = lbp_score

            # get local minima for MSE
            mse_crit = mse_f.deriv().r
            mse_r_crit = mse_crit[mse_crit.imag==0].real
            mse_test = mse_f.deriv(2)(mse_r_crit) 
            mse_x_min = mse_r_crit[mse_test>0]
            mse_y_min = mse_f(mse_x_min)
            print("mse_x_min, mse_y_min:", mse_x_min, mse_y_min)
            if len(mse_x_min) == 1:
              mse_pos = mse_x_min[0]
            elif len(mse_x_min) == 0:
              # probably an end point
              mse_y_min = min(mse) 
              min_index = mse.index(mse_y_min)
              mse_pos = pos[min_index]
            print("mse polyfit", len(mse))
            print("pos polyfit", len(pos))
            if rad_interval(mse_pos, 0) > rad_interval(start_pos, end_pos):
              print("predicted mse out of range.")
              mse_pos = end_pos

          # compute rotation and new mse
          curr_move = curr_frame.copy()
          curr_move = add_border(curr_move, self.border_buffer)
          if mode == "ROTATE":
            curr_move = rotate_about_robot(curr_move, mse_pos, self.robot_location)
          elif mode == "OFFSET_H":
            curr_move = replace_border(curr_move, curr_move.shape[0], curr_move.shape[1], self.robot_location[0]+mse_pos, self.robot_location[1])
          elif mode == "OFFSET_W":
            curr_move = replace_border(curr_move, curr_move.shape[0], curr_move.shape[1], self.robot_location[0], self.robot_location[1] + mse_pos)

          # measure output
          mse_score, ssim_score, lbp_score = self.score_metrics(action, prev_move, curr_move, dbg_compare)
          metric_score = ((mse_score)*(ssim_score)*(lbp_score))
          print("mse_pos", mse_pos)
          pos.append(mse_pos)
          mse.append(mse_score)
          ssim.append(ssim_score)
          lbp.append(lbp_score)
          metric.append(metric_score)
          if mse_score < best_mse_score:
            best_mse_score = mse_score 
            best_mse_pos = mse_pos 
          if ssim_score < best_ssim_score:
            best_ssim_score = ssim_score 
            best_ssim_pos = mse_pos 
          if lbp_score < best_lbp_score:
            best_lbp_score = lbp_score 
            best_lbp_pos = mse_pos 
          if metric_score < best_metric_score:
            best_metric_score = metric_score 
            best_metric_pos = mse_pos 

          # todo: store frame, pos, metric scores in history for future analysis
          print("pos, score1:", (best_mse_pos, best_mse_score), (best_ssim_pos, best_ssim_score)) 
          print("pos, score2:", (best_lbp_pos, best_lbp_score), (best_metric_pos, best_metric_score))
          print("pos, score3:", (mse_pos, mse_score, ssim_score, lbp_score))
        if self.frame_num >= self.stop_at_frame:
          ### show results
          plt.figure(figsize=(15,9))
          plt.subplot(211);plt.title("Curve fit");plt.xlabel("samples - k")
          mse_plot = []
          ssim_plot = []
          lbp_plot = []
          for i,j in enumerate(mse):
            mse_plot.append(mse[i])
            ssim_plot.append(ssim[i]/self.cfg.SSIM_THRESH)
            lbp_plot.append(lbp[i]/self.cfg.LBP_THRESH)
          plt.plot((mse_plot),"b", label="mse")
          plt.plot((ssim_plot),"r", label="ssim")
          # plt.plot((lbp_plot),"g", label="lbp")
          plt.plot(pos,"c", label="pos");plt.legend()
          plt.plot(metric,"y", label="metric");plt.legend()
          plt.legend(); plt.tight_layout(); plt.show()
          cv2.waitKey(0)
        if best_score is None:
          # current algorithm aimed at best mse
          best_score = best_mse_score
          best_pos = best_mse_pos
          
        print("Final LMS pos, score:", best_pos, best_score)
        return best_pos 

    #########################
    # Hardcoding / Manually tuning / gathering data to figure out algorithm
    # to lower NewMap, increase off_h  
    # to move NewMap right, increase off_w
    def manual_tuning(self, frame_num):
        [rot, off_h, off_w] = [0, 0, 0]
        if frame_num == 112:
          [rot, off_h, off_w] = [0, 0, 0] # best
        if frame_num == 113:
          # slope based
          # [off_h, off_w] = [-20, -5]   Too high, too right
          # [off_h, off_w] = [-5, -20]   OK height, too right
          [rot, off_h, off_w] = [6.5, -5, -70]  
          # equation based
          [rot, off_h, off_w] = [.0713, -5, -70]  
          [rot, off_h, off_w] = [.11, 0, 0]  
        if frame_num == 114:
          # slope based
          # [off_h, off_w] = [3, -60]  Too far right
          [rot, off_h, off_w] = [13, -5, -130]  # best so far
          # equation based
          [rot, off_h, off_w] = [.1426, -9, -69]  
          [rot, off_h, off_w] = [.1426, -9, -140]  
          [rot, off_h, off_w] = [.22, 0, 0]  
        if frame_num == 115:
          # can focus on the cube too
          # slope based
          # [off_h, off_w] = [-5, -200] # too far left
          # [off_h, off_w] = [0, -180] # close
          [rot, off_h, off_w] = [19.5, -1, -185] # a bit blury
          # equation based
          [rot, off_h, off_w] = [.2139, -15, -68]  
          [rot, off_h, off_w] = [.2139, -24, -208]  
          [rot, off_h, off_w] = [.33, 0, 0]  
        if frame_num == 116:
          # slope based
          # [off_h, off_w] = [-5, -250] # too high, too left
          # [off_h, off_w] = [5, -200] # too high, too right
          #[off_h, off_w] = [15, -220] # much closer
          [rot, off_h, off_w] = [26, 15, -215] # much closer
          # equation based
          [rot, off_h, off_w] = [.2852, -20, -68] # much closer
          [rot, off_h, off_w] = [.2852, -40, -282] # much closer
          [rot, off_h, off_w] = [.44, 0, 0]  

        print(frame_num, "manual_tuning: ", rot, off_h, off_w)
        return rot, off_h, off_w

    #########################
    def mean_sq_err(self, imageA, imageB):
        # the 'Mean Squared Error' between the two images is the
        # sum of the squared difference between the two images;
        # NOTE: the two images must have the same dimension
        # err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        print("np.sum diff sqr", err)
        # Normalized Least Squared Error
        err /= np.sum((imageA.astype("float")) ** 2)
        # Least Squared Error
        # err /= float(imageA.shape[0] * imageA.shape[1])
        print("mse shape", float(imageA.shape[0] * imageA.shape[1]), np.sum((imageA.astype("float")) ** 2))

        # return the MSE, the lower the error, the more "similar"
        # the two images are
        return err

    def compare_images(self, imageA, imageB, title=None):
        # compute the mean squared error and structural similarity
        # index for the images
        m = self.mean_sq_err(imageA, imageB)
        try:
          s = ssim(imageA, imageB)
          s = 1 - s   # make "lower is better"
        except:
          s = None
        if False and self.frame_num >= self.stop_at_frame:
          # setup the figure
          fig = plt.figure(title)
          plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
          # show first image
          ax = fig.add_subplot(1, 2, 1)
          plt.imshow(imageA, cmap = plt.cm.gray)
          plt.axis("off")
          # show the second image
          ax = fig.add_subplot(1, 2, 2)
          plt.imshow(imageB, cmap = plt.cm.gray)
          plt.axis("off")
          # show the images
          plt.show()
        return m, s

     
    def find_best_rotation(self, action, prev_frame, curr_frame, frame_num=0):
        #####################################################################
        # IMAGE ROTATION AROUND ROBOT LOCATION
        # Line Analysis, Keypoint Analysis, Least Metric Squared Error
        #
        # All we need is to know the angle of rotation to merge the maps.
        # Maps don't have to be precise. A 50% rating is given to the most
        # recently evaluated image. 
        #
        # In general, a strong line like a table border is very accurate
        # and quick.
        #
        # If there's sufficient Keypoints, the computation is quick and
        # reasonable accurate.
        #
        # LMSE requires somewhat expensive iterations to compute a reasonabl3
        # result.
        #####################################################################
        # line detection and keypoint detection includes black borders
        # KP, Lines, features can handle full images with borders
        # Metrics require rectangular intersected images with no borders
        # KP, features can get previous and current unrotated maps
        #   and match them using optical flow.  Then rotate.
        ####################################################################
        # initialize images and map for new move
        new_angle_ln = None
        new_angle_kp = None
        new_angle_lms = None
        new_angle = None
        best_angle = None
        best_mse = None
        height,width,ch = self.curr_move.shape
        # for mode in ["SIFT", "FEATURE", "ROOTSIFT", "ORB", "SURF"]:
        # for mode in ["SIFT", "FEATURE", "ROOTSIFT", "ORB"]:
        metrics = {"LINE":{"MSE":None},"KP":{"MSE":None},"LMS":{"MSE":None}}
        for mode in ["LINE", "KP", "LMS"]:
          ##################################################################
          if mode == "LINE":
            new_angle_ln = self.analyze_move_by_line(self.prev_move, self.curr_move, action, frame_num=0)
            print("new ln angle: ", new_angle_ln)
            print("##########################")
            new_angle = new_angle_ln
          ##################################################################
          elif mode == "KP":
            new_angle_kp = self.analyze_move_by_kp(self.prev_move, self.curr_move, action, frame_num=0, kp_mode="BEST")
            print("new kp angle: ", mode, new_angle_kp)
            print("##########################")
            new_angle = new_angle_kp
          ##################################################################
          elif mode == "LMS":
            # start_pos is a local prev/cur comparison, not based on
            # global positioning on map.
            if action == "LEFT":
              start_pos = 0
              end_pos= np.pi/16
            elif action == "RIGHT":
              start_pos = 0
              end_pos= -np.pi/16
            start_N = 0
            if (metrics["LINE"]["MSE"] is not None and
                metrics["KP"]["MSE"] is not None):
              interval = rad_interval(end_pos,start_pos)*.125 
              if rad_interval(new_angle_kp, new_angle_ln) < self.cfg.RADIAN_THRESH:
                # No need to run LMS; LINE and KP agree.
                print("LINE and KP agree on angle:", new_angle_ln, new_angle_kp)
                break
              if rad_interval(new_angle_kp, new_angle_ln) < interval:
                mid_pos = rad_sum2(action, new_angle_kp, new_angle_ln)/2
                start_pos = rad_dif2(action, mid_pos, interval)
                end_pos = rad_sum2(action, mid_pos, interval)
                print("MSE search near LINE and KP angles only:", new_angle_ln, new_angle_kp)
                start_N = 8
            new_angle_lms = self.find_lmse(action, "ROTATE", self.prev_move,
                          self.curr_move, start_pos=start_pos,
                          end_pos=end_pos,
                          frame_num=frame_num, start_N=0)
            # new angle should never be outside of the start/end range
            if (rad_interval(end_pos, start_pos) <= rad_interval(new_angle_lms, 0)):
              print("Error: angle out of range:", end_pos, start_pos, new_angle_lms)

            new_angle = new_angle_lms
            print("new mse angle: ", mode, new_angle)
            print("##########################")
          ##################################################################
          elif mode == "MANUAL":
            new_angle, off_h, off_w = self.manual_tuning(self.frame_num)
          else:
            continue
          ##################################################################
          if new_angle is None:
            continue
          # cv2.imshow("curr_move:", self.curr_move)
          new_map = add_border(self.curr_move.copy(), self.border_buffer)
          # cv2.imshow("orig new_map:", new_map)
          rotated_new_map = new_map.copy()  # for resilience against bugs
          print("robot location", self.robot_location, new_angle, self.robot_orientation)
          pt2 = [int(self.robot_location[1]), int(self.robot_location[0])]
          pt2[1] += self.cfg.MAP_ROBOT_H_POSE
          rot_angle = rad_sum(self.robot_orientation, new_angle)
          print("2orientation:", self.robot_orientation)
          rotated_new_map = rotate_about_robot(rotated_new_map, rot_angle, pt2)
          metrics[mode]["MSE"], metrics[mode]["SSIM"], metrics[mode]["LBP"] = self.score_metrics(action, self.map, rotated_new_map)
          if best_mse is None or metrics[mode]["MSE"] < best_mse:
            best_mse = metrics[mode]["MSE"]
            best_angle = new_angle
          print(mode, "mse, ssim, lbp metrics: ",
            metrics[mode]["MSE"], metrics[mode]["SSIM"], metrics[mode]["LBP"] )

        if action in ["LEFT", "RIGHT"]:
          self.robot_orientation = rad_sum2(action, self.robot_orientation, best_angle)
          print("3orientation:", self.robot_orientation)
          self.map = self.merge_maps(self.map, rotated_new_map)
       
        return 

    def analyze_fwd_rev(self, action, prev_frame, curr_frame, frame_num=0):
          #####################################################################
          # Line Analysis, Keypoint Analysis, Least Metric Squared Error
          #
          # In general, a strong line like a table border is very accurate
          # and quick.
          #
          # If there's sufficient Keypoints, the computation is quick and
          # reasonable accurate.
          #
          # LMSE requires somewhat expensive iterations to compute a reasonabl3
          # result.
          #####################################################################
          # initialize images and map for new move
          height,width,ch = self.curr_move.shape
          metrics = {"LINE":{},"KP":{},"LMS":{}}
          for mode in ["LINE", "KP", "LMS"]:
            ##################################################################
            if mode == "LINE":
              pixels_moved_ln = self.analyze_move_by_line(self.prev_move, self.curr_move, action, frame_num=0)
              print("pixels moved by line: ", pixels_moved_ln)
              print("##########################")
              pixels_moved = pixels_moved_ln
            ##################################################################
            elif mode == "KP":
              pixels_moved_kp = self.analyze_move_by_kp(self.prev_move, self.curr_move, action, frame_num=0, kp_mode="BEST")
              print("pixels_moved_kp: ", pixels_moved_kp)
              print("##########################")
              pixels_moved = pixels_moved_kp
            ##################################################################
            elif mode == "LMS":
              # start_pos is a local prev/cur comparison, not based on
              # global positioning on map.
              if action == "FORWARD":
                start_pos = self.curr_move.shape[0]
                end_pos = 0
              elif action == "REVERSE":
                start_pos = self.curr_move.shape[0]
                end_pos = 0
              pixels_moved_mse = self.find_lmse(action, "OFFSET_H", 
                            self.prev_move, self.curr_move, start_pos=start_pos,
                            end_pos=end_pos, frame_num=frame_num)
              pixels_moved = pixels_moved_mse 
              print("pixels_moved_mse: ", pixels_moved_mse)
              print("##########################")
            ##################################################################
            elif mode == "MANUAL":
              new_angle, off_h, off_w = self.manual_tuning(self.frame_num)
              pixels_moved = off_h

            ##################################################################
            # cv2.imshow("curr_move:", self.curr_move)
            new_map = add_border(self.curr_move.copy(), self.border_buffer)
            if pixels_moved is not None:
              angle = self.robot_orientation
              print("4orientation:", self.robot_orientation)
              [pix_h, pix_w] = [round(pixels_moved * math.cos(angle)), round(pixels_moved * math.sin(angle))]
              self.robot_location[0] += round(pix_h)
              self.robot_location[1] += round(pix_w)
              self.robot_location_no_border[0] += round(pix_h)
              self.robot_location_no_border[1] += round(pix_w)
              new_map = replace_border(new_map, new_map.shape[0], new_map.shape[1], self.robot_location[0], self.robot_location[1])

            # cv2.imshow("orig new_map:", new_map)
            pt2 = [int(self.robot_location[1]), int(self.robot_location[0])]
            pt2[1] += self.cfg.MAP_ROBOT_H_POSE
            print("2orientation:", self.robot_orientation)
            # rotate to global map position
            new_map = rotate_about_robot(new_map, self.robot_orientation, pt2)
            metrics[mode]["MSE"], metrics[mode]["SSIM"], metrics[mode]["LBP"] = self.score_metrics(action, self.map, new_map)
            print(mode, "mse, ssim, lbp metrics: ",
              metrics[mode]["MSE"], metrics[mode]["SSIM"], metrics[mode]["LBP"] )
            if (# metrics[mode]["MSE"] <= self.cfg.MSE_THRESH and
                metrics[mode]["SSIM"] <= self.cfg.SSIM_THRESH and
                metrics[mode]["LBP"]  <= self.cfg.LBP_THRESH):
              break

          self.map = self.merge_maps(self.map, new_map)
          pt2 = [int(self.robot_location[1]), int(self.robot_location[0])]
          self.map_overlay = self.map.copy()
          # draw center of robot on map_overlay
          self.map_overlay = cv2.circle(self.map_overlay,pt2,3,(255,0,0),-1)
          # draw rotated robot on map_overlay
          rot_rect = (pt2, self.cfg.MAP_ROBOT_SHAPE, radians_to_degrees(-self.robot_orientation))
          print("5orientation:", self.robot_orientation)
          box = cv2.boxPoints(rot_rect) 
          box = np.int0(box)
          cv2.drawContours(self.map_overlay,[box],0,(0,0,255),2)

          rotated_robot_rectangle = cv2.circle(rotated_new_map,pt2,3,(255,0,0),-1)
          if action in ["FORWARD", "REVERSE"]:
            self.robot_position
            self.map = self.merge_maps(self.map, new_map)

    #################################
    # Map merging
    #################################
    def merge_maps(self, map, new_map):
        map_shape, map_border = real_map_border(map)
        new_map_shape, new_map_border = real_map_border(new_map)
        # find overlap
        # map_maxh, map_minh, map_maxw, map_minw = get_min_max_borders(map_border)
        # new_map_maxh, new_map_minh, new_map_maxw, new_map_minw = get_min_max_borders(new_map_border)
        map_maxw, map_minw, map_maxh, map_minh = get_min_max_borders(map_border)
        new_map_maxw, new_map_minw, new_map_maxh, new_map_minh = get_min_max_borders(new_map_border)
        print("new_map/map minw:", new_map_minw, map_minw)
        print("new_map/map maxw:", new_map_maxw, map_maxw)
        print("new_map/map minh:", new_map_minh, map_minh)
        print("new_map/map maxh:", new_map_maxh, map_maxh)
        print("new_map/map shape:", new_map.shape, map.shape)
        final_map = map.copy()
        buffer = 3  # eliminate the black border at the merge points
        for h in range(new_map_minh-1, new_map_maxh-1):
          for w in range(new_map_minw-1, new_map_maxw-1):
            if final_map[h,w].all() == 0:
              final_map[h,w] = new_map[h,w]
            elif new_map[h,w].all() == 0:
              pass
            elif (final_map[h-buffer,w].all() == 0 or
                  final_map[h+buffer,w].all() == 0 or
                  final_map[h, w+buffer].all() == 0 or
                  final_map[h, w-buffer].all() == 0):
              final_map[h,w] = new_map[h,w]
            elif (new_map[h-buffer,w].all() == 0 or
                  new_map[h+buffer,w].all() == 0 or
                  new_map[h, w+buffer].all() == 0 or
                  new_map[h, w-buffer].all() == 0):
              # avoid adding border around new_map
              pass
            else:
              final_map[h,w] = .5*final_map[h,w] + .5*new_map[h,w]
        return final_map

    def add_robot_to_overlay(self):
          # make overlay map and add robot in current position.
          # in future, can add position of each move leading to this point

          pt2 = [int(self.robot_location[1]), int(self.robot_location[0])]
          # draw center of robot on map_overlay
          self.map_overlay = cv2.circle(self.map_overlay,pt2,3,(255,0,0),-1)
          # draw rotated robot on map_overlay
          rot_rect = (pt2, self.cfg.MAP_ROBOT_SHAPE, radians_to_degrees(-self.robot_orientation))
          print("6orientation:", self.robot_orientation)
          box = cv2.boxPoints(rot_rect)
          box = np.int0(box)
          cv2.drawContours(self.map_overlay,[box],0,(0,0,255),2)

          rotated_robot_rectangle = cv2.circle(self.map_overlay,pt2,3,(255,0,0),-1)
          cv2.imshow("map_overlay:", self.map_overlay)
          if self.frame_num >= 119:
            cv2.waitKey(0)

    #################################
    # CREATE MAP - main map driver routine
    #################################

    def analyze(self, frame_num, action, prev_img_pth, curr_img_pth, done):
        self.frame_num = frame_num
        curr_image = cv2.imread(curr_img_pth)
        curr_image_KP = Keypoints(curr_image)
        bird_eye_vw = self.get_birds_eye_view(curr_image)
        rotated_new_map = self.curr_move
        self.move_state(action, bird_eye_vw)
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
          self.map = add_border(self.curr_move, self.border_buffer)
          self.map_overlay = self.map.copy()
          self.map_height,self.map_width = self.map.shape[:2]

          # work with AnalyzeMove to store robot location
          self.robot_length = self.cfg.MAP_ROBOT_H_POSE
          self.robot_location = [(self.border_buffer+self.robot_length + self.curr_move_height), (self.border_buffer + self.curr_move_width/2)]
          self.robot_location_no_border = [(self.robot_length + self.curr_move_height), (self.curr_move_width/2)]
          print("robot_location:",self.robot_location, self.map_height, self.map_width, self.border_buffer, self.robot_length)

          self.VIRTUAL_MAP_SIZE = self.border_buffer * 1 + self.map_height
          self.map_virtual_map_center = self.VIRTUAL_MAP_SIZE / 2

          self.robot_orientation = 0  # starting point in radians
          print("orientation: ", self.robot_orientation)

          self.add_robot_to_overlay()
          return 
        else:
          ###########################
          # Analyze and Integrate Each Subsequent Frame
          #
          # We're mapping. So, we want base movement. Gather stats.
          # This is post-facto processing, so analysis doesn't have to be
          # realtime.  The idea is to gather enough information to support
          # realtime control after the analysis.
          ###########################
          if action in ["LEFT","RIGHT"]:
            self.find_best_rotation(action, self.prev_move,
                            self.curr_move, frame_num=frame_num)
          elif action in ["FORWARD","REVERSE"]:
            self.analyze_fwd_rev(action, self.prev_move,
                            self.curr_move, frame_num=frame_num)

          # self.map has been computed.
          # make overlay map and add robot in current position.
          # in future, can add position of each move leading to this point

          pt2 = [int(self.robot_location[1]), int(self.robot_location[0])]
          self.map_overlay = self.map.copy()
          self.add_robot_to_overlay()
          return

