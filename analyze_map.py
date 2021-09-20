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
        self.stop_at_frame = 117
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
        self.prev_image = None

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

        self.map_KP = None
        self.frame_num = 0
        self.color_quant_num_clust = 0
        self.cvu = CVAnalysisTools()
        self.cfg = Config()
        self.curr_transform = {}
        self.transform_hist = {}
        self.final_angle = 0
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
    def get_transform_hist(self, frame_num):
        return self.transform_hist[frame_num]

    def set_transform_hist(self):
        if len(self.transform_hist) != self.frame_num:
          print("transform_hist err: ", len(self.transform_hist), self.frame_num)
        self.transform_hist[self.frame_num] = self.curr_transform.copy()

    def get_transform(self, xform, key):
        try:
          return xform[key]
        except:
          return None

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
        # self.prev_move_KP = self.curr_move_KP
        self.curr_move = new_above_view_img.copy()
        # self.curr_move_KP = Keypoints(self.curr_move)
        # TODO: compute distances for each action

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
    def score_lines(self, phase, map, rotated_new_map, action):
        mol = new_map.copy()
        rnm = rotated_new_map.copy()
        if self.color_quant_num_clust is None:
          # Compute Color Quant
          floor_clusters = self.cvu.color_quantification(new_map, self.color_quant_num_clust)
        min_slope_dif_radians = None
        min_slope_dif = self.cfg.INFINITE
        best_line, best_line_radians = None, None

    def get_line_angle(self, prev_map_frame, curr_map_frame, action, frame_num):
        pm = prev_map_frame.copy()
        cm = curr_map_frame.copy()
        # Compute Color Quant
        angle = self.lna.best_line_angle(pm, cm, action, self.robot_location, frame_num=0)
        if action == "LEFT":
          return -angle
        elif action == "RIGHT":
          return angle
        else:
          print("expected L/R turn")
          return None

    #########################
    # Features
    #########################
    def score_features(self, phase, map, rotated_new_map):
        mol = new_map.copy()
        rnm = rotated_new_map.copy()
        # feature_results =
        kp_results["score"] = mse_score
        kp_results["recommendation"] = None

    #########################
    # KeyPoints
    # Input: Map, Intersected Rotated New Map, Phase 
    #########################
    def score_keypoints(self, map, rotated_new_map):
        mol = new_map.copy()
        rnm = rotated_new_map.copy()
        kp_results["score"] = kp_score
        kp_results["recommendation"] = None
        return kpe_score

    def get_kp_angle(self, prev_map_frame, curr_map_frame, action, frame_num, kp_mode="BEST"):
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
          angle = pm_kp.best_keypoint_angle(cm_kp, action, self.robot_location_no_border, frame_num=0)
          if angle is None:
            continue
        if angle is None:
          return None
        if action == "LEFT":
          return -angle
        elif action == "RIGHT":
          return angle
        else:
          print("expected L/R turn")
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
        print("mol_border:", mol_border)
        print("rnm_border:", rnm_border)
        intersect_border = intersect_borders(mol_border, rnm_border)
        if len(intersect_border) == 0:
          cv2.imshow("nonintersecting mol", mol)
          cv2.imshow("nonintersecting rnm", rnm)
          cv2.waitKey(0)
        
        intersect_mol = image_in_border(intersect_border, gray_mol)
        intersect_rnm = image_in_border(intersect_border, gray_rnm)
        if action == "LEFT":
          shape, intersect_mol_border = real_map_border(intersect_mol)
          shape, intersect_rnm_border = real_map_border(intersect_rnm)
          mol_maxh, mol_minh, mol_maxw, mol_minw = get_min_max_borders(intersect_mol_border)
          rnm_maxh, rnm_minh, rnm_maxw, rnm_minw = get_min_max_borders(intersect_rnm_border)
          print("orig mol min/max", mol_maxh, mol_minh, mol_maxw, mol_minw)
          print("orig rnm min/max", rnm_maxh, rnm_minh, rnm_maxw, rnm_minw)
          # TODO: base % on full mol_border (not intersected border)
          # remove 20% of left 5% of top for RNM 
          rnm_minw = round(rnm_minw + .2*(rnm_maxw-rnm_minw))
          rnm_minh = round(rnm_minh + .05*(rnm_maxh-rnm_minh))
          # border in format of a linear ring
          intersect_rnm_border = [[[rnm_maxh,rnm_minw]],[[rnm_maxh, rnm_maxw]], [[rnm_minh, rnm_maxw]], [[rnm_minh, rnm_minw]], [[rnm_maxh,rnm_minw]]]
          # remove 20% of right, 5% of bottom for MOL 
          mol_maxw = round(mol_maxw - .2*(mol_maxw-mol_minw))
          mol_maxh = round(mol_maxh - .05*(mol_maxh-mol_minh))
          # border in format of a linear ring
          intersect_mol_border = [[[mol_maxh,mol_minw]],[[mol_maxh, mol_maxw]], [[mol_minh, mol_maxw]], [[mol_minh, mol_minw]], [[mol_maxh,mol_minw]]]
          print("mol_border", mol_border)
          print("rnm_border", intersect_rnm_border)
          print("intersect_mol h,w", intersect_mol.shape[0], intersect_mol.shape[1])
          print("intersect_rnm h,w", intersect_rnm.shape[0], intersect_rnm.shape[1])

          intersect_mol = image_in_border(intersect_mol_border, intersect_mol)
          intersect_rnm = image_in_border(intersect_rnm_border, intersect_rnm)
          print("final intersect_mol h,w", intersect_mol.shape[0], intersect_mol.shape[1])
          print("final intersect_rnm h,w", intersect_rnm.shape[0], intersect_rnm.shape[1])

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
        lbp_results = {}
        lbp_results["score"] = lbp_score
        lbp_results["recommendation"] = None
        ssim_results = {}
        ssim_results["score"] = ssim_score
        ssim_results["recommendation"] = None
        mse_results = {}
        mse_results["score"] = mse_score
        mse_results["recommendation"] = None
        # return mse_results, ssim_results, lbp_results
        return mse_score, ssim_score, lbp_score

    ##############
    # find_best_rotation()
    ##############
    def find_best_rotation(self, action, prev_frame, curr_frame, start_radians, end_radians, frame_num=0):
        def in_bounds(rot_radians, start_radians, end_radians):
            if (rot_radians > max(start_radians, end_radians) or
                rot_radians < min(start_radians, end_radians)):
              return False
            return True

        N = 50
        dbg_compare = False
        best_mse_score = self.cfg.INFINITE
        best_mse_rot = None
        best_ssim_score = self.cfg.INFINITE
        best_ssim_rot = None
        best_lbp_score = self.cfg.INFINITE
        best_lbp_rot = None
        best_metric_score = self.cfg.INFINITE
        best_metric_rot = None
        best_score = None
        best_rot = None
        rot_mse_radians = 0
        rot_ssim_radians = 0
        rot_lbp_radians = 0
        rot_metric_radians = 0
        mse = []
        ssim = []
        lbp = []
        metric = []
        rot = []
        rand_i = 0
        prev_move = add_border(prev_frame, self.border_buffer)
        # prev_move = prev_frame
        for k in range(N+4):
          # measure input
          if False and k < N:
            rot_radians = rad_sum(start_radians, rad_dif(end_radians,start_radians)*k/N)
            print("rot_radians",rot_radians) 
            # if k in [74, 80, 85, 99]:
            if k in [25, 30, 35]:
              dbg_compare = True
            else:
              dbg_compare = False
          elif False and k >= N:
            dbg_compare = True
            if k-N == 0:
              rot_radians = best_lbp_rot
            elif k-N == 1:
              rot_radians = best_ssim_rot
            elif k-N == 2:
              rot_radians = best_mse_rot
            elif k-N == 3:
              rot_radians = best_metric_rot
          elif k < 16:
            rot_radians = rad_sum(start_radians, rad_dif(end_radians,start_radians)*k*.125)
          else:
            # fit a curve
            mse_z = np.polyfit(mse, rot, 4)
            mse_f = np.poly1d(mse_z)                            
            ssim_z = np.polyfit(ssim, rot, 4)
            ssim_f = np.poly1d(ssim_z)                            
            lbp_z = np.polyfit(lbp, rot, 4)
            lbp_f = np.poly1d(lbp_z)                            
            # predict new value
            prev_rot_mse_radians = rot_mse_radians
            prev_rot_ssim_radians = rot_ssim_radians
            prev_rot_lbp_radians = rot_lbp_radians

            # get local minima for MSE
            mse_crit = mse_f.deriv().r
            mse_r_crit = mse_crit[mse_crit.imag==0].real
            mse_test = mse_f.deriv(2)(mse_r_crit) 
            mse_x_min = mse_r_crit[mse_test>0]
            mse_y_min = mse_f(mse_x_min)

            rot_mse_radians = mse_f(mse_x_min)
            # rot_mse_radians = mse_f(self.cfg.MSE_THRESH * 0.75) 
            rot_ssim_radians = ssim_f(self.cfg.SSIM_THRESH * 0.75) 
            rot_lbp_radians = lbp_f(self.cfg.LBP_THRESH * 0.75) 
            # try biggest change
            d1 = abs(prev_rot_mse_radians - rot_mse_radians )
            d2 = abs(prev_rot_ssim_radians - rot_ssim_radians )
            d3 = abs(prev_rot_lbp_radians - rot_lbp_radians )
            prev_rot_radians = rot_radians
            if True or d1 == min(d1,d2,d3) and in_bounds(rot_mse_radians, start_radians, end_radians):
              print("computed MSE radians", mse_x_min, mse_y_min)
              for i,rads in enumerate(mse_x_min):
                if rads < start_radians or rads > end_radians:
                  continue
                if rads < best_mse_score:
                  best_mse_score = mse_y_min[i]
                  best_mse_rot = rads
              rot_radians = best_mse_score
              break
            elif d2 == min(d1,d2,d3) and in_bounds(rot_ssim_radians, start_radians, end_radians):
              print("SSIM wins")
              rot_radians = rot_ssim_radians
            elif d3 == min(d1,d2,d3) and in_bounds(rot_lbp_radians, start_radians, end_radians):
              print("LBP wins")
              rot_radians = rot_lbp_radians
            # ensure within bounds
            if (rot_radians > max(start_radians, end_radians) or
                rot_radians < min(start_radians, end_radians) or
                prev_rot_radians - rot_radians < self.cfg.RADIAN_THRESH):
              rand_i += 1
              if rand_i > 20:
                break
              rot_radians = rad_sum(start_radians, random.random()*rad_dif(end_radians, start_radians))
              print(rand_i, k, "random", rot_radians)
          # compute rotation and new mse
          curr_move = curr_frame.copy()
          curr_move = add_border(curr_move, self.border_buffer)
          rot_move = rotate_about_robot(curr_move, rot_radians, self.robot_location)
          # measure output
          mse_score, ssim_score, lbp_score = self.score_metrics(action, prev_move, rot_move, dbg_compare)
          # metric_score = ((mse_score/self.cfg.MSE_THRESH)+(ssim_score/self.cfg.SSIM_THRESH)+(lbp_score))
          # metric_score = ((mse_score)+(ssim_score/self.cfg.SSIM_THRESH)+(lbp_score/self.cfg.LBP_THRESH))
          metric_score = ((mse_score)*(ssim_score)*(lbp_score))
          rot.append(rot_radians)
          mse.append(mse_score)
          ssim.append(ssim_score)
          lbp.append(lbp_score)
          metric.append(metric_score)
          if mse_score < best_mse_score:
            rand_i = 0
            print("best MSE")
            best_mse_score = mse_score 
            best_mse_rot = rot_radians 
          if ssim_score < best_ssim_score:
            rand_i = 0
            print("best SSIM")
            best_ssim_score = ssim_score 
            best_ssim_rot = rot_radians 
          if lbp_score < best_lbp_score:
            rand_i = 0
            print("best LBP")
            best_lbp_score = lbp_score 
            best_lbp_rot = rot_radians 
          if metric_score < best_metric_score:
            rand_i = 0
            print("best Metric")
            best_metric_score = metric_score 
            best_metric_rot = rot_radians 
          # todo: store frame, rot, metric scores in history for future analysis
          print("rot, score1:", (best_mse_rot, best_mse_score), (best_ssim_rot, best_ssim_score)) 
          print("rot, score2:", (best_lbp_rot, best_lbp_score), (best_metric_rot, best_metric_score))
          print("rot, score3:", (rot_radians, mse_score, ssim_score, lbp_score))
#          if mse_score <= self.cfg.MSE_THRESH:
#            print("MSE thresh")
#            best_score = best_mse_score
#            best_rot = best_mse_rot
#            break
          if ssim_score <= self.cfg.SSIM_THRESH:
            print("SSIM thresh")
            best_score = best_ssim_score
            best_rot = best_ssim_rot
            break
          if False and lbp_score <= self.cfg.LBP_THRESH:
            print("LBP thresh")
            best_score = best_lbp_score
            best_rot = best_lbp_rot
            break
        if True:
          ### show results
          plt.figure(figsize=(15,9))
          plt.subplot(211);plt.title("Curve fit");plt.xlabel("samples - k")
          mse_plot = []
          ssim_plot = []
          lbp_plot = []
          for i,j in enumerate(mse):
            # mse_plot.append(mse[i]/self.cfg.MSE_THRESH)
            mse_plot.append(mse[i])
            ssim_plot.append(ssim[i]/self.cfg.SSIM_THRESH)
            lbp_plot.append(lbp[i]/self.cfg.LBP_THRESH)
          plt.plot((mse_plot),"b", label="mse")
          plt.plot((ssim_plot),"r", label="ssim")
          # plt.plot((lbp_plot),"g", label="lbp")
          plt.plot(rot,"c", label="rot");plt.legend()
          plt.plot(metric,"y", label="metric");plt.legend()
          plt.legend(); plt.tight_layout(); plt.show()
          cv2.waitKey(0)
        if best_score is None:
          # best_score = best_metric_score
          # best_rot = best_metric_rot
 
          # current algorithm aimed at best mse
          best_score = best_mse_score
          best_rot = best_mse_rot
          
#          mse_ratio = best_mse_score / self.cfg.MSE_THRESH
#          ssim_ratio = best_ssim_score / self.cfg.SSIM_THRESH
#          lbp_ratio = best_lbp_score / self.cfg.LBP_THRESH
#          if mse_ratio == min(mse_ratio, ssim_ratio, lbp_ratio):
#            best_score = best_mse_score
#            best_rot = best_mse_rot
#          elif ssim_ratio == min(mse_ratio, ssim_ratio, lbp_ratio):
#            best_score = best_ssim_score
#            best_rot = best_ssim_rot
#          elif lbp_ratio == min(mse_ratio, ssim_ratio, lbp_ratio):
#            best_score = best_lbp_score
#            best_rot = best_lbp_rot
        print("Final rot, score:", best_rot, best_score)
        return best_rot 

    # delete following if LMSE works
    def find_best_rotation_binary(self, map, new_map, start_radians, end_radians, frame_num=0):
      best_mse, best_ssim, best_lbp = self.cfg.INFINITE, self.cfg.INFINITE, self.cfg.INFINITE
      score_hist = []
      radians_min = start_radians
      radians_max = end_radians
      
      start_with_midpt = True  
      do_min = True  # some metrics require comparing min/max to find best
      do_max = True  # direction to rotate.
      while True:
        if do_min:
          radians = radians_min
          do_min = False
        elif do_max:
          radians = radians_max
          do_max = False
        else:
          radians = self.rad_sum(radians_min,radians_max) / 2
        print("min, mid, max radians:", radians_min, radians, radians_max)
        if self.color_quant_num_clust > 0:
          new_map_rot_cq = rotate_about_robot(floor_clusters, radians, self.robot_location)
        new_map_rot = rotate_about_robot(new_map, radians, self.robot_location)
        shape, map_border = real_map_border(new_map_rot)
        # linesP, imglinesp = self.cvu.get_lines(new_map_rot)

        #################
        # how do you tell if you've gone too far?
        # with lines, the slope results are "above goal" or "below goal", allowing
        #   to figure out if you should bump the min or max.
        # with metrics, the answer is just "above goal".
        # So, when find new min, move down both min/max/mid ?
        # 
        # m \          /
        # e  \        /
        # t   \      /
        # r    \    /
        # i     \  /
        # c      \/
        #      radian
        # Use Least Mean Squares (LMS) filters is to find steepest descent .
        #
        mse_score, ssim_score, lbp_score = self.score_map(map, new_map_rot)
        score_hist.append([radians, mse_score, ssim_score, lbp_score])
        if mse_score < best_mse:
          best_mse = mse_score
          prev_best_mse_radians = radians
          best_mse_radians = radians
          if mse_radians_min >= mse_radians_max:
            mse_radians_min = best_mse
            radians_min = radians
          else:
            mse_radians_min = best_mse
            radians_min = radians
          if mse_score < dummy:
            best_mse_radians = radians
        if ssim_score < best_ssim:
          best_ssim = ssim_score
          prev_best_ssim_radians = radians
          best_ssim_radians = radians
        if lbp_score < best_lbp:
          best_lbp = lbp_score 
          prev_best_lbp_radians = radians
          best_lbp_radians = radians
        print("mse ssim lbp: ", (best_mse, round(best_mse_radians, 4)),
                                (best_ssim, round(best_ssim_radians,4)), 
                                (best_lbp, round(best_lbp_radians, 4)))

        if (# mse_score  <= self.cfg.MSE_THRESH and 
            ssim_score <= self.cfg.SSIM_THRESH and
            lbp_score  <= self.cfg.LBP_THRESH):
            break

      return best_line, final_radians

      if False:
        #################
        # print(linesP)
        in_brdr_cnt = 0
        min_slope_dif = self.cfg.INFINITE
        rot_min_slope_dif = self.cfg.INFINITE
        max_dist = 0
        slope_hist = []
        for [[l0,l1,l2,l3]] in linesP:
          if self.cvu.line_in_border(map_border, (l0,l1), (l2,l3)):
            in_brdr_cnt += 1
            # print("l0-4:", l0,l1,l2,l3)
            dx = l0 - l2
            dy = l1 - l3
            dist = np.sqrt(dx**2 + dy**2)
            if max_dist < dist:
              dist_line = [l0,l1,l2,l3]
              dist_dx = dx
              dist_dy = dy
              dist_slope = self.arctan2(map_dx, map_dy)
              max_dist = dist

            diff_slope = self.rad_dif(map_slope, self.arctan2(dx, dy))
            slope_hist.append([radians, diff_slope, in_brdr_cnt])
            print("map, cur slopes:", map_slope, self.arctan2(dx,dy))
            # print("diff slope/min:", diff_slope, min_slope_dif)
            if abs(diff_slope) < abs(rot_min_slope_dif):
              rot_min_slope_dif = diff_slope
            if abs(diff_slope) < abs(min_slope_dif):
              min_slope_dif = diff_slope
              min_slope_dif_radians = radians
              best_line = [l0,l1,l2,l3]
              if dy != 0:
                print("best line so far:", best_line, radians, diff_slope, "dxy:", map_slope, (dx/dy))
        print("in border line cnt: ", in_brdr_cnt)
        if (dist_line[0] == best_line[0] and dist_line[1] == best_line[1]
            and dist_line[2] == best_line[2] and dist_line[3] == best_line[3]):
           same_line = True
       
        if frame_num == 113:
          cv2.imshow("bad rotated new map:",imglinesp)
          cv2.waitKey(0)
        if frame_num >= self.stop_at_frame:
          # pt = [int(self.robot_location[1]), int(self.robot_location[0])] # [w,h]
          # imglinesp = cv2.circle(imglinesp,pt,3,(255,0,0),-1)
          # cv2.imshow("rotated new map:",imglinesp)
          # cv2.waitKey(0)
          pass
        if in_brdr_cnt == 0:
          # did we rotate too far?
          radians_max = radians
          break
        elif ((min_slope_dif == 0 or self.rad_dif(radians_max,radians_min) <= .001)
          and (round(best_mse_radians,3) == round(min_slope_dif_radians,3) or
               round(best_ssim_radians,3) == round(min_slope_dif_radians,3) or
               round(best_lbp,3) == round(min_slope_dif_radians,3))):
          print("radians min max: ", radians_min, radians_max, min_slope_dif)
          # radians min max:  14.970703125 15 16.365384615384613
          # how is min_slope_dif bigger than max radians?
          #  - radians is angle of rotation, min slope dif is slope of resulting line
          break
        elif rot_min_slope_dif > 0:
          radians_min = radians
        elif rot_min_slope_dif <= 0:
          radians_max = radians
      print("best mse: ", best_mse, best_mse_radians)
      print("best ssim:", best_ssim, best_ssim_radians)
      print("best lbp: ", best_lbp, best_lbp_radians)
      if in_brdr_cnt == 0 and best_line is not None:
        print("######## Using Best Slope So Far #########")
        print("radians min max: ", radians_min, radians_max)
        print("New Map best line/radians:", best_line, min_slope_dif_radians)
        final_radians = min_slope_dif_radians
      elif min_slope_dif == self.cfg.INFINITE:
        print("######## New Map Slope not found #########")
        best_line, final_radians = None, None
      else:
        print("######## New Map Slope found #########")
        print("New Map info:", best_line, min_slope_dif_radians)
        final_radians = min_slope_dif_radians
      return best_line, final_radians
    
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
        s = ssim(imageA, imageB)
        # if self.frame_num >= 113:
        if self.frame_num >= self.stop_at_frame:
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
        return m, (1-s)

    def slope_metric_matching(self, map_overlay, rotated_new_map, map_line, map_slope):
        mrot, moff_h, moff_w = self.manual_tuning(self.frame_num)
        mse_score, ssim_score, lbp_score = self.cfg.INFINITE,self.cfg.INFINITE,self.cfg.INFINITE
        for n in range(5):
          off_h = int(moff_h / 4 * n)
          off_w = int(min(moff_w,-20) / 4 * n)
          print("SLOPE METRICS off_h, off_w:", off_h, off_w)
          rnm = rotated_new_map.copy()
          rnm = replace_border(rnm,
                            self.map_overlay.shape[0], self.map_overlay.shape[1],
                            off_h,off_w)
          # the general idea is that the slope computes the height axis and
          # feature points control the width axis.
          off_b = self.get_height_dif_by_line(rnm, map_line, map_slope, off_w)
          off_h += off_b   # solves 112 slightly off with huge oscilations for unknown reasons
          print("off_b, off_h, off_w", off_b, off_h, off_w)
          # off_h and off_w are absolute offsets for replace_border 
          # The KPs should be at same H as the new offset.
          rnm = rotated_new_map.copy()
          rnm = replace_border(rnm, self.map_overlay.shape[0], self.map_overlay.shape[1],
                            off_h,off_w)
  
          print("###############")
          print("Slope Metric offsets")
          map_pts = []
          rot_pts = []
          mse_score, ssim_score, lbp_score = self.score_map(map_overlay, rnm)
          if mse_score < best_mse:
            best_mse = mse_score
            best_mse_off_h = off_h
            best_mse_off_w = off_w
          if ssim_score < best_ssim:
            best_ssim = ssim_score
            best_ssim_off_h = off_h
            best_ssim_off_w = off_w
          if lbp_score < best_lbp:
            best_lbp = lbp_score
            best_lbp_off_h = off_h
            best_lbp_off_w = off_w
        print("best mse: ", best_mse, best_mse_off_h, best_mse_off_w)
        print("best ssim:", best_ssim, best_ssim_off_h, best_ssim_off_w)
        print("best lbp: ", best_lbp, best_lbp_off_h, best_lbp_off_w)
        return best_mse, (best_mse_off_h, best_mse_off_w), best_ssim, (best_ssim_off_h, best_ssim_off_w), best_lbp, (best_lbp_off_h, best_lbp_off_w)

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
    
    def apply_homography(self, prev_img, curr_img, frame_num=0):
        pi = prev_img.copy()
        ci = curr_img.copy()
        pig = cv2.cvtColor(pi, cv2.COLOR_BGR2GRAY)
        cig = cv2.cvtColor(ci, cv2.COLOR_BGR2GRAY)
        feature_params = dict( maxCorners = 100,
               # qualityLevel = 0.3,
               qualityLevel = 0.01,
               minDistance = 7,
               blockSize = 7 )
        
        pig_kp = Keypoints(pig)
        cig_kp = Keypoints(cig)
        kp_matches,notsogood = pig_kp.compare_kp(cig_kp)
        # print("num matches:", len(kp_matches))
        pig_features2 = cv2.goodFeaturesToTrack(pig, mask = None, **feature_params)
        cig_features2 = cv2.goodFeaturesToTrack(cig, mask = None, **feature_params)

#        pig_disp = prev_img.copy()
#        cig_disp = curr_img.copy()
#        # for pt in pig_features:
#        for pt in kp_matches:
#          ppt = [int(pt[0][0]), int(pt[0][1])]
#          # print("ppt", ppt)
#          pig_disp=cv2.circle(pig_disp,ppt,3,(0,255,0),-1)
        pig_features = []
        cig_features = []
        for kpm in kp_matches:
          kp1 = pig_kp.keypoints[kpm[0].queryIdx]
          kp2 = cig_kp.keypoints[kpm[0].trainIdx]
          pig_features.append([int(kp1.pt[0]), int(kp1.pt[1])])
          cig_features.append([int(kp2.pt[0]), int(kp2.pt[1])])
        pig_features = np.int32(pig_features)
        cig_features = np.int32(cig_features)
        print("pigf", pig_features)
        print("cigf", cig_features)
#        for pt in cig_features:
#          ppt = [int(pt[0][0]), int(pt[0][1])]
#          # print("cpt", ppt)
#          cig_disp=cv2.circle(cig_disp,ppt,3,(0,255,0),-1)
#        cv2.imshow("cig_disp", cig_disp )
#        cv2.imshow("pig_disp", pig_disp )
#        cv2.waitKey(0)

        print("num of pig/cig features:", len(pig_features), len(cig_features))

        # add border to color images before applying transforms
        # pi = add_border(pi, self.border_buffer)
        # ci = add_border(ci, self.border_buffer)

# Only because the object is planar, the camera pose can be retrieved from the homograph
        H, mask = cv.findHomography(cig_features, pig_features, cv.RANSAC,5.0)
        H2, mask2 = cv.findHomography(cig_features2, pig_features2, cv.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        # border_shape, cig_border = real_map_border(cig)
        # cig_maxw, cig_minw, cig_maxh, cig_minh = get_min_max_borders(cig_border)

        # h = cig_maxh - cig_minh
        # w = cig_maxw - cig_minw
        # cig_pts = np.float32([ [cig_minh,cig_minw],[cig_minh,cig_maxw],[cig_maxh,cig_maxw],[cig_maxh,cig_minw] ]).reshape(-1,1,2)
        h,w = ci.shape[:2]
        cig_pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        cig_pts_px = cv.perspectiveTransform(cig_pts,H)

        print("Homog H", H, cig_pts_px)

        # print("size=",sz)
        ci = cv.warpPerspective(ci, H, (h,w));
        # ci = cv.warpPerspective(ci, cig_pts_px, (h,w));
        if self.map_overlay is None:
          self.map_overlay = pi
        # self.map_overlay = self.merge_maps(self.map_overlay, ci)
        # cv2.imshow("orig", curr_img )
        # cv2.imshow("prev", prev_img )
        # cv2.imshow("homography", ci )
        # ci = cv.warpAffine(ci, H, sz);
        cv2.imshow("Warp Persp", ci )
        
        sz = curr_img.shape[:2]
        K_lst = [[ 6.5746697944293521e+002, 0., 3.1950000000000000e+002 ],
                 [ 0., 6.5746697944293521e+002, 2.3950000000000000e+002 ],
                 [ 0., 0., 1.]]
        # K = np.array(K_lst, dtype="float32").reshape((3,3))
        K = np.array(K_lst, dtype=np.float32).reshape((3,3))
        distortion_coef_lst = [-4.1802327176423804e-001, 5.0715244063187526e-001, 
                                0., 0.,  -5.7843597214487474e-001]
        # newcamera, roi = cv2.getOptimalNewCameraMatrix(K, distortion_coef_lst, (sz), 0)
        # newimg = cv2.undistort(img, K, d, None, newcamera)

        num, Rs, Ts, Ns  = cv2.decomposeHomographyMat(H, K)
        print("num",num)
        print("Rs",Rs)
        print("Ts",Ts)
        print("Ns",Ns)
        # ci = cv2.warpAffine(ci, Rs[0], (ci.shape[0], ci.shape[1]))

        # quick solution to get the camera pose:
        # https://docs.opencv.org/master/d9/dab/tutorial_homography.html
        #  Normalization to ensure that ||c1|| = 1
        norm = np.sqrt(H[0,0]*H[0,0] + H[1,0]*H[1,0] + H[2,0]*H[2,0])
        print("H",norm)
        H1 = np.divide(H,norm)
        print("H",H, norm)
        c1  = H1[:,0]
        c2  = H1[:,1]
        c3  = np.cross(c1,c2) 
        otw = H1[:,2]   # translation vector
        # dist = np.sqrt(otw[0]**2 + otw[1]**2 + otw[2]**2)
        dist = otw[0]
        print("otw, dist:", otw, dist)
        oRw = np.zeros((3, 3), dtype = "float32")
        for i in range(3):
          oRw[i,0] = c1[i]
          oRw[i,1] = c2[i]
          oRw[i,2] = c3[i]
        print("oRw:", oRw)  # Rotation matrix
        # ci = cv2.warpAffine(ci, R, (ci.shape[0], ci.shape[1]))

        # a 3D point expressed in the world frame into the camera frame
#        cam_extrinsics = np.zeros((4, 4), dtype = "float32")
#        for i in range(4):
#          for j in range(4):
#            if i < 3 and j < 3:
#              cam_extrinsics[i,j] = oRw[i,j]
#            elif i < 3 and j == 3:
#              cam_extrinsics[i,j] = otw[j]
#            elif i == 3 and j == 3:
#              cam_extrinsics[i,j] = 1
#            else:
#              cam_extrinsics[i,j] = 0
#        # [x, y, z, 1] = cam_extrinsics * [X, Y, Z, 1]

        # https://stackoverflow.com/questions/58538984/how-to-get-the-rotation-angle-from-findhomography
        # with SVD decomposition of the 2x2 upper-left block to recover a rotation:
        u, _, vh = np.linalg.svd(H[0:2, 0:2])
        R = u @ vh
        angle = -math.atan2(R[1,0], R[0,0])
        print("homography angle:", angle)

        u1, _, vh1 = np.linalg.svd(H1[0:2, 0:2])
        R1 = u1 @ vh1
        angle1 = -math.atan2(R1[1,0], R1[0,0])
        print("homography angle1:", angle1, norm)

        u2, _, vh2 = np.linalg.svd(H2[0:2, 0:2])
        R2 = u2 @ vh2
        angle2 = math.atan2(R2[1,0], R2[0,0])
        print("homography angle2:", angle2)

        norm2 = np.sqrt(H2[0,0]*H2[0,0] + H2[1,0]*H2[1,0] + H2[2,0]*H2[2,0])
        H3 = np.divide(H2,norm2)
        u3, _, vh3 = np.linalg.svd(H3[0:2, 0:2])
        R3 = u3 @ vh3
        angle3 = math.atan2(R3[1,0], R3[0,0])
        print("homography angle3:", angle3, norm2)

# manual angle: -0.11
# homography angle: -0.13783406390920439
# homography angle1: -0.13783406390920436
# homography angle2: 2.9152893930653474


        # cv2.imshow("homography", ci )
        # cv2.waitKey(0)
        return angle, dist, ci

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
          # self.robot_location = [(self.border_buffer + self.curr_move_height/2),(self.border_buffer+self.robot_length + self.curr_move_height)]
          print("robot_location:",self.robot_location, self.map_height, self.map_width, self.border_buffer, self.robot_length)

          self.VIRTUAL_MAP_SIZE = self.border_buffer * 1 + self.map_height
          self.map_virtual_map_center = self.VIRTUAL_MAP_SIZE / 2

          # Compare to self for intial keypoint rotation score; 
          # only if min rotation is 0
          self.map_KP = Keypoints(self.map)
          kp_matches,notsogood = self.map_KP.compare_kp(self.map_KP)
          print("num matches:", len(kp_matches))
          # orient to self
          map_pts, map_pts2 = self.map_KP.get_n_match_kps(kp_matches, self.map_KP, 3)
          self.robot_orientation = 0  # starting point in radians
          print("orientation: ", self.robot_orientation)
          self.curr_transform = {}
          self.set_transform_hist()
          self.prev_image = curr_image.copy()

          return # should fall through eventually

        ###########################
        # Analyze and Integrate Each Subsequent Frame
        ###########################
        else:
          # initialize images and map for new move
          height,width,ch = self.curr_move.shape
          # for mode in ["SIFT", "FEATURE", "ROOTSIFT", "ORB", "SURF"]:
          # for mode in ["SIFT", "FEATURE", "ROOTSIFT", "ORB"]:
          metrics = {"LINE":{},"KP":{},"LMS":{}}
          for mode in ["LINE", "KP", "LMS"]:
            if mode == "LINE":
              new_angle_ln = self.get_line_angle(self.prev_move, self.curr_move, action, frame_num=0)
              print("new ln angle: ", new_angle_ln)
              print("##########################")
              new_angle = new_angle_ln
            elif mode == "KP":
              new_angle_kp = self.get_kp_angle(self.prev_move, self.curr_move, action, frame_num=0, kp_mode="BEST")
              print("new kp angle: ", mode, new_angle_kp)
              print("##########################")
              new_angle = new_angle_kp
            elif mode == "LMS":
              if action == "LEFT":
                # start_angle = -np.pi/16
                start_angle = 0
                end_angle= np.pi/16
              elif action == "RIGHT":
                start_angle = 0
                end_angle= np.pi/16
#              start_angle = self.robot_orientation
#              end_angle=rad_sum(start_angle,np.pi/16)
#              if (metrics["LINE"]["MSE"] <= metrics["KP"]["MSE"] and 
#                metrics["LINE"]["SSIM"] <= metrics["KP"]["SSIM"] and 
#                metrics["LINE"]["LBP"]  <= metrics["LINE"]["LBP"]):
#                # LINE is better than KP
#                if new_angle_ln < new_angle_kp:
#                  if action == "LEFT":
#                    start_angle = new_angle_ln
#                elif new_angle_ln > new_angle_kp:
#                  if action == "LEFT":
#                    end_angle=new_angle_kp
#              elif (metrics["LINE"]["MSE"] >= metrics["KP"]["MSE"] and 
#                metrics["LINE"]["SSIM"] >= metrics["KP"]["SSIM"] and 
#                metrics["LINE"]["LBP"]  >= metrics["LINE"]["LBP"]):
#                # LINE is better than KP
#                if new_angle_ln > new_angle_kp:
#                  start_angle = self.robot_orientation
#                  end_angle=rad_sum(start_angle,np.pi/4),
#              else:
#                start_angle = self.robot_orientation
#                end_angle=rad_sum(start_angle,np.pi/4),
              new_angle_mse = self.find_best_rotation(action, self.prev_move,
                            self.curr_move, start_radians=start_angle,
                            end_radians=end_angle,
                            frame_num=frame_num)
              new_angle = new_angle_mse
              print("new mse angle: ", mode, new_angle_mse)
              print("##########################")
            elif mode == "MANUAL":
              new_angle, off_h, off_w = self.manual_tuning(self.frame_num)

            # cv2.imshow("curr_move:", self.curr_move)
            new_map = add_border(self.curr_move, self.border_buffer)
            # cv2.imshow("orig new_map:", new_map)
            rotated_new_map = new_map.copy()  # for resilience against bugs
            # self.robot_location = [(self.border_buffer + self.curr_image.shape[1]/2),dist]
            print("robot location", self.robot_location, new_angle, self.final_angle)
            self.prev_image = curr_image.copy()
            pt2 = [int(self.robot_location[1]), int(self.robot_location[0])]
            pt2[1] += self.cfg.MAP_ROBOT_H_POSE
            rot_angle = self.final_angle + new_angle
            rotated_new_map = rotate_about_robot(rotated_new_map, rot_angle, pt2)
            metrics[mode]["MSE"], metrics[mode]["SSIM"], metrics[mode]["LBP"] = self.score_metrics(action, self.map, rotated_new_map)
            # print("mse, ssim, lbp metrics: ",mse_results["score"], ssim_results["score"], lbp_results["score"])
            # if (mse_results["score"] <= self.cfg.MSE_THRESH and ssim_results["score"] <= self.cfg.SSIM_THRESH and
            #     lbp_results["score"] <= self.cfg.LBP_THRESH):
            print(mode, "mse, ssim, lbp metrics: ",
              metrics[mode]["MSE"], metrics[mode]["SSIM"], metrics[mode]["LBP"] )
            if (# metrics[mode]["MSE"] <= self.cfg.MSE_THRESH and 
                metrics[mode]["SSIM"] <= self.cfg.SSIM_THRESH and 
                metrics[mode]["LBP"]  <= self.cfg.LBP_THRESH):
              break

          self.final_angle += new_angle
          self.robot_orientation = self.final_angle
          self.map = self.merge_maps(self.map, rotated_new_map)
          pt2 = [int(self.robot_location[1]), int(self.robot_location[0])]
          self.map_overlay = self.map.copy()
          # draw center of robot on map_overlay
          self.map_overlay = cv2.circle(self.map_overlay,pt2,3,(255,0,0),-1)
          # draw rotated robot on map_overlay
          rot_rect = (pt2, self.cfg.MAP_ROBOT_SHAPE, radians_to_degrees(-self.robot_orientation))
          box = cv2.boxPoints(rot_rect) 
          box = np.int0(box)
          cv2.drawContours(self.map_overlay,[box],0,(0,0,255),2)

          rotated_robot_rectangle = cv2.circle(rotated_new_map,pt2,3,(255,0,0),-1)
          # self.map_overlay = self.merge_maps(self.map_overlay, ci)
          cv2.imshow("map_overlay:", self.map_overlay)
          cv2.waitKey(0)
          return

          #####################################################################
          # IMAGE ROTATION AROUND ROBOT LOCATION
          # Line Analysis, Keypoint Analysis, Feature Analysis, Metric Analysis
          #####################################################################
          # line detection and keypoint detection includes black borders
          # KP, Lines, features can handle full images with borders
          # Metrics require rectangular intersected images with no borders
          # KP, features can get previous and current unrotated maps
          #   and match them using optical flow.  Then rotate.
          ####################################################################

          h = 0
          w = 1
          map_line, map_slope, max_dist = self.analyze_map_lines(self.map_overlay, frame_num)
          if map_slope is not None:
            new_map_img = new_map.copy()
            # initially assume all left turns
            start_angle = self.robot_orientation 
            best_rotation = self.find_best_rotation(action, self.map_overlay, 
                          new_map_img, start_angle=start_angle,
                          end_angle=self.rad_sum(start_angle,np.pi/4),
                          frame_num=frame_num)
            
            if best_rotation is not None:
              final_angle = best_rotation
              print("Final best angle:", final_angle)
              self.robot_orientation = final_angle

              rotated_new_map = new_map.copy()  # for resilience against bugs
              rotated_new_map = rotate_about_robot(rotated_new_map, final_angle, self.robot_location)
              rotated_new_map_disp = rotated_new_map.copy()
              rot_only_new_map = rotated_new_map.copy() # for straight merge experiment
              cv2.line(rotated_new_map_disp, (best_line[0], best_line[1]), (best_line[2], best_line[3]), (0,255,0), 3, cv2.LINE_AA)
              # self.show_rot(new_map)

            ##############
            # Best Rotated Image found via line or KP analysis
            # Next, Merge and display images
            rnm_kp = {}
            mol_kp = {}
            kp_offsets = {}
            if False:
            # if best_line_angle is not None or best_kp_angle is not None:
              print("frame_num:",frame_num)
              # (self.border_buffer+3*self.robot_gripper_pad_len + self.map_width)]
              rnm_kp = replace_border(rotated_new_map,
                       self.map_overlay.shape[0], self.map_overlay.shape[1],
                                  0, 0)
              for kp_mode in ["SIFT", "ROOTSIFT", "ORB"]:
                mol = self.map_overlay.copy()
                rnm = rnm_kp.copy()
                print("kp_mode: ", kp_mode)
                rnm_kp[kp_mode] = Keypoints(rnm, kp_mode)
                mol_kp[kp_mode] = Keypoints(mol, kp_mode)
                kp_off_h, kp_off_w, map_pts, new_map_rot_pts = self.evaluate_map_kp_offsets(mol_kp[kp_mode], rnm, rnm_kp[kp_mode])
                txt = "RESULT: " + kp_mode + " KP match"
                kp_offsets[kp_mode] = [kp_off_h, kp_off_w, map_pts, new_map_rot_pts]
                print(txt,(kp_off_h, kp_off_w))
                rnm = rnm_kp[kp_mode].drawKeypoints()
                mol = mol_kp[kp_mode].drawKeypoints()
                if frame_num >= self.stop_at_frame:
                  txt = kp_mode + "_" + "rotate_new_mapkp"
                  cv2.imshow(txt, rnm)
                  txt = kp_mode + "_" + "mol_kp"
                  cv2.imshow(txt, mol)
                  cv.waitKey(0)
                  cv2.destroyAllWindows()

              # If kp_mode didn't work, try more generic features, combined with line slope
              f1off_h = 0 
              f1off_w = 0
              f2off_h = 0 
              f2off_w = 0
              f3off_h = 0 
              f3off_w = 0
              # if len(map_pts) == 0:
              if False:
                moff_h,moff_w = self.manual_tuning(frame_num)
                # if len(map_pts) == 0:
                if False:
                  f1off_h, f1off_w, f1map_pts, new_map_rot_pts = self.feature_offsets(self.map_overlay, rotated_new_map, map_line, map_slope, best_line, frame_num)
                  print("RESULT: Purely based on features: ",(f1off_h, f1off_w),(moff_h, moff_w))
                  f2off_h, f2off_w, map_pts, new_map_rot_pts = self.feature_offsets(self.map_overlay, rotated_new_map, map_line, map_slope, best_line, frame_num, use_slope=True)
                  print("RESULT: slope/var features: ",(f2off_h, f2off_w),(moff_h, moff_w))
                  f3off_h, f3off_w, map_pts, new_map_rot_pts = self.feature_offsets(self.map_overlay, rotated_new_map, map_line, map_slope, best_line, frame_num, use_slope=True, offset_width=f2off_w)
                  print("RESULT: slope/var features w offset_width: ",(f3off_h, f3off_w),(moff_h, moff_w))
                else:
                  foff_h = kp_off_h
                  foff_w = kp_off_w
                  print("RESULT: KP match: ",(foff_h, foff_w),(moff_h, moff_w))

                # compute from scratch; ignore above variance/manual/keypoint calculations.
                fsoff_h, fsoff_w = 0, 0
                fsoff_h, fsoff_w, map_pts, new_map_rot_pts = self.feature_slope_offsets(self.map_overlay, rotated_new_map, map_line, map_slope, fsoff_h, fsoff_w, frame_num)
                print("RESULT: feature_slope: ",(fsoff_h, fsoff_w),(moff_h, moff_w))

        if True: 
            if True:
              if True:  # dummy values if prev analysis skipped
                kp_off_h, kp_off_w = 0, 0
                f1off_h = 0
                f1off_w = 0
                f2off_h = 0
                f2off_w = 0
                f3off_h = 0
                f3off_w = 0

              if False:
                self.slope_metric_matching(self.map_overlay, rotated_new_map, map_line, map_slope)

            # if len(map_pts) == 0:
              if True:
                print("KEYPOINT, FEATURE AND LINE TUNING FAILED, FALLBACK TO MANUAL TUNING", frame_num)
                rot,moff_h,moff_w = self.manual_tuning(frame_num)
                print("moff_h, moff_w:", moff_h, moff_w)

              # for merge_map_mode in ["MANUAL","ROT_ONLY","KP", "FEATURE", "FEATURE1", "FEATURE2", "FEATURE_SLOPE"]:
              # only one can be done at a time; can add a "primary" such as "MANUAL" and
              # dif merge_map_mode can do a delta each time.
              for merge_map_mode in ["MANUAL"]:
                print("mergemap mode:", merge_map_mode)
                if (merge_map_mode == "ROT_ONLY"):
                  ronm = rot_only_new_map 
                  off_h, off_w = 0,0
                elif (merge_map_mode == "KP"):
                  off_h, off_w = kp_off_h, kp_off_w
                elif (merge_map_mode == "FEATURE1"):
                  off_h, off_w = f1off_h, f1off_w
                elif (merge_map_mode == "FEATURE2"):
                  off_h, off_w = f2off_h, f2off_w
                elif (merge_map_mode == "FEATURE3"):
                  off_h, off_w = f3off_h, f3off_w
                elif (merge_map_mode == "FEATURE_SLOPE"):
                  off_h, off_w = fsoff_h, fsoff_w
                elif (merge_map_mode == "MANUAL"):
                  off_h, off_w = moff_h, moff_w
                merge_map_mode = "MANUAL"
                show_results = True
                if show_results or frame_num >= self.stop_at_frame:
                  # cv2.imshow("old overlay:", self.map_overlay)
                  # Final rotation
                  if (merge_map_mode != "ROT_ONLY"):
                    ronm = replace_border(rotated_new_map,
                           self.map_overlay.shape[0], self.map_overlay.shape[1],
                           off_h,off_w)
                  ronm_disp = ronm.copy() # display rotation
                  # circle accepts [w,h]
                  pt = [int(self.robot_location[1]), int(self.robot_location[0])]
                  ronm_disp=cv2.circle(ronm_disp,pt,3,(255,0,0),-1)
                  # txt = "rot reborder h:" + str(off_h) + " w:" + str(off_w)
                  # cv2.imshow(txt, ronm_disp)
                  # Note: if robot_location is correct and doing LEFT with correct angle,
                  #       then just merge?  Nope, didn't work.  With the rotation matrix,
                  #       the results are more like a swivel instead of a rotation. So,
                  #       a Rot Left looks like image is offsetting downward; L/R doesn't
                  #       move much.
                # map_ovrly = cv2.addWeighted(self.map_overlay,0.5,ronm,0.5,0.5)
                self.map_overlay = self.merge_maps(self.map_overlay, ronm)
                if show_results or frame_num >= self.stop_at_frame:
                  map_ovrly = self.merge_maps(self.map_overlay, ronm_disp)
                  # cv2.line(map_ovrly, (map_line[0], map_line[1]), (map_line[2], map_line[3]), (0,255,0), 3, cv2.LINE_AA)
                  txt = merge_map_mode + "_" + str(frame_num)
                  # cv2.imshow(txt,map_ovrly)
                  # cv2.imshow("rotated around robot", ronm_disp)
        if True: 
          cv2.waitKey(0)
          cv2.destroyAllWindows()
          return


