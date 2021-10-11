#!/usr/bin/env python
from operator import itemgetter, attrgetter
import cv2 as cv
import numpy as np 
import math
from shapely.geometry import *
from cv_analysis_tools import *
from config import *
from utilborders import *
from utilradians import *

class FeatureMatch:
        def __init__(self, kp1, kp2, dist):
            self.queryIdx = kp1
            self.trainIdx = kp2
            self.distance = dist

          # print("queryIdx:", kpm[0].queryIdx, kpm[1].queryIdx)
          # print("imgIdx:", kpm[0].imgIdx, kpm[1].imgIdx)
          # print("trainIdx:", kpm[0].trainIdx, kpm[1].trainIdx)
          # print("num kp:", len(self.KP.keypoints),len(KP2.keypoints))
          # print("kp1t:",self.keypoints[kpm[0].queryIdx].pt)
          # print("kp2q:",KP2.keypoints[kpm[1].trainIdx].pt)
          # print("dist:", kpm[0].distance, kpm[1].distance)

#    def __init__(self, qidx, tidx, dist):
#        self.fm = []
#
#    def __getitem__(self,index):
#        if len(self.fm) < index:
#          return self.fm.bricksId[index]
##
#    def __setitem__(self,index,value):
#        if len(self.fm) < index:
#          self.fm[index] = value
#        if len(self.fm) == index:



class Keypoints:
    ######################################
    # functions that work for SIFT and ORB
    ######################################
    def __init__(self, img, kp_mode = "SIFT"):
      self.keypoints = []
      self.pc_map = None
      self.pc_header = None
      self.kp_pc_points = None
      self.border = None
      self.kp_mode = kp_mode
      self.cfg = Config()
      self.img = img.copy()
      self.cvu = CVAnalysisTools()
      self.feature_matches = []
      if kp_mode == "SIFT":
        self.KP = cv.SIFT_create()
        try:
          self.img= cv.cvtColor(self.img,cv.COLOR_BGR2GRAY)
        except:
          pass
        # cv.imshow("kp img:", self.img)
        # cv.waitKey(0)
      elif kp_mode == "ROOTSIFT":
        # initialize the SIFT feature extractor
        self.KP = cv.SIFT_create()
        self.img= cv.cvtColor(self.img,cv.COLOR_BGR2GRAY)
      elif kp_mode == "ORB":
        self.KP = cv.ORB_create()         # Initiate SIFT detector
      elif kp_mode == "SURF":
        # SURF requires special installation and compilation b/c
        # SURF is not free.  Not used by default.
        # self.KP = cv.SURF_create()         # Initiate SIFT detector
        self.KP = cv.xfeatures2d.SURF_create(400)         # Initiate SIFT detector
      elif kp_mode == "FEATURE":
        feature_params = dict( maxCorners = 100,
               qualityLevel = 0.02,
               # qualityLevel = 0.03,   # too few results
               # qualityLevel = 0.01,   # too many results
               # qualityLevel = 0.3,    # decent quality, but not many results
               minDistance = 7,
               blockSize = 7 )
      else:
        print("Unknown KP mode: ", kp_mode)
      # compute descriptors
      if kp_mode == "FEATURE":
        try:
          self.img= cv.cvtColor(self.img,cv.COLOR_BGR2GRAY)
        except:
          pass
        kps = cv2.goodFeaturesToTrack(self.img, mask = None, **feature_params)
        self.keypoints = []
        for kp in kps:
          # print("kp", kp)
          self.keypoints.append(cv2.KeyPoint(kp[0][0],kp[0][1],1))
        self.descriptor = None
      else:
        self.keypoints, self.descriptor = self.KP.detectAndCompute(self.img,None)
      if kp_mode == "ROOTSIFT" and len(self.keypoints) >= 0:
        # apply the Hellinger kernel by first L1-normalizing and taking the square-root
        eps = 1e-7
        self.descriptor /= (self.descriptor.sum(axis=1, keepdims=True) + eps)
        self.descriptor = np.sqrt(self.descriptor)
        # self.descriptor /= (np.linalg.norm(descs, axis=1, ord=2) + eps)

    def get_points(self):
      kp_list = [[kp.pt[0], kp.pt[1]] for kp in self.keypoints]
      return kep_list

    def get_kp(self):
      return self.keypoints

    def get_des(self):
      return self.descriptor

    def get_kp_mode(self):
      return self.kp_mode

    def get_image(self):
      return self.image

    def find_min_var(self,KP2,include_dist=True):
      pass

    # There should be KPs on both the prev_frame and curr_frame that represent the same point.
    # These KPs should line up after rotation. The rotation should equal the delta angle from 
    # the robot location.  Equidistance from robot location is expected.
    # 
    # When turning left, the new frame's keypoints will appear to be right of the prev frame's kps.
    # 
    def best_keypoint_move(self, KP2, action, robot_location, frame_num):
        #####
        def angle_match_count(angle_match, rads, radian_thresh):
          found = False
          for i, [a,c] in enumerate(angle_match):
            if rad_interval(rads, a) <= radian_thresh:
              found = True
              angle_match[i][1] += 1
          if not found:
            angle_match.append([rads, 0])
          return angle_match

        def move_match_count(move_match, pix_moved, pix_thresh):
          found = False
          for i, [m,c] in enumerate(move_match):
            if pix_moved <= pix_thresh:
              found = True
              move_match[i][1] += 1
          if not found:
            move_match.append([pix_moved, 0])
          return move_match

        #####

        good_matches, bad_matches = self.compare_kp(KP2,include_dist=True)
        KP1_kps = []
        KP2_kps = []
        angle_match = []
        move_match = []
        if self.border is None:
          border_shape, self.border = real_map_border(self.img)
        # print("num matches: ", len(good_matches))
        for i,kpm in enumerate(good_matches):
          # print("queryIdx:", kpm[0].queryIdx, kpm[1].queryIdx)
          # print("imgIdx:", kpm[0].imgIdx, kpm[1].imgIdx)
          # print("trainIdx:", kpm[0].trainIdx, kpm[1].trainIdx)
          # print("num kp:", len(self.KP.keypoints),len(KP2.keypoints))
          # print("kp1t:",self.keypoints[kpm[0].queryIdx].pt)
          # print("kp2q:",KP2.keypoints[kpm[1].trainIdx].pt)
          # print("dist:", kpm[0].distance, kpm[1].distance)
          ((queryIdx, trainIdx, dist), (n_queryIdx, n_trainIdx, n_dist), ratio) = kpm
          kp1 = self.keypoints[queryIdx]
          kp2 = KP2.keypoints[trainIdx]
          # (x,y) and (h,w) mismatch
          h,w = 1,0 
          rl = [robot_location[h], robot_location[w]]
          # rl = robot_location
          if point_in_border(self.border,kp1.pt) and point_in_border(self.border,kp2.pt):
            # if action is "LEFT", the prevframe kp1 should be left of curr frame kp2.
            #   assert: abs(kp1[h] - kp2[h]) < abs(kp1[w] - kp2[w])
            #   assert: kp1[w] < kp2[w]
            #
            # rads = rad_isosceles_triangle(kp1.pt, robot_location, kp2.pt)
            # w is pt[0], h is pt[1]
            if (action == "LEFT" and 
                (abs(kp1.pt[h] - kp2.pt[h]) > abs(kp1.pt[w] - kp2.pt[w]) or kp1.pt[w] >= kp2.pt[w])):
              continue
            elif (action == "RIGHT" and 
                (abs(kp1.pt[h] - kp2.pt[h]) > abs(kp1.pt[w] - kp2.pt[w]) or kp1.pt[w] <= kp2.pt[w])):
              print("Right assert: KP fails", kp1.pt, kp2.pt)
              continue
            if action in ["LEFT", "RIGHT"]:
              rads = rad_isosceles_triangle(kp1.pt, rl, kp2.pt)
              if rads is None:
                continue
              angle_match = angle_match_count(angle_match, rads, self.cfg.RADIAN_THRESH)
            if action in ["FORWARD"]:
              if abs(kp1.pt[w] - kp2.pt[w]) > 2 or kp1.pt[h] <= kp2.pt[h]:
                print("FWD assert: KP fails", kp1.pt, kp2.pt)
                continue
            elif action in ["REVERSE"]:
              if abs(kp1.pt[w] - kp2.pt[w]) > 2 or kp1.pt[h] >= kp2.pt[h]:
                print("Rev assert: KP fails", kp1.pt, kp2.pt)
                continue
            if action in ["FORWARD","REVERSE"]:
              pix_moved = abs(kp1.pt[h] - kp2.pt[h])
              print("Fwd/Rev KP match:", kp1.pt, kp2.pt, pix_moved)
              move_match = move_match_count(move_match, pix_moved, 2)

        if action in ["LEFT", "RIGHT"] and len(angle_match) >= 1:
          # was: return best count; now: return list for mse eval 
          kp_angle_lst = [a[0] for a in angle_match]
          print("KP Angle match:", angle_match)
          return(kp_angle_lst)
#          max_a, max_c = self.cfg.INFINITE, -self.cfg.INFINITE
#          kp_angle_lst = []
#          for a,c in angle_match:
#            if c >= max_c:
#              if action in ["LEFT"]:
#                max_a = a
#              elif action in ["RIGHT"]:
#                max_a = -a
#              if c > max_c:
#                kp_angle_lst = []
#              else:
#                # return ties for further evaluation
#                kp_angle_lst.append(max_a)
#              max_c = c

        if action in ["FORWARD", "REVERSE"] and len(move_match) >= 1:
          kp_move_lst = [mv[0] for mv in move_match]
          print("KP move match:", kp_move_lst)
          return(kp_move_lst)

#          max_mv, max_c = self.cfg.INFINITE, -self.cfg.INFINITE
#          kp_move_lst = []
#          for mv,c in move_match:
#            if c >= max_c:
#              if action in ["LEFT"]:
#                max_mv = mv
#              elif action in ["RIGHT"]:
#                max_mv = -mv
#              if c > max_c:
#                kp_move_lst = []
#              else:
#                # return ties for further evaluation
#                kp_move_lst.append(max_mv)
#              max_c = c
#          print("move match:", max_mv, max_c, move_match)
#          return(kp_move_lst)

        return None

    def compare_homography(self,KP2,include_dist=True):
        print("num map,rot features:", len(map_features), len(rot_features))
        delta = []
        first_time_through = True
        # print("map_features:", map_features)
        print("len map_features:", len(map_features))
        for i, map_pt_lst in enumerate(map_features):
          map_pt = [int(map_pt_lst[0][0]), int(map_pt_lst[0][1])]
          # print("i map_pt:", i, map_pt)
          if not point_in_border(self.border, map_pt):
            # print("map pt not in border:", map_pt, self.border)
            mol=cv2.circle(mol,map_pt,3,(0,255,0),-1)
            continue
          cv2.circle(mol,map_pt,3,(255,0,0),-1)
          for j, rot_pt_lst in enumerate(rot_features):
            rot_pt = [int(rot_pt_lst[0][0]), int(rot_pt_lst[0][1])]
            if not point_in_border(rot_border, rot_pt):
              # print("rot pt not in border:", rot_pt, rot_border)
              rnm=cv2.circle(rnm,rot_pt,3,(0,255,0),-1)
              continue
            if first_time_through:
              rnm=cv2.circle(rnm,rot_pt,3,(255,0,0),-1)
            # distance needs to be directional, and consider both x,y separately
            # change of slope and line segment (sounds like a keypoint!)
            # min change of both varx & vary
            # the diff_distance_variation needs to be in the same direction
            dist = math.sqrt((map_pt[0]-rot_pt[0])**2+(map_pt[1]-rot_pt[1])**2)
            if (map_pt[0] > rot_pt[0]):
              dist = -dist 
            slope = self.arctan2((map_pt[0] - rot_pt[0]) , (map_pt[1]-rot_pt[1]))
            grouping = slope * dist
            delta.append([grouping, i, j])
        return rotation, off_h, off_w 
        return good, not_so_good

    def compare_kp(self,KP2,include_dist=True):
      if self.kp_mode == "FEATURE":
        good = []
        notsogood = []
        h,w = self.img.shape
        print("h,w:", h,w)
        for i, kp1 in enumerate(self.keypoints):
          for j, kp2 in enumerate(KP2.keypoints):
            dist1 = np.sqrt((h - kp1.pt[0])**2 + (w - kp1.pt[1])**2)
            dist2 = np.sqrt((h - kp2.pt[0])**2 + (w - kp2.pt[1])**2)
            if dist1 == 0:
              ratio = self.INFINITE
            else:
              ratio = abs(dist1-dist2)/dist1
            m = ((i, j, dist1),(i, j, dist2), ratio)
            if ratio <= .1:
              good.append(m)
            else:
              notsogood.append(m)
        return good, notsogood
      ################
      des2 = KP2.get_des()
      if self.descriptor is None or des2 is None:
        return [],[] 
      # BFMatcher with default params
      bf = cv.BFMatcher()
      matches = bf.knnMatch(self.descriptor,des2,k=2)
      # print("des info",self.descriptor, des2)
      print("len matches:", len(matches))
      # Apply ratio test
      good = []
      notsogood = []
      for m,n in matches:
          d = m.distance/n.distance
          if m.distance < 0.75*n.distance:
              good.append(((m.queryIdx, m.trainIdx, m.distance), (n.queryIdx, n.trainIdx, n.distance), d))
          else:
              notsogood.append(((m.queryIdx, m.trainIdx, m.distance), (n.queryIdx, n.trainIdx, n.distance), d))
      good = sorted(good, key=itemgetter(2))
      notsogood = sorted(notsogood, key=itemgetter(2))
      if not include_dist:
         good = [g[:-1] for g in good]
         notsogood = [g[:-1] for g in notsogood]
      return good, notsogood

    def get_n_match_kps(self, matches, KP2, n, return_list=True, border=None):
        KP1_kps = []
        KP2_kps = []
        skipped=0
        # print("num matches: ", len(matches))
        for i,kpm in enumerate(matches):
          # print("queryIdx:", kpm[0].queryIdx, kpm[1].queryIdx)
          # print("imgIdx:", kpm[0].imgIdx, kpm[1].imgIdx)
          # print("trainIdx:", kpm[0].trainIdx, kpm[1].trainIdx)
          # print("num kp:", len(self.KP.keypoints),len(KP2.keypoints))
          # print("kp1t:",self.keypoints[kpm[0].queryIdx].pt)
          # print("kp2q:",KP2.keypoints[kpm[1].trainIdx].pt)
          # print("dist:", kpm[0].distance, kpm[1].distance)
          ((queryIdx, trainIdx, dist), (n_queryIdx, n_trainIdx, n_dist), ratio) = kpm
          if i >= n+skipped:
            if return_list:
              KP1_kps = np.float32(KP1_kps)
              KP2_kps = np.float32(KP2_kps)
            return KP1_kps, KP2_kps
          kp1 = self.keypoints[queryIdx]
          kp2 = KP2.keypoints[trainIdx]
          if border is None or point_in_border(border,kp1.pt):
            if return_list:
              if kp1.pt not in KP1_kps and kp2 not in KP2_kps:
                KP1_kps.append([kp1.pt[0], kp1.pt[1]])
                KP2_kps.append([kp2.pt[0], kp2.pt[1]])
              else:
                skipped += 1
            else:
              if kp1.pt not in KP1_kps and kp2.pt not in KP2_kps:
                # KP1_kps.append(kp1.pt)
                # KP2_kps.append(kp2.pt)
                KP1_kps.append(kp1)
                KP2_kps.append(kp2)
              else:
                skipped += 1
          else:
            skipped += 1
          ###
        print("insufficient matches: ", (len(matches)-skipped))
        if return_list:
          KP1_kps = np.float32(KP1_kps)
          KP2_kps = np.float32(KP2_kps)
        return KP1_kps, KP2_kps

    def drawKeypoints(self, color=(0,255,0)):
        if self.kp_mode == "SIFT":
           return cv.drawKeypoints(self.img,self.keypoints,None,color=(0,255,0), flags=0)
        elif self.kp_mode == "ORB":
           return cv.drawKeypoints(self.img,self.keypoints,None,color=(0,255,0), flags=0)
        elif self.kp_mode == "SURF":
           return cv.drawKeypoints(self.img,self.keypoints,None,color=(0,255,0), flags=0)
           # return cv.drawMatchesKnn(self.img,self.get_kp(),
           #                        self.KP,self.get_kp(), good_matches, None,
           #                        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

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

    ############################
    # MAP KEYPOINT ANALYSIS FUNCTIONS
    ############################

    def evaluate_map_kp_offsets(self, map_KP, new_map_rot, new_map_rot_KP = None):
        shape, map_border = self.real_map_border(new_map_rot)
        if new_map_rot_KP is None:
          new_map_rot_KP = Keypoints(new_map_rot, kp_mode=map_KP.get_kp_mode())
        good_matches,notsogood = map_KP.compare_kp(new_map_rot_KP)
        print("kp_offsets: len good matches:", len(good_matches))
        map_pts, new_map_rot_pts = map_KP.get_n_match_kps(good_matches, new_map_rot_KP, 6, return_list=False, border=map_border)
        for pnt in new_map_rot_pts:
          pt = [int(pnt.pt[0]), int(pnt.pt[1])] # circle does [w,h], but KP also does [w,h]
          new_map = cv2.circle(new_map_rot,pt,3,(255,0,0),-1)
        delta_h, delta_w = 0,0
        x,y = 0,1
        for i, map_pt in enumerate(map_pts):
          # convert KP xy to image hw
          delta_h += (map_pt.pt[y] - new_map_rot_pts[i].pt[y])
          delta_w += (map_pt.pt[x] - new_map_rot_pts[i].pt[x])
          print("kp_off: ", i, int(delta_h/(i+1)), int(delta_w/(i+1)))
        if len(map_pts) > 0:
          delta_h = int(delta_h / len(map_pts))
          delta_w = int(delta_w / len(map_pts))
        print("avg kp_offsets hw: ", delta_h, delta_w) 
        if self.frame_num >= self.stop_at_frame:
          cv2.imshow("Matching KeyPoints", new_map_rot)
          cv2.waitKey(0)
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
        angle = self.rad_sum(start_angle, end_angle)/2 
        new_map_rot = self.rotate_about_robot(new_map, angle)

        delta_h, delta_w, map_pts, new_map_rot_pts = self.evaluate_map_kp_offsets(map_KP, new_map_rot)
        if len(map_pts) > 0 and (best_kp_delta_h is None or delta_h < best_kp_delta_h):
            best_kp_angle, best_kp_delta_h, best_kp_delta_w = angle, delta_h, delta_w 
            best_map_pts, best_new_map_pts = map_pts, new_map_rot_pts
            print("new best kp angle:",best_map_pts,best_new_map_pts)
        # we're now in radians, so need much smaller threshold
        diff_angle = self.rad_interval(max_angle, min_angle)
  
        if abs(self.rad_interval(max_angle,min_angle)) <= .001 or delta_h == 0:
          break
        elif delta_h > 0:
          min_angle = angle
        elif delta_h < 0:
          max_angle = angle
      return best_kp_angle, best_kp_delta_h, best_kp_delta_w, best_map_pts, best_new_map_pts

    def best_kp_angle(self, map_KP, new_map, start_angle, end_angle):
              self.map_overlay_KP = Keypoints(self.map_overlay)
              start_angle = .5
              end_angle = start_angle+np.pi/9
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
                  if self.frame_num >= self.stop_at_frame:
                    cv2.imshow("map kp", rotated_new_map_disp)
                  rotated_new_map_disp = rotated_new_map.copy()
                  rotated_new_map_disp_KP = Keypoints(rotated_new_map_disp)
                  rotated_new_map_disp = rotated_new_map_disp_KP.drawKeypoints()
                  if self.frame_num >= self.stop_at_frame:
                    cv2.imshow("best kp, rotated", rotated_new_map_disp)
                    cv2.waitKey(0)


    ######################################
    # old code specific to ORB
    ######################################

    # descriptor match implementations:
    # https://github.com/opencv/opencv/blob/master/modules/features2d/src/matchers.cpp
    def map_to_clusters(self, clusters):
      # image to cluster mapping
      kp_list = self.get_kp()
      # look through known KPs for matching descriptors for clusters
      for kp in kp_list:
        print("ARD: TODO map_to_clusters")
        # look through clusters for matching points

    def deep_copy_kp(self, KP, kp_list):
      # ARD TODO
      # deep copy descriptors?  Not sure this is correct
      n = 0
      des = KP.get_descriptor()
      num_desc = len(des)
      len_desc = len(des[0])
      # copy descriptors into continuous bytes
      s = [0]*(num_desc*len_desc)
      for i in range(num_desc):
        for c in range(len_desc):
          s[n] = des[i,c]
          n = n + 1
      # copy byte offset of each descriptor
      new_desc = [0]*len(s)
      for i in range(0,len(s)):
        new_desc[i]=int(s[i])

      # https://stackoverflow.com/questions/23561236/deep-copy-of-an-opencv2-orb-data-structure-in-python
      # Get descriptors from second y image using the detected points
      # from the x image
      # f, d = orb.compute(im_y, f)
      # direct deep copy of pixel feature locations
      f = KP.get_features()
      # centroid = self.cluster['centroid']
      # return [cv.KeyPoint(x = (k.pt[0]-centroid.x), y = (k.pt[1]-centroid.y),
      return [cv.KeyPoint(x = k.pt[0], y = k.pt[1],
            _size = k.size, _angle = k.angle,
            _response = k.response, _octave = k.octave,
            _class_id = k.class_id) for k in f if k in kp_list], new_desc


      return kp_list

    def get_features(self):
      return self.keypoints

    def get_descriptor(self):
      return self.descriptor

    # note: self called from w clusters, compare with curr full pc clusters 
    # w is from the previous pc analysis.
    def compare_cluster_kp(self,pc_KP,kp_c_pc_mapping):
          pts      = []
          distance = []
          pc_clust = []
          pc_kp_info = self.compare_kp(map_KP)
          for [pc_kp,c_kp,dist] in pc_kp_info:
            # if pc_c is None then the keypoint was not in the cluster shape
            for [pc_kp,pc_c,pc_pt,obb] in kp_info:
              print("pc_kp/c_kp:", pc_kp, c_kp)
              if (pc_kp[0] == c_kp[0] and pc_kp[1] == c_kp[1]):
                pc_clust.append(pc_c)
                distance.append(dist)
                pts.append([pc_pt, pc_kp, c_kp])
                print("bf_match: matching kp")
                # only need top 3 matches for indiv clusters
                if len(pc_clust) == 3:
                  break
          return pc_clust, distance, pts
        # Rejected compare_kp code
        # if i < len(bf_matches) - 1:
        #   ratio = bf_matches[i].distance/bf_matches[i+1].distance
        # else:
        #   ratio = None
        # if i < len(matches) - 1 and m.distance < 0.75 * matches[i+1].distance:
        # if ratio <= .75:

    # TODO: move some kp code from world.py and keypoints.py to cluster.py
    # note: self called from individual pc clusters
    def compare_cluster_kp(self,pc_KP,kp_c_pc_mapping):
          pts      = []
          distance = []
          pc_clust = []
          pc_kp_info = self.compare_kp(pc_KP)
          for [pc_kp,c_kp,dist] in pc_kp_info:
            # if pc_c is None then the keypoint was not in the cluster shape
            for [pc_kp,pc_c,pc_pt,obb] in kp_c_pc_mapping:
              print("pc_kp/c_kp:", pc_kp, c_kp)
              if (pc_kp[0] == c_kp[0] and pc_kp[1] == c_kp[1]):
                pc_clust.append(pc_c)
                distance.append(dist)
                pts.append([pc_pt, pc_kp, c_kp])
                print("bf_match: matching kp")
                # only need top 3 matches for indiv clusters
                if len(pc_clust) == 3:   
                  break
          return pc_clust, distance, pts



    def compare_w_pc_kp(self, map_KP, map_pc_info, kp_w_info):
          if pc_KP == None or len(kp_pc_info) == 0 or len(kp_w_info) == 0:
            return None
          kp_w_pc_info_match = []
          w_pc_kp_info = self.compare_kp(pc_KP)
          if len(w_pc_kp_info) == 0:
            return None
          print("# w,pc,kpmatches", len(self.get_kp()), len(pc_KP.get_kp()), len(w_pc_kp_info))
          for w_pc_i,[w_kp1,pc_kp1,w_dist1] in enumerate(w_pc_kp_info):
            print("w_kp1",w_kp1," matches pc_kp1",pc_kp1," with distance", w_dist1)
            pc_clust2, pc_dist2, pc_pts2 = kp_pc_info
            print("w_clust pc_clust distance", w_pc_i, pc_clust2, pc_dist2)
            # ([pc_pt, pc_kp, c_kp])
            pc_pt2, pc_w_kp2, pc_pc_kp2 = None, None, None
            for j in range(len(pc_pts2)):
              if len(pc_pts2[j]) == 3:
                # 1 list per kp pt, may be empty
                pc_pt2, pc_w_kp2, pc_pc_kp2 = pc_pts2[j]
                print(j, "pc_pts2:",pc_pt2, pc_w_kp2, pc_pc_kp2)
              # else:   # probably empty [[], [],...,[]]
              #  print("pc_pts2:",pc_pts2)
            w_clust2, w_dist2, w_pts2 = kp_w_info
            w_pt2, w_w_kp2, w_pc_kp2 = None, None, None
            for j in range(len(w_pts2)):
              if len(w_pts2[j]) == 3:
                # 1 list per kp pt, may be empty
                w_pt2, w_w_kp2, w_pc_kp2 = w_pts2[j]
                print(j, "w_pts2:",w_pt2, w_w_kp2, w_pc_kp2)
              # else:   # probably empty [[], [],...,[]]
              #   print("w_pts2:",w_pts2)

            # so, w_kp1 == pc_kp1, find matching 3d pts for w_kp2 and pc_kp2
            # Note: the x/y locations of w_kp1 and pc_kp1 aren't same
            pc_match = None
            for pc_j in range(len(pc_clust2)):
              if (pc_kp1 != None and pc_pc_kp2 != None 
                 and pc_kp1[0] == pc_pc_kp2[pc_j][0] 
                 and pc_pc_kp1 != None and pc_pc_kp2 != None
                 and pc_pc_kp1[1] == pc_pc_kp2[pc_j][1]):
                pc_match = pc_j
            w_match = None
            for w_k in range(len(w_clust2)):
              if (w_kp1 != None and w_pc_kp2 != None 
                 and w_kp1[0] == w_pc_kp2[w_k][0] 
                 and w_pc_kp1 != None and w_pc_kp2 != None
                 and w_pc_kp1[1] == w_pc_kp2[w_k][1]):
                w_match = w_k

            if (pc_match == None or w_match == None):
              print("w_pc_kp: no match", w_pc_i, pc_w_kp2, pc_pc_kp2)
              kp_w_pc_info_match.append([w_kp1, pc_kp1, w_dist1, w_pc_i, pc_w_kp2, pc_pc_kp2, None])
            else:
              # did kp move?
              dist = distance_3d(w_pt2[w_k], pc_pt2[pc_j])
              print("w_pc_kp: movement", w_pc_i, dist, w_pt2[w_k], pc_pt2[pc_j])
              kp_w_pc_info_match.append([w_kp1, pc_kp1, w_dist1, w_pc_i, w_w_kp2, w_pc_kp2, dist])
          return kp_w_pc_info_match

    def old_compare_kp_orb(self,KP2):
      # from matplotlib import pyplot as plt

      # find the keypoints and descriptors with ORB
      kp1 = self.get_kp()            
      des1 = self.get_descriptor()   
      kp2 = KP2.get_kp()             
      des2 = KP2.get_descriptor()    
      # create BFMatcher object
      bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
      # Match descriptors.
      bf_matches = bf.match(self.get_des,des2)
      bf_matches = sorted(bf_matches, key = lambda x:x.distance)
      # print("bf_matches",len(bf_matches), len(kp1), len(kp2))

      # Initialize lists
      pc_kp_info = []
      # https://stackoverflow.com/questions/31690265/matching-features-with-orb-python-opencv
      # This test rejects poor matches by computing the ratio between the 
      # best and second-best match. If the ratio is below some threshold,
      # the match is discarded as being low-quality.
      # Sort them in the order of their distance. Lower distances are better.
      for i,m in enumerate(bf_matches):
        # print("bf_match distance", m.distance)
        # good.append([kp1])
        # Get the matching keypoints for each of the images
        # queryIdx - row of the kp1 interest point matrix that matches
        # trainIdx - row of the kp2 interest point matrix that matches
        img1_idx = m.queryIdx
        img2_idx = m.trainIdx
        pc_kp_info.append([kp1[img1_idx], kp2[img2_idx], m.distance, ])
      return pc_kp_info
