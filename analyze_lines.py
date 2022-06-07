from utilradians import *
from utilborders import *
from config import *
from cv_analysis_tools import *
import scipy.stats as stats
from alset_state import *
from arm_nav import *
from matplotlib import pyplot as plt
import cv2
import numpy as np
import copy
from math import sin, cos, pi, sqrt
from utilborders import *
from cv_analysis_tools import *
from dataset_utils import *
from PIL import Image, ImageChops 
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

########################
# Map Line Analysis
#
# On a tabletop and roads, edges are typically well deliniated and visible.
# For the initial tabletop apps, the line analysis has proven more reliable
# than the keypoint analysis.
#
# Note: using stego instead of color_quant_num_clust 
#################################
class AnalyzeLines():

    #########################
    # Map Line Analysis 
    #########################
    def evaluate_lines_in_image(self, image, border):
      linesP, imglinesp = self.cvu.get_lines(image)
      # print("border:", border)
      max_num_lines = 10
      lines = []
      line = {}
      for i, [[l0,l1,l2,l3]] in enumerate(linesP):
        # make sure it's not just a line along the border
        if not(border is None or line_in_border(border, (l0,l1), (l2,l3))):
          print("not line in border")
          continue
        line["line"] = [l0,l1,l2,l3]
        dx = l0 - l2
        dy = l1 - l3
        line["len"] = np.sqrt(dx**2 + dy**2)
        line["slope"] = rad_arctan2(dx, dy)
        line["intersect"] = []
        # limit to top X longest lines
        min_len = self.cfg.INFINITE
        min_len_num = self.cfg.INFINITE
        if len(lines) < max_num_lines:
          lines.append(line.copy())
        else:
          for ln_num, ln in enumerate(lines):
            if min_len > ln["len"]:
              min_len_num = ln_num
              min_len = ln["len"]
          if min_len < line["len"]:
            lines[min_len_num] = line.copy()

      for i,i_line in enumerate(lines):
        [l0,l1,l2,l3] = i_line["line"] 
        i_ls = LineString([(l0, l1), (l2, l3)])
        for j,j_line in enumerate(lines):
          if i >= j:
            continue
          # matching intersections within borders can determine proper angle as the points will line up.
          # intersections will have same distance to robot location.
          [l0,l1,l2,l3] = j_line["line"] 
          j_ls = LineString([(l0, l1), (l2, l3)])
          intersect = i_ls.intersection(j_ls)
          if (not intersect.is_empty and intersect.geom_type.startswith('Point') and (border is None or 
              point_in_border(border, [round(intersect.coords[0][0]), round(intersect.coords[0][1])]))):
            print("intersect:", intersect, i, i_ls, j, j_ls)
            i_line["intersect"].append([i,j, [round(intersect.coords[0][0]), round(intersect.coords[0][1])]])
          if not intersect.is_empty and intersect.geom_type.startswith('Line'):
            # combine the lines into one
            print("Combine two lines into one:", i_ls, j_ls, intersect)

      print("lines:", lines)
      return lines

    # used by mapping
    def best_line_pos(self, prev_map, curr_map, action, robot_location=None, frame_num=0):
      def angle_match_count(angle_match, rads, radian_thresh):
          found = False
          for i, [a,c] in enumerate(angle_match):
            if rad_interval(rads, a) <= radian_thresh:
              found = True
              angle_match[i][1] += 1
          if not found:
            angle_match.append([rads, 1])
          return angle_match

      def offset_h_match_count(offset_h_match, offset_h):
          found = False
          for i, [h,c] in enumerate(offset_h_match):
            if abs(offset_h - h) <= 1:
              found = True
              offset_h_match[i][1] += 1
          if not found:
            offset_h_match.append([offset_h, 1])
          return offset_h_match

      def  offset_h_lines(offset_match, cm_ln, pm_ln, robot_location):
          # For the robot w location, compute offset h 
          # h1 - h2 = m1*w1  - m2*w2
          robot_w = robot_location[1]
          offset_h = robot_w * (cm_ln[1]-cm_ln[3])/(cm_ln[0]-cm_ln[2]) - robot_w * (pm_ln[1]-pm_ln[3])/(pm_ln[0]-pm_ln[2])
          print("offset_h @ robot_w", offset_h, robot_w, (cm_ln[1]-cm_ln[3])/(cm_ln[0]-cm_ln[2]), (pm_ln[1]-pm_ln[3])/(pm_ln[0]-pm_ln[2]))
          offset_h_match_count(offset_match, offset_h)
          return offset_h

      def offset_h_intersect(offset_match, pt1, pt2):
          offset_h = pt1[0] - pt2[0]
          offset_h_match_count(offset_match, offset_h)
          return offset_h

      def best_offset_h(offset_match):
          if len(offset_match) >= 1:
            offset_h_list = [off[0] for off in offset_match]
            return offset_match[0][0]
            # for h1, c1 in offset_match:
          else:
            return None

      confidence = 1
      radian_thresh = self.cfg.RADIAN_THRESH
      I_intersect_rads = None 
      J_intersect_rads = None
      if self.color_quant_num_clust is not None:
        minK = self.color_quant_num_clust
        maxK = self.color_quant_num_clust
      else:
        minK = 3
        maxK = 6
      print("minK maxK", minK, maxK)
      comparison = {}
      for K in range(minK, maxK):
        border_shape, border = real_map_border(prev_map)
        pm_floor_clusters = self.cvu.color_quantification(prev_map, K)
        cm_floor_clusters = self.cvu.color_quantification(curr_map, K)
        # Phase 1: within same images, find intersections and lines slopes
        cm = self.evaluate_lines_in_image(cm_floor_clusters, border)
        pm = self.evaluate_lines_in_image(pm_floor_clusters, border)
        comparison["intersect_rads"] = []
        comparison["line_rads"] = []
        # Phase 2: across 2 images, make full comparison of
        #          angles of intersections and line slopes
        floor_clusters = cm_floor_clusters.copy()
        for cm_i, cm_line in enumerate(cm):
          cv2.line(floor_clusters, (cm_line["line"][0], cm_line["line"][1]), (cm_line["line"][2], cm_line["line"][3]), (0,255,0), 3, cv2.LINE_AA)
          for pm_i, pm_line in enumerate(pm):
            # 
            #-- compare intersections
            for [cm_i1, cm_i2, cm_pt] in cm_line["intersect"]:
              for [pm_j1, pm_j2, pm_pt] in pm_line["intersect"]:
                angle = rad_isosceles_triangle(cm_pt, robot_location, pm_pt)
                comparison["intersect_rads"].append([cm_i, pm_i, cm_pt, pm_pt, angle])
            comparison["line_rads"].append([cm_i, pm_i, rad_dif(cm_line["slope"], pm_line["slope"])])

            # todo: compare relative line positions for non-intersecting lines
            #   - todo: provide confidence level of matches
            # todo: compare line lens
            #   - todo: determine multiple cm lines map to same pm line 
            #   - todo: determine multiple pm lines map to same cm line 
            # todo: bad_match
        cv2.imshow("floor lines", floor_clusters)
        # cv2.waitKey(0)

        # Phase 3: match lines
        print("intersect_rads, line_rads:", comparison["intersect_rads"], comparison["line_rads"])
        angle_match = []
        offset_match = []
        for i, i_list in enumerate(comparison["intersect_rads"]):
          I_cm_i, I_pm_i, I_cm_pt, I_pm_pt, I_intersect_rads = i_list
          for j, j_list in enumerate(comparison["intersect_rads"]):
            if i >= j:
              continue
            if I_intersect_rads is None or J_intersect_rads is None:
              continue
            J_cm_i, J_pm_i, J_cm_pt, J_pm_pt, J_intersect_rads = j_list
            # compare the differences in slope, not the slopes themselves
            if rad_interval(I_intersect_rads, J_intersect_rads) <= radian_thresh:
              print("good intersect match:",I_intersect_rads, J_intersect_rads, I_cm_i, J_cm_i, I_pm_i, J_pm_i)
              angle_match = angle_match_count(angle_match, I_intersect_rads, radian_thresh)
              offset_h = offset_h_intersect(offset_match, J_cm_pt, J_pm_pt)
            # else:
            #  print("bad intersect match:",I_intersect_rads, J_intersect_rads)
          for j, [J_cm_i, J_pm_j, J_rads] in enumerate(comparison["line_rads"]):
            if i >= j:
              continue
            if I_intersect_rads is None or J_rads is None:
              continue
            if len(cm[J_cm_i]["intersect"]) == 0 or len(pm[J_pm_j]["intersect"]) == 0:
              continue
            j_rads = j_list[2]
            if rad_interval(I_intersect_rads, J_rads) <= radian_thresh:
              print("good intersect-slope match:",I_intersect_rads, J_rads, I_cm_i, J_cm_i, J_pm_j, I_pm_i,  cm[J_cm_i], pm[J_pm_j])

              angle_match = angle_match_count(angle_match, I_intersect_rads, radian_thresh)
              [tmp_i1,tmp_j1, cm_pt1] = cm[J_cm_i]["intersect"][0]
              [tmp_i2,tmp_j2, pm_pt2] = pm[J_pm_j]["intersect"][0]
              offset_h = offset_h_intersect(offset_match, cm_pt1, pm_pt2)
            # else:
            #   print("bad intersect match:",I_intersect_rads, J_rads)
        for i, [I_cm_i, I_pm_j, I_rads] in enumerate(comparison["line_rads"]):
          for j, [J_cm_i, J_pm_j, J_rads] in enumerate(comparison["line_rads"]):
            if i >= j:
              continue
            if I_cm_i == J_cm_i and I_pm_j == J_pm_j:
              continue
            if I_rads is None or J_rads is None:
              continue
            # this is a delta slope, but it doesn't need to be less than radian thresh.
            # the lines should be verified to be the same based on location, length, and slope.
            # Ideally the slope would be similar to pevious turn in same direction.
            if rad_interval(I_rads, J_rads) <= radian_thresh:
              print("good slope match:",I_rads, J_rads, I_cm_i, I_pm_j, J_cm_i, J_pm_j)
# good slope match: 0.1853479499956947 0.18426569933597836 22 54 236 77
#                                                                J_cm_i


              angle_match = angle_match_count(angle_match, I_rads, radian_thresh)
              cmln = cm[I_cm_i]["line"]
              pmln = pm[J_pm_j]["line"]
              offset_h = offset_h_lines(offset_match, cmln, pmln, robot_location)
            # else:
            #   print("bad intersect match:",I_rads, J_rads)

        if action in ["LEFT", "RIGHT"]:
          # return angle
          if len(angle_match) >= 1:
            # Was: do a count; now caller does mse comparison
            angle_list = [a[0] for a in angle_match]
            print("angle match:", angle_match, angle_list)
            return(angle_list)

            # max_a, max_c = self.cfg.INFINITE, - self.cfg.INFINITE
            # for a,c in angle_match:
            #   if c > max_c:
            #     max_c = c
            #     max_a = a
            # print("Angle match:", max_a, max_c, angle_match)
          # elif len(comparison["intersect_rads"]) >= 1:
          #   # ntersection_rads included in angle_match
          #   angle_list = [a[0] for a in angle_match]
          #   return(comparison["intersect_rads"][0][4]) 
          # elif len(comparison["line_rads"]) >= 1:
          #   return(comparison["line_rads"][0][2]) 
          else:
            continue
        elif action in ["FORWARD", "REVERSE"]:
          # offset_h = best_offset_h(offset_match)
          if offset_match is None or len(offset_match) == 0:
            return None
          offset_h_list = [off[0] for off in offset_match]
          return offset_h_list
      return None


    # used by mapping
    def compare_lines(self, map, new_map, action="LEFT", mode="ROTATE", frame_num=0):
      self.curr_frame_num = frame_num
      for K in range(3,6):
        floor_clusters = self.cvu.color_quantification(map, K)
        linesP, imglinesp = self.cvu.get_lines(floor_clusters)
        if linesP is None:
          continue
        shape, map_border = real_map_border(map)
        max_dist, map_line, map_dx, map_dy, map_slope, in_brdr_cnt = self.cvu.find_longest_line(linesP, map_border)
        print("#### Map Line Analysis: ", len(linesP), in_brdr_cnt) 
        print("in border line cnt: ", in_brdr_cnt)
        if map_dx is None or map_dy is None:
            print("no map line: ", K)
            continue
        self.color_quant_num_clust = K
        self.last_line = map_line
        print("longest line",map_line)
        if frame_num >= 180:
          cv2.line(floor_clusters, (map_line[0], map_line[1]), (map_line[2], map_line[3]), (0,255,0), 3, cv2.LINE_AA)
          pt = [int(self.robot_location[1]), int(self.robot_location[0])] # circle uses [w,h]
          imglinesp = cv2.circle(floor_clusters,pt,3,(255,0,0),-1)
          # cv2.imshow("best map line", floor_clusters)
          # cv2.imshow("all map lines", imglinesp)
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
      best_line, best_line_angle = None, None
      best_mse, best_ssim, best_lbp = self.INFINITE, self.INFINITE, self.INFINITE
      score_hist = []

      # do binary search
      angle_min = start_angle
      angle_max = end_angle
      while True:
        angle = rad_sum(angle_min,angle_max) / 2
        print("min, mid, max angle:", angle_min, angle, angle_max)
        if self.color_quant_num_clust > 0:
          floor_clusters = self.cvu.color_quantification(new_map, self.color_quant_num_clust)
          new_map_rot_cq = self.rotate_about_robot(floor_clusters, angle)
        new_map_rot = self.rotate_about_robot(new_map, angle)
        shape, map_border = real_map_border(new_map_rot)
        linesP, imglinesp = self.cvu.get_lines(new_map_rot)

        #################
        mse_score, ssim_score, lbp_score = self.score_map(map, new_map_rot)
        score_hist.append([angle, mse_score, ssim_score, lbp_score])
        if mse_score < best_mse:
          best_mse = mse_score
          best_mse_angle = angle
        if ssim_score < best_ssim:
          best_ssim = ssim_score
          best_ssim_angle = angle
        if lbp_score < best_lbp:
          best_lbp = lbp_score 
          best_lbp_angle = angle
        print("mse ssim lbp: ", (best_mse, round(best_mse_angle, 4)),
                                (best_ssim, round(best_ssim_angle,4)), 
                                (best_lbp, round(best_lbp_angle, 4)))

        #################
        # print(linesP)
        in_brdr_cnt = 0
        min_slope_dif = self.INFINITE
        rot_min_slope_dif = self.INFINITE
        max_dist = 0
        slope_hist = []
        for [[l0,l1,l2,l3]] in linesP:
          if line_in_border(map_border, (l0,l1), (l2,l3)):
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

            diff_slope = rad_interval(map_slope, self.arctan2(dx, dy))
            slope_hist.append([angle, diff_slope, in_brdr_cnt])
            print("map, cur slopes:", map_slope, self.arctan2(dx,dy))
            # print("diff slope/min:", diff_slope, min_slope_dif)
            if abs(diff_slope) < abs(rot_min_slope_dif):
              rot_min_slope_dif = diff_slope
            if abs(diff_slope) < abs(min_slope_dif):
              min_slope_dif = diff_slope
              min_slope_dif_angle = angle
              best_line = [l0,l1,l2,l3]
              if dy != 0:
                print("best line so far:", best_line, angle, diff_slope, "dxy:", map_slope, (dx/dy))
        print("in border line cnt: ", in_brdr_cnt)
        if (dist_line[0] == best_line[0] and dist_line[1] == best_line[1]
            and dist_line[2] == best_line[2] and dist_line[3] == best_line[3]):
           same_line = True
       
        # if frame_num == 113:
        #   cv2.imshow("bad rotated new map:",imglinesp)
        #   cv2.waitKey(0)
        if frame_num >= self.stop_at_frame:
          # pt = [int(self.robot_location[1]), int(self.robot_location[0])] # [w,h]
          # imglinesp = cv2.circle(imglinesp,pt,3,(255,0,0),-1)
          # cv2.imshow("rotated new map:",imglinesp)
          # cv2.waitKey(0)
          pass
        if in_brdr_cnt == 0:
          # did we rotate too far?
          angle_max = angle
          break
        elif ((min_slope_dif == 0 or rad_interval(angle_max,angle_min) <= .001)
          and (round(best_mse_angle,3) == round(min_slope_dif_angle,3) or
               round(best_ssim_angle,3) == round(min_slope_dif_angle,3) or
               round(best_lbp,3) == round(min_slope_dif_angle,3))):
          print("angle min max: ", angle_min, angle_max, min_slope_dif)
          # angle min max:  14.970703125 15 16.365384615384613
          # how is min_slope_dif bigger than max angle?
          #  - angle is angle of rotation, min slope dif is slope of resulting line
          break
        elif rot_min_slope_dif > 0:
          angle_min = angle
        elif rot_min_slope_dif <= 0:
          angle_max = angle
      print("best mse: ", best_mse, best_mse_angle)
      print("best ssim:", best_ssim, best_ssim_angle)
      print("best lbp: ", best_lbp, best_lbp_angle)
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

    # used by mapping
    def get_height_dif_by_line(self, rnm, map_line, map_slope, offset_width=None):
        floor_clusters = self.cvu.color_quantification(rnm, self.color_quant_num_clust)
        shape, new_map_rot_border = real_map_border(rnm)
        linesP, imglinesp = self.cvu.get_lines(rnm)
        # print(linesP)
        print("map line:", map_line)
        min_slope_dif = self.INFINITE
        for [[l0,l1,l2,l3]] in linesP:
          if line_in_border(new_map_rot_border, (l0,l1), (l2,l3)):
            dx = l0 - l2
            dy = l1 - l3
            dist = np.sqrt(dx**2 + dy**2)
            # diff_slope = map_slope - np.arctan2(dx, dy)
            diff_slope = rad_dif(map_slope, self.arctan2(dx, dy))
            print("map, cur slopes:", map_slope, self.arctan2(dx,dy), [l0,l1,l2,l3])
            if abs(diff_slope) < abs(min_slope_dif):
              min_slope_dif = diff_slope
              best_line = [l0,l1,l2,l3]
              best_slope = self.arctan2(dx, dy)
              print("best match: ", diff_slope, best_slope)
        if (map_line[3] - map_line[1]) == 0:
            map_b = self.INFINITE
        elif offset_width is not None:
            robot_w = offset_width
            # y = mx + b ; b = y - mx
            # mb = map_line[2] - map_line[1]*(map_line[1]-map_line[3])/(map_line[0]-map_line[2])
            # nb = best_line[2] - best_line[1]*(best_line[1]-best_line[3])/(best_line[0]-best_line[2])
            # off_b = mb - nb
            # print("mb nb off_b:", mb, nb, off_b)

            off_b = robot_w * (map_line[1]-map_line[3])/(map_line[0]-map_line[2]) - robot_w * (best_line[1]-best_line[3])/(best_line[0]-best_line[2])
        else:
            # For the robot w location, compute the h location
            # h1 = m1*w1 + b1   ; h2 = m2*w2 + b2
            # h1 - m1*w1 = h2 - m2*w2
            # h1 - h2 = m1*w1  - m2*w2
            robot_w = self.robot_location[1]
            off_b = robot_w * (map_line[1]-map_line[3])/(map_line[0]-map_line[2]) - robot_w * (best_line[1]-best_line[3])/(best_line[0]-best_line[2])
            print("off_b", off_b)
        return round(off_b)

    ################
    # line analysis for tracking objects, rectangular boundaries, tabletops
    ################
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
    
    def get_frame_lines(self, curr_img, gripper_img = None):
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
    
    # Probably should delete
    def analyze_all_frames(self, dataset, frame_lines, actions, gripper_img, drop_off_img, img_paths):
        angle_sd = SortedDict()
        box_line_intersections = []
        arm_pos = []
        num_ds = len(frame_lines)
        prev_lines = None
        hough_lines = None
        num_passes = 4
        # ret, gripper_img = cv2.threshold(gripper_img, 100, 255, cv2.THRESH_TOZERO)
        ret, drop_off_img = cv2.threshold(drop_off_img, 100, 255, cv2.THRESH_TOZERO)
        line_group = [[]]  # includes gripper line_group[0]
        gripper_lines = self.get_hough_lines(gripper_img)
        gripper_lines_image = np.zeros_like(gripper_img)
        min_gripper_y = 100000000
        for line in gripper_lines:
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
        drop_off_gray = cv2.cvtColor(drop_off_img, cv2.COLOR_BGR2GRAY)
        drop_off_lines = self.get_hough_lines(drop_off_gray)
        drop_off_lines_image = np.zeros_like(drop_off_gray)
        for line in drop_off_lines:
          for x1,y1,x2,y2 in line:
            cv2.line(drop_off_lines_image,(x1,y1),(x2,y2),(255,0,0),5)
        cv2.imshow("drop_off lines", drop_off_lines_image)
        # cv2.waitKey(0)
    
        action_cnt = -1
        for ds_num, ds in enumerate(dataset):
          for pass_num in range(num_passes):
            num_imgs = len(frame_lines[ds_num])
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
              hough_lines = frame_lines[ds_num][img_num]
              if hough_lines is None:
                hough_lines = []
              if img_num > 0 and frame_lines[ds_num][img_num-1] is not None:
                prev_hough_lines = frame_lines[ds_num][img_num-1]
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
                      gripper_lines.append(copy.deepcopy(g_line))
                      # line_group[0].append(g_line)
                      found = True
                      break
                  # if found:
                  #   continue
                  for g_line in gripper_lines:
                    if len(g_line) == 0:
                      continue
                    if self.is_same_line(h_line, g_line):
                      # gripper_lines.append(g_line)
                      gripper_lines.append(copy.deepcopy(g_line))
                      # line_group[0].append(g_line)
                      found = True
                      break
                  # if found:
                  #   continue
    
                  for g_line in prev_hough_lines:
                    if len(g_line) == 0:
                      continue
                    if self.is_same_line(h_line, g_line):
                      gripper_lines.append(copy.deepcopy(g_line))
                      found = True
                  # if found:
                  #   continue
    
                if pass_num == 1:
                #####################################
                # process unknown lines
                #####################################
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
                      hough_lines = frame_lines[is_ds_num][is_img_num]
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
                hough_lines = frame_lines[ds_num][img_num]
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

    def transform_end_points(self, from_frame_num, from_line_end_points, to_frame_num):
        transformed_lines = []
        for fr_num in range(from_frame_num, to_frame_num):
          if self.M_inv is not None:
            self.M_inv[fr_num]
            # inverse matrix of simple rotation is reversed rotation.
            points = []
            for x1,y1,x2,y2 in from_line_end_points:
              # points = np.array([[35.,  0.], [175., 0.], [105., 200.], [105., 215.], ])
              points.append([x1,y1])
              points.append([x2,y2])
            # add ones
            ones = np.ones(shape=(len(points), 1))
            points_ones = np.hstack([points, ones])
            # transform points
            transformed_points = M_inv.dot(points_ones.T).T
            transformed_lines = []
            x = 1
            for x,y in transformed_points:
              if x == 1:
                ln = [x,y]
              else:
                ln.append(x)
                ln.append(y)
                transformed_lines.append(ln)
              x = 3-x
          else:
            transformed_lines = []
            for i, [x1,y1,x2,y2] in enumerate(line_end_points):
              [delta_x,delta_y] = self.pix_moved[frame_num]
              transformed_lines.append[x1+delta_x, y1+delta_y, x2+delta_x, y2+delta_y]
        return transformed_lines

    def save_movement(self, frame_num, action, base_movement, arm_movement):
        robot_location = None
        while len(self.M_inv) <= frame_num:
          self.M_inv.append(None)
        while len(self.pix_moved) <= frame_num:
          self.pix_moved.append(None)
        while len(self.robot_loc) <= frame_num:
          self.robot_loc.append(None)
        if action in ["LEFT","RIGHT"]:
          if base_movement is not None:
            pixels_moved, angle_moved, robot_location = base_movement
            self.M_inv[frame_num] = cv2.getRotationMatrix2D(robot_location, radians_to_degrees(angle_moved), 1)
          self.pix_moved[frame_num] = None
        else:
          if base_movement is not None:
            pixels_moved, angle_moved, robot_location = base_movement
          elif arm_movement is not None:
            pixels_moved, angle_moved, robot_location = arm_movement
          else:  # gripper movement
            print("None base and arm movement", frame_num, action)
            pixels_moved = None
            robot_location = self.robot_loc[frame_num-1]
          self.M_inv[frame_num] = None
          self.pix_moved[frame_num] = pixels_moved
        if robot_location is not None:
          self.robot_loc[frame_num] = copy.deepcopy(robot_location)

    def black_out_gripper(self, plot_img, stego_gripper):
        print("stego_gripper", stego_gripper)
        if stego_gripper is None:
          return plot_img 
        img1 = (plot_img > 127).astype(int) 
        img2 = np.logical_not((stego_gripper > 127).astype(int))
        black_out_plot_img = np.logical_and(img1,img2)
        return black_out_plot_img 

    def analyze(self, ds_num, frame_num, curr_plot_img, curr_img, prev_img, action, base_movement, arm_movement, gripper, is_stego = False):
        # movement is [move_vert, rot_angle]
        # arm_movement is verticle
        # gripper is pixel mapping
        # frame_line
        #   ds_num, frame_num => [{},{}...]
        #     line_num
        #     type: stego vs img edge
        #     hough_line
        #     angle
        #     same_frame_broken_line  [[line#, extended_line]]
        #     same_frame_parallel [[line#, dist]]
        #     same_frame_intersect [[line#, xy_intersect]]
        #     dif_frame_broken_line [[frame, line#, extended_line]]
        #     dif_frame_parallel [[frame, line#, dist]]
        #     prev_dif_frame_same_line [[frame_num, line_num]] 
        #     next_dif_frame_same_line [[frame_num, line_num]]
        #########################
        # Initialize
        # note: analysis doesn't continue across datasets 
        #########################
        self.save_movement(frame_num, action, base_movement, arm_movement)
        while ds_num >= len(self.frame_lines):
          self.frame_lines.append([])
          self.stego_gripper.append([])
        while frame_num >= len(self.frame_lines[ds_num]):
          self.frame_lines[ds_num].append([])
          self.stego_gripper[ds_num].append(None)
        self.stego_gripper[ds_num][frame_num] = copy.deepcopy(gripper)
        # remove gripper
        curr_img_no_gripper = self.black_out_gripper(curr_plot_img, gripper)
        if gripper is not None:
          cv2.imshow("gripper", gripper)
          # cv2.imshow("curr_img_no_gripper", curr_img_no_gripper)
          plt.imshow(curr_img_no_gripper)
          gray = cv2.cvtColor(curr_img.copy(), cv2.COLOR_BGR2GRAY)
          curr_lines = self.get_hough_lines(gray)
          curr_lines_image = self.get_hough_lines_img(curr_lines, gray)
          self.display_lines("curr_lines_no_gripper", curr_lines, curr_img)
          cv2.waitKey(30)
          for i,ln in enumerate(curr_lines):
            self.frame_lines[ds_num][frame_num].append({})
            fl = self.frame_lines[ds_num][frame_num][i]
            fl["line_num"] = i
            fl["hough_line"] = copy.deepcopy(ln)
            fl["angle"] = np.arctan2((ln[0][0]-ln[0][2]), (ln[0][1]-ln[0][3]))
            fl["is_stego"] = is_stego
            fl["same_frame_broken_line"] = []
            fl["same_frame_parallel_line"] = []
            fl["same_frame_intersecting_line"] = []
            fl["dif_frame_broken_line"] = []
            fl["dif_frame_parallel_line"] = []
            fl["dif_frame_same_line"] = []
        else:
          print("gripper is None")
          curr_lines = None
  
        ##########################
        # same frame analysis
        ##########################
        for i1, fr_ln1 in enumerate(self.frame_lines[ds_num][frame_num]):
          h_line1 = fr_ln1["hough_line"]
          for i2, fr_ln2 in enumerate(self.frame_lines[ds_num][frame_num]):
            if i2 <= i1:
              continue
            ##############
            # broken lines
            h_line2 = fr_ln2["hough_line"]
            extended_line = self.is_broken_line(h_line1, h_line2)
            if extended_line is not None:
              fr_ln1["same_frame_broken_line"].append([i2, extended_line])
              fr_ln2["same_frame_broken_line"].append([i1, extended_line])
              print("same_frame_broken_line", i1, i2, extended_line)
            ################
            # parallel lines 
            dist = self.parallel_dist(h_line1, h_line2)
            if dist is not None:
              fr_ln1["same_frame_parallel_line"].append([i2, dist])
              fr_ln2["same_frame_parallel_line"].append([i1, (-dist[0],-dist[1])])
              print("same_frame_parallel_line", i1, i2, dist)
            ################
            # intersecting lines (circular linked list)
            xy_intersect = self.line_intersection(h_line1, h_line2)
            if xy_intersect is not None:
              fr_ln1["same_frame_intersecting_line"].append([i2, xy_intersect])
              fr_ln2["same_frame_intersecting_line"].append([i1, xy_intersect])
              print("same_frame_intersecting_line", i1, i2, xy_intersect)

        ##########################
        # inter-frame analysis
        ##########################
        if curr_lines is not None:
          from_lines = copy.deepcopy(curr_lines)
          matched_lines = []
          for fn in range(frame_num):
            if len(matched_lines) == len(curr_lines):
              break
            from_frame = frame_num - fn
            to_frame = frame_num - fn - 1
            transformed_lines = self.transform_end_points(from_frame, from_lines, to_frame)
            for i1, fr_ln1 in enumerate(transformed_lines):
              if i1 in matched_lines:
                continue
              # check if transformed line is out of bounds
              for ti in range(4):
                if (fr_ln1[0][ti] < 0 or fr_ln1[0][ti] > 224):
                  fr_ln1["dif_frame_same_line"] = [to_frame, None, 10000000]
                  print("dif_frame_broken_line", frame_num, i1, to_frame, i2, dist)
                  matched_lines.append(i1)
                  continue
              h_line1 = fr_ln1["hough_line"]
              for i12,fr_ln2 in enumerate(self.frame_lines[ds_num][to_frame_num]):
                ##############
                # broken lines
                h_line2 = fr_ln2["hough_line"]
                extended_line = self.is_broken_line(h_line1, h_line2)
                if extended_line is not None:
                  fr_ln1["dif_frame_broken_line"].append([to_frame, i2, extended_line])
                  fr_ln2["dif_frame_broken_line"].append([frame_num, i1, extended_line])
                  print("dif_frame_broken_line", frame_num, i1, to_frame, i2, extended_line)
                ################
                # parallel lines
                dist = self.parallel_dist(h_line1, h_line2)
                if dist is not None:
                  fr_ln1["dif_frame_parallel_line"].append([to_frame, i2, dist])
                  fr_ln2["dif_frame_parallel_line"].append([frame_num, i1, (-dist[0], -dist[1])])
                  print("dif_frame_parallel_line", frame_num, i1, to_frame, i2, dist)
                  ##############
                  # same lines
                  if self.is_same_line(x1_line, h_line):
                    fr_ln1["dif_frame_same_line"].append([to_frame, i2, dist])
                    fr_ln2["dif_frame_same_line"].append([frame_num, i1, (-dist[0],-dist[1])])
                    print("dif_frame_same_line", frame_num, i1, to_frame, i2, dist)
                    matched_lines.append(i1)
        fl = self.frame_lines[ds_num][frame_num]
        return fl

  
    def display_persist_lines(self):
        def get_plline(pl_stats, angle_list, angle_sd):
            # [mean_disp_x, mean_disp_y, stddev_disp_x, stddev_disp_y, mean_x, mean_y, mean_line, mean_line_length, mean_angle, counter, [pl_min_img_num, pl_max_img_num]] = pl_stats
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
  
    def load_state(self):
        # TODO
        pass

    def __init__(self):
        # [[ds_num, img_num]]
        self.frame_lines = [[]]
        self.stego_gripper = [[]]
        self.cfg = Config()
        self.robot_loc = []
        self.alset_state = AlsetState()
        self.cvu = CVAnalysisTools(self.alset_state)
        func_idx_file = "sample_code/TT_BOX.txt"
        self.dsu = DatasetUtils(app_name="TT", app_type="FUNC")
        dataset = [[]]
        self.curr_dataset = dataset[0]
        self.fwd_actions = [[]]
        unmoved_pix = None
        slow_moved_pix = None
        self.M_inv = []
        self.pix_moved = []

    def move_to_analyze_alset(self):
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
        frame_lines = []
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
            # REMOVE to handle more ds_num
            ################################
            if pass_num == 0:
              frame_lines.append([])
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
                frame_lines[ds_num].append([])
    
              ##################################
              #  Compute Gripper
              ##################################
              elif pass_num == 1:
                compute_gripper_and_or()
                # overwrite previous attempt at getting gripper.
                print("frame_lines sz", len(frame_lines), len(frame_lines[ds_num]), img_num)
                # 1,0,0
                frame_lines[ds_num][img_num] = self.get_frame_lines(img, gripper)
                frame_lines_image = self.get_hough_lines_img(frame_lines[ds_num][img_num], img)
                cv2.imshow("box hough lines", frame_lines_image)
              elif pass_num == 2:
                persist_line,persist_line_stats, angle_sd = self.analyze_frame_lines(dataset, frame_lines, actions, gripper, drop_off_img[ds_num], img_paths)
                self.display_persist_lines(dataset, persist_line, persist_line_stats, angle_sd, img)
              elif False and pass_num == 3:
                stego_img, unique_color, stego_key = stego(img, unique_color)
                # cv2.imshow("stego orig input img", img)
                cv2.imshow("stego img", stego_img)
                stego_gray = cv2.cvtColor(stego_img, cv2.COLOR_BGR2GRAY)
                # do again but with each color isolated?
                stego_hough_lines = self.get_frame_lines(stego_img)
                stego_hough_lines_image = self.get_hough_lines_img(stego_hough_lines, stego_gray)
                cv2.imshow("stego hough lines", stego_hough_lines_image)
                # cv2.imshow("stego gray", stego_gray)
    
                # cv2.imshow("stego adj input img", adj_img)
                # cv2.imshow("stego input img", rl_img)
                # cv2.imshow("stego rl img", rl_img2)
                cv2.waitKey(0)
    
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
    
  # if __name__ == '__main__':
  #   box = AnalyzeLines()
  
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

# 4 instantiations of object:
# - Stego
#   - Arm actions
#
#   - Body actions
#     - handles rotations
#       - via "conversion" functions or absolute map?
#       - do line analysis before doing rotation
# - Image
#   - Arm actions
#   - Body actions
