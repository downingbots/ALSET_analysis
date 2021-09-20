from utilradians import *
from utilborders import *
from config import *
from cv_analysis_tools import *


########################
# Map Line Analysis
#
# On a tabletop and roads, edges are typically well deliniated and visible.
# For the initial tabletop apps, the line analysis has proven more reliable
# than the keypoint analysis.
#################################
class LineAnalysis():
    def __init__(self):
      self.color_quant_num_clust = None
      self.cfg = Config()
      self.cvu = CVAnalysisTools()


    #########################
    # Map Line Analysis 
    #########################
    def evaluate_lines_in_image(self, image, border):
      linesP, imglinesp = self.cvu.get_lines(image)
      # print("border:", border)
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
        lines.append(line.copy())

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
  

    def best_line_angle(self, prev_map, curr_map, action, robot_location, frame_num=0):
      def angle_match_count(angle_match, rads, radian_thresh):
          found = False
          for i, [a,c] in enumerate(angle_match):
            if rad_dif(rads, a) <= radian_thresh:
              found = True
              angle_match[i][1] += 1
          if not found:
            angle_match.append([rads, 0])
          return angle_match

      likely_angle = None
      confidence = 1
      radian_thresh = self.cfg.RADIAN_THRESH
      if self.color_quant_num_clust is not None:
        minK = self.color_quant_num_clust
        maxK = self.color_quant_num_clust
      else:
        minK = 3
        maxK = 6
      comparison = {}
      for K in range(minK, maxK):
        border_shape, border = real_map_border(prev_map)
        pm_floor_clusters = self.cvu.color_quantification(prev_map, K)
        cm_floor_clusters = self.cvu.color_quantification(curr_map, K)
        cm = self.evaluate_lines_in_image(cm_floor_clusters, border)
        pm = self.evaluate_lines_in_image(pm_floor_clusters, border)
        comparison["intersect_rads"] = []
        comparison["line_rads"] = []
        for cm_i, cm_line in enumerate(cm):
          for pm_i, pm_line in enumerate(pm):
            # 
            # compare intersections
            for [cm_i1, cm_i2, cm_pt] in cm_line["intersect"]:
              for [pm_j1, pm_j2, pm_pt] in pm_line["intersect"]:
                angle = rad_isosceles_triangle(cm_pt, robot_location, pm_pt)
                comparison["intersect_rads"].append([cm_i, pm_i, cm_pt, pm_pt, angle])
            # compare line angles
            comparison["line_rads"].append([cm_i, pm_i, rad_dif(cm_line["slope"], pm_line["slope"])])

            # todo: compare relative line positions for non-intersecting lines
            #   - todo: provide confidence level of matches
            # todo: compare line lens
            #   - todo: determine multiple cm lines map to same pm line 
            #   - todo: determine multiple pm lines map to same cm line 
            # todo: bad_match

        print("intersect_rads, line_rads:", comparison["intersect_rads"], comparison["line_rads"])
        angle_match = []
        for i, i_list in enumerate(comparison["intersect_rads"]):
          I_cm_i, I_pm_i, I_cm_pt, I_pm_pt, I_intersect_rads = i_list
          for j, j_list in enumerate(comparison["intersect_rads"]):
            if i >= j:
              continue
            J_cm_i, J_pm_i, J_cm_pt, J_pm_pt, J_intersect_rads = j_list
            # compare the differences in slope, not the slopes themselves
            if rad_dif(I_intersect_rads, J_intersect_rads) <= radian_thresh:
              print("good intersect match:",I_intersect_rads, J_intersect_rads, I_cm_i, J_cm_i, I_pm_i, J_pm_i)
              angle_match = angle_match_count(angle_match, I_intersect_rads, radian_thresh)
            # else:
            #  print("bad intersect match:",I_intersect_rads, J_intersect_rads)
          for j, [J_cm_i, J_pm_j, J_rads] in enumerate(comparison["line_rads"]):
            if i >= j:
              continue
            j_rads = j_list[2]
            if rad_dif(I_intersect_rads, J_rads) <= radian_thresh:
              print("good intersect-slope match:",I_intersect_rads, J_rads, I_cm_i, J_cm_i, J_pm_j, I_pm_i)
              angle_match = angle_match_count(angle_match, I_intersect_rads, radian_thresh)
            # else:
            #   print("bad intersect match:",I_intersect_rads, J_rads)
        for i, [I_cm_i, I_pm_j, I_rads] in enumerate(comparison["line_rads"]):
          for j, [J_cm_i, J_pm_j, J_rads] in enumerate(comparison["line_rads"]):
            if i >= j:
              continue
            if I_cm_i == J_cm_i and I_pm_j == J_pm_j:
              continue
            # this is a delta slope, but it doesn't need to be less than radian thresh.
            # the lines should be verified to be the same based on location, length, and slope.
            # Ideally the slope would be similar to pevious turn in same direction.
            if rad_dif(I_rads, J_rads) <= radian_thresh:
              print("good slope match:",I_rads, J_rads, I_cm_i, I_pm_j, J_cm_i, J_pm_j)
              angle_match = angle_match_count(angle_match, I_rads, radian_thresh)
            # else:
            #   print("bad intersect match:",I_rads, J_rads)

        if len(angle_match) >= 1:
          max_a, max_c = self.cfg.INFINITE, - self.cfg.INFINITE
          for a,c in angle_match:
            if c > max_c:
              max_c = c
              max_a = a
          print("Angle match:", max_a, max_c, angle_match)
          return(max_a)
        elif len(comparison["intersect_rads"]) >= 1:
          return(comparison["intersect_rads"][0][4]) 
        elif len(comparison["line_rads"]) >= 1:
          return(comparison["line_rads"][0][2]) 
        else:
          continue
      return None


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
        if frame_num >= self.stop_at_frame:
          cv2.line(floor_clusters, (map_line[0], map_line[1]), (map_line[2], map_line[3]), (0,255,0), 3, cv2.LINE_AA)
          pt = [int(self.robot_location[1]), int(self.robot_location[0])] # circle uses [w,h]
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

            diff_slope = rad_dif(map_slope, self.arctan2(dx, dy))
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
          angle_max = angle
          break
        elif ((min_slope_dif == 0 or rad_dif(angle_max,angle_min) <= .001)
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


