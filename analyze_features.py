    def feature_offsets(self, map_overlay, rotated_new_map, map_line, map_slope, rot_line, frame_num, use_slope=False, offset_width=None):
        print("###############")
        print("feature offsets")
        moff_h, moff_w = self.manual_tuning(frame_num)

        map_pts = []
        rot_pts = []
        off_h = 0
        mol = map_overlay.copy()
        rnm = rotated_new_map.copy()
        # x = (y - b)/M
        if use_slope:
          off_h = self.get_height_dif_by_line(rnm, map_line, map_slope, offset_width)
          if offset_width is not None:
            off_w = offset_width
          else:
            off_w = 0
          if off_h > 10*moff_h or off_w > 10*moff_w:
            return off_h, off_w, [],[]
          # off_h and off_w are absolute offsets for replace_border
          # The KPs should be at same H as the new offset.
          rnm = self.replace_border(rnm,
                            self.map_overlay.shape[0], self.map_overlay.shape[1],
                            off_h,off_w)

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
            # distance needs to be directional, and consider both x,y separately
            # change of slope and line segment (sounds like a keypoint!)
            # min change of both varx & vary
            # the diff_distance_variation needs to be in the same direction
            dist = math.sqrt((map_pt[0]-rot_pt[0])**2+(map_pt[1]-rot_pt[1])**2)
            if (map_pt[0] > rot_pt[0]):
              dist = -dist 
            slope = self.arctan2((map_pt[0] - rot_pt[0]) , (map_pt[1]-rot_pt[1]))
            ranking = slope * dist
            delta.append([ranking, i, j])
        if self.frame_num >= self.stop_at_frame:
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
            print("map_feat:", mfeat, rfeat)
            # compute the mean difference
            sum_h += mfeat[0] - rfeat[0]
            sum_w += mfeat[1] - rfeat[1]
          foff_h = -int(sum_h/n)
          foff_w = -int(sum_w/n)

          moff_h, moff_w = self.manual_tuning(frame_num)
          if use_slope:
            # off_h computed earlier
            if offset_width is not None:
              off_w += foff_w  
            else:
              off_w = foff_w  
          else:
            off_h = foff_h
            off_w = foff_w
          print("n off xy, manual xy", n, [foff_h, foff_w], [moff_h, moff_w], [off_h, off_w])

        return off_h, off_w, map_pts, rot_pts
        # return moff_h, moff_w, map_pts, rot_pts
    

