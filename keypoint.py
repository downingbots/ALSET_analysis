#!/usr/bin/env python
import cv2 as cv

class Keypoints:
    def __init__(self, img):
      self.keypoints = []
      self.pc_map = None
      self.pc_header = None
      self.kp_pc_points = None
      orb = cv.ORB_create()         # Initiate SIFT detector
      # orb = cv.ORB(1000,1.2)         # Initiate SIFT detector
      # find the keypoints and descriptors with SIFT or ORB
      # print("len(cropped_img):", len(cropped_img)) # always 136?
      self.keypoints, self.descriptor = orb.detectAndCompute(img,None)

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
      # return [cv2.KeyPoint(x = (k.pt[0]-centroid.x), y = (k.pt[1]-centroid.y),
      return [cv2.KeyPoint(x = k.pt[0], y = k.pt[1],
            _size = k.size, _angle = k.angle,
            _response = k.response, _octave = k.octave,
            _class_id = k.class_id) for k in f if k in kp_list], new_desc


    def get_kp(self):
      kp_list = [[kp.pt[0], kp.pt[1]] for kp in self.keypoints]
      return kp_list

    def get_features(self):
      return self.keypoints


    def get_descriptor(self):
      return self.descriptor

    def update_pc(self, data):
      self.pc_header = data.header

    def compare_kp(self,KP2):
      # from matplotlib import pyplot as plt

      # find the keypoints and descriptors with ORB
      kp1 = self.get_kp()            
      des1 = self.get_descriptor()   
      kp2 = KP2.get_kp()             
      des2 = KP2.get_descriptor()    
      # create BFMatcher object
      bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
      # Match descriptors.
      bf_matches = bf.match(des1,des2)
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
