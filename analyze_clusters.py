from sklearn import metrics, linear_model
from sklearn.metrics import r2_score, mean_squared_error
# from sklearn.cluster import DBSCAN, kmeans
from sklearn.cluster import *
from analyze_keypoints import *
import cv2 as cv2
from math import sqrt
import scipy.stats
# from kneebow.rotor import Rotor
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import *

class AnalyzeClusters():

  def __init__(self, alset_state):
      self.alset_state = alset_state
      self.cluster = {}
      self.KP = None
      # store the number of points and radius
      self.radius = 2
      self.n_points = 8 * self.radius
      self.color_references = []
      self.display_lbp = False
      # example color reference:
      #   'brick': local_binary_pattern(brick, self.n_points, self.radius, METHOD)

  def create_cluster(self, id = None, status = None, centroid = None, 
               kp = None, kp_attr = None, location = None, 
               num_grasps = None, grasps = None, grasp_attr = None,
               state = None): 
      num_locations = None
      # self.cluster = {}  # done during init
      self.cluster['id'] = id
      self.cluster['centroid'] = centroid          # current centroid 

      # 'kp' is the kp derived from a single world cluster
      self.cluster['kp'] = kp                      # Keypoint class
      self.cluster['kp_attr'] = []                 # mean, stddev for distances 

      self.cluster['map_location'] = []
      if num_locations == None:
        self.cluster['num_locations'] = 0            
        self.cluster['location'] = []              # a history of locations
      if num_grasps == None:
        self.cluster['num_grasps'] = 0
        self.cluster['grasps'] = []                # history of grasp
        self.cluster['grasp_attr'] = []            # ["name",value] pair
      if location is not None:
        self.cluster['num_locations'] += 1
        self.cluster['location'].append(location)  # center=x,y,z + orientation
      # self.cluster['state'] = state                # e.g., ISOLATED 

      # self.cluster['interaction'] = interaction    # with cluster_id list
      # self.cluster['interaction_attr'] = interaction_attr  # ["name",value] pair
      # pointer to self.cluster not a deep copy!
      # self.cluster['normalized_shape'] = []
      # self.cluster['normalized_kp'] = []

  #####################
  # CLUSTER UTILITIES 
  #####################

  # analyze image into a set of clusters
  def analyze(self, frame_num, action, prev_img_path, curr_img_path, done):
      # self.analyze_color(None, curr_img_path)
      curr_img = cv2.imread(curr_img_path)

      self.KP = Keypoints(curr_img)

      # from: https://stackoverflow.com/questions/40142835/image-not-segmenting-properly-using-dbscan
      labimg = cv2.cvtColor(curr_img, cv2.COLOR_BGR2LAB)

      n = 0
      while(n<4):
          labimg = cv2.pyrDown(labimg)
          n = n+1

      feature_image=np.reshape(labimg, [-1, 3])
      rows, cols, chs = labimg.shape
      
      db = DBSCAN(eps=5, min_samples=50, metric = 'euclidean',algorithm ='auto')
      db.fit(feature_image)
      labels = db.labels_

      indices = np.dstack(np.indices(labimg.shape[:2]))
      xycolors = np.concatenate((labimg, indices), axis=-1) 
      feature_image2 = np.reshape(xycolors, [-1,5])
      db.fit(feature_image2)
      labels2 = db.labels_

      # no colors, no models; not so easy to segment!

      # Number of clusters in labels, ignoring noise if present.
      n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
      n_noise_ = list(labels).count(-1)
      print('#clusters, noise: ', n_clusters_, n_noise_)
#      print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
#      print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
#      print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
#      print("Adjusted Rand Index: %0.3f"
#            % metrics.adjusted_rand_score(labels_true, labels))
#      print("Adjusted Mutual Information: %0.3f"
#            % metrics.adjusted_mutual_info_score(labels_true, labels))
#      print("Silhouette Coefficient: %0.3f"
#            % metrics.silhouette_score(X, labels))

      if n_clusters_ > 0 and False: # ARD
        plt.figure(2)
        plt.subplot(2, 1, 1) # numrows numcols index
        plt.imshow(curr_img)
        plt.axis('off')

        plt.subplot(2, 1, 2)
        plt.imshow(np.reshape(labels2, [rows, cols]))
        plt.axis('off')
        plt.show()
      else:

        # try color quantification
        Z = curr_img.reshape((-1,3))
        # convert to np.float32
        Z = np.float32(Z)
        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        for K in range(1,9):
          # K: number of clusters
          # compactness : sum of squared distance from each point to their centers.
          # labels : the label array where each element marked '0', '1'.....
          # centers : This is array of centers of clusters.
          ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
          # Now convert back into uint8, and make original image
          # print("compactness, labels, centers:", ret, label.flatten, center)
          # ret is a single float, label is ?, center is RGB
          center = np.uint8(center)
          res = center[label.flatten()]
          res2 = res.reshape((curr_img.shape))
          if self.display_lbp:
            cv.imshow('res2',res2)
            cv.waitKey(0)
            cv.destroyAllWindows()

      if self.KP != None:
        kp = self.KP.get_kp() 

#      for c_id in set(db1.labels_):
#        if c_id != -1:
#          for i, label in enumerate(db1.labels_):
#              if db1.labels_[i] == c_id:
#                  # print("label", c_id, i, gray_curr_img[i])
#                  counter[c_id] += 1
#                  running_sum[c_id] += gray_curr_img[i]
#                  # print(c_id, "shape append", curr_img_gray[i])
#                  self.clusters[c_id].cluster['shape'].append(gray_curr_img[i])
#        c = self.clusters[c_id]
#        c.cluster['kp_c_mapping'] = kp_list[c_id]
#        center = running_sum[c_id] / counter[c_id]
#        c.cluster['center'] = center
#        # print("center for clust", c , " is ", self.clusters[c].cluster['center'])
#        # normalize shape
#        c.normalize()
#        # need to normalize KP:
#        print("cluster ",c_id," len", len(c.cluster['shape']))
#      print("num_clusters:",len(self.clusters))
      return True

