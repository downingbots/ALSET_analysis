from sklearn import metrics, linear_model
from sklearn.metrics import r2_score, mean_squared_error
# from sklearn.cluster import DBSCAN, kmeans
from sklearn.cluster import *
from keypoint import *
import cv2 as cv2
from math import sqrt
import scipy.stats
from kneebow.rotor import Rotor
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import *

class AnalyzeClusters():

  def __init__(self):
      self.cluster = {}
      self.KP = None
      # store the number of points and radius
      self.radius = 2
      self.n_points = 8 * self.radius
      self.color_references = []
      self.display_lbp = False
      # example color reference:
      #   'brick': local_binary_pattern(brick, self.n_points, self.radius, METHOD)


  # plot the color histograms using opencv 
  def show_color_histogram(self, image):
      for i, col in enumerate(['b', 'g', 'r']):
          hist = cv2.calcHist([image], [i], None, [256], [0, 256])
          plt.plot(hist, color=col)
          plt.xlim([0, 256])
      plt.show()

  #########################
  # settings for Local Binary Pattern : color analysis
  
  # a measure of histogram distributions 
  def kullback_leibler_divergence(p, q):
      p = np.asarray(p)
      q = np.asarray(q)
      filt = np.logical_and(p != 0, q != 0)
      return np.sum(p[filt] * np.log2(p[filt] / q[filt]))
  
  
  # lbp allows automatic classification and identification of textures.
  # The patterns are rotation and grayscale invariant.
  def match(self, refs, img):
      best_score = 10
      best_name = None
      lbp = local_binary_pattern(img, self.n_points, self.radius, METHOD)
      n_bins = int(lbp.max() + 1)
      hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
      for name, ref in refs.items():
          ref_hist, _ = np.histogram(ref, density=True, bins=n_bins,
                                     range=(0, n_bins))
          score = kullback_leibler_divergence(hist, ref_hist)
          if score < best_score:
              best_score = score
              best_name = name
      return best_name
  
  # Local Binary Pattern
  def get_local_binary_pattern(self, image, eps=1e-7):
      # compute the Local Binary Pattern representation
      # of the image, and then use the LBP representation
      # to build the histogram of patterns
      # lbp = feature.local_binary_pattern(image, self.n_points,
      lbp = local_binary_pattern(image, self.n_points, self.radius, method="uniform")
      (hist, _) = np.histogram(lbp.ravel(),
          bins=np.arange(0, self.n_points + 3),
          range=(0, self.n_points + 2))
      # print("lbp shape:", len(lbp), len(lbp[0])) # 224x224
      if self.display_lbp:
        cv2.imshow("get LBP image:", lbp)

      # normalize the histogram
      hist = hist.astype("float")
      hist /= (hist.sum() + eps)

      # return the histogram of Local Binary Patterns
      return hist

  def analyze_color(self, refs, img_path):
      image = cv2.imread(img_path)
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      hist = self.get_local_binary_pattern(gray)
      if self.display_lbp:
        self.show_color_histogram(image)

      (means, stds) = cv2.meanStdDev(image)
      print("means, stds:", means, stds) # for each RGB channel
      # Example feature vectors:
      #   local_binary_pattern
      #   a vector of R,G,B means, stdev
      #   stats = np.concatenate([means, stds]).flatten()

      if self.display_lbp:
        plt.plot(hist,'b-')
        plt.ylabel('Feature Vectors')
        plt.show()
        cv.waitKey(0)
      return
  
      # We want automatically gather and label:
      #
      # - Ground:
      #    - Safe-to-drive ground
      #    - Avoided ground
      # - Robot:
      #     - robot grippers colors
      #     - robot arm colors (and shapes, if known by movement)
      #       -- arm position from known parked-point
      #     - Other robot colors
      # - Objects:
      #     - Colors, shape
      #     - Objects picked up
      #     - Container objects
      #       -- Note: a Dump Truck is a container object :-)
      #       -- Note: a Shovel is a container object :-)
      #     - Avoided, Desired, Ignored
      #     - pushable, immovable attributes, drive over, 
      #     - grabable, pickable

      brick = data.brick()
      grass = data.grass()
      gravel = data.gravel()
      
      refs = {
          'brick': local_binary_pattern(brick, n_points, radius, METHOD),
          'grass': local_binary_pattern(grass, n_points, radius, METHOD),
          'gravel': local_binary_pattern(gravel, n_points, radius, METHOD)
      }
      
      # classify rotated textures
      print('Rotated images matched against references using LBP:')
      print('original: brick, rotated: 30deg, match result: ',
            match(refs, rotate(brick, angle=30, resize=False)))
      print('original: brick, rotated: 70deg, match result: ',
            match(refs, rotate(brick, angle=70, resize=False)))
      print('original: grass, rotated: 145deg, match result: ',
            match(refs, rotate(grass, angle=145, resize=False)))
      
      # plot histograms of LBP of textures
      fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3,
                                                             figsize=(9, 6))
      plt.gray()
      
      ax1.imshow(brick)
      ax1.axis('off')
      hist(ax4, refs['brick'])
      ax4.set_ylabel('Percentage')
      
      ax2.imshow(grass)
      ax2.axis('off')
      hist(ax5, refs['grass'])
      ax5.set_xlabel('Uniform LBP values')
      
      ax3.imshow(gravel)
      ax3.axis('off')
      hist(ax6, refs['gravel'])
      
      plt.show()

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
        self.analyze_color(None, curr_img_path)

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
          print("compactness, labels, centers:", ret, label.flatten, center)
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

