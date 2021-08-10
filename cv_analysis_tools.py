import numpy as np
import cv2
import math
from config import *
from matplotlib import pyplot as plt
from skimage.feature import local_binary_pattern

class CVAnalysisTools():
  def __init__(self):
      # foreground/background for movement detection
      self.background = None
      self.foreground = None
      self.prev_foreground = None
      self.cfg = Config()
      # store the number of points and radius for color histogram
      self.n_points = None
      self.radius = None
      self.TEXTURE_METHOD = 'uniform'
      self.textures = {}

  ###############################################
  # should remove from automated func
  def optflow(self, old_frame_path, new_frame_path):
      if old_frame_path is None:
        print("optflow: old_frame None")
        return True
      old_frame = cv2.imread(old_frame_path)
      new_frame = cv2.imread(new_frame_path)
      # cap = cv2.VideoCapture('slow.flv')
      # params for ShiTomasi corner detection
      feature_params = dict( maxCorners = 100,
                             qualityLevel = 0.3,
                             minDistance = 7,
                             blockSize = 7 )
      # Parameters for lucas kanade optical flow
      lk_params = dict( winSize  = (15,15),
                        maxLevel = 2,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
      # Create some random colors
      color = np.random.randint(0,255,(100,3))
      # Take first frame and find corners in it
      # ret, old_frame = cap.read()
      old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
      p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
      # Create a mask image for drawing purposes
      mask = np.zeros_like(old_frame)
  
      frame_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
      # calculate optical flow
      try:
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
      except:
        print("OPT FLOW FAILS")
        return False
      # Select good points
      good_new = p1[st==1]
      good_old = p0[st==1]
      # draw the tracks
      dist = 0
      numpts = 0
      frame1 = new_frame
      for i,(new,old) in enumerate(zip(good_new,good_old)):
          a,b = new.ravel()
          c,d = old.ravel()
          dist += math.hypot(a-c,b-d)
          numpts += 1
          # mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
          # frame1 = cv2.circle(frame1,(a,b),5,color[i].tolist(),-1)
      img = cv2.add(new_frame,mask)
      # cv2.imshow('frame',img)
      # k = cv2.waitKey(30) & 0xff
      # Now update the previous frame and previous points
      # old_gray = frame_gray.copy()
      p0 = good_new.reshape(-1,1,2)
      # cv2.destroyAllWindows()
      if numpts != 0:
        dist /= numpts
      else:
        dist = 0
      print("optflow dist %f " % (dist))
      # note: PPF also used to ensure that moving
      # tried 0.75, 0.9, 1
      # OPTFLOWTHRESH = 0.8
      if dist > self.cfg.OPTFLOWTHRESH:
        return True
      else:
        return False

  # ARD: does not work well
  def moved_pixels_over_time(self, prev_img_path, curr_img_path, init=False):
      # height, width, channels = prev_img.shape
      if self.background is None or init:
        prev_img = cv2.GaussianBlur(prev_img, (5, 5), 0)
        self.background = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
        print("init")
      elif init and self.foreground is not None:
        # self.background = cv2.absdiff(self.foreground_inv, self.background)
        self.background = cv2.add(self.foreground, self.background)
        print("bg")
      elif self.prev_foreground is not None:
        fgdiff = cv2.absdiff(self.foreground, self.prev_foreground)
        # self.foreground = cv2.absdiff(self.foreground, fgdiff)
        # cv2.imshow("Gripper FGdiff", fgdiff)
        # self.background = cv2.absdiff(self.background, fgdiff)
        # self.background = cv2.add(fgdiff, self.background)
        # cv2.accumulateWeighted(fgdiff, self.background.astype(float), 0.75)
        print("prevfg")
        pass
      if self.foreground is not None:
         self.prev_foreground = self.foreground.copy()
      curr_img = cv2.GaussianBlur(curr_img, (5, 5), 0)
      gray_curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
      diff = cv2.absdiff(gray_curr_img, self.background)
      # diff = cv2.absdiff(self.background, gray_curr_img)
      # diff = cv2.add(gray_curr_img, self.background)
      self.foreground_inv = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY_INV)[1]
      self.foreground_bin = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)[1]
      self.foreground = self.foreground_inv
      #  self.foreground = self.foreground_inv
      # else:
      #   self.foreground = self.foreground_bin
      # self.foreground = cv2.absdiff(self.background, self.foreground)
      # self.foreground = cv2.add(self.foreground, self.background)
      # cv2.accumulateWeighted(self.foreground, self.background.astype(float), 0.1)
      # cv2.accumulateWeighted(self.foreground, self.background.astype(float), 0.5)
#     #### Adding Contours didn't help!
#      contours, hierarchy = cv2.findContours(self.foreground, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#      contour_list = []
#      for contour in contours:
#        M = cv2.moments(contour)
#        area = cv2.contourArea(contour)
#        # if area > 100 :
#        #     contour_list.append(contour)
#        contour_list.append(contour)
#      image = self.foreground.copy()
#      cv2.drawContours(image, contour_list, -1, (0, 255, 0), 2)
#      cv2.imshow("Gripper Output", image)
      cv2.imshow("Gripper FG", self.foreground)
      # cv2.imshow("Gripper prev", prev_img)
      # cv2.imshow("Gripper curr", curr_img)
      # cv2.imshow("Gripper Output", self.foreground)
      cv2.imshow("Gripper Background", self.background)
      cv2.waitKey(0)
      return self.foreground

  def moved_pixels(self, prev_img_path, curr_img_path, init=False):
      prev_img = cv2.imread(prev_img_path)
      curr_img = cv2.imread(curr_img_path)
      # thresh = 10
      thresh = 20
      prev_img = cv2.GaussianBlur(prev_img, (5, 5), 0)
      self.background = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
      curr_img = cv2.GaussianBlur(curr_img, (5, 5), 0)
      gray_curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
      diff = cv2.absdiff(gray_curr_img, self.background)
      self.foreground = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY_INV)[1]
      # cv2.imshow("Gripper FG", self.foreground)
      # cv2.imshow("Gripper IM", gray_curr_img)
      # cv2.waitKey(0)
      return self.foreground

  def unmoved_pixels(self, prev_img_path, curr_img_path, init=False):
      prev_img = cv2.imread(prev_img_path)
      curr_img = cv2.imread(curr_img_path)
      # thresh = 10
      thresh = 20
      prev_img = cv2.GaussianBlur(prev_img, (5, 5), 0)
      self.background = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
      curr_img = cv2.GaussianBlur(curr_img, (5, 5), 0)
      gray_curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
      diff = cv2.absdiff(gray_curr_img, self.background)
      self.foreground = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)[1]
      cv2.imshow("Gripper FG", self.foreground)
      cv2.imshow("Gripper IM", gray_curr_img)
      cv2.waitKey(0)
      return self.foreground

  ###############################################
  # Color histogram for texture mapping (e.g., roads, floor)
  ###############################################
  # plot the color histograms using opencv
  def draw_image_histogram(image, channels, color='k'):
      hist = cv2.calcHist([image], channels, None, [256], [0, 256])
      plt.plot(hist, color=color)
      plt.xlim([0, 256])
  
  def show_color_histogram(image):
      for i, col in enumerate(['b', 'g', 'r']):
          draw_image_histogram(image, [i], color=col)
      plt.show()

  def kullback_leibler_divergence(self, p, q):
      p = np.asarray(p)
      q = np.asarray(q)
      filt = np.logical_and(p != 0, q != 0)
      return np.sum(p[filt] * np.log2(p[filt] / q[filt]))
  
  def match_texture(self, img):
      best_score = 10
      best_name = None
      lbp = local_binary_pattern(img, self.n_points, self.radius, self.TEXTURE_METHOD)
      n_bins = int(lbp.max() + 1)
      hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
      for label in self.textures.keys():
          ref_hist, _ = np.histogram(self.textures[label], density=True, bins=n_bins,
                                     range=(0, n_bins))
          score = kullback_leibler_divergence(hist, ref_hist)
          if score < best_score:
              best_score = score
              best_name = label
      return best_name
 
  def register_texture(self, label, image):
      lbp = local_binary_pattern(image, self.n_points, self.radius, self.TEXTURE_METHOD)
      self.textures[label] = lbp

  ###############################################
