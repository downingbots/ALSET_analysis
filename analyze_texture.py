import cv2
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
import numpy as np 
      
class LocalBinaryPattern:
  def __init__(self):
      self.textures = {}
      self.TEXTURE_METHOD = 'uniform'

  def set_params(self, image, radius=None):
      # store the number of points and radius
      if radius is None:
        radius = int(.75 * min(image.shape[0], image.shape[1])/2)
      self.eps = 1e-7
      self.n_points = 8*radius
      self.radius = radius
      self.image = image
      self.hist = self.describe(image)
      # self.targetMeanLight = 0
      # self.targetMeanLightCnt = 0
      
  def describe(self, image, radius=None):
      # compute the Local Binary Pattern representation
      # of the image, and then use the LBP representation
      # to build the histogram of patterns
      lbp = local_binary_pattern(image, self.n_points, self.radius, method="uniform")
      n_bins = int(lbp.max() + 1)
      # hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
      (hist, _) = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
      
      # normalize the histogram
      hist = hist.astype("float")
      hist /= (hist.sum() + self.eps)
      
      # return the histogram of Local Binary Patterns
      return hist
      
  def draw_histogram(self):
      # channel GBR, but LBP is on gray image?
      plt.gray(self.hist)
      plt.xlim([0, 256])
      
  def kullback_leibler_divergence(self, p, q):
      p = np.asarray(p)
      q = np.asarray(q)
      # print("minding my Ps and Qs", len(p), len(q))
      filt = np.logical_and(p != 0, q != 0)
      return np.sum(p[filt] * np.log2(p[filt] / q[filt]))
      
  def get_score(self, orig_image, comparison_image, radius=None):
      self.set_params(image, radius)
      orig_hist = self.describe(orig_image, radius)
      comp_hist = self.describe(comparison_image, radius)
      score = self.kullback_leibler_divergence(orig_hist, comp_hist)
      return score 
      
  def match(self, img):
      best_score = 10
      best_name = None
      self.set_params(img)
      lbp = local_binary_pattern(img, self.n_points, self.radius, self.TEXTURE_METHOD)
      n_bins = int(lbp.max() + 1)
      hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
      for label in self.textures.keys():
          ref_hist, _ = np.histogram(self.textures[label], density=True, bins=n_bins,
                                     range=(0, n_bins))
          score = self.kullback_leibler_divergence(hist, ref_hist)
          # lower score is better
          if score < best_score:
              best_score = score
              best_name = label
      return best_name, best_score

  def register(self, label, image, run_id=None, frame_num=None):
      try:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      except:
        gray_image = image
      self.set_params(gray_image)
      lbp = local_binary_pattern(gray_image, self.n_points, self.radius, self.TEXTURE_METHOD)
      if run_id is not None and frame_num is not None:
        key = label + "_" + str(run_id) + "_" + str(frame_num)
      else:
        key = label
      self.textures[key] = lbp
      return lbp

