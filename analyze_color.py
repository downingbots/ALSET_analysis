import cv2
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
import numpy as np 
      
class LocalBinaryPattern:
  def __init__(self, image, radius):
      # store the number of points and radius
      # self.numPoints = numPoints
      self.eps = 1e-7
      self.numPoints = 8*radius
      self.radius = radius
      self.image = image
      self.hist = self.describe(image)
      self.targetMeanLight = 0
      self.targetMeanLightCnt = 0
      
  def describe(self, image):
      # compute the Local Binary Pattern representation
      # of the image, and then use the LBP representation
      # to build the histogram of patterns
      lbp = local_binary_pattern(image, self.numPoints, self.radius, method="uniform")
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
      
  def get_score(self, comparison_image):
      best_score = 10
      best_name = None
      comp_hist = self.describe(comparison_image)
      score = self.kullback_leibler_divergence(self.hist, comp_hist)
      return score 
      
