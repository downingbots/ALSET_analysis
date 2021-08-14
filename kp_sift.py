import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from operator import itemgetter, attrgetter

class kp_sift():
  def __init__(self, img):
      self.sift = cv.SIFT_create()
      # self.sift = cv.SIFT()
      # find the keypoints and descriptors with SIFT
      gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
      self.keypoints, self.descriptor = self.sift.detectAndCompute(gray,None)
      # print("keypoints:", self.keypoints)

  def get_kp(self):
      return self.keypoints

  def get_des(self):
      return self.descriptor

  def compare_kp(self,kp_sift2,include_dist=True):
      des2 = kp_sift2.get_des()
      if self.descriptor is None or des2 is None:
        return [],[] 
      # print("kp_sift2:", kp_sift2, des2)
      # BFMatcher with default params
      bf = cv.BFMatcher()
      # print("des info",self.descriptor, des2)
      matches = bf.knnMatch(self.descriptor,des2,k=2)
      print("len matches:", len(matches))
      # Apply ratio test
      good = []
      notsogood = []
      for m,n in matches:
          d = m.distance/n.distance
          if m.distance < 0.75*n.distance:
              good.append((m, n, d))
          else:
              notsogood.append((m, n, d))
      good = sorted(good, key=itemgetter(2))
      notsogood = sorted(notsogood, key=itemgetter(2))
      if not include_dist:
         good = [g[:-1] for g in good]
         notsogood = [g[:-1] for g in notsogood]
      return good, notsogood

      # cv.drawMatchesKnn expects list of lists as matches.
      # img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
      # cv2.imshow(img3),plt.show()
      # cv2.waitKey(0)

  # the scale should be the same
  # the robot movement should be within 3 standard deviations
  def location_compare(self, kp_sift2, xydif, xyvar, include_dist=True):
      sd = [None, None]
      max_dist = [None, None]
      for i in range(2):
       sd[i] = np.sqrt(xyvar[i])
       if xydif[i] < 0:
         max_dist[i] -= 3*sd[i]
       else:
         max_dist[i] += 3*sd[i]
      # ARD: not done yet; more a concept for now

      # ignore KPs right on the edge
      

