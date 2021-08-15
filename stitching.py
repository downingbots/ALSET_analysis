# derived from:
# https://github.com/kushalvyas/Python-Multiple-Image-Stitching/blob/master/code/pano.py
import cv2
import numpy as np 
import sys
import time

class img_matchers:
  def __init__(self):
    self.sift = cv2.SIFT_create()
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=50)
    self.flann = cv2.FlannBasedMatcher(index_params, search_params)

  def match(self, i1, i2, direction=None):
    imageSet1 = self.getSiftFeatures(i1)
    imageSet2 = self.getSiftFeatures(i2)
    print("Direction : ", direction)
    matches = self.flann.knnMatch(
      imageSet2['des'],
      imageSet1['des'],
      k=2
      )
    good = []
    for i , (m, n) in enumerate(matches):
      if m.distance < 0.7*n.distance:
        good.append((m.trainIdx, m.queryIdx))

    if len(good) > 4:
      pointsCurrent = imageSet2['kp']
      pointsPrevious = imageSet1['kp']

      matchedPointsCurrent = np.float32(
        [pointsCurrent[i].pt for (__, i) in good]
      )
      matchedPointsPrev = np.float32(
        [pointsPrevious[i].pt for (i, __) in good]
        )

      H, s = cv2.findHomography(matchedPointsCurrent, matchedPointsPrev, cv2.RANSAC, 4)
      return H
    return None

  def getSiftFeatures(self, im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    kp, des = self.sift.detectAndCompute(gray,None)
    return {'kp':kp, 'des':des}


class Img_Stitch:
  def __init__(self, img1, img2, img1_dim=None, img2_dim=None):
    # self.images = [cv2.resize(img1,(480, 320)), cv2.resize(img2,(480, 320))]
    self.images = [img1, img2]
    self.count = len(self.images)
    self.left_list, self.right_list, self.center_im = [], [],None
    self.matcher_obj = img_matchers()
    print("Number of images : %d"%self.count)
    self.centerIdx = self.count/2 
    print("Center index image : %d"%self.centerIdx)
    self.center_im = self.images[int(self.centerIdx)]
    for i in range(self.count):
      if(i<=self.centerIdx):
        self.left_list.append(self.images[i])
      else:
        self.right_list.append(self.images[i])
    print("Image lists prepared")

  def leftshift(self):
    # self.left_list = reversed(self.left_list)
    a = self.left_list[0]
    for b in self.left_list[1:]:
      H = self.matcher_obj.match(a, b, 'left')
      print("Homography is : ", H)
      xh = np.linalg.inv(H)
      print("Inverse Homography :", xh)
      ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]));
      ds = ds/ds[-1]
      print("final ds=>", ds)
      f1 = np.dot(xh, np.array([0,0,1]))
      f1 = f1/f1[-1]
      xh[0][-1] += abs(f1[0])
      xh[1][-1] += abs(f1[1])
      ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]))
      offsety = abs(int(f1[1]))
      offsetx = abs(int(f1[0]))
      dsize = (int(ds[0])+offsetx, int(ds[1]) + offsety)
      print("image dsize =>", dsize)
      print("b shape =>", b.shape)
# final ds=> [661.53637938 601.94388498   1.        ]
# image dsize => (696, 666)
# b shape =>     (672, 597, 3)
# tmp shape =>   (666, 696, 3)
# off x,y: 2 18
# ValueError: could not broadcast input array from shape (672,597,3) into shape (648,597,3)
# post-border tmp shape => (666, 702, 3)
# ValueError: could not broadcast input array from shape (672,597,3) into shape (648,597,3)


      tmp = cv2.warpPerspective(a, xh, dsize)
      print("tmp shape =>", tmp.shape)
      # cv2.imshow("warped", tmp)
      # cv2.imshow("b", b)
      # cv2.waitKey()
      if b.shape[0] > tmp.shape[0]:
        border_size = b.shape[0] - tmp.shape[0] + offsety
        tmp = cv2.copyMakeBorder( tmp, top=0, bottom=border_size, left=0, right=0,
                                  borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
      if b.shape[1] > tmp.shape[1]:
        border_size = b.shape[1] - tmp.shape[1] + offsetx
        tmp = cv2.copyMakeBorder( tmp, top=0, bottom=0, left=0, right=border_size,
                                  borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
      print("off x,y:", offsetx, offsety)
      print("post-border tmp shape =>", tmp.shape)
      # cv2.imshow("warped", tmp)
      # cv2.imshow("b", b)
      # cv2.waitKey()
      tmp[offsety:b.shape[0]+offsety, offsetx:b.shape[1]+offsetx] = b
      a = tmp

    self.leftImage = tmp

    
  def rightshift(self):
    for each in self.right_list:
      H = self.matcher_obj.match(self.leftImage, each, 'right')
      print("Homography :", H)
      txyz = np.dot(H, np.array([each.shape[1], each.shape[0], 1]))
      txyz = txyz/txyz[-1]
      dsize = (int(txyz[0])+self.leftImage.shape[1], int(txyz[1])+self.leftImage.shape[0])
      tmp = cv2.warpPerspective(each, H, dsize)
      cv2.imshow("tp", tmp)
      cv2.waitKey()
      # tmp[:self.leftImage.shape[0], :self.leftImage.shape[1]]=self.leftImage
      tmp = self.mix_and_match(self.leftImage, tmp)
      print("tmp shape",tmp.shape)
      print("self.leftimage shape=", self.leftImage.shape)
      self.leftImage = tmp
    # self.showImage('left')



  def mix_and_match(self, leftImage, warpedImage):
    i1y, i1x = leftImage.shape[:2]
    i2y, i2x = warpedImage.shape[:2]
    print(leftImage[-1,-1])

    t = time.time()
    black_l = np.where(leftImage == np.array([0,0,0]))
    black_wi = np.where(warpedImage == np.array([0,0,0]))
    print(time.time() - t)
    print(black_l[-1])

    for i in range(0, i1x):
      for j in range(0, i1y):
        try:
          if(np.array_equal(leftImage[j,i],np.array([0,0,0])) and  np.array_equal(warpedImage[j,i],np.array([0,0,0]))):
            # print "BLACK"
            # instead of just putting it with black, 
            # take average of all nearby values and avg it.
            warpedImage[j,i] = [0, 0, 0]
          else:
            if(np.array_equal(warpedImage[j,i],[0,0,0])):
              # print "PIXEL"
              warpedImage[j,i] = leftImage[j,i]
            else:
              if not np.array_equal(leftImage[j,i], [0,0,0]):
                bw, gw, rw = warpedImage[j,i]
                bl,gl,rl = leftImage[j,i]
                # b = (bl+bw)/2
                # g = (gl+gw)/2
                # r = (rl+rw)/2
                warpedImage[j, i] = [bl,gl,rl]
        except:
          pass
    # cv2.imshow("waRPED mix", warpedImage)
    # cv2.waitKey()
    return warpedImage

  def trim_left(self):
    pass

  def showImage(self, string=None):
    if string == 'left':
      cv2.imshow("left image", self.leftImage)
      # cv2.imshow("left image", cv2.resize(self.leftImage, (400,400)))
    elif string == "right":
      cv2.imshow("right Image", self.rightImage)
    cv2.waitKey()


#  s = Img_Stitch(args)
#  s.leftshift()
#  # s.showImage('left')
#  s.rightshift()
#  print "done"
#  cv2.imwrite("test12.jpg", s.leftImage)
#  print "image written"
#       cv2.destroyAllWindows()
