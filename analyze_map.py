# import the necessary packages
import numpy as np
from numpy import asarray
from PIL import Image
import cv2
import argparse
from keypoint import *
from kp_sift import *
from imutils import paths
import imutils
from matplotlib import pyplot as plt
from cv_analysis_tools import *

# based on: 
# https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
class AnalyzeMap():
    def __init__(self):
        self.KPs = None
        self.map = None
        # real map array info
        # real map array sizes change over time
        self.map_arr = None
        self.map_sift = None
        self.map_rows = None
        self.map_cols = None
        self.map_ch = None
        self.border_buffer = None   # move_img
        self.border_multiplier = 1   # move_img
        # To compute the distances,var of moves
        self.move_rows = None
        self.move_cols = None
        self.curr_move = None
        self.curr_move_sift = None
        self.prev_move = None
        self.prev_move_sift = None
        # dist/degrees = 0, var = 1 in lists below
        self.forw_dist = []  
        self.back_dist = []
        self.left_degrees = []
        self.right_degrees = []
        # the robot starts in the middle of an empty square of VIRTUAL_MAP_SIZE pixels
        # the virtual map locations do not change.
        self.VIRTUAL_MAP_SIZE = None
        self.virtual_map_center = None
        self.robot_location = None
        self.robot_orientation = None
        self.robot_location_hist = []
        self.robot_orientation_hist = []
        self.clusters = []
        self.grabable_objects = []
        self.container_objects = []
        self.dif_avg = []
        self.dif_var = []
        self.min_pt = []
        self.max_pt = []
        self.cvu = CVAnalysisTools()

    def real_to_virtual_map_coordinates(self, pt):
        self.virtual_map_center = None
        x = pt[0] - self.virtual_map_center[0] + self.VIRTUAL_MAP_SIZE/2
        y = pt[1] - self.virtual_map_center[1] + self.VIRTUAL_MAP_SIZE/2
        return x,y

    def virtual_to_real_map_coordinates(self, pts):
        x = pt[0] + self.virtual_map_center[0] - self.VIRTUAL_MAP_SIZE/2
        y = pt[1] + self.virtual_map_center[1] - self.VIRTUAL_MAP_SIZE/2
        return x,y

    def order_points(self, pts):
        # initialzie a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
        rect = np.zeros((4, 2), dtype = "float32")

        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # return the ordered coordinates
        return rect

    def four_point_transform(self, image, pts):
        # obtain a consistent order of the points and unpack them
        # individually
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect
        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order

        dst = np.array([
              [0, 0], [maxWidth-1, 0], [maxWidth-1, maxHeight-1], [0, maxHeight-1]
              ], dtype = "float32")
       
        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(dst, rect)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight),  cv2.WARP_INVERSE_MAP)
        # return the warped image
        return warped

    def kp_match_stats(self, kp_matches, sift1=None, sift2=None):
        if len(kp_matches) >= 3:
            # kps are sorted by distance. Lower distances are better.
            num_kp_matches = len(kp_matches)
            avg_x_dif = 0
            avg_y_dif = 0
            x_dev = []
            y_dev = []
            min_x = 1000000000000000000
            max_x = -1000000000000000000
            min_y = 1000000000000000000
            max_y = -1000000000000000000
            # check if use defaults
            if sift1 is None:
              sift1=self.map_sift
            if sift2 is None:
              sift2=self.curr_move_sift
            for kpm in kp_matches:
              if len(sift1.keypoints) <= kpm[0].queryIdx or len(sift2.keypoints) <= kpm[0].trainIdx:
                print("trI, numkp1, numkp2", kpm[0].queryIdx, kpm[0].trainIdx, len(sift1.keypoints), len(sift2.keypoints))
                continue
              sift1_kp = sift1.keypoints[kpm[0].queryIdx].pt
              sift2_kp = sift2.keypoints[kpm[0].trainIdx].pt
              # print("map,new,dist:", sift1_kp, sift2_kp, dist)
              x_dev.append(sift1_kp[0] - sift2_kp[0])
              y_dev.append(sift1_kp[1] - sift2_kp[1])
              avg_x_dif += sift1_kp[0] - sift2_kp[0]
              avg_y_dif += sift1_kp[1] - sift2_kp[1]
              min_x = min(min_x, sift1_kp[0])
              max_x = max(max_x, sift1_kp[0])
              min_y = min(min_y, sift1_kp[1])
              max_y = max(max_y, sift1_kp[1])
            # ARD: todo: autotune to minimize the avg kp difs
            avg_x_dif /= num_kp_matches
            avg_y_dif /= num_kp_matches

            x_dev = [(x_dev[i] - avg_x_dif) ** 2 for i in range(len(x_dev))]
            y_dev = [(y_dev[i] - avg_y_dif) ** 2 for i in range(len(y_dev))]
            x_var = sum(x_dev) / num_kp_matches
            y_var = sum(y_dev) / num_kp_matches

            print("avg x,y dif: ", avg_x_dif, avg_y_dif, x_var, y_var)
            print("max/min x,y: ", [max_x, max_y],[min_x, min_y])
        else:
            print("insufficient keypoints")
            return [None, None], [None, None], [None, None], [None, None]
            # x = undefined_var
        return [avg_x_dif, avg_y_dif], [x_var, y_var], [min_x, min_y], [max_x, max_y]

    def move_state(self, action, new_above_view_img):
        self.prev_move = self.curr_move
        self.prev_move_sift = self.curr_move_sift
        self.curr_move = new_above_view_img
        self.curr_move_sift = kp_sift(self.curr_move)
        # TODO: compute distances for each action

    def get_n_match_kps(self, matches, sift1, sift2, n, ret_kp=False):
        sift1_kps = []
        sift2_kps = []
        # print("num matches: ", len(matches))
        for i,kpm in enumerate(matches):
          # print("queryIdx:", kpm[0].queryIdx, kpm[1].queryIdx)
          # print("imgIdx:", kpm[0].imgIdx, kpm[1].imgIdx)
          # print("trainIdx:", kpm[0].trainIdx, kpm[1].trainIdx)
          # print("num kp:", len(sift1.keypoints),len(sift2.keypoints))
          # print("kp1t:",sift1.keypoints[kpm[0].queryIdx].pt)
          # print("kp2q:",sift2.keypoints[kpm[1].trainIdx].pt)
          # print("dist:", kpm[0].distance, kpm[1].distance)
          if i >= n:
            if not ret_kp:
              sift1_kps = np.float32(sift1_kps)
              sift2_kps = np.float32(sift2_kps)
            return sift1_kps, sift2_kps
          kp1 = sift1.keypoints[kpm[0].queryIdx]
          kp2 = sift2.keypoints[kpm[0].trainIdx]
          if ret_kp:
            sift1_kps.append(kp1)
            sift2_kps.append(kp2)
          else:
            sift1_kps.append(kp1.pt)
            sift2_kps.append(kp2.pt)
        print("insufficient matches: ", len(matches))
        if not ret_kp:
          sift1_kps = np.float32(sift1_kps)
          sift2_kps = np.float32(sift2_kps)
        return sift1_kps, sift2_kps

    ###########
    def get_contours(self,img):
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
        # find contours in the thresholded image
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	        cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        return cnts
        
    def draw_contours(self,img,cnt,text="",def_clr=(0,255,0)):
        for i,c in enumerate(cnt):
            # compute the center of the contour
            M = cv2.moments(c)
            # cX = int((M["m10"] / M["m00"]) )
            # cY = int((M["m01"] / M["m00"]) )
            c = c.astype("int")
            # print(i,"c",c)
            itext = text + str(i)
            cv2.drawContours(img, [c], -1, def_clr, 2)
            # cv2.putText(img, itext, (cX, cY),
            #       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    def contours_difs(self, img,cnt1, cnt2):
        def ci_dist(ci1,ci2):
          x1 = ci1[0][0][0] 
          x2 = ci2[0][0][0] 
          y1 = ci1[0][0][1] 
          y2 = ci2[0][0][1]
          return np.sqrt((x1-x2)**2+(y1-y2)**2)
        c1_match = []
        c1_unmatched = []
        c1_min = []
        c1_dist = []
        for i,c1 in enumerate(cnt1):
          ci1 = c1.astype("int")
          c1_min_dist = 1000000000000000
          c1_min_val = None
          found = False
          for i,c2 in enumerate(cnt2):
            ci2 = c2.astype("int")
            # print( ci1[0][0][0], ci2[0][0][1])
            if ci1[0][0][0] == ci2[0][0][0] and ci1[0][0][1] == ci2[0][0][1]:
              found = True
              c1_match.append(ci1)
              break
            elif c1_min_dist > ci_dist(ci1,ci2):
              c1_min_dist = ci_dist(ci1,ci2)
              c1_min_val = ci2
          if not found:
              c1_unmatched.append(ci1)
              c1_min.append(c1_min_val)
        # self.draw_contours(img,c1_match)
        # self.draw_contours(img,c1_unmatched,def_clr=(255,0,0))
        x_dist = 0
        y_dist = 0
        x_dev = []
        y_dev = []
        for i,c1 in enumerate(c1_min):
          dx = c1[0][0][0] - cnt1[i][0][0][0]  
          dy = c1[0][0][1] - cnt1[i][0][0][1]
          x_dist += dx
          y_dist += dy
          x_dev.append(dx)
          y_dev.append(dy)
        x_dist /= len(c1_min)
        y_dist /= len(c1_min)
        x_dev = [(x_dev[i] - x_dist) ** 2 for i in range(len(x_dev))]
        y_dev = [(y_dev[i] - y_dist) ** 2 for i in range(len(y_dev))]
        x_var = sum(x_dev) / len(x_dev)
        y_var = sum(y_dev) / len(y_dev)
        print("Contour: avg dist,var:", (x_dist, y_dist), (x_var, y_var))
        self.draw_contours(img,c1_min,def_clr=(255,0,0))
        return c1_match, c1_unmatched, c1_min
   
    def add_border(self, img, bordersize):
        print("bordersize:", bordersize, img.shape[:2])
        row, col = img.shape[:2]
        bottom = img[row-2:row, 0:col]
        mean = cv2.mean(bottom)[0]
        border = cv2.copyMakeBorder(
            img,
            top=bordersize,
            bottom=bordersize,
            left=bordersize,
            right=bordersize,
            borderType=cv2.BORDER_CONSTANT,
            # value=[mean, mean, mean]
            value=[0, 0, 0]  # black
        )
        return border

    def rotate_bound(self, image, M):
        # ARD: assumes rotate over center, (not correct)
        # grab the dimensions of the image and then determine the
        # center
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        # perform the actual rotation and return the image
        return cv2.warpAffine(image, M, (nW, nH), borderMode=cv2.BORDER_CONSTANT)

    ###########

    def create_map(self, frame_num, action, prev_img_pth, curr_img_pth, done):
        curr_image = cv2.imread(curr_img_pth)
        curr_image_sift = kp_sift(curr_image)
        w = curr_image.shape[0]
        h = curr_image.shape[1]

        # avg (x,y) dif, var:  (2.37,1.62) (1715,423)
        pts = np.array([(0,0),(w,0),(w*28/32,h*21/32),(w*4/32,h*21/32)], dtype = "float32")
        # print("w,h:", w,h)
        # print("pts:", pts)
        # apply the four point tranform to obtain a "birds eye view" of the image
        above_view = self.four_point_transform(curr_image, pts)
        self.move_state(action, above_view)
        # show the original and warped images
        # image2 = cv2.drawKeypoints(curr_image,curr_image_sift.keypoints,None,color=(0,255,0), flags=0)
        # above_view2 = cv2.drawKeypoints(self.curr_move,self.curr_move_sift.keypoints,None,color=(0,255,0), flags=0)
        # cv2.imshow("Original", image2)
        # cv2.imshow("Warped", above_view2 )
        # cv2.waitKey(0)
        if self.map is None:
          self.move_rows,self.move_cols,ch = self.curr_move.shape
          # add a big border
          self.border_buffer = max(self.move_rows,self.move_cols)*self.border_multiplier
          print("border bufer:", self.border_buffer)
          self.map = self.add_border(self.curr_move, self.border_buffer)
          self.map_rows,self.map_cols,self.map_ch = self.map.shape
          self.map_sift = kp_sift(self.map)
          self.map_arr = asarray(self.map)

          self.robot_location = [(self.map_rows / 2+self.border_buffer), (self.border_buffer+self.map_cols)]
          self.VIRTUAL_MAP_SIZE = self.border_buffer * 2 + self.map_rows
          self.map_virtual_map_center = self.VIRTUAL_MAP_SIZE / 2
          kp_matches,notsogood = self.map_sift.compare_kp(self.map_sift)
          print("num matches:", len(kp_matches))
          # orient to self
          map_pts, map_pts2 = self.get_n_match_kps(kp_matches, self.map_sift, self.map_sift, 3)
          self.robot_orientation = cv.getAffineTransform(map_pts,map_pts)
          print("orientation: ", self.robot_orientation)
        else:
            rows,cols,ch = self.curr_move.shape
            new_map = self.add_border(self.curr_move, self.border_buffer)
            new_map_sift = kp_sift(new_map)
            # self.map_sift = kp_sift(self.map)
            kp_matches,notsogood = self.map_sift.compare_kp(new_map_sift)
            print("curr move:")
            new_rows,new_cols,new_ch = new_map.shape
            map_pts, new_map_pts = self.get_n_match_kps(kp_matches, self.map_sift, new_map_sift, 3)
            self.dif_avg, self.dif_var, self.min_pt, self.max_pt = self.kp_match_stats(kp_matches, new_map_sift, self.map_sift)
            print("new_map_pts:", new_map_pts)
            print("mappts:", map_pts)
            new_map_orientation = cv.getAffineTransform(map_pts, new_map_pts)
            # cv2.imshow("map", self.map)
            new_map = cv.warpAffine(new_map,new_map_orientation,(new_rows, new_cols), borderMode=cv2.BORDER_CONSTANT)
            # new_map = self.rotate_bound(self.curr_move,M)
            cv2.imshow("new_map", new_map)
            # cv2.waitKey(0)

            # ORB use BRIEF descriptors. But BRIEF performs poorly with rotation.
            # at this point, the rotated keypoints on the warped new_map can't be compared 
            # the map's keypoints.

            # The good news is that the SIFT patent has expired and can be used.
            # Try Panaramic stitching.
            stitch_images = [self.map, new_map]
  
            stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
            # stitcher = Stitcher_create()
            (status, stitched) = stitcher.stitch(stitch_images)
            # if the status is '0', then OpenCV successfully performed image
            # stitching
            if status == 0:
              stitch_rows,stitch_cols,stitch_ch = stitched.shape
              print("stitch rows,cols:", (stitch_rows, stitch_cols), (self.map_rows, self.map_cols), (new_rows, new_cols))
              # write the output stitched image to disk
              # cv2.imwrite(args["output"], stitched)
	      # display the output stitched image to our screen
              cv2.imshow("Stitched", stitched)
              cv2.waitKey(0)
              cv2.destroyAllWindows()
            # otherwise the stitching failed, likely due to not enough keypoints)
            # being detected
            else:
              print("[INFO] image stitching failed ({})".format(status))
            if True:
              self.map_sift = kp_sift(self.map)
              new_map_sift = kp_sift(new_map)
              # good_matches,notsogood = new_map_sift.compare_kp(self.map_sift,include_dist=False)
              good_matches,notsogood = self.map_sift.compare_kp(new_map_sift,include_dist=False)
              print("new map:")
              self.dif_avg, self.dif_var, self.min_pt, self.max_pt = self.kp_match_stats(good_matches, self.map_sift, new_map_sift)
              # self.dif_avg, self.dif_var, self.min_pt, self.max_pt = self.kp_match_stats(kp_matches, new_map_sift, self.map_sift)
              map_pts, new_map_pts = self.get_n_match_kps(good_matches, self.map_sift, new_map_sift, 6, ret_kp=True)

              # cv.drawMatchesKnn expects list of lists as matches.
              # img3 = cv.drawMatchesKnn(self.map,self.map_sift.get_kp(),
              #                     new_map,new_map_sift.get_kp(), good_matches, None,
              #                     flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
              # cv2.imshow("keymaps1",img3)
              # cv2.waitKey(0)
              above_view1 = cv2.drawKeypoints(self.map,map_pts,None,color=(0,255,0), flags=0)
              above_view2 = cv2.drawKeypoints(new_map,new_map_pts,None,color=(0,255,0), flags=0)
              for i, mpt in enumerate(map_pts):
                new_pt = new_map_pts[i].pt
                print(i,"kp dif:", (mpt.pt[0]-new_pt[0]), (mpt.pt[1]-new_pt[1]))


              ###################
              # _cnt => _contours
              # due to image rotation, the contours get warped, so no easy way to 
              # transform contours.
              #
              if False:
                map_cnt = self.get_contours(self.map)
                new_map_cnt = self.get_contours(new_map)
                new_map_cnt_match, new_map_cnt_unmatched, new_map_cnt_min = self.contours_difs(above_view1, new_map_cnt, map_cnt)
                map_cnt_match, map_cnt_unmatched, map_cnt_min = self.contours_difs(above_view2, map_cnt, new_map_cnt)
                # self.draw_contours(above_view1, map_cnt, "av1:")
                # self.draw_contours(above_view2, new_map_cnt, "av2:")

              ###################
              # Edge/line detection
              # note: includes black edges from border
              if False:
                edges1 = cv2.Canny(self.map,100,200)
                edges2 = cv2.Canny(new_map,100,200)
                cv2.namedWindow("above1", cv2.WINDOW_NORMAL) 
                cv2.namedWindow("above2", cv2.WINDOW_NORMAL) 
                cv2.imshow("above1",edges1)
                cv2.imshow("above2",edges1)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

              ###################
              # line detection
              # note: includes black edges from border
              if True:
                linesP, imglinesp = self.cvu.get_lines(self.map)
                cv2.imshow("Map LinesP - Probabilistic Line Transform", imglinesp);
                linesP, imglinesp = self.cvu.get_lines(new_map)
                cv2.imshow("NewMap LinesP - Probabilistic Line Transform", imglinesp);
                cv2.waitKey(0)
                cv2.destroyAllWindows()



#            # self.map_arr = np.ndarray(self.map_arr).reshape(self.map_rows, self.map_cols)
#            # self.map_arr = self.map_arr.reshape(self.map_rows, self.map_cols)
#            # map_arr2 = self.map_arr.reshape(self.map_rows, self.map_cols)
