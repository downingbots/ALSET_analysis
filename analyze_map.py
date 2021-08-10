# import the necessary packages
import numpy as np
import cv2
import argparse

# based on: 
# https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
class AnalyzeMap():

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
#        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
#        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
#        maxWidth = max(int(widthA), int(widthB))
        maxWidth = br[0]
        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
#        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
#        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
#        maxHeight = max(int(heightA), int(heightB))
        maxHeight = br[1]
        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order

#        dst = np.array([
#        	[0, 0],
#        	[maxWidth - 1, 0],
#        	[maxWidth - 1, maxHeight - 1],
#        	[0, maxHeight - 1]], dtype = "float32")
       
        maxHeight =  8/14 * maxHeight
        minWidth  =  9/16 * maxWidth 
        diff = round((maxWidth - minWidth)/2)
        x1 = diff
        x2 = maxWidth - diff
        y1 = maxHeight - diff
        y2 = diff
        
        dst = np.array([
                [ x1, y1], [x2, y1], [x2, y2], [x1, y2]
                ], dtype = "float32")
        print("dst:",dst)

        # compute the perspective transform matrix and then apply it
        # M = cv2.getPerspectiveTransform(rect, dst)
        M = cv2.getPerspectiveTransform(dst, rect)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight),  cv2.WARP_INVERSE_MAP)
        # return the warped image
        return warped

    def map(self, frame_num, action, prev_img, curr_img, done):
        image = cv2.imread(curr_img)
        w = image.shape[0]
        h = image.shape[1]
        pts = np.array([(w*3/16,0),(w*13/16,h*15/16),(w*13/16,0),(w*3/16,h*15/16)], dtype = "float32")
        print("w,h:", w,h, pts)
        # apply the four point tranform to obtain a "birds eye view" of
        # the image
        warped = self.four_point_transform(image, pts)
        # show the original and warped images
        cv2.imshow("Original", image)
        cv2.imshow("Warped", warped)
        cv2.waitKey(0)
