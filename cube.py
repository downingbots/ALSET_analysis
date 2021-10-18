#""
#Two times we know for sure where cube is:
# Mapping fix
#  - rotate to current orientation
#  - relocate to current location on map
#    -- Is this computed correctly based on previous position? code looks ok
#  - Looks like 2 lefts after Forward were too big?
#
#######
#  - compute offset_h as total distance from location
#  - Rotate total angle from origin
#    -> actually compute w
#  - compute orientation
#    -> keep track
#
#Forward pass
#  - Gather Frames and metadata as go
#    - rerun with data already gathered
#    - gather frames in order
#    - record known missing data
#      - Get cube / box at different sizes and location
#        -- Update location to more accurate location
#
#    - go backwards and fill in missing data
#      - don't store images, just metadata like path names
#
#    - ignore unknown objects --> too much noise  
#
#  - Grippers closed with bb gap and no optflow
#    - light should be shining on it.
#    - get LBP for shining on cube
#      - distinguish between light near/not near
#    - get width / height / color / LBP of cube in grasp
#      - Do LBP on cube at all pixel radius??
#    - bounding boxes
#    - store KPs on cube / box
#
#  - block out cube in grasp (and below) while mapping
#    - label different => as possible drivable space?
#      - Don't map
#      - compare with known LBP after PARK is complete
#
#  - LBP for SAFE TO DRIVE
#    LBP for cube
#    LBP for box
#
#
#  - Do similar with dropping cube in box.
#    - track box
#    - get box location
#      - want arm location when doing the drop
#    - get
#
#  - As first phase?
#    - no, need gripper BB
#    - differentiate between ground and object?
#
#  - Do backward run
#    - scan the App.
#      - ID object name to pick up or drop off
#      - look for operations involving cube or box
#      - Look for DROP or PICK-UP or PARK
#      - Go back one at a time to find 
#
#      fast forward to  
#
#  BB cube with grippers 
#  BB cube 
#  BB L/R gripper 
#  BB drivable space
#
#  - when pickup begins, the cube should be below the robot within range of open grippers
#    -inside bounding box
#
#
#
#  - We can track backward where the cube is based on reverse-mapping
#    - get width of cube
#    - get estimated location of cube
#    - Edge detection 
#
#  - gripper bounding boxes separated.
#    - gripper is 
#    switch
#
#
#
#
#Note: mapping: cube should be same size?
#
#""

#
# https://blog.pollithy.com/python/numpy/detect-orientation-of-cube-opencv
# 
# Use the a priori knowledge about cubes to find them by their edges not the 
# size of masks or the color of faces.
# 
#     Find the most important edges
#     Find all perpendicular lines that are equally long and close to each others
#     Connect the upper ends of lines and lower ends of lines
#     Cluster the ends of lines with DBSCAN to find the edges of the cube
import cv2
import numpy as np
from sklearn.cluster import *
import math, random
# from shapely.geometry import *
from analyze_keypoints import *
from cv_analysis_tools import *
from utilborders import *


def find_kp(img, kp_mode="SIFT"):
    if kp_mode == "BEST":
      kp_mode_list = ["SIFT", "ORB"] 
    else:
      kp_mode_list = [kp_mode]
    for mode in kp_mode_list:
      KPs = Keypoints(img,kp_mode=mode)
      kp_img = KPs.drawKeypoints()
      cv2.imshow("keypoints",kp_img)
      cv2.waitKey(0)



# allow detection of tiny squares
def find_square(img):
    # already done by find_cube:
    # convert the stitched image to grayscale and threshold it
    # such that all pixels greater than zero are set to 255
    # (foreground) while all others remain 0 (background)
    shape, approximations = None, None
    squares = []
    # find all external contours in the threshold image then find
    # the *largest* contour which will be the contour/outline of
    # the stitched image
    imagecontours, hierarchy = cv2.findContours(img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    # for each of the contours detected, the shape of the contours is approximated 
    # using approxPolyDP() function and the contours are drawn in the image using 
    # drawContours() function
    # For our border case, there may be a few dots or small contours that won't
    # be considered part of the border.
    # print("real_map_border count:", len(imagecontours))
    if len(imagecontours) > 1:
      # print("hierarchy:", hierarchy)
      for i, c  in enumerate(imagecontours):
        area = cv2.contourArea(c)
        M = cv2.moments(c)
        # print(i, "area, moment:", area, M, len(c))
        # print(i, "area:", area, len(c))
    for count in imagecontours:
      # epsilon = 0.01 * cv2.arcLength(count, True)
      epsilon = 0.01 * cv2.arcLength(count, True)
      approximations = cv2.approxPolyDP(count, epsilon, True)
      # e.g. [[[224 224]] [[252 372]] [[420 372]] [[447 224]]]
      #the name of the detected shapes are written on the image
      i, j = approximations[0][0] 
      if len(approximations) == 3:
        shape = "Triangle"
      elif len(approximations) == 4:
        shape = "Trapezoid"
        area = cv2.contourArea(approximations)
        # if area > 100 &  cv2.isContourConvex(approximations):
        if cv2.isContourConvex(approximations):
          maxCosine = -100000000000000
          for j in range(2, 5):
            cosine = abs(middle_angle(approximations[j%4][0], approximations[j-2][0], approximations[j-1][0]))
            print("cosine ", cosine, ":", approximations[j%4][0], approximations[j-2][0], approximations[j-1][0])
            maxCosine = max(maxCosine, cosine)
          # if cosines of all angles are small
          # (all angles are ~90 degree) then write quandrange
          # vertices to resultant sequence
          if maxCosine < 0.3 and maxCosine >= 0:
            shape = "Square"
            squares.append(approximations)
            print("square found:", approximations)
          else:
            print("maxCos:", maxCosine)
        else:
          print("non convex contour", approximations)
      elif len(approximations) == 5:
        shape = "Pentagon"
      elif 6 < len(approximations) < 15:
        shape = "Ellipse"
      else:
        shape = "Circle"
      # if len(imagecontours) > 1:
      #   cv2.putText(thresh,shape,(i,j),cv2.FONT_HERSHEY_COMPLEX,1,0,2)
      #   cv2.waitKey(0)
      # print("map shape:", shape, approximations)
      #displaying the resulting image as the output on the screen
      # imageread = mapimg.copy()
      # print("contour:", count)
      # print("approx contour:", approximations)
      # return shape, approximations
      print("shape:", shape, squares)
    return squares

def preprocess_cube2(img_path):
      cvu = CVAnalysisTools()
      img,mean_diff,rl_bb = cvu.adjust_light(img_path)
      orig_img = img.copy()
      img = cv2.Canny(img, 50, 200, None, 3)
      # thresh = 10
      thresh = 20
      img = cv2.GaussianBlur(img, (5, 5), 0)
      # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      gray_img = cv2.dilate(img,None,iterations = 2)
      gray_img = cv2.threshold(gray_img, thresh, 255, cv2.THRESH_BINARY_INV)[1]
      return gray_img, orig_img

def preprocess_cube(img):
    img_gray = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2GRAY)
    noise_removal = cv2.bilateralFilter(img_gray, 9,75,75)
    # thresh_image = cv2.adaptiveThreshold(noise_removal, 255,
    thresh_image = cv2.adaptiveThreshold(img_gray, 255,
      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
      cv2.THRESH_BINARY, 11, 2)
    
    # this dilate and erode section is not optimal and 
    # the sizes of the kernels is the result multiple attempts
    kernel = np.ones((10,1), np.uint8)
    dilated_thresh_image = cv2.dilate(thresh_image, kernel, iterations=1)
    # cv2.imshow("dti1", dilated_thresh_image)
    # cv2.waitKey(0)
    kernel = np.ones((10,1), np.uint8)
    dilated_thresh_image = cv2.erode(dilated_thresh_image, kernel, iterations=1)
    # cv2.imshow("dti2", dilated_thresh_image)
    # cv2.waitKey(0)
    kernel = np.ones((5,5), np.uint8)
    dilated_thresh_image = cv2.erode(dilated_thresh_image, kernel, iterations=1)
    # cv2.imshow("dti3", dilated_thresh_image)
    # cv2.waitKey(0)
    kernel = np.ones((20,1), np.uint8)
    dilated_thresh_image = cv2.dilate(thresh_image, kernel, iterations=1)
    # cv2.imshow("dti4", dilated_thresh_image)
    # cv2.waitKey(0)
    kernel = np.ones((25,1), np.uint8)
    dilated_thresh_image = cv2.erode(dilated_thresh_image, kernel, iterations=1)
    # cv2.imshow("dti5", dilated_thresh_image)
    # cv2.waitKey(0)
    kernel = np.ones((5,5), np.uint8)
    dilated_thresh_image = cv2.erode(dilated_thresh_image, kernel, iterations=1)
    return dilated_thresh_image, img
   
    
def find_cube(img):
    return
    gray, img = preprocess_cube2(img)
    # cv2.imshow("dti", img)
    # cv2.waitKey(0)
    find_kp(img)
    # find_square(gray)

    # invert the black and white image for the LineDetection
    inverted_dilated_thresh_image = cv2.bitwise_not(gray)
    find_square(inverted_dilated_thresh_image)
    
    img2 = img.copy()
    # Control the lines we want to find (minimum size and minimum distance between two lines)
    # minLineLength = 100
    minLineLength = 10
    # minLineLength = 5
    # maxLineGap = 80
    maxLineGap = 8
    # maxLineGap = 5
    # threshold = 100
    threshold = 10
    # threshold = 5
    
    # Keep in mind that this is opencv 2.X not version 3 (the results of the api differ)
    # ARD: fixed for python3
    lines = cv2.HoughLinesP(inverted_dilated_thresh_image, 
        rho = 1,
        theta = 1 * np.pi/180,
        lines=np.array([]),
        threshold = threshold,
        minLineLength = minLineLength,
        maxLineGap = maxLineGap)
    
    # cv2.imshow("inverted_dilated_thresh_image:",inverted_dilated_thresh_image)
    # cv2.waitKey(0)
    
    # Now select the perpendicular lines:
    
    # storage for the perpendicular lines
    correct_lines = np.array([])
    
    if lines is not None and lines.any():
    # iterate over every line
     for l in lines:
      for x1,y1,x2,y2 in l:
        print("line:",x1,y1,x2,y2)
        
        # calculate angle in radian (if interesten in this see blog entry about arctan2)
        angle = np.arctan2(y1 - y2, x1 - x2)
        # convert to degree
        degree = abs(angle * (180 / np.pi))
        
        # only use lines with angle between 85 and 95 degrees 
        if 85 < degree < 95:
          # draw the line on img2
          print("line: perpendicular")
          cv2.line(img2,(x1,y1),(x2,y2),(0,255,0),2)
                
          # correct upside down lines (switch lower and upper ends)
          if y1 < y2:
            temp = y2
            y2 = y1
            y1 = temp
            temp = x2
            x2 = x1
            x1 = temp
                    
          # store the line 
          correct_lines = np.concatenate((correct_lines, np.array([x1,y1,x2,y2], \
                           dtype = "uint32")))        
          
          # draw the upper and lower end on img2
          cv2.circle(img2, (x1,y1), 2, (0,0,255), thickness=2, lineType=8, shift=0)
          cv2.circle(img2, (x2,y2), 2, (255,0,0), thickness=2, lineType=8, shift=0)
    
    # lots of storage for findings
    squares = np.array([])
    lower_points = np.array([])
    upper_points = np.array([])
    top_lines = np.array([])
    bottom_lines = np.array([])
    areas = np.array([])
    
    # reshape the numpy array to a matrix with four columns
    correct_lines = correct_lines.reshape(-1, 4)
    
    
    for a_x1, a_y1, a_x2, a_y2 in correct_lines:
      print("correct_line: ", a_x1, a_y1, a_x2, a_y2)
      line_length = np.linalg.norm(np.array([a_x1, a_y1])-np.array([a_x2, a_y2]))
    
      for b_x1, b_y1, b_x2, b_y2 in correct_lines:
        line_length_b = np.linalg.norm(np.array([b_x1, b_y1])-np.array([ b_x2, b_y2]))
            
        # O(n^2)
        # Compare all lines with each others
            
        # only those with similar length
        if 0.9 > max(line_length, line_length_b)/min(line_length, line_length_b) > 1.1:
          continue
        
        # distance between the top points of the lines
        dist = np.linalg.norm(np.array([ a_x1, a_y1 ]) - np.array([b_x1, b_y1]))
        
        # lines that are too close to eachs others (or even the same line) excluded
        # also exclude those too distant
        if 20 < dist < line_length:
              
          # distance between lower points
          dist = np.linalg.norm(np.array([ a_x2, a_y2 ]) - np.array([b_x2, b_y2]))
                
          # if the lower points also match
          if 20 < dist < line_length:
                  # NOW: create the line between the uppder and lower ends
            top_lines = np.concatenate((top_lines, np.array([a_x1,a_y1,b_x1,b_y1], \
                           dtype = "uint32")))
            angle_top_line = np.arctan2(int(a_y1) - int(b_y1), int(a_x1) - int(b_x1))
            degree_top_line = abs(angle_top_line * (180 / np.pi))
    
            bottom_lines = np.concatenate((bottom_lines, np.array([a_x1,a_y1,b_x1,b_y1], \
                           dtype = "uint32")))
            angle_bottom_line = np.arctan2(int(a_y1) - int(b_y1), int(a_x1) - int(b_x1))
            degree_bottom_line = abs(angle_bottom_line * (180 / np.pi))
            
            # hack around 0 degree
            if degree_top_line == 0 or degree_bottom_line == 0:
              degree_top_line += 1
              degree_bottom_line += 1
              
            # if the upper and lower connection have an equal angle 
            # they are interesting corners for a cube's face
            if 0.8 > max(degree_top_line, degree_bottom_line)/min(degree_top_line, \
                             degree_bottom_line) > 1.2:
              print("too much difference in line degrees")
              continue
                        
            # draw the upper line and store its ends
            cv2.line(img2, (int(a_x2), int(a_y2)), (int(b_x2), int(b_y2)), (0,0,255), 1)
            upper_points = np.concatenate((upper_points, np.array([a_x2, a_y2], \
                             dtype = "uint32")))
            upper_points = np.concatenate((upper_points, np.array([b_x2, b_y2], \
                             dtype = "uint32")))
            
            # draw the lower line and store its ends
            cv2.line(img2, (int(a_x1), int(a_y1)), (int(b_x1), int(b_y1)), (255,0,0), 1)
            lower_points = np.concatenate((lower_points, np.array([a_x1, a_y1], \
                             dtype = "uint32")))
            lower_points = np.concatenate((lower_points, np.array([b_x1, b_y1], \
                             dtype = "uint32")))
                    
            # store the spanned tetragon
            area = np.array([  
              int(a_x1), int(a_y1),
              int(b_x1), int(b_y1),
              int(a_x2), int(a_y2), 
              int(b_x2), int(b_y2)
            ], dtype = "int32")
            areas = np.concatenate((areas, area))
            print("spanned tetragon:", area)

    def centroidnp(arr1, arr2):
      # this method calculates the center of an array of points
      # print("centroidnp: ", arr.shape, arr.size)
      # length = arr.shape[0]
      # length = arr.size
      # sum_x = np.sum(arr[:, 0])
      # sum_y = np.sum(arr[:, 1])
      sum_x = np.sum(arr1[:])
      sum_y = np.sum(arr2[:])
      length_x = len(arr1)
      length_y = len(arr2)
      if length_x == length_y:
        print("len", length_y)
      else:
        print("len", length_x, length_y)
      return (int(np.round(sum_x/length_x)), int(np.round(sum_y/length_y)))
    
    # Promising results of the cluster algorithm
    corners = np.array([])
    lower_corners = np.array([])
    upper_corners = np.array([])
    
    # --------------------------------------------------
    # Cluster the lower points
    # --------------------------------------------------
    
    # reshape the array to int32 matrix with two columns
    vectors = np.int32(lower_points.reshape(-1, 2))
    
    if vectors.any():
      # API of DBSCAN from scikit-learn
      # http://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
        
      # Run DBSCAN with eps=30 means that the minimum distance between two clusters is 30px
      # and that points within 30px range will be part of the same cluster
      db = DBSCAN(eps=75, min_samples=10).fit(vectors)
      core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
      core_samples_mask[db.core_sample_indices_] = True
      labels = db.labels_
    
      # Number of clusters in labels, ignoring noise if present.
      n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    
      # iterate over the clusters
      for i in set(db.labels_):
        if i == -1:
          # -1 is noise
          continue
                
        color = (random.randint(0, 255),random.randint(0, 255),random.randint(0, 255))
        index = db.labels_ == i
            
        # draw the members of the cluster
        for (point_x, point_y) in zip(vectors[index,0], vectors[index,1]):
          cv2.circle(img2,  (point_x, point_y), 5, color, thickness=1, lineType=8, shift=0)
    
        # calculate the centroid of the members
        print("centroidnp:", vectors[index,0], vectors[index,1])
        # cluster_center = centroidnp(np.array(zip(np.array(vectors[index,0]),\
        #             np.array(vectors[index,1]))))
        cluster_center = centroidnp(np.array(vectors[index,0]),\
                    np.array(vectors[index,1]))
        print("cluster_center:", cluster_center)
            
        # draw the the cluster center
        cv2.circle(img2,  cluster_center, 5, color, thickness=10, lineType=8, shift=0)
          
        # store the centroid as corner
        corners = np.concatenate((corners, np.array([cluster_center[0], cluster_center[1]],\
                    dtype = "uint32")))
        lower_corners = np.concatenate((lower_corners, 
                    np.array([cluster_center[0], cluster_center[1]], dtype = "uint32")))
                            
    # --------------------------------------------------
    # Cluster the upper points
    # = same as with lower points
    # --------------------------------------------------
    
    vectors = np.int32(upper_points.reshape(-1, 2))
    
    if vectors.any():
      db = DBSCAN(eps=75, min_samples=10).fit(vectors)
      core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
      core_samples_mask[db.core_sample_indices_] = True
      labels = db.labels_
    
      n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    
      for i in set(db.labels_):
        if i == -1:
          continue
        color = (random.randint(0, 255),random.randint(0, 255),random.randint(0, 255))
        index = db.labels_ == i
        for (point_x, point_y) in zip(vectors[index,0], vectors[index,1]):
          cv2.circle(img2,  (point_x, point_y), 5, color, thickness=1, lineType=8, shift=0)
        # cluster_center = centroidnp(np.array(zip(np.array(vectors[index,0]),\
        #             np.array(vectors[index,1]))))
        cluster_center = centroidnp(np.array(vectors[index,0]), np.array(vectors[index,1]))
        print("cluster_center:", cluster_center)
        cv2.circle(img2,  cluster_center, 5, color, thickness=10, lineType=8, shift=0)
        corners = np.concatenate((corners, np.array([cluster_center[0], cluster_center[1]], \
                    dtype = "uint32")))
        upper_corners = np.concatenate((upper_corners, \
                    np.array([cluster_center[0], cluster_center[1]], dtype = "uint32")))
    
      cv2.imshow("cube:",img2)
      cv2.waitKey(0)

# img_path = "/tmp/d5f6fec0-b602-11eb-abe9-16f63a1aa8c9.jpg"
# img = cv2.imread(img_path)
# find_cube(img)
