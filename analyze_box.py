#!/usr/bin/env python
from alset_state import *
from matplotlib import pyplot as plt
import cv2
import STEGO.src
from alset_stego import *
import numpy as np
import copy
from math import sin, cos, pi, sqrt
from utilborders import *
from cv_analysis_tools import *
from dataset_utils import *
from PIL import Image
import imutils
from skimage.feature import peak_local_max
from scipy import ndimage
from scipy.cluster.hierarchy import ward
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN
from sklearn import metrics, linear_model
import matplotlib.image as mpimg
from sortedcontainers import SortedList, SortedSet, SortedDict

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''
#def find_epsilon(self, pc):
def find_epsilon(distances):
      from rotor import Rotor
      from sklearn.neighbors import NearestNeighbors

#      neigh = NearestNeighbors(n_neighbors=2)
#      nbrs = neigh.fit(pc)
#      distances, indices = nbrs.kneighbors(pc)
      distances = np.sort(distances, axis=0)
      distance2 = distances[:, 1] # taking the second column of the sorted distances
      plt.rcParams['figure.figsize'] = (5,3) # setting the figure size
      plt.plot(distance2) # plotting the distances
      plt.show() # showing the plot

      print("distances len:", len(distances))
      j = None
      k = None
      for i in range(len(distances[:])):
        distances[i][0] = i
        if distances[i][1] == 0:
          j = i
        elif distances[i][1] > 1:
          if k == None:
            k = i
            print("eps dist below/abv 1: ", i, len(distances - i))
          # distances = distances[:(i-1)]
          # break
        # if distances[i][0] > 0 or distances[i][1] > 0:
          # print("distances: ",i,distances[i])
      if j != None:
        distances = distances[(j+1):]
      rotor = Rotor()
      rotor.fit_rotate(distances)
      elbow_index = rotor.get_elbow_index()
      # rotor.plot_elbow()
      # rotor.plot_knee()
      # print("distances: ",distances)
      # distance1 = distances[:,1]
      # distance2 = distances[1,:]
      # print("elbow: ",elbow_index, distance1[elbow_index], distance2[elbow_index])
      print("epsilon: ",elbow_index, distances[elbow_index][1])
      return distances[elbow_index][1]


def analyze_clusters(self, feature_img):
    # feature_image=np.reshape(labimg, [-1, 3])
    
    db = DBSCAN(eps=5, min_samples=50, metric = 'euclidean',algorithm ='auto')
    db.fit(feature_image)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    if False:
      print('#clusters, noise: ', n_clusters_, n_noise_)
      print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
      print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
      print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
      print("Adjusted Rand Index: %0.3f"
            % metrics.adjusted_rand_score(labels_true, labels))
      print("Adjusted Mutual Information: %0.3f"
            % metrics.adjusted_mutual_info_score(labels_true, labels))
      print("Silhouette Coefficient: %0.3f"
            % metrics.silhouette_score(X, labels))

    if n_clusters_ > 0:
      plt.figure(2)
      plt.subplot(2, 1, 1) # numrows numcols index
      plt.imshow(feature_img)
      plt.axis('off')

      plt.subplot(2, 1, 2)
      plt.imshow(np.reshape(labels2, [rows, cols]))
      plt.axis('off')
      plt.show()

    for c_id in set(db.labels_):
      if c_id != -1:
        for i, label in enumerate(db.labels_):
            if db.labels_[i] == c_id:
                # convert to gray_feature_img????
                # print("label", c_id, i, feature_img[i])
                counter[c_id] += 1
                running_sum[c_id] += feature_img[i]
                # print(c_id, "shape append", feature_img[i])
                self.clusters[c_id].cluster['shape'].append(feature_img[i])
      c = self.clusters[c_id]
      c.cluster['kp_c_mapping'] = kp_list[c_id]
      center = running_sum[c_id] / counter[c_id]
      c.cluster['center'] = center
      # print("center for clust", c , " is ", self.clusters[c].cluster['mean'])
      # normalize shape
      c.normalize()
      # need to normalize KP:
      print("cluster ",c_id," len", len(c.cluster['shape']))
    print("num_clusters:",len(self.clusters))
    return True

def analyze_x_y_dist(x_y_distances, imgnums):
    db = DBSCAN(eps=5, min_samples=50, metric = 'euclidean',algorithm ='auto')
    db.fit(x_y_distances)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    if False:
      print('#clusters, noise: ', n_clusters_, n_noise_)
      print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
      print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
      print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
      print("Adjusted Rand Index: %0.3f"
            % metrics.adjusted_rand_score(labels_true, labels))
      print("Adjusted Mutual Information: %0.3f"
            % metrics.adjusted_mutual_info_score(labels_true, labels))
      print("Silhouette Coefficient: %0.3f"
            % metrics.silhouette_score(X, labels))

    if False and n_clusters_ > 0:
      # plt.figure(1)
      # plt.subplot(2, 1, 1) 
      plt.imshow(np.reshape(labels, x_y_distances))
      plt.axis('off')
      plt.show()

    clusters = {}
    # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    clusters['x_y_mapping'] = [[]  for i in range(n_clusters_)]
    clusters['mean'] = [[None, None]  for i in range(n_clusters_)]
    clusters['stddev'] = [[None, None]  for i in range(n_clusters_)]
    counter = [0 for i in range(n_clusters_)]
    running_sum = [[0, 0] for i in range(n_clusters_)]
    running_sum2 = [[0, 0] for i in range(n_clusters_)]
    best_cluster = None
    min_dist = 1000000000000
    # histogram for pixel dist
    pixel_dist = [[0 for i in range(30)],[0 for i in range(30)]]
    pixel_dist_pair = [[[0,[]] for i in range(11)] for j in range(11)]
    num_imgs = imgnums[-1]
    img_dist = []
    img_cid_dist = []
    for c_id in set(db.labels_):
      if c_id != -1:
        for i, label in enumerate(db.labels_):
            if db.labels_[i] == c_id:
                # convert to gray_feature_img????
                # print("label", c_id, i, x_y_distances[i])
                counter[c_id] += 1
                for k in range(2):
                  running_sum[c_id][k] += x_y_distances[i][k]
                  running_sum2[c_id][k] += x_y_distances[i][k] * x_y_distances[i][k]
                  histogrm = int(max(0, min(29, (x_y_distances[i][k] + 15))))
                  pixel_dist[k][histogrm] += 1
                x = int(max(0, min(10, (x_y_distances[i][0] + 5))))
                y = int(max(0, min(10, (x_y_distances[i][1] + 5))))
                pixel_dist_pair[x][y][0] += 1
                pixel_dist_pair[x][y][1].append(imgnums[i])
                while len(img_dist) < imgnums[i]+1:
                  img_dist.append([])
                  img_cid_dist.append({})
                img_dist[imgnums[i]].append((int(x_y_distances[i][0]), int(x_y_distances[i][1])))
                try:
                  img_cid_dist[imgnums[i]][c_id].append((int(x_y_distances[i][0]), int(x_y_distances[i][1])))
                except:
                  img_cid_dist[imgnums[i]][c_id] = []
                  img_cid_dist[imgnums[i]][c_id].append((int(x_y_distances[i][0]), int(x_y_distances[i][1])))
           
        for k in range(2):
          clusters['mean'][c_id][k] = running_sum[c_id][k] / counter[c_id]
          clusters['stddev'][c_id][k] = sqrt(running_sum2[c_id][k] / counter[c_id] - running_sum[c_id][k] * running_sum[c_id][k] / counter[c_id] / counter[c_id])
#          for i in range(30):
#            val = i - 15
#            print("pixel dist cnt: ", val, k, pixel_dist[k][i])
        d = abs(clusters['mean'][c_id][0])+abs(clusters['mean'][c_id][1])
        if min_dist > d:
          min_dist = d
          best_cluster = c_id
          # best_x_y_dist = clusters['mean'][c_id]
        print("mean for clust", c_id, " is ", clusters['mean'][c_id], clusters['stddev'][c_id], counter[c_id])
#        for x in range(11):
#          for y in range(11):
#            if pixel_dist_pair[x][y][0] > 0:
#              print("pixel cnt: ", (x-5, y-5), pixel_dist_pair[x][y])
#        for i_n, xy_lst in enumerate(img_dist):
#          if len(xy_lst) > 0:
#            print("img#",i_n, img_dist[i_n])
    if True:
        print("IMG CID DIST")
        for i_n, c_id_lst in enumerate(img_cid_dist):
          for c_n, c_id in enumerate(img_cid_dist[i_n]):
            # for xy_n, xy_lst in enumerate(img_cid_dist):
              # if len(xy_lst) > 0:
                print("img#",i_n, c_id, img_cid_dist[i_n][c_id])
    print("num_clusters:",len(clusters), best_cluster)
    return clusters, best_cluster

# display with: https://github.com/cnr-isti-vclab/meshlab/releases
def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

def box_contours(imgL, imgR):
    INFINITE = 1000000000000000000
    gray_img = cv2.cvtColor(obj_img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.Canny(gray_img, 50, 200, None, 3)
    # thresh = 10
    thresh = 20
    gray_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    gray_img = cv2.dilate(gray_img,None,iterations = 2)
    gray_img = cv2.threshold(gray_img, thresh, 255, cv2.THRESH_BINARY_INV)[1]
    # gray_img = cv2.bitwise_not(gray_img)
    imagecontours, hierarchy = cv2.findContours(gray_img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

def cluster_gripper():
      Z = img.copy()
      Z = Z.reshape((-1,3))
      # convert to np.float32
      Z = np.float32(Z)
      # define criteria, number of clusters(K) and apply kmeans()
      criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
      K = num_clusters
      # compactness : sum of squared distance from each point to their centers.
      # labels : the label array where each element marked '0', '1'.....
      # centers : This is array of centers of clusters.
      ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
      # Now convert back into uint8, and make original image
      # print("compactness, centers:", ret, center)
      # ret is a single float, label is ?, center is RGB
      center = np.uint8(center)
      res = center[label.flatten()]
      res2 = res.reshape((img.shape))
      return res2

def findparallel(lines, allowed_delta):
  lines1 = []
  for i in range(len(lines)):
    for j in range(len(lines)):
        if (i == j):continue
        if abs(lines[i][1] - lines[j][1]) <= allowed_delta:
             # You've found a parallel line!
             lines1.append((i,j))
  return lines1

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[0][2], line2[0][0] - line2[0][2])
    ydiff = (line1[0][1] - line1[0][3], line2[0][1] - line2[0][3])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       return None
       # raise Exception('lines do not intersect')

    # d = (det(line1[0][0:1], line1[0][2:3]), det(line2[0][0:1],line2[0][2:3]))
    d = (det(line1[0][0:2], line1[0][2:4]), det(line2[0][0:2],line2[0][2:4]))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    # print("intersection: ", line1, line2, [x,y])
    if ( x >= max(line1[0][0], line1[0][2]) + 2
      or x <= min(line1[0][0], line1[0][2]) - 2
      or x >= max(line2[0][0], line2[0][2]) + 2
      or x <= min(line2[0][0], line2[0][2]) - 2
      or y <= min(line1[0][1], line1[0][3]) - 2
      or y >= max(line1[0][1], line1[0][3]) + 2
      or y <= min(line2[0][1], line2[0][3]) - 2
      or y >= max(line2[0][1], line2[0][3])) + 2:
      # intersection point outside of line segments' range
      return None
       
    xdiff = (line1[0][0] - line1[0][2], line2[0][0] - line2[0][2])
    ydiff = (line1[0][1] - line1[0][3], line2[0][1] - line2[0][3])
    return [x, y]


# Is c between a and b?
def isBetween(a, b, c):
    crossproduct = (c.y - a.y) * (b.x - a.x) - (c.x - a.x) * (b.y - a.y)

    epsilon = .5
    # compare versus epsilon for floating point values, or != 0 if using integers
    if abs(crossproduct) > epsilon:
        return False

    dotproduct = (c.x - a.x) * (b.x - a.x) + (c.y - a.y)*(b.y - a.y)
    if dotproduct < 0:
        return False

    squaredlengthba = (b.x - a.x)*(b.x - a.x) + (b.y - a.y)*(b.y - a.y)
    if dotproduct > squaredlengthba:
        return False

    return True

def is_same_line(line1, line2):
    if is_parallel(line1, line2) and line_intersection(line1, line2) is not None:
      # print("same line: line1, line2", line1, line2)
      return True

def is_broken_line(line1, line2):
    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]
    def get_dist(x1,y1, x2, y2):
        return sqrt((x2-x1)**2 + (y2-y1)**2)

    if not is_parallel(line1, line2):
      return None
    if line_intersection(line1, line2) is not None:
      return None

    dist0 = get_dist(line1[0][0], line1[0][1], line2[0][0], line1[0][1])
    dist1 = get_dist(line1[0][0], line1[0][1], line2[0][2], line1[0][3])
    dist2 = get_dist(line1[0][2], line1[0][3], line2[0][0], line1[0][1])
    dist3 = get_dist(line1[0][2], line1[0][3], line2[0][2], line1[0][3])

    extended_line = None
    if dist0 == max(dist0,dist1,dist2,dist3):
      extended_line = [[line1[0][0], line1[0][1], line2[0][0], line1[0][1]]]
    elif dist1 == max(dist0,dist1,dist2,dist3):
      extended_line = [[line1[0][0], line1[0][1], line2[0][2], line1[0][3]]]
    elif dist2 == max(dist0,dist1,dist2,dist3):
      extended_line = [[line1[0][2], line1[0][3], line2[0][0], line1[0][1]]]
    elif dist3 == max(dist0,dist1,dist2,dist3):
      extended_line = [[line1[0][2], line1[0][3], line2[0][2], line1[0][3]]]
   
    if not (is_parallel(line1, extended_line) and is_parallel(line2, extended_line)):
      return None
    # print("broken line: ", line1, line2, extended_line)
    return extended_line

def same_line_score(line1, line2, delta=[0,0]):
    # smaller is better / more likely to be exact same line
    angle = np.arctan2((line1[0][0]-line2[0][2]), (line1[0][1]-line2[0][3]))
    if line_intersection(line1, line2) is not None:
      intsect = 1.0
    else:
      line2_delta = [[line2[0][0] + delta[0], lin2[0][1] + delta[1]]]
      broken_line = is_broken_line(u_line, h_line)
      if (broken_line is None):
        intsect = 3.0
      else:
        l1_len = np.sqrt((line1[0][0] - (line1[0][2]))**2 +
                         (line1[0][1] - (line1[0][3]))**2)
        l2_len = np.sqrt((line2_delta[0][0] - (line2_delta[0][2]))**2 +
                         (line2_delta[0][1] - (line2_delta[0][3]))**2)
        l3_len = np.sqrt((broken_line[0] - (broken_line[2]))**2 +
                         (broken_line[1] - (broken_line[3]))**2)
        intsect = 1.0 + max(2.0 * (l3_len / (l1_len+l2_len)), 2.0)

    # dist of end points
    d1 = np.sqrt((line1[0][0] - (line2[0][0] + delta[0]))**2 +
                 (line1[0][1] - (line2[0][1] + delta[1]))**2)
    d2 = np.sqrt((line1[0][2] - (line2[0][2] + delta[0]))**2 +
                 (line1[0][3] - (line2[0][3] + delta[1]))**2)
    # smaller is better
    score = angle*intsect*(d1+d2)
    return score

def is_parallel(line1, line2):
    angle1 = np.arctan2((line1[0][0]-line1[0][2]), (line1[0][1]-line1[0][3]))
    angle2 = np.arctan2((line2[0][0]-line2[0][2]), (line2[0][1]-line2[0][3]))
    allowed_delta = .1
    if abs(angle1-angle2) <= allowed_delta:
      # print("is_parallel line1, line2", line1, line2, angle1, angle2)
      return True
    if abs(np.pi-abs(angle1-angle2)) <= allowed_delta:
      # note: .01 and 3.14 should be considered parallel
      # print("is_parallel line1, line2", line1, line2, angle1, angle2)
      return True
    return False


# mean disp, angle, linlen: False None None 0.048744851309931586 41.048751503547585 [[203, 182, 205, 223]] 1
# mean disp, angle, linlen: False None None 0.04874585130993159 41.048751503547585 [[204, 182, 206, 223]] 1

def parallel_dist(line1, line2, dbg=False):
    if not is_parallel(line1, line2):
      return None
    # line1, line2 [[151 138 223 149]] [[ 38  76 139  96]]

    # y = mx + c
    # pts = [(line1[0][0], line1[0][2]), (line1[0][1], line1[0][3])]
    pts = [(line1[0][0], line1[0][1]), (line1[0][2], line1[0][3])]
    x_coords, y_coords = zip(*pts)
    A = np.vstack([x_coords,np.ones(len(x_coords))]).T
    l1_m, l1_c = np.linalg.lstsq(A, y_coords)[0]
    if dbg:
      print("x,y,m,c", x_coords, y_coords, l1_m, l1_c)

    pts = [(line2[0][0], line2[0][1]), (line2[0][2], line2[0][3])]
    x_coords, y_coords = zip(*pts)
    A = np.vstack([x_coords,np.ones(len(x_coords))]).T
    l2_m, l2_c = np.linalg.lstsq(A, y_coords)[0]
    if dbg:
      print("x,y,m,c", x_coords, y_coords, l2_m, l2_c)

    # coefficients = np.polyfit(x_val, y_val, 1)
    # Goal: set vert(y) the same on both lines, compute horiz(x).
    # with a vertical line, displacement will be very hard to compute
    # unless same end-points are displaced.
    if ((line1[0][0] >= line2[0][0] >= line1[0][2]) or
        (line1[0][0] <= line2[0][0] <= line1[0][2])):
      x1 = line2[0][0]
      y1 = line2[0][1]
      y2 = y1
      x2 = (y2 - l1_c) / l1_m
      # y2 = l1_m * x1 + l1_c
      # x2 = (y1 - l2_c) / l2_m
    elif ((line1[0][0] >= line2[0][2] >= line1[0][2]) or
          (line1[0][0] <= line2[0][2] <= line1[0][2])):
      x1 = line2[0][2]
      y1 = line2[0][3]
      y2 = y1
      x2 = (y2 - l1_c) / l1_m
      # y2 = l1_m * x1 + l1_c
      # x2 = (y1 - l2_c) / l2_m
    elif ((line2[0][0] >= line1[0][0] >= line2[0][2]) or
          (line2[0][0] <= line1[0][0] <= line2[0][2])):
      x1 = line1[0][0]
      y1 = line1[0][1]
      y2 = y1
      x2 = (y2 - l2_c) / l2_m
    elif ((line2[0][0] >= line1[0][2] >= line2[0][2]) or
          (line2[0][0] <= line1[0][2] <= line2[0][2])):
      x1 = line1[0][2]
      y1 = line1[0][3]
      y2 = y1
      x2 = (y2 - l2_c) / l2_m
    elif ((line1[0][1] >= line2[0][1] >= line1[0][3]) or
          (line1[0][1] <= line2[0][1] <= line1[0][3])):
      x1 = line2[0][0]
      y1 = line2[0][1]
      y2 = y1
      x2 = (y2 - l1_c) / l1_m
    elif ((line1[0][1] >= line2[0][3] >= line1[0][3]) or
          (line1[0][1] <= line2[0][3] <= line1[0][3])):
      y1 = line2[0][3]
      x1 = line2[0][2]
      y2 = y1
      x2 = (y2 - l1_c) / l1_m
    elif ((line2[0][1] >= line1[0][1] >= line2[0][3]) or
          (line2[0][1] <= line1[0][1] <= line2[0][3])):
      y1 = line1[0][1]
      x1 = line1[0][0]
      y2 = y1
      x2 = (y2 - l2_c) / l2_m
    elif ((line2[0][1] >= line1[0][3] >= line2[0][3]) or
          (line2[0][1] <= line1[0][3] <= line2[0][3])):
      y1 = line1[0][3]
      x1 = line1[0][2]
      y2 = y1
      x2 = (y2 - l2_c) / l2_m
    else:
      return None
    # print("parallel_dist", (x1-x2),(y1-y2))
    return x1-x2, y1 - y2

def get_scale():
    # based upon increased gap of parallel lines within same group
    pass

def get_displacement():
    # based upon parallel lines
    pass

def find_parallel_lines(lines):
    lines_ = lines[:, 0, :]
    angle = lines_[:, 1]
    # Perform hierarchical clustering
    angle_ = angle[..., np.newaxis]
    y = pdist(angle_)
    Z = ward(y)
    cluster = fcluster(Z, 0.5, criterion='distance')
    parallel_lines = []
    for i in range(cluster.min(), cluster.max() + 1):
        temp = lines[np.where(cluster == i)]
        parallel_lines.append(temp.copy())
    return parallel_lines


def watershed2(img):
    # Read the image from disk
    cv2.imshow("Original image", img)  # Display image
    img_float = np.float32(img)  # Convert image from unsigned 8 bit to 32 bit float
    criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 10, 1)
    # Defining the criteria ( type, max_iter, epsilon )
    # cv2.TERM_CRITERIA_EPS - stop the algorithm iteration if specified accuracy, epsilon, is reached.
    # cv2.TERM_CRITERIA_MAX_ITER - stop the algorithm after the specified number of iterations, max_iter.
    # cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER - stop the iteration when any of the above condition is met.
    # max_iter - An integer specifying maximum number of iterations.In this case it is 10
    # epsilon - Required accuracy.In this case it is 1
    k = 50  # Number of clusters
    ret, label, centers = cv2.kmeans(img_float, k, None, criteria, 50, cv2.KMEANS_RANDOM_CENTERS)
    # apply kmeans algorithm with random centers approach
    center = np.uint8(centers)
    # Convert the image from float to unsigned integer
    res = center[label.flatten()]
    # This will flatten the label
    res2 = res.reshape(img.shape)
    # Reshape the image
    cv2.imshow("K Means", res2)  # Display image
    cv2.imwrite("1.jpg", res2)  # Write image onto disk
    meanshift = cv2.pyrMeanShiftFiltering(img, sp=8, sr=16, maxLevel=1, termcrit=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 5, 1))
    # Apply meanshift algorithm on to image
    cv2.imshow("Output of meanshift", meanshift)
    # Display image
    cv2.imwrite("2.jpg", meanshift)
    # Write image onto disk
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Convert image from RGB to GRAY
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # apply thresholding to convert the image to binary
    fg = cv2.erode(thresh, None, iterations=1)
    # erode the image
    bgt = cv2.dilate(thresh, None, iterations=1)
    # Dilate the image
    ret, bg = cv2.threshold(bgt, 1, 128, 1)
    # Apply thresholding
    marker = cv2.add(fg, bg)
    # Add foreground and background
    canny = cv2.Canny(marker, 110, 150)
    # Apply canny edge detector
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Finding the contors in the image using chain approximation
    marker32 = np.int32(marker)
    # converting the marker to float 32 bit
    cv2.watershed(img,marker32)
    # Apply watershed algorithm
    m = cv2.convertScaleAbs(marker32)
    ret, thresh = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Apply thresholding on the image to convert to binary image
    thresh_inv = cv2.bitwise_not(thresh)
    # Invert the thresh
    res = cv2.bitwise_and(img, img, mask=thresh)
    # Bitwise and with the image mask thresh
    res3 = cv2.bitwise_and(img, img, mask=thresh_inv)
    # Bitwise and the image with mask as threshold invert
    res4 = cv2.addWeighted(res, 1, res3, 1, 0)
    # Take the weighted average
    final = cv2.drawContours(res4, contours, -1, (0, 255, 0), 1)

    watershed_edges_img = np.zeros_like(gray)
    accuracy_factor = .1
    for c in contours:
        if len(contours) > 1:
          area = cv2.contourArea(c)
          # print("area: ", area)
          if area <= 6:
            continue
          accuracy = accuracy_factor * cv2.arcLength(c, False)
          approx = cv2.approxPolyDP(c, accuracy, False)
#          for a in range(1, len(approx)):
#            i1, j1 = approx[a-1][0]
#            i2, j2 = approx[a][0]
#            cv2.line(watershed_edges_img, (i1, j1), (i2, j2), (255,0,0), 3, cv2.LINE_AA)
#          print("len approx", len(approx))
#          if len(approx) == 1:
          watershed_edges_img = cv2.drawContours(watershed_edges_img, c, -1, (255, 0, 0), 2)
    watershed_edges_img = canny
    cv2.imshow("watershededges1", watershed_edges_img)
    hough_lines = get_hough_lines(watershed_edges_img, 20)
    hough_lines_image = get_hough_lines_img(hough_lines, gray)
    cv2.imshow("watershedhoughlines1", hough_lines_image)
    watershed_edges_img2 = np.zeros_like(gray)
    watershed_edges_img2 = cv2.drawContours(watershed_edges_img2, contours, -1, (255, 0, 0), 2)
    cv2.imshow("watershededges2", watershed_edges_img2)

    hough_lines = get_hough_lines(watershed_edges_img2)
    hough_lines_image = get_hough_lines_img(hough_lines, gray)
    cv2.imshow("watershedhoughlines2", hough_lines_image)
    # cv2.waitKey(0)

    accuracy_factor = .1
    # print("len contours",  len(contours))
    for c in contours:
        obj_contours = gray.copy()
        if len(contours) > 1:
          area = cv2.contourArea(c)
          if area < 16:
            continue
          # Calculate accuracy as a percent of the contour perimeter
          accuracy = accuracy_factor * cv2.arcLength(c, True)
          approx = cv2.approxPolyDP(c, accuracy, True)
          cv2.drawContours(obj_contours, c, -1, (0,255,0), 3)
          cv2.imshow("contours", obj_contours)
          # cv2.waitKey()  # Wait for key stroke

    # Draw the contours on the image with green color and pixel width is 1
    cv2.imshow("Watershed", final)  # Display the image
    cv2.imwrite("3.jpg", final)  # Write the image
    # cv2.waitKey()  # Wait for key stroke

def track_table(img):
    # perspective, width/length scale
    pass

def track_wall(img):
    pass

def track_ground_barrier(img):
    pass

def track_lines(img):
    pass

def track_road(img):
    pass


def get_hough_lines(img, max_line_gap = 10):
      # rho_resolution = 1
      # theta_resolution = np.pi/180
      # threshold = 155
      rho = 1  # distance resolution in pixels of the Hough grid
      theta = np.pi / 180  # angular resolution in radians of the Hough grid
      #  threshold = 15  # minimum number of votes (intersections in Hough grid cell)
      threshold = 10  # minimum number of votes (intersections in Hough grid cell)
      # min_line_length = 50  # minimum number of pixels making up a line
      min_line_length = 40  # minimum number of pixels making up a line
      # max_line_gap = 10  # maximum gap in pixels between connectable line segments
      # max_line_gap = 5  # maximum gap in pixels between connectable line segments

      # Output "lines" is an array containing endpoints of detected line segments
      hough_lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                    min_line_length, max_line_gap)
      return hough_lines

def get_hough_lines_img(hough_lines, gray):
    hough_lines_image = np.zeros_like(gray)
    if hough_lines is not None:
      for line in hough_lines:
        for x1,y1,x2,y2 in line:
          cv2.line(hough_lines_image,(x1,y1),(x2,y2),(255,0,0),3)
    else:
      print("watershed houghlines None")
    return hough_lines_image

def get_box_lines(curr_img, gripper_img):
      gray = cv2.cvtColor(curr_img.copy(), cv2.COLOR_BGR2GRAY)
      edges = cv2.Canny(gray, 50, 200, None, 3)
      edges = cv2.dilate(edges,None,iterations = 1)
      cv2.imshow("edges1", edges)
      edges = cv2.bitwise_and(edges, cv2.bitwise_not(gripper_img))
      cv2.imshow("edges2", edges)
      hough_lines = get_hough_lines(edges)
      hough_lines_image = np.zeros_like(curr_img)
      # print("hough_lines", hough_lines)
      if hough_lines is None:
        return None
      for line in hough_lines:
        for x1,y1,x2,y2 in line:
          cv2.line(hough_lines_image,(x1,y1),(x2,y2),(255,0,0),5)
      cv2.imshow("houghlines", hough_lines_image)
      # cv2.waitKey(0)
      return hough_lines

def unmoved_pixels(unmoved_pix, slow_moved_pix, action, prev_img, curr_img):
      # from last_gripper_open to beginning
      gray = cv2.cvtColor(curr_img.copy(), cv2.COLOR_BGR2GRAY)
      edges = cv2.Canny(gray, 50, 200, None, 3)
      edges = cv2.dilate(edges,None,iterations = 1)
      d_edges = cv2.dilate(edges,None,iterations = 3)
      sm_edges = cv2.dilate(edges,None,iterations = 6)
      prev_gray = cv2.cvtColor(prev_img.copy(), cv2.COLOR_BGR2GRAY)
      prev_edges = cv2.Canny(prev_gray, 50, 200, None, 3)
      d_prev_edges = cv2.dilate(prev_edges,None,iterations = 3)
      sm_prev_edges = cv2.dilate(prev_edges,None,iterations = 6)
      if action in ["FORWARD", "REVERSE", "LEFT", "RIGHT"] and unmoved_pix is None:
        unmoved_pix = d_edges
      if action.startswith("UPPER_") or action.startswith("LOWER_"):
        slow_moved_pix = sm_edges
      if unmoved_pix is not None:
        cv2.imshow("unmoved pix", unmoved_pix)
        cv2.imshow("edges", edges)
        # cv2.waitKey(0)
        diff = cv2.absdiff(unmoved_pix, d_edges)
        diff_diff = cv2.absdiff(d_edges, d_prev_edges)
        full_diff = cv2.absdiff(gray, prev_gray)
        cv2.imshow("full_img_diff", full_diff)
        cv2.imshow("diff", diff_diff)
        
      else:
        diff = None
      prev_diff = cv2.absdiff(prev_edges, sm_prev_edges)
      if slow_moved_pix is not None:
        prev_sm_diff = cv2.absdiff(sm_prev_edges, d_prev_edges)
        sm_diff = cv2.absdiff(slow_moved_pix, sm_edges)
        diff_diff = cv2.absdiff(sm_edges, sm_prev_edges)
        cv2.imshow("diff", diff_diff)
      else:
        sm_diff = None
      for h in range(prev_img.shape[0]):
        for w in range(prev_img.shape[1]):
          if diff is not None and int(diff[h][w]) > 30 and unmoved_pix[h][w] > 60:
            # edge is 0 or 255
            # diff can result in increase or decrease depending on edge
            if action in ["FORWARD", "REVERSE", "LEFT", "RIGHT"]:
              unmoved_pix[h][w] = int(.95*unmoved_pix[h][w] + .05*d_edges[h][w])
          elif diff is not None and int(diff[h][w]) > 30 and prev_diff[h][w] > 30:
            # unmoved_pix[h][w] = int(.75*unmoved_pix[h][w] + .25*edges[h][w])
            if action in ["FORWARD", "REVERSE", "LEFT", "RIGHT"]:
              unmoved_pix[h][w] = int(.95*unmoved_pix[h][w] + .05*d_edges[h][w])
            # print("edge, dif, prevdif:", edges[h][w], diff[h][w], prev_diff[h][w])
          if sm_diff is not None and int(sm_diff[h][w]) > 30 and slow_moved_pix[h][w] > 60:
            if action.startswith("UPPER_") or action.startswith("LOWER_"):
              slow_moved_pix[h][w] = int(.95*slow_moved_pix[h][w] + .05*sm_edges[h][w])
          elif sm_diff is not None and int(sm_diff[h][w]) > 30 and prev_sm_diff[h][w] > 30:
            if action.startswith("UPPER_") or action.startswith("LOWER_"):
              slow_moved_pix[h][w] = int(.95*slow_moved_pix[h][w] + .05*sm_edges[h][w])

      if unmoved_pix is not None and slow_moved_pix is not None:
        cv2.imshow("unmoved pix", unmoved_pix)
        ret, thresh = cv2.threshold(unmoved_pix, 100, 255, cv2.THRESH_TOZERO)
        ret, thresh2 = cv2.threshold(slow_moved_pix, 100, 255, cv2.THRESH_TOZERO)
        cv2.imshow("thresh", thresh)
        cv2.imshow("thresh2", thresh2)
      cv2.imshow("edges", edges)
      # Run Hough on edge detected image
      hough_lines = get_hough_lines(edges)
      hough_lines_image = np.zeros_like(curr_img)
      # print("hough_lines", hough_lines)
      for line in hough_lines:
        for x1,y1,x2,y2 in line:
          cv2.line(hough_lines_image,(x1,y1),(x2,y2),(255,0,0),5)
      cv2.imshow("houghlines", hough_lines_image)
      # cv2.waitKey(0)
      return unmoved_pix, slow_moved_pix, hough_lines, d_edges

def exact_line(l1, l2):
    if l1[0][0]==l2[0][0] and l1[0][1]==l2[0][1] and l1[0][2]==l2[0][2] and l1[0][3]==l1[0][3]:
      return True
    return False

def display_lines(img_label, line_lst, curr_img):
    lines_image = np.zeros_like(curr_img)
    for line in line_lst:
      print("line:", line)
      # for x1,y1,x2,y2 in line[0]:
      x1,y1,x2,y2 = line[0]
      cv2.line(lines_image,(x1,y1),(x2,y2),(255,0,0),5)
    cv2.imshow(img_label, lines_image)

def display_line_pairs(img_label, line_pair_lst, curr_img, mode=2):
    lines_image = np.zeros_like(curr_img)
    for line0, line1, rslt in line_pair_lst:
      if mode == 0:
        print("line0:", img_label, line0)
      elif mode == 1:
        print("line1:", img_label, line1)
      elif mode == 2:
        print("line0,line1:", img_label, line0, line1)
      if mode == 0 or mode == 2:
        x1,y1,x2,y2 = line0[0]
        cv2.line(lines_image,(x1,y1),(x2,y2),(255,0,0),3)
      if mode == 1 or mode == 2:
        x1,y1,x2,y2 = line1[0]
        cv2.line(lines_image,(x1,y1),(x2,y2),(130,0,0),5)
    cv2.imshow(img_label, lines_image)

def display_line_group(gripper_img, lines, estimated_lines=None):
    # lines is equiv of: Line_Group["GRIPPER"]["LINES"][(ds_num,img_num)]
    # estimated_lines is equiv of: Line_Group["GRIPPER"]["ESTIMATED_LINES"][(ds_num,img_num)]
    lines_img = gripper_img.copy()
    for line_vals in lines.values():
      line, disparity = line_vals
      print("line:", line, disparity)
      x1,y1,x2,y2 = line[0]
      cv2.line(lines_img,(x1,y1),(x2,y2),(255,0,0),3)
    if estimated_lines is not None:
      for line in estimated_lines:
        print("est line:", line)
        x1,y1,x2,y2 = line[0]
        cv2.line(lines_img,(x1,y1),(x2,y2),(130,0,0),3)
    if estimated_lines is None:
      cv2.imshow("LineGroup", lines_img)
    else:
      cv2.imshow("LineGroup+EstLines", lines_img)

def find_bounding_box(box_lines):
    # we have a guess about the box lines.  
    # Now compute bounding box for each frame.
    pass

#
#       ___   w2 = 3
#      /   \
#     /     \   h = 3D
#    /       \
#    123456789   w1=9
#
#    w1/d1 = w2/d2  => but d1 is near zero????. depends on angle of camera.
#                      get pi camera angle
#                      pi FOV: 62.2deg x 48.8deg
#    w1/w2 = d1/d2 = (d2-d1)
#                   => use to get distance traveled
#                   => # pixels moved?
#
#    If you know the end width, you can compute the distance to the end
#
def analyze_table():
    pass



def analyze_moved_lines(moved_lines, best_moved_lines, num_ds, num_imgs, gripper_img, actions):

    # gripper_lines are umoved between frames.
    # These lines show up consistently unmoved between frames.
    # Other less-consistent lines associated with the gripper are possible.
    gripper_lines = get_hough_lines(gripper_img)
    gripper_lines_image = np.zeros_like(gripper_img)
    for line in gripper_lines:
      for x1,y1,x2,y2 in line:
        cv2.line(gripper_lines_image,(x1,y1),(x2,y2),(255,0,0),5)
    cv2.imshow("gripper lines", gripper_lines_image)

    # 0 = unknown
    # 1 = gripper / unmoving
    # 2+ = group of lines moving as a group between frames

    # Note: a line must of appeared in at least two consec frames
    # a particular line can be ID'ed by (ds_num, img_num, ml_num)

    def is_known_line(known_lines, line_id, moved_line):
        # line = 
        # for kl_id, kl in enumerate(known_lines):
          # if exact_line(l1, l2):
        pass

    def same_line_group(disparity1, disparity2):
        if abs(disparity1[0] - disparity2[0]) <= max(1,.25*(disparity1[0]+disparity2[0])/2):
          if abs(disparity1[1] - disparity2[1]) <= max(1,.25*(disparity1[1]+disparity2[1])/2):
            return True
        return False

    def add_line(line_groups, known_line, curr_key, prev=None,next=None):
        # add a known_line to image index
        try:
          # current entry for known line already exists
          # add link to previous image of known line
          if prev is not None:
            known_line[curr_key][0].append(prev)
          # add link to next image of known line
          if next is not None:
            known_line[curr_key][1].append(curr)
        except:
          # create current entry for known line
          # add link to previous image of known line
          if prev is not None and curr is not None:
            known_line[curr_key] = [[prev],[curr]]
          elif prev is not None:
            known_line[curr_key] = [[prev],[]]
          elif curr is not None:
            known_line[curr_key] = [[],[curr]]
          else:
            raise
        if prev is not None:
          prev_key = (curr_key[0], curr_key[1]-1, prev)
          try:
            # prev entry for known line already exists
            known_line[prev_key][1].append(curr_key[2])
          except:
            # create prev entry for known line
            known_line[prev_key] = [[prev],[curr]]
          try:
            found = line_group["BOX"]["LINES"][prev_key]
            line_group["BOX"]["LINES"].append(curr_key)
          except:
            pass
        if next is not None:
          next_key = (curr_key[0], curr_key[1]+1, next)
          try:
            # prev entry for known line already exists
            known_line[next_key][0].append(curr_key[2])
          except:
            # create prev entry for known line
            known_line[next_key] = [[curr_key[2]],[]]

   
        
          
    # find_epsilon(distances)

    # problem: multiple known_line mappings?
    #   -> make prev/next a list

    line_groups = [[]]   # known_line_id, [deltas]
    unknown_lines = {} 
    known_lines = {} 
    known_lines = {}   # (ds_num, img_num, ml_num) = [[prev],[next]]
    # Line_Group = {"BOX","GRIPPER","TABLE","CUBE"}
    Line_Group = {}
    # Line_Group["BOX"] = {"LINES", "COMPOSITE", "CAMERA_POSITION"}
    Line_Group["G0"] = {}
    # Line_Group["G0"]["LINES"] = {} # [(ds_num, img_num)][ml_num]: [line, disparity]
    # Line_Group["G0"]["ESTIMATED_LINES"] = {} # ds_num, img_num : [line, est_disparity]
    # Line_Group["G0"]["COMPOSITE"] = []  # lineseg [x1,y1,z1][x2,y2,z2]
    # Line_Group["G0"]["CAMERA_POSITION"] = [] # ds_num, img_num, pos
    num_LG = 0
    Line_Group["GRIPPER"] = {}
    Line_Group["GRIPPER"]["LINES"] = {} # ds_num, img_num, ml_num : line
    Line_Group["GRIPPER"]["ESTIMATED_LINES"] = {} # ds_num, img_num, ml_num : line
    num_other_lg = 0
    line = []
    x_y_distances = []
    for ds_num in range(num_ds):
      action_cnt = -1
      if ds_num == 0:
        # line_groups.append([]
        pass
      x_y_distances = []
      x_y_distances_imgnum = []
      x_y_disp = []
      x_y_disp_imgnum = []
      disp_actions = {"UPPER_ARM_UP", "UPPER_ARM_DOWN", "LOWER_ARM_UP", "LOWER_ARM_DOWN", "FORWARD", "REVERSE", "LEFT", "RIGHT"}
      x_y_disp_action = {}
      x_y_disp_action_imgnum = {}
      for a in disp_actions:
        x_y_disp_action[a] = []
        x_y_disp_action_imgnum[a] = []
      for img_num in range(num_imgs):
        min_disp = 1000000000
        action_cnt += 1
        for ml_num, lst in enumerate(moved_lines[ds_num][img_num]):
          if lst is None:
            continue

          # prevline = [moved_lines[ds_num][img_num][ml_num][1][1],
          #             moved_lines[ds_num][img_num][ml_num][1][2]]

          disparity = moved_lines[ds_num][img_num][ml_num][2]
          x_y_distances.append(disparity)
          x_y_distances_imgnum.append(img_num)

        for ml_num, lst in enumerate(best_moved_lines[ds_num][img_num]):
          if lst is None:
            continue
          disparity = best_moved_lines[ds_num][img_num][ml_num][2]
          x_y_disp.append(disparity)
          x_y_disp_imgnum.append(img_num)
          try:  # No gripper actions
            action = actions[ds_num][img_num]
            x_y_disp_action[action].append(disparity)
            x_y_disp_action_imgnum[action].append(img_num)
          except:
            pass

          # print("line disparity", [img_num, ml_num], prevline, disparity)
          # print("line disparity", [img_num, ml_num], moved_lines[ds_num][img_num][ml_num])
      # print("x_y_disp", len(x_y_disp))
      #########
      # ALL_LINE COMBINATIONS shows that taking the minimum parallel line between frames
      # is the way to go
      #########
      # print("ALL_LINE COMBINATIONS")
      # clusters, best_cluster = analyze_x_y_dist(x_y_distances, x_y_distances_imgnum)

      #########
      # BEST_LINE DISPARITY gets a set of clusters for approx different line groups over frames
      # For any particular image, if the x_y disp is within about 25%, the lines are probably
      # in the same line group.
      #########
      print("BEST_LINE DISPARITY")
      best_clusters, best_cluster = analyze_x_y_dist(x_y_disp, x_y_disp_imgnum)

      #########
      # The actions make a difference in how the lines are viewed frame to frame
      # Seems hard to come up with reliable rules of movement
      # Probably need estimated camera and line positions (e.g., based on intersections)
      #########
      for a in disp_actions:
        if len(x_y_disp_action[a]) > 10:
          print(a, "DISPARITY", len(x_y_disp_action[a]))
          clusters, best_cluster = analyze_x_y_dist(x_y_disp_action[a], x_y_disp_action_imgnum[a])

      #########
      # While clustering helped verify some theories and provide guidelines, when you restrict
      # to "BEST LINES" and per action, then all lines are clumped into the same cluster as the
      # lines move over time.
      #
      # Do line grouping based on "BEST LINES" and x_y disparity and gripper analysis.
      # Do intersections to get points for scale.  
      # Do scale by comparing dist between parallel lines within same image and comparing dist across images.
      # 
      # gripper -> compute gripper bounds. Look for lines with (0,0) +-1 within / across /below bounds.
      # 
 

      ##################################################33
      # Gripper Check
      # are the lines potentially part of the gripper?
      action_cnt = -1
      print("Gripper check")
      for ds_num in range(num_ds):
        if ds_num == 0:
          pass
        for img_num in range(num_imgs):
          img_line_group = []
          action_cnt += 1
          print("dsnum, cnt, len", ds_num, action_cnt, len(actions))
          print("action[ds_num] len", ds_num, action_cnt, len(actions[ds_num]))
          min_y1 = 100000000000
          min_x1 = 100000000000
          max_x1 = -100000000000
          if actions[ds_num][action_cnt] == "GRIPPER_OPEN":
            if img_num == 0:
              Line_Group["GRIPPER"]["LINES"][(ds_num,img_num)] = {}
              Line_Group["GRIPPER"]["ESTIMATED_LINES"][(ds_num,img_num)] = []
              continue
            prev_LG = Line_Group["GRIPPER"]["LINES"][(ds_num,img_num-1)]
            Line_Group["GRIPPER"]["LINES"][(ds_num,img_num)] = prev_LG
            prev_LG = Line_Group["GRIPPER"]["ESTIMATED_LINES"][(ds_num,img_num-1)]
            Line_Group["GRIPPER"]["ESTIMATED_LINES"][(ds_num,img_num)] = prev_LG
            for LG_num in range(num_LG):
              LG_name = "G" + str(LG_num)
              LG = Line_Group[LG_name]["LINES"][(ds_num,img_num)]
              prev_LG = Line_Group[LG_name]["LINES"][(ds_num,img_num-1)]
              Line_Group[LG_name]["LINES"][(ds_num,img_num)] = prev_LG
              prev_LG = Line_Group[LG_name]["ESTIMATED_LINES"][(ds_num,img_num-1)]
              Line_Group[LG_name]["ESTIMATED_LINES"][(ds_num,img_num)] = prev_LG
            continue
          if action_cnt < len(actions[ds_num]) and actions[ds_num][action_cnt] not in ["GRIPPER_OPEN", "GRIPPER_CLOSE"]:
            for line in gripper_lines:
              for x1,y1,x2,y2 in line:
                min_y1 = min(y1, y2, min_y1)  # top most line
                min_x1 = min(x1, x2, min_x1)  # left most line
                max_x1 = max(x1, x2, max_x1)  # right most line

          try:
            LG = Line_Group["GRIPPER"]["LINES"][(ds_num,img_num)]
          except:
            Line_Group["GRIPPER"]["LINES"][(ds_num,img_num)] = {}
            LG = Line_Group["GRIPPER"]["LINES"][(ds_num,img_num)]
          for bml_num, lst in enumerate(best_moved_lines[ds_num][img_num]):
            if lst is None:
              continue
            bm_lines= best_moved_lines[ds_num][img_num][bml_num][1] 
            disparity = best_moved_lines[ds_num][img_num][bml_num][2] 
            print("bml:", bml_num, bm_lines, disparity)
            found = False
            for bm_line in bm_lines:
                x1,y1,x2,y2 = bm_line
                if min_y1 < y1 and min_y1 < y2:
                  print("gripper line min_y1, y1, y2:", min_y1, y1, y2)
                  if abs(disparity[0]) <= 1.5 and abs(disparity[1]) <= 1.5:
                    print("new gripper line (disparity):", bm_lines, disparity)
                    # Line_Group["GRIPPER"]["LINES"][(ds_num,img_num)][bml_num] = [copy.deepcopy(bm_lines), disparity]
                    LG[bml_num] = [copy.deepcopy(bm_lines), disparity]
                    found = True
                  elif min_x1 < x1 < max_x1 and min_x1 < x2 < max_x1:
                    print("new gripper line (inside):", bm_lines, disparity, min_x1, max_x1, x1, x2)
                    LG[bml_num] = [copy.deepcopy(bm_lines), disparity]
                    found = True
                  else:
                    for g_line in gripper_lines:
                      # for x1,y1,x2,y2 in line:
                        print("bm_lines, g_line:", bm_lines, g_line)
                        if is_same_line(bm_lines, g_line):
                          print("new gripper line (same):", bm_lines)
                          LG[bml_num] = [copy.deepcopy(bm_lines), disparity]
                          found = True
                        elif is_broken_line(bm_lines, g_line):
                          print("new gripper line (broken):", bm_lines)
                          LG[bml_num] = [copy.deepcopy(bm_lines), disparity]
                          found = True
                      # if found:
                        # break
              # if found:
                # break
            if found:
              continue

            # not a gripper line.  find a possible line group among the "best moved" lines.
            # best_moved_line contains [best_prev_line, curr_line, disparity])
            print("find line group matching:", img_line_group)
            found = False
            # for ilg_num, [ilg_bml, ilg_disparity, ilg_line] in enumerate(img_line_group):
            for ilg_num, ilg in enumerate(img_line_group):
              for ilg_lst in ilg:
                ilg_bml, ilg_disparity, ilg_line = ilg_lst
                if same_line_group(disparity, ilg_disparity):
                  print("append to img LG", ilg_num, ilg_lst)
                  img_line_group[ilg_num].append([bml_num, disparity, copy.deepcopy(bm_lines)])
                  found = True
                  break
            if not found:
              # new line group with initial line
              img_line_group.append([[bml_num, disparity, copy.deepcopy(bm_lines)]])
              print("new img LG", img_line_group[-1])

          print("image LG:", img_line_group, ds_num, img_num)
          ###########
          # Line groups within an image detected. Merge with to Line_Group.
          # Line_Group["G1"]["LINES"] maps to lines in "best_moved_lines"
          # Line_Group["G2"]["LINES"][(ds_num, img_num)][bml_num)] = [line, disparity]
          # Line_Group["G3"]["ESTIMATED_LINES"][(ds_num, img_num)] = [[line, disparity]]
          LG_found = False
          curr_LG = None
          # for ilg_num, [ilg_bml, ilg_disparity, ilg_line] in enumerate(img_line_group):
          for ilg_num, ilg in enumerate(img_line_group):
            no_match_for_ilg_bml = []
            for ilg_lst in ilg:
              # iterate through lines within image's line group to find a matching LG
              ilg_bml = ilg_lst[0]
              # print("ilg_bml", ilg_bml)
              # print("bml 0", len(best_moved_lines))
              # print("bml 1", len(best_moved_lines[ds_num]))
              # print("bml 2", len(best_moved_lines[ds_num][img_num]))
              # print("bml 3a", best_moved_lines[ds_num][img_num])
              # print("num bml 3b", ilg_bml, len(best_moved_lines[ds_num][img_num]))
              # print("bml 3c", best_moved_lines[ds_num][img_num][ilg_bml])
              # print("bml 4", best_moved_lines[ds_num][img_num][ilg_bml][0])
              prevline = best_moved_lines[ds_num][img_num][ilg_bml][0]
              currline = best_moved_lines[ds_num][img_num][ilg_bml][1]
              currdisp = best_moved_lines[ds_num][img_num][ilg_bml][2]
              if LG_found:
                # LG already set. Copy currline to it.
                curr_LG[ilg_bml] = [copy.deepcopy(currline), currdisp]
                continue

              # Check for exact match in Line_Group
              prev_LG_key = (ds_num, img_num-1)
              for LG_num in range(num_LG):
                LG_name = "G" + str(LG_num)
                try:
                  prev_LG = Line_Group[LG_name]["LINES"][prev_LG_key]
                except:
                  continue

                for LG_vals in prev_LG.values():
                  LG_line, LG_disparity = LG_vals
                  if exact_line(LG_line, prevline):
                    print("found LG for", LG_name, LG_line)
                    # we found the matching LG group. Add next image values.
                    LG_found = True
                    curr_LG_key = (ds_num, img_num)
                    try:
                      curr_LG = Line_Group[LG_name]["LINES"][curr_LG_key]
                    except:
                      Line_Group[LG_name]["LINES"][curr_LG_key] = {}
                      curr_LG = Line_Group[LG_name]["LINES"][curr_LG_key]
                    # copy over img_line_grp into LG
                    curr_LG[ilg_bml] = [copy.deepcopy(currline), currdisp]
                    break
              if LG_found:
                continue
                   
              # fell through, so no LG_found.
              # if no exact line match in any Line Group, try matching on ESTIMATED_LINES
              for LG_num in range(num_LG):
                LG_name = "G" + str(LG_num)
                try:
                  prev_LG = Line_Group[LG_name]["ESTIMATED_LINES"][prev_LG_key]
                except:
                  continue
                for LG_line in prev_LG:
                  line_score = same_line_score(LG_line, prevline)
                  print("same_line_score for", LG_line, prevline, line_score)
                  if line_score < 3:
                    print("found Estimated LG for", LG_name, LG_line, prevline, line_score)
                    # we found the matching LG group. Add next image values.
                    LG_found = True
                    curr_LG_key = (ds_num, img_num)
                    try:
                      curr_LG = Line_Group[LG_name]["LINES"][curr_LG_key]
                    except:
                      Line_Group[LG_name]["LINES"][curr_LG_key] = {}
                      curr_LG = Line_Group[LG_name]["LINES"][curr_LG_key]
                    # copy over img_line_grp into LG
                    curr_LG[ilg_bml] = [copy.deepcopy(currline), currdisp]
                    break

              if LG_found:
                for ilg_num, ilg_bml in enumerate(no_match_for_ilg_bml):
                  # Need to split off line from line Group if disparity doesn't continue to match
                  # the rest of the group.
                  if ilg_bml == -1:
                    continue
                  ilg_prevline = best_moved_lines[ds_num][img_num][ilg_bml][0]
                  ilg_currline = best_moved_lines[ds_num][img_num][ilg_bml][1]
                  ilg_currdisp = best_moved_lines[ds_num][img_num][ilg_bml][2]
                  if same_line_group(ilg_currdisp, currdisp):
                    ilg_curr_LG[ilg_bml] = [copy.deepcopy(ilg_currline), ilg_currdisp]
                    no_match_for_ilg_bml[ilg_num] = -1
              else:
                no_match_for_ilg_bml.append(ilg_bml)
              continue  # go back to try next line within the image_line_group

            # Check if no LG_found for a particular img_line_group. Make New LG group.
            print("len no_match_for_ilg_bml:", len(no_match_for_ilg_bml))
            
            if len(no_match_for_ilg_bml) > 0:
                # Check for exact match in Line_Group
                new_LG_key = (ds_num, img_num)
                LG_name = "G" + str(num_LG)
                num_LG += 1
                Line_Group[LG_name] = {}
                Line_Group[LG_name]["LINES"] = {}
                Line_Group[LG_name]["ESTIMATED_LINES"] = {}
                new_LG = Line_Group[LG_name]["LINES"]
                new_LG[new_LG_key] = {}
                for ilg_num, ilg_bml in enumerate(no_match_for_ilg_bml):
                  if ilg_bml == -1:
                    continue
                  ilg_line = best_moved_lines[ds_num][img_num][ilg_bml][1]
                  ilg_disp = best_moved_lines[ds_num][img_num][ilg_bml][2]
                  new_LG[new_LG_key][ilg_bml] = [copy.deepcopy(ilg_line), ilg_disp]
                  print("make new LG for", ilg_bml, ilg_disp, ilg_line)
                  no_match_for_ilg_bml[ilg_num] = -1
            continue # go back to try next image_line_group
  
          print("num_LG", num_LG)
          prev_LG_key = (ds_num, img_num-1)
          curr_LG_key = (ds_num, img_num)
          for LG_num in range(num_LG):
            LG_name = "G" + str(LG_num)
            try:
              prev_est_LG = Line_Group[LG_name]["ESTIMATED_LINES"][prev_LG_key]
            except:
              Line_Group[LG_name]["ESTIMATED_LINES"][prev_LG_key] = []
              prev_est_LG = Line_Group[LG_name]["ESTIMATED_LINES"][prev_LG_key]
            try:
              prev_LG = Line_Group[LG_name]["LINES"][prev_LG_key]
            except:
              Line_Group[LG_name]["LINES"][prev_LG_key] = {}
              prev_LG = Line_Group[LG_name]["LINES"][prev_LG_key]
            try:
              curr_est_LG = Line_Group[LG_name]["ESTIMATED_LINES"][curr_LG_key]
            except:
              Line_Group[LG_name]["ESTIMATED_LINES"][curr_LG_key] = []
              curr_est_LG = Line_Group[LG_name]["ESTIMATED_LINES"][curr_LG_key]
            try:
              curr_LG = Line_Group[LG_name]["LINES"][curr_LG_key]
            except:
              Line_Group[LG_name]["LINES"][curr_LG_key] = {}
              curr_LG = Line_Group[LG_name]["LINES"][curr_LG_key]

            # compute expected displacement
            sum_disp = [0,0]
            cnt_disp = 0
            avg_disp = [0,0]
            for LG_vals in curr_LG.values():
              print("LG_items", LG_vals, LG_num)
              LG_line, LG_disparity = LG_vals
              cnt_disp += 1
              sum_disp[0] += LG_disparity[0]
              sum_disp[1] += LG_disparity[1]
            if cnt_disp == 0:
              print("warning: no active lines in LG", LG_name)
              for LG_vals in prev_LG.values():
                LG_line, LG_disparity = LG_vals
                cnt_disp += 1
                sum_disp[0] += LG_disparity[0]
                sum_disp[1] += LG_disparity[1]
            if cnt_disp > 0:
              avg_disp[0] = sum_disp[0] / cnt_disp
              avg_disp[1] = sum_disp[1] / cnt_disp
            else:
              avg_disp[0] = 0
              avg_disp[1] = 0

            # all previous lines must be accounted for in current lines
            prevlines_found = []
            for ilg_bml in curr_LG.keys():
              prevlines_found.append(best_moved_lines[ds_num][img_num][ilg_bml][0])

            # store two-way pointers instead of yet more loops?
            for LG_vals in prev_LG.values():
              LG_line, LG_disparity = LG_vals
              found = False
              for prevline in prevlines_found:
                if exact_line(LG_line, prevline):
                  found = True
                  break
              if found:
                continue
              # Not found in curr LINES, adjust based on Disparity and add to estimated lines
              est_LG_line = [[round(LG_line[0][0]+avg_disp[0]), round(LG_line[0][1]+avg_disp[1]),
                              round(LG_line[0][2]+avg_disp[0]), round(LG_line[0][3]+avg_disp[1])]]
              curr_est_LG.append(est_LG_line)


          cv2.destroyAllWindows()
          curr_LG_key = (ds_num, img_num)
          print("GRIPPER LG: ")
          LG = Line_Group["GRIPPER"]["LINES"][curr_LG_key]
          display_line_group(gripper_img, LG)
          # est_LG = Line_Group["GRIPPER"]["LINES"][curr_LG_key]
          # display_line_group(gripper_img, LG, est_LG)
          cv2.waitKey(0)
          lines_image = np.zeros_like(gripper_img)
          for LG_num in range(num_LG):
            LG_name = "G" + str(LG_num)
            print("LG: ", LG_name, num_LG)
            LG = Line_Group[LG_name]["LINES"][curr_LG_key]
            # display_line_group(lines_image, LG)
            est_LG = Line_Group[LG_name]["ESTIMATED_LINES"][curr_LG_key]
            # display_line_group(lines_image, LG, est_LG)
            # cv2.waitKey(0)

      # Can we combine lines and determine the 3D image?  
      #   Line_Group["BOX"]["COMPOSITE"] = []  # lineseg [x1,y1,z1][x2,y2,z2]
      # Can we estimate the Camera Position?
      #   Line_Group["BOX"]["CAMERA_POSITION"] = [] # ds_num, img_num, pos

      # Display Gripper: Unmoved, Gripper Line Group, Est Gripper Line Group (diff colors)
      # Display Line_Groups: Gripper Line Group, Est Gripper Line Group (diff colors/shades of gray)
      # find_line_intersects(
      color = (random.randint(0, 255),random.randint(0, 255),random.randint(0, 255))

      # is_gripper

      return

      if False:
          # note: to be in diff_frame_parallel, 
          # a particular line can be ID'ed by (ds_num, img_num, ml_num, 1 or 2)
          # ml_num can change img to img, and may not exist at all
          f1_ln, f2_ln, dist = lst

          # 
          # see if a f1_ln is a known line
          if ds_num > 0:
            status = "UNKNOWN"
            # compare current moved_lines to previous image's moved lines
            for f2_prev_ml_num, f2_lst in enumerate(moved_lines[ds_num][img_num-1]):
              if f2_lst is None:
                continue
              prev_f1_ln, prev_f2_ln, prev_dist = lst
              # if same line is in prev and current image, store as
              # part of the "known line" doubly linked list.
              if exact_line(f1_ln, prev_f2_ln):
                # note: bml_num for same line is different for each img_num
                f1_id = (ds_num, img_num, bml_num) 
                # add_line(known_line, f1_id, prev=f2_prev_ml_num,next=None)
                status = "KNOWN"

           # tracking individual lines is not our goal. 
           # Assume tracking known lines helps maintaining line groups.
           # is known line part of a line group?
           # mapping known-line to a line group
           # mapping known-line to a known line doubly linked list
           # known_line_id = [first_known_instance, most_recent_instance, line_group]
           # a known line could be linked to an earlier known line
           # how to clean up the combined known line?  Do we need to?
           
              # track an individual line in a group over time.
              # all lines in line group should match their disparity 
              # Estimate positions of all line in line group (if missing in an image)

              # line_group: real_line, estimated_linem, composite_line
              #             line_counts
              # line group: location relative to gripper at time of drop-off
              # 

           # our goal is to find and track line_groups.
           # A line_group could be mapped to a Table, Box, gripper,
           # the robot light (as defined by a cluster) or something else.
           # Need to track line_group over time.
           # Line group have same disparity/scale characteristics.
           # Line group ideally should be able to track the same known lines 
           # over time, but not all lines will be detected in every frame.
           # 

        # Composite Line group
        # Even if a line is not detected, or a subset of a previous line,
        # it should have "moved" in a way that is consistent with the other
        # lines in the group.  Predict that movement (ideally in 3D.)


        # Current Lines (at end)
        #   f1_id = (ds_num, img_num, ml_num) 
        # Previous lines:
        #   f1_id = (ds_num, img_num, ml_num) 
        # Composite
        #   Map img-space to 3D space based upon predicted camera position
        # Predicted Camera Position
        #
        # Line_Group = {"BOX","GRIPPER","TABLE","CUBE"}
        # Line_Group["BOX"] = {"LINES", "COMPOSITE", "CAMERA_POSITION"}
        # Line_Group["BOX"]["LINES"] = {} # ds_num, img_num, ml_num
        # Line_Group["BOX"]["COMPOSITE"] = []  # lineseg [x1,y1,z1][x2,y2,z2]
        # Line_Group["BOX"]["CAMERA_POSITION"] = [] # ds_num, img_num, pos
        
        # disparity = []
        # for ml_num, lst in enumerate(moved_lines[ds_num][img_num]):
          # if ds_num > 0:
            # disparity between lines along x and y axis was previously 
            # computed and part of moved_lines data
         
               


                #
                # TODO: compute new scale, disparity
                # disparity = (((x/2-w/2)*Z - (x'/2-w/2))*Z') / f
                # scale: Z' = (dist(pt1,pt2)/(dist(pt1',pt2')*Z
                # raspberry pi:
                # focal_length_pixels = 3.04mm / 0.00112mm = 2714.3px
                #
                # compute angles / intersections within a group
                # Need 2 intersections along the same line  
                #   - can compute both disparity and scale!!
                #   - can project to theoretical intersection points
                #   - if > 2, can compare scale/disparities for acccuracy
                # Alternative: distance between 2 parallel lines
                #   - can compute scale
                #   - can compute disparity only if end-points are consistent
                #     - can compute based on end-ponts and compare

                # TODO: check if consistent with line_group
                # TODO: determine which line_group it belongs in
                #   - find intersections
                #     - compare scales / disparity between frames
                #   - find parallel lines between frames, compute x,y deltas
                #     of parallel lines.
                # TODO: compute (guess?) 3D position of line, camera
                #   - if 3 intersections along same line group
                #
                # TODO: matches with earlier frames (intermittent detection)

                # TODO: Figure out which 3D lines are part of box
                #   - Directly above at point of drop
                #   - Keep track of line group 
                #   - line group can expand over time (e.g. include the base)
                # TODO: compute bounding box. track over movement.
                # TODO: track intersect with gripper (esp. upper bound)
                # 
                # TODO: handle blocking of bounding box by gripper
                #       gripper has own bounding box


                # Tabletop/Rectanglular bounds:
                #   - big line group with long, intersecting lines
                #   - track based on retracted, L/R/F/B
                #   - separate from box
                # break
          # if status == "UNKNOWN":
            # TODO: eventually add to Unknown list 
            # pass

          ##################################################33
          # Line Group Check

          ##################################################33
          # Table check?  prototype here? (same as rectangle line/taple?)
          # with table understanding, improve mapping

          ##################################################33
          # 3D Pose estimation: camera, lines
          if False: 
            if actions[ds_num][action_cnt] == "UPPER_ARM_UP":
              pass
            elif actions[ds_num][action_cnt] == "UPPER_ARM_DOWN":
              pass
            elif actions[ds_num][action_cnt] == "LOWER_ARM_UP":
              pass
            elif actions[ds_num][action_cnt] == "LOWER_ARM_DOWN":
              pass
            elif actions[ds_num][action_cnt] == "GRIPPER_OPEN":
              pass
            elif actions[ds_num][action_cnt] == "GRIPPER_CLOSE":
              pass
            elif actions[ds_num][action_cnt] == "FORWARD":
              track_table()
              estimate_box()
            elif actions[ds_num][action_cnt] == "REVERSE":
              track_table()
              estimate_box()
            elif actions[ds_num][action_cnt] == "LEFT":
              track_table()
              estimate_box()
            elif actions[ds_num][action_cnt] == "RIGHT":
              track_table()
              estimate_box()

    # TODO: generalize beyond lines to circles or contours

    return disparity, scale

# not debugged
def contour_gripper(gripper_img):
    try:
      gray = cv2.cvtColor(gripper_img, cv2.COLOR_BGR2GRAY)
    except:
      gray = gripper_img
    # thresh = 10
    thresh = 128
    gray = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)[1]
    gray = cv2.dilate(gray,None, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Prune contours
    maxArea = 0.0
    for c in contours:
      if len(contours) > 1:
        area = cv2.contourArea(c)
        if area > maxArea:
          MaxArea = area

    minArea = 0.2 * maxArea;

    pruned_contours = []
    for c in contours:
      if len(contours) > 1:
        area = cv2.contourArea(c)
        if area > minArea:
          pruned_contours.append(copy.deepcopy(c))
        
    # Smooth the contours
    smoothedContours = []
    for i in range(len(pruned_contours)):
      x = []
      y = []

      n = prunedContours[i].size()
      for j in range(n):
        x.append(prunedContours[i][j].x)
        y.append(prunedContours[i][j].y)

      kernel = cv2.getGaussianKernel(11, 4.0)
      window = np.outer(kernel, kernel.transpose())

      xSmooth = cv2.filter2D(x, -1, window)
      ySmooth = cv2.filter2D(y, -1, window)
      points = []
      for j in range(n):
        points.append(xSmooth[j], ySmooth[j])
      points = np.array(points)
      points = np.float32(points[:, np.newaxis, :])
      return points


def analyze_box_lines(box_lines, actions, gripper_img, drop_off_img, img_paths):
    angle_sd = SortedDict()
    box_line_intersections = []
    world_lines = []
    arm_pos = []
    num_ds = len(box_lines)
    prev_lines = None
    hough_lines = None
    num_passes = 4
    # ret, gripper_img = cv2.threshold(gripper_img, 100, 255, cv2.THRESH_TOZERO)
    ret, drop_off_img = cv2.threshold(drop_off_img, 100, 255, cv2.THRESH_TOZERO)
    line_group = [[]]  # includes gripper line_group[0]
    gripper_lines = get_hough_lines(gripper_img)
    gripper_lines2 = get_hough_lines(gripper_img)
    gripper_lines_image = np.zeros_like(gripper_img)
    min_gripper_y = 100000000
    for line in gripper_lines2:
      for x1,y1,x2,y2 in line:
        cv2.line(gripper_lines_image,(x1,y1),(x2,y2),(255,0,0),5)
        for y in [y1,y2]:
          if y < min_gripper_y:
            MIN_GRIPPER_Y = y

    cv2.imshow("gripper lines", gripper_lines_image)
    unknown_lines = []
    same_frame_parallel = []
    diff_frame_parallel = []
    min_diff_frame_parallel = []
    broken_lines = []
    gripper_lines = []
    gripper_lines2 = []
    drop_off_lines = get_hough_lines(drop_off_img)
    drop_off_lines_image = np.zeros_like(drop_off_img)
    for line in drop_off_lines:
      for x1,y1,x2,y2 in line:
        cv2.line(drop_off_lines_image,(x1,y1),(x2,y2),(255,0,0),5)
    cv2.imshow("drop_off lines", drop_off_lines_image)
    # cv2.waitKey(0)

    # findparallel(lines, allowed_delta):
    # line_intersection(line1, line2):
    # is_same_line(line1, line2):
    # is_broken_line(line1, line2):
    # is_parallel(line1, line2):
    # parallel_dist(line1, line2):
    # get_scale():
    # get_displacement():

    action_cnt = -1
    persist_line_stats = []
    persist_line = []
    persist_line_most_recent = [] # [[ds_num, img_num]]
    persist_line_min_max_ds_img = []
    persist_line_counter = []
    for ds_num, ds in enumerate(dataset):
      for pass_num in range(num_passes):
        num_imgs = len(box_lines[ds_num])
        if pass_num == 0:
          unknown_lines.append([])
          same_frame_parallel.append([])
          diff_frame_parallel.append([])
          min_diff_frame_parallel.append([])
          broken_lines.append([])
        for img_num in range(num_imgs):
          new_img_num = True
          if pass_num == 0:
            unknown_lines[ds_num].append([])
            same_frame_parallel[ds_num].append([])
            diff_frame_parallel[ds_num].append([])
            min_diff_frame_parallel[ds_num].append([])
            broken_lines[ds_num].append([])
          action_cnt += 1
          prev_lines = hough_lines
          hough_lines = box_lines[ds_num][img_num]
          if hough_lines is None:
            hough_lines = []
          if img_num > 0 and box_lines[ds_num][img_num-1] is not None:
            prev_hough_lines = box_lines[ds_num][img_num-1]
          else:
            prev_hough_lines = []
          for hl_num, h_line in enumerate(hough_lines):
            print(pass_num, hl_num, "new h_line:", h_line)
            # Gripper lines now computed differently
            unknown_lines[ds_num][img_num].append(copy.deepcopy(h_line))
            
            # Gripper Lines should be the same from frame to frame
            if len(h_line) == 0:
              continue
            if pass_num == 0 and False:
              # Gripper lines now computed differently
              unknown_lines[ds_num].append([])
              # find gripper (line_group[0])
              found = False
              # TODO: increase strength of unmoved lines.
              # remove moved lines, esp over FWD/REV/L/R.
              for g_line in line_group[0]:
                if len(g_line) == 0:
                  continue
                if is_same_line(h_line, g_line):
                  # gripper_lines.append(g_line)
                  gripper_lines2.append(copy.deepcopy(g_line))
                  # line_group[0].append(g_line)
                  found = True
                  break
              # if found:
              #   continue
              # for g_line in gripper_lines:
              for g_line in gripper_lines2:
                if len(g_line) == 0:
                  continue
                if is_same_line(h_line, g_line):
                  # gripper_lines.append(g_line)
                  gripper_lines2.append(copy.deepcopy(g_line))
                  # line_group[0].append(g_line)
                  found = True
                  break
              # if found:
              #   continue

              for g_line in prev_hough_lines:
                if len(g_line) == 0:
                  continue
                if is_same_line(h_line, g_line):
                  # gripper_lines.append(g_line)
                  gripper_lines2.append(copy.deepcopy(g_line))
                  # line_group[0].append(g_line)
                  found = True
              # if found:
              #   continue

            if pass_num == 1:
              # for u_line in unknown_lines[ds_num][-3:-1]:
              for u_line in unknown_lines[ds_num][img_num]:
                if len(u_line) == 0:
                  continue
                if is_parallel(u_line, h_line) and not is_same_line(u_line, h_line):
                  dist = parallel_dist(u_line, h_line)
                  if dist is not None:
                    found = False
                    for x1_line, x2_line, d in same_frame_parallel[ds_num][img_num]:
                      if exact_line(x1_line, h_line) and exact_line(x2_line, u_line):
                        found = True
                        break
                    if not found:
                      print("same_frame_parallel", ds_num, img_num)
                      same_frame_parallel[ds_num][img_num].append([copy.deepcopy(u_line), copy.deepcopy(h_line), dist])
                # broken lines
                broken_line = is_broken_line(u_line, h_line)
                if broken_line is not None:
                  found = False
                  for x1_line, x2_line, b_line in broken_lines[ds_num][img_num]:
                    if exact_line(x1_line, h_line) and exact_line(x2_line, u_line):
                      found = True
                      break
                  if not found:
                    # print("broken_line", ds_num, img_num)
                    broken_lines[ds_num][img_num].append([copy.deepcopy(u_line), copy.deepcopy(h_line), broken_line])
              min_dist = 100000000
              best_p_line = None
              for p_line in prev_hough_lines:
                  if len(p_line) == 0:
                    continue
                  dist = parallel_dist(p_line, h_line)
                  if dist is not None:
                    found = False
                    for x1_line, x2_line, d in diff_frame_parallel[ds_num][img_num]:
                      if exact_line(x1_line, h_line) and exact_line(x2_line, p_line):
                        found = True
                        break
                    if not found :
                      print("diff_frame_parallel", ds_num, img_num)
                      diff_frame_parallel[ds_num][img_num].append([copy.deepcopy(p_line), copy.deepcopy(h_line), dist])
                      if min_dist > sqrt(abs(dist[0])+abs(dist[1])):
                        min_dist = sqrt(abs(dist[0])+abs(dist[1]))
                        best_p_line = copy.deepcopy(p_line)
              if best_p_line is not None:
                disp = parallel_dist(best_p_line, h_line)
                min_diff_frame_parallel[ds_num][img_num].append([copy.deepcopy(best_p_line), copy.deepcopy(h_line), disp])
              # cv2.waitKey(0)

            if pass_num == 2:
              # display broken lines
              display_line_pairs("broken lines", broken_lines[ds_num][img_num], drop_off_img)
              # note: we could generalize this analysis to be
              # based upon edge contours instead of lines.
              # display diff_frame_parallel lines
              display_line_pairs("best moved lines", min_diff_frame_parallel[ds_num][img_num], drop_off_img)
              disparity, scale = analyze_moved_lines(diff_frame_parallel, min_diff_frame_parallel, num_ds, num_imgs, gripper_img, actions)

          if pass_num == 0:
            if img_num == num_imgs-1:
              print("##################")
              print("Image Sorted Lines")
              print("##################")
              for is_ds_num, ds in enumerate(dataset):
                for is_img_num in range(num_imgs):
                  hough_lines = box_lines[is_ds_num][is_img_num]
                  if hough_lines is None:
                    print("No hough_lines for", is_ds_num, is_img_num)
                    continue
                  else:
                    for hl_num, h_line in enumerate(hough_lines):
                      print((is_ds_num, is_img_num, hl_num), h_line)

          if pass_num == 2:
            print("pass_num", pass_num)
      
            ###################
            # Angle Sorted Dict
            ###################
            hough_lines = box_lines[ds_num][img_num]
            if hough_lines is None:
              print("No hough_lines for", ds_num, img_num)
              continue
            else:
              # print("##################")
              # print("Angle Sorted Lines")
              # print("##################")
              for hl_num, h_line in enumerate(hough_lines):
                angle = np.arctan2((h_line[0][0]-h_line[0][2]), (h_line[0][1]-h_line[0][3]))
                if angle <= 0:
                  angle += np.pi
                try_again = True
                while try_again:
                  try:
                    # ensure slightly unique angle so no lines are overwritten
                    val = angle_sd[angle]
                    angle += .000001
                    try_again = True
                  except:
                    try_again = False
                angle_sd[angle] = (ds_num, img_num, h_line)

  
            if img_num == num_imgs-1:
              print("##################")
              print("Angle Sorted Lines")
              print("##################")
              skv = angle_sd.items()
              for item in skv:
                print(item)
  
            ###########
            # Intersections within same image
            if hough_lines is None:
              print("No hough_lines for", ds_num, img_num)
              # continue
            else:
              for hl_num, h_line in enumerate(hough_lines):
                for hl_num2, h_line2 in enumerate(hough_lines):
                  if hl_num2 > hl_num:
                    xy_intersect = line_intersection(h_line, h_line2)
                    if xy_intersect is not None:
                      box_line_intersections.append([ds_num, img_num, xy_intersect, h_line, h_line2])
            if img_num == num_imgs-1:
              print("##################")
              print("Line Intersections")
              print("##################")
              for bli in box_line_intersections:
                 print(bli)

            ############
            # find lines
            ############
            if img_num == num_imgs-1:
              num_lines = 0
              # persist_line = []
              # persist_line_most_recent = [] # [[ds_num, img_num]]
              # persist_line[lineno] = {(ds_num, img_num), [key1, key2]}
              # new persist_line[lineno][(d,i for d in range(num_ds) for i in range(num_imgs))] = []
              persist_line_3d = []  
              # 3d_persist_line[line_num] = [composite_3D_line]
              max_angle_dif = .1   # any bigger, can be purged from cache
              line_cache = []      # [(line_num, angle)] 
              akv = angle_sd.items()
              min_dist_allowed = 3
              a_cnt = 0
              for angle, item in akv:
                a_cnt += 1
                if a_cnt % 50 == 0 or len(persist_line) < 7:
                  print(a_cnt, "angle:", angle, len(persist_line))
                # angle, pl_angle => too big to be considered parallel even though same line.... 
                ds_num, img_num, a_line = item
                min_dist = 1000000000000
                best_pl_num = -1
                best_pl_line = -1
                best_pl_ds_num = -1
                best_pl_img_num = -1
                max_counter = -1
                max_counter_num = -1
                # instead of going through each persist line,
                # go to persist line with angles above/below curr angle.
                for pl_num, pl in enumerate(persist_line):
                  mr_pl_ds_num, mr_pl_img_num = persist_line_most_recent[pl_num]
                  pl_angle_lst = persist_line[pl_num][(mr_pl_ds_num, mr_pl_img_num)]
                  if len(persist_line) < 7:
                    print("pl ds,img num:", mr_pl_ds_num, mr_pl_img_num)
                    # print("pl_angle_lst:", pl_angle_lst)
                  for pl_angle in pl_angle_lst:
                    pl_item = angle_sd[pl_angle]
                    pl_ds_num, pl_img_num, pl_line = pl_item
                    dist = parallel_dist(pl_line, a_line)
                    # if len(persist_line) < 7:
                    if True:
                      cont = True
                      for i in range(4):
                        if abs(pl_line[0][i] - a_line[0][i]) > 4:
                         cont = False
                      cont = False
                      if cont:
                        dist = parallel_dist(pl_line, a_line, True)
                        print("pl_angle, item:", pl_angle, pl_item)
                        print("pl_line:", pl_line, a_line)
                        print("pl_dist:", dist)
                        if dist is not None:
                          print(min_dist,sqrt(dist[0]**2+dist[1]**2))
                    if dist is None:
                      # lines may be broken continutions of each other 
                      extended_line = is_broken_line(pl_line, a_line)
                      if extended_line is not None:
                        dist = parallel_dist(pl_line, extended_line)
                        if len(persist_line) < 7:
                          print("dist:", dist, pl_line, extended_line)
                    if dist is not None and min_dist > sqrt(dist[0]**2+dist[1]**2):
                      min_dist = sqrt(dist[0]**2+dist[1]**2)
                      best_pl_num = pl_num
                      best_pl_ds_num = mr_pl_ds_num
                      best_pl_img_num = mr_pl_img_num
                # if ds_num == best_pl_ds_num: 
                #   print("min_dist", min_dist, abs(img_num - best_pl_img_num), best_pl_img_num)
                if ds_num == best_pl_ds_num and min_dist < abs(img_num - best_pl_img_num+1) * min_dist_allowed:
                  # same line
                  # print("p_l",persist_line[best_pl_num])
                  # print("p_l2", persist_line[best_pl_num][(best_pl_ds_num, best_pl_img_num)])
                  try:
                    lst = persist_line[best_pl_num][(ds_num, img_num)]
                    lst.append(angle)
                  except:
                    persist_line[best_pl_num][(ds_num, img_num)] = []
                    persist_line[best_pl_num][(ds_num, img_num)].append(angle)
                  persist_line_most_recent[best_pl_num] = [ds_num, img_num]
                else:
                  persist_line_most_recent.append([ds_num, img_num])
                  if len(persist_line) < 7:
                      print("best pl, ds, img:", best_pl_num, best_pl_ds_num, best_pl_img_num)
                  persist_line.append({})
                  persist_line[-1][(ds_num, img_num)] = []
                  persist_line[-1][(ds_num, img_num)].append(angle)
              # persist_line_stats = []
              mean_gripper_line = []
              mean_gripper_line1 = []
              mean_gripper_line2 = []
              # persist_line_counter = []
              # persist_line_min_max_ds_img = []
              non_gripper = []
              none_disp = []
              big_count = []
              for pl_num, pl in enumerate(persist_line):
                print("PERSISTENT LINE #", pl_num)
                persist_line_stats.append(None)
                counter = 0
                running_sum_x = 0
                running_sum_y = 0
                running_sum2_x = 0
                running_sum2_y = 0
                got_disp = False
                dispcnt = 0
                running_sum_counter = 0
                running_sum2_counter = 0
                running_sum_disp_x = 0
                running_sum_disp_y = 0
                running_sum2_disp_x = 0
                running_sum2_disp_y = 0
                running_line_length = 0
                running_line = [0,0,0,0]
                running_angle = 0
                # for ds = 0, get max/min img#
                persist_line_min_max_ds_img.append([1000000, -1])
                a_ds_num = -1
                a_img_num = -1
                a_line = []
                for pl_ds_num in range(num_ds):
                  if pl_ds_num > 0:
                    # TODO: num_imgs depends on ds_num; get ds_num 0 to work first.
                    print("pl_ds_num > 0", pl_ds_num)
                    break
                  for pl_img_num in range(num_imgs):
                    try:
                      angle_list = pl[(pl_ds_num, pl_img_num)]
                    except:
                      continue
                    # first and last line appearance
                    if pl_img_num  < persist_line_min_max_ds_img[pl_num][0]:
                      persist_line_min_max_ds_img[pl_num][0] = pl_img_num
                    if pl_img_num  > persist_line_min_max_ds_img[pl_num][1]:
                      persist_line_min_max_ds_img[pl_num][1] = pl_img_num

                    for pl_angle in angle_list:
                      prev_a_line = copy.deepcopy(a_line)
                      prev_ds_num = a_ds_num
                      prev_img_num = a_img_num
                      asd_item = angle_sd[pl_angle]
                      a_ds_num, a_img_num, a_line = asd_item
                      if prev_ds_num != -1:
                        disp = parallel_dist(prev_a_line, a_line)
                        if disp is not None:
                          got_disp = True
                          running_sum_disp_x += disp[0]
                          running_sum_disp_y += disp[1]
                          running_sum2_disp_x += disp[0] * disp[0]
                          running_sum2_disp_y += disp[1] * disp[1]
                          dispcnt += 1
                      running_sum_x += (a_line[0][0] + a_line[0][2])/2
                      running_sum_y += (a_line[0][1] + a_line[0][3])/2
                      for i in range(4):
                        running_line[i] += a_line[0][i]
                      x_dif = abs(a_line[0][0] - a_line[0][2])
                      y_dif = abs(a_line[0][1] - a_line[0][3])
                      running_line_length += sqrt(x_dif*x_dif + y_dif*y_dif)
                      running_angle += pl_angle
                      counter += 1
                if counter == 0:
                  print("counter:", counter)
                  print("angle_list:", angle_list)
                  print("pl, ds, img:", pl_num, a_ds_num, a_img_num)
                  continue
                if dispcnt == 0:
                  stddev_disp_x = None
                  stddev_disp_y = None
                  mean_disp_x = None
                  mean_disp_y = None
                else:
                  stddev_disp_x = sqrt(running_sum2_disp_x / dispcnt - running_sum_disp_x * running_sum_disp_x / dispcnt / dispcnt)
                  stddev_disp_y = sqrt(running_sum2_disp_y / dispcnt - running_sum_disp_y * running_sum_disp_y / dispcnt / dispcnt)
                  mean_disp_x = running_sum_disp_x / dispcnt
                  mean_disp_y = running_sum_disp_y / dispcnt
                mean_x = running_sum_x / counter
                mean_y = running_sum_y / counter
                mean_line_length = running_line_length / counter
                mean_angle = running_angle / counter
                mean_line = [[0,0,0,0]]
                for i in range(4):
                  mean_line[0][i] = int(running_line[i]/counter)
                print("mean disp, angle, linlen:", got_disp, mean_disp_x, mean_disp_y, mean_angle, mean_line_length, mean_line, counter) 
                persist_line_stats[pl_num] = [mean_disp_x, mean_disp_y, stddev_disp_x, stddev_disp_y, mean_x, mean_y, copy.deepcopy(mean_line), mean_line_length, mean_angle, counter, copy.deepcopy(persist_line_min_max_ds_img[pl_num])]
                persist_line_counter.append(counter)
                running_sum_counter += counter
                running_sum2_counter += counter * counter
                if got_disp and (abs(mean_disp_x) + abs(mean_disp_y) < 1):
                  mean_gripper_line2.append(mean_line)
                  if got_disp and (abs(mean_disp_x) + abs(mean_disp_y) < .5):
                    mean_gripper_line1.append(mean_line)
                    if got_disp and (abs(mean_disp_x) + abs(mean_disp_y) < .001):
                      mean_gripper_line.append(mean_line)
                elif got_disp:
                  non_gripper.append(mean_line)
                else:
                  none_disp.append(mean_line)
                
                if counter > 50:
                  big_count.append(mean_line)
                  if max_counter < counter:
                    max_counter = counter
                    max_counter_num = pl_num
              counter_cnt = len(persist_line_counter)
              mean_counter = running_sum_counter / counter_cnt
              stddev_counter = sqrt(running_sum2_counter / counter_cnt - running_sum_counter * running_sum_counter / counter_cnt / counter_cnt)
              print("counter mean, stdev", mean_counter, stddev_counter)
              # MIN_GRIPPER_Y
              for pl_num in [max_counter_num]:
                print("PERSIST LINE", pl_num)
                for pl_item in persist_line[pl_num].items():
                  pl_key, angle_list = pl_item
                  for a_num, pl_angle in enumerate(angle_list):
                    asd_item = angle_sd[pl_angle]
                    a_ds_num, a_img_num, a_line = asd_item
                    print(pl_key, a_num, a_line, persist_line_min_max_ds_img[pl_num])

              # print("mean_gripper_line1", len(mean_gripper_line1))
              # print("mean_gripper_line2", len(mean_gripper_line2))
              # print("non_gripper_line", len(non_gripper))
              print("none_disp", len(none_disp))
              print("mean_gripper_line", len(mean_gripper_line))
              display_lines("Mean_Gripper_Lines", mean_gripper_line, drop_off_img)
              # display_lines("Mean_Gripper_Lines1", mean_gripper_line1, drop_off_img)
              # display_lines("Mean_Gripper_Lines2", mean_gripper_line2, drop_off_img)
              # display_lines("Mean_NonGripper_Line", non_gripper, drop_off_img)
              display_lines("Mean_BigCount", big_count, drop_off_img)
              cv2.waitKey(0)

              mean_counter = running_sum_counter / counter_cnt
              stddev_counter = sqrt(running_sum2_counter / counter_cnt - running_sum_counter * running_sum_counter / counter_cnt / counter_cnt)
              print("counter mean, stdev", mean_counter, stddev_counter)
              # MIN_GRIPPER_Y
            pl_last_img_line = {}
            for pl_ds_num in range(num_ds):
              for pl_img_num in range(num_imgs):
                bb = []
                bb_maxw, bb_minw, bb_maxh, bb_minh = -1, 10000, -1, 10000
                for pl_num in range(len(persist_line)):
                  pl_stats = persist_line_stats[pl_num]
                  if pl_stats is not None:
                    [mean_disp_x, mean_disp_y, stddev_disp_x, stddev_disp_y, mean_x, mean_y, mean_line, mean_line_length, mean_angle, counter, [pl_min_img_num, pl_max_img_num]] = pl_stats
                    print("PL", pl_num, counter, mean_line, pl_min_img_num, pl_max_img_num)
                  else:
                    continue
                  if mean_y > MIN_GRIPPER_Y:
                    continue
                  if counter < mean_counter:
                    continue
                  a_line = None
                  try:
                    print("pl angle_list:")
                    angle_list = persist_line[pl_num][(pl_ds_num, pl_img_num)]
                    print(angle_list)
                    l_maxw, l_minw, l_maxh, l_minh = -1, 10000, -1, 10000
                    for a_num, pl_angle in enumerate(angle_list):
                      asd_item = angle_sd[pl_angle]
                      a_ds_num, a_img_num, a_line = asd_item
                      l_maxw = max(a_line[0][0], a_line[0][2], l_maxw)
                      l_minw = min(a_line[0][0], a_line[0][2], l_minw)
                      l_maxh = max(a_line[0][1], a_line[0][3], l_maxh)
                      l_minh = min(a_line[0][1], a_line[0][3], l_minh)
                      if l_maxh > MIN_GRIPPER_Y:
                        l_maxh = MIN_GRIPPER_Y
                    pl_last_img_line[pl_num] = [l_maxw, l_minw, l_maxh, l_minh]
                  except:
                    print("except pl angle_list:")
                    try:
                      [l_maxw, l_minw, l_maxh, l_minh] = pl_last_img_line[pl_num]
                    except:
                      continue
                  if l_maxw == -1 or l_maxh == -1:
                    print("skipping PL", pl_num)
                    continue
                bb = make_bb(bb_maxw, bb_minw, bb_maxh, bb_minh)
                img_path = img_paths[pl_ds_num][pl_img_num]
                img = cv2.imread(img_path)
                bb_img = get_bb_img(img, bb)
                print(pl_img_num, "bb", bb)
                cv2.imshow("bb", bb_img)
                cv2.waitKey(0)

        x = 1/0
        cv2.waitKey(0)
        if False:
              pass

              # display gripper lines
              # display_lines("gripper lines", gripper_lines[ds_num][img_num], drop_off_img)
              # display_lines("gripper lines", gripper_lines2, drop_off_img)


              # display parallel lines
              display_line_pairs("parallel lines", same_frame_parallel[ds_num][img_num], drop_off_img)

              # display diff_frame_parallel lines
              display_line_pairs("moved lines", diff_frame_parallel[ds_num][img_num], drop_off_img)
              found = (len(same_frame_parallel[ds_num][img_num])
                     + len(diff_frame_parallel[ds_num][img_num])
                     + len(broken_lines[ds_num][img_num]))
              if found and new_img_num: 
                print("ds_num, img_num", ds_num, img_num)
                cv2.waitKey(0)
                new_img_num = False
              
            ######################
            # elif pass_num == 2:
        # wrong indentation
        if False:
            if actions[ds_num][action_cnt] == "UPPER_ARM_UP":
              pass
            elif actions[ds_num][action_cnt] == "UPPER_ARM_DOWN":
              pass
            elif actions[ds_num][action_cnt] == "LOWER_ARM_UP":
              pass
            elif actions[ds_num][action_cnt] == "LOWER_ARM_DOWN":
              pass
            elif actions[ds_num][action_cnt] == "GRIPPER_OPEN":
              pass
            elif actions[ds_num][action_cnt] == "GRIPPER_CLOSE":
              pass
            elif actions[ds_num][action_cnt] == "FORWARD":
              track_table()
              estimate_box()
            elif actions[ds_num][action_cnt] == "REVERSE":
              track_table()
              estimate_box()
            elif actions[ds_num][action_cnt] == "LEFT":
              track_table()
              estimate_box()
            elif actions[ds_num][action_cnt] == "RIGHT":
              track_table()
              estimate_box()

            if pass_num == 0:
              # find gripper lines, which are always same at in this phase
              # unchanged over whole sequence
              # gripper_lines = 
              pass

  
            # find line groups that are constant with respect to each other
            # use scale to estimate distance
            # is_parallel
              # -> distance
            # is_perpendicular
              # -> at corners of line
            # intersect


            # discontinued_line

            # vanishing point 
            # [1  0  0  0   [ D       [Dx
            #  0  1  0  0  *  0 ]  =   Dy
            #  0  0  1  0]             Dz]
      
            

  
            # TODO: eliminate moving line groups
  
            # find relationship of line groups
            # is one behind/above another?
  
            # estimate drop point into box

            # estimate position of camera

def black_out_light(rl, image):
      black = [0,0,0]
      IMG_HW = img_sz()
      rlmask = rl["LABEL"].copy()
      rlmask = rlmask.reshape((IMG_HW,IMG_HW))

      for h in range(image.shape[0]):
        for w in range(image.shape[1]):
          if 124 <= unmoved_pix[h][w] <= 255:
            image[h, w] = black


def black_out_gripper(unmoved_pix, image):
      ret, contour_thresh = cv2.threshold(unmoved_pix.copy(), 125, 255, 0)
      contour_thresh = cv2.dilate(contour_thresh,None,iterations = 10)
      contours, hierarchy = cv2.findContours(contour_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
      accuracy_factor = .1
      x,y,ch = image.shape
      obj_contours = image.copy()
      for c in contours:
        if len(contours) > 1:
          area = cv2.contourArea(c)
          if area < 16:
            continue
          # Calculate accuracy as a percent of the contour perimeter
          accuracy = accuracy_factor * cv2.arcLength(c, True)
          approx = cv2.approxPolyDP(c, accuracy, True)
          cv2.drawContours(obj_contours, c, -1, (0,255,0), 3)
          cv2.imshow("contours", obj_contours)

      black = [0,0,0]
      for h in range(image.shape[0]):
        for w in range(image.shape[1]):
          if 124 <= unmoved_pix[h][w] <= 255:
            image[h, w] = black

      cv2.imshow("blacked out gripper", image)
      # cv2.waitKey(0)
      return image




      Z = img.copy()
      Z = Z.reshape((-1,3))
      # convert to np.float32
      Z = np.float32(Z)
      # define criteria, number of clusters(K) and apply kmeans()
      criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
      K = num_clusters
      # compactness : sum of squared distance from each point to their centers.
      # labels : the label array where each element marked '0', '1'.....
      # centers : This is array of centers of clusters.
      ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
      # Now convert back into uint8, and make original image
      # print("compactness, centers:", ret, center)
      # ret is a single float, label is ?, center is RGB
      center = np.uint8(center)
      res = center[label.flatten()]
      res2 = res.reshape((img.shape))
      return res2


def depth(imgL, imgR):

    if True:
      # disparity range is tuned for 'aloe' image pair
      window_size = 3
      # window_size = 6
      min_disp = 16
      # num_disp = 112-4*min_disp
      num_disp = 112-min_disp
      # num_disp = 48
      num_disp = 64
      # num_disp = 112+ 4*min_disp
      # num_disp = 16*(100//16)
      print('computing disparity...')
      stereo = cv2.StereoSGBM_create(
          minDisparity = min_disp,
          numDisparities = num_disp,
          blockSize = 16,
          # blockSize = 25,
          P1 = 8*3*window_size**2,
          P2 = 32*3*window_size**2,
          disp12MaxDiff = 1,
          uniquenessRatio = 10,
          speckleWindowSize = 100,
          speckleRange = 32
      )
      disparity = stereo.compute(imgL, imgR)
      disp = disparity.astype(np.float32) / 16.0

    elif True:
      # disparity range is tuned for 'aloe' image pair
      window_size = 3
      min_disp = 16
      num_disp = 112-min_disp
      print('computing disparity...')
      stereo = cv2.StereoSGBM_create(
          minDisparity = min_disp,
          numDisparities = num_disp,
          blockSize = 16,
          P1 = 8*3*window_size**2,
          P2 = 32*3*window_size**2,
          disp12MaxDiff = 1,
          uniquenessRatio = 10,
          speckleWindowSize = 100,
          speckleRange = 32
      )
      disparity = stereo.compute(imgL, imgR)
      disp = disparity.astype(np.float32) / 16.0

    else:
                min_disp = 16
                l_img = imgL
                r_img = imgR
                blocksize = 25
                numDisp = 16 * (100//16)
                print("blocksize:", blocksize, numDisp)
                stereo = cv2.StereoBM_create(numDisparities=numDisp, blockSize=blocksize)
                disparity = stereo.compute(r_img, l_img)  # L, R
                cv2.imshow(prev_action,prev_img)
                cv2.imshow(action,img)
                cv2.imshow("Disparity",disparity)
                lrdisparity = stereo.compute(l_img, r_img)  # L, R
                cv2.imshow("LRDisparity",lrdisparity)
                stereo = cv2.StereoBM_create(numDisparities=numDisp+16, blockSize=blocksize)
                rldisparitym = stereo.compute(r_img, l_img)  # L, R
                cv2.imshow("rlDisparitym",rldisparitym)
                full_disparity = stereo.compute(img, prev_img)  # L, R
                cv2.imshow("full_Disparity",full_disparity)
                full_disparity2 = stereo.compute(prev_img, img)  # L, R
                # cv2.waitKey(0)
                disp = disparity.astype(np.float32) / 16.0



    print('generating 3d point cloud...',)
    h, w = imgL.shape[:2]
    f = 0.8*w                          # guess for focal length
    Q = np.float32([[1, 0, 0, -0.5*w],
                    [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
                    [0, 0, 0,     -f], # so that y-axis looks up
                    [0, 0, 1,      0]])
    points = cv2.reprojectImageTo3D(disp, Q)
    colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
    mask = disp > disp.min()
    out_points = points[mask]
    out_colors = colors[mask]
    out_fn = 'out.ply'
    write_ply('out.ply', out_points, out_colors)
    print('%s saved' % '.ply')
    cv2.imshow('left', imgL)
    cv2.imshow('right', imgR)
    ## cv2.imshow('disparity', np.hstack((imgL, imgR, (disp-min_disp)/num_disp)))
    # cv2.imshow('disparity', disparity)
    cv2.imshow('disparity', (disp-min_disp)/num_disp)
    # cv2.waitKey()
    # cv2.destroyAllWindows()



def point_cloud(img, line_seg):
    print('generating 3d point cloud...',)
    h, w = imgL.shape[:2]
    f = 0.8*w                          # guess for focal length
    # raspberry pi:
    # focal_length_pixels = 3.04mm / 0.00112mm = 2714.3px

    Q = np.float32([[1, 0, 0, -0.5*w],
                    [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
                    [0, 0, 0,     -f], # so that y-axis looks up
                    [0, 0, 1,      0]])

    # 0.482897 1.794769 -1.313481
    # points = cv2.reprojectImageTo3D(disp, Q)
    delta_point = .1
    line_seg_3d = []
    for line in line_seg_3d:
      for x1,y1,z1, x2,y2,z2 in line:
        for x in range(x1, x2, delta_point):
          for y in range(y1, y2, delta_point):
            for z in range(z1, z2, delta_point):
              points.append(x,y,z) 

    colors = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = disp > disp.min()
    out_points = points[mask]
    out_colors = colors[mask]
    out_fn = 'out.ply'
    write_ply('out.ply', out_points, out_colors)
    print('%s saved' % 'obstacle.ply')


def above_gripper_bb():
      # imgsz = 640
      imgsz = img_sz()
      above_grippers_bb = make_bb(imgsz-1, 0, int(imgsz/2) - 10, 0)
      print("Above Gripper BB:", above_grippers_bb)
      return above_grippers_bb

def find_best_offset(action, l_img, r_img, low_val=None, high_val=None):
    min_mse = 10000000000000000000
    val_mse = None
    # imgsz = 640
    imgsz = img_sz()
    l_bb = above_gripper_bb()
    r_bb = above_gripper_bb()
    l_ag_img = get_bb_img(l_img.copy(), l_bb)
    r_ag_img = get_bb_img(r_img.copy(), r_bb)
    # plt.imshow(l_ag_img,'gray')
    # plt.pause(5)
    if action not in ["LEFT", "RIGHT", "GRIPPER_OPEN", "GRIPPER_CLOSE"]:
        l_ag_img = cv2.rotate(l_ag_img, cv2.ROTATE_90_CLOCKWISE)
        r_ag_img = cv2.rotate(r_ag_img, cv2.ROTATE_90_CLOCKWISE)
    if low_val is None:
      low_val = 0
    if high_val is None:
      high_val = int(imgsz/2)
    for val in range(low_val, high_val):
      new_l_bb = make_bb(l_ag_img.shape[1], val, l_ag_img.shape[0], 0)
      new_l_img = get_bb_img(l_ag_img.copy(), new_l_bb)
      new_r_bb = make_bb(r_ag_img.shape[1] - val, 0, r_ag_img.shape[0], 0)
      new_r_img = get_bb_img(r_ag_img.copy(), new_r_bb)
      # print("MSE val1:", val, new_l_bb, new_r_bb, new_l_img.shape, new_r_img.shape)
      val_mse = cvu.mean_sq_err(new_l_img, new_r_img)
      # print("MSE val:", val, val_mse, new_l_bb, new_r_bb)
      if min_mse >  val_mse:
        min_mse = val_mse
        # best_l_bb  = copy.deepcopy(new_l_bb)
        # best_r_bb  = copy.deepcopy(new_r_bb)
        best_val  = val
        # best_l_img  = new_l_img.copy()
        # best_r_img  = new_r_img.copy()
        # print("MSE BEST_OBJ_BB:", val, val_mse)
    print("MSE BEST_OBJ_BB:", best_val, val_mse)
    return best_val, min_mse, l_ag_img, r_ag_img

if __name__ == '__main__':
    cvu = CVAnalysisTools()
    func_idx_file = "sample_code/TT_BOX.txt"
    dsu = DatasetUtils(app_name="TT", app_type="FUNC")
    dataset = [[]]
    curr_dataset = dataset[0]
    unmoved_pix = None
    slow_moved_pix = None
    num_datasets = 0
    num_images = 0
    prev_func_name = ""
    unique_color = {}
    with open(func_idx_file, 'r') as file1:
      while True:
        ds_idx = file1.readline()
        if not ds_idx:
          break
        if ds_idx[-1:] == '\n':
          ds_idx = ds_idx[0:-1]
        # ./apps/FUNC/GOTO_BOX_WITH_CUBE/dataset_indexes/FUNC_GOTO_BOX_WITH_CUBE_21_05_16a.txt
        func_name = dsu.get_func_name_from_idx(ds_idx)
        # func_name = dsu.dataset_idx_to_func(ds_idx)
        if prev_func_name == "DROP_CUBE_IN_BOX" and func_name != "DROP_CUBE_IN_BOX":
          curr_dataset = dataset.append([])
        delta_arm_pos = {"UPPER_ARM_UP":0, "UPPER_ARM_DOWN":0,
                         "LOWER_ARM_UP":0, "LOWER_ARM_DOWN":0}
        arm_pos = []
        with open(ds_idx, 'r') as file2:
          while True:
            img_line = file2.readline()
            # 21:24:34 ./apps/FUNC/QUICK_SEARCH_FOR_BOX_WITH_CUBE/dataset/QUICK_SEARCH_FOR_BOX_WITH_CUBE/LEFT/9e56a302-b5fe-11eb-83c4-16f63a1aa8c9.jpg
            if not img_line:
               break
            print("img_line", img_line)
            curr_dataset.append(img_line)
            [time, app, mode, func_name, action, img_name, img_path] = dsu.get_dataset_info(img_line,mode="FUNC") 
            if action in ["UPPER_ARM_UP", "UPPER_ARM_DOWN", "LOWER_ARM_UP", "LOWER_ARM_DOWN"]:
              delta_arm_pos[action] += 1
            arm_pos.append(delta_arm_pos.copy())

    print("dataset",dataset)
    img = None
    img_paths = []
    prev_img = None
    num_passes = 4
    box_lines = []
    actions = []
    # img_copy = []
    edge_copy = []
    rl_copy = []
    compute_gripper = True
    for pass_num in range(num_passes):
      for ds_num, ds in enumerate(dataset):
        # img_copy_num = len(ds) // 24
        ################################
        # REMOVE
        ################################
        if ds_num > 0:
          continue #concentrate on first ds initially
        ################################
        if pass_num == 0:
          box_lines.append([])
          actions.append([])
          img_paths.append([])
        for img_num, img_line in enumerate(reversed(ds)):
          [time, app, mode, func_name, action, img_name, img_path] = dsu.get_dataset_info(img_line,mode="FUNC") 
          # print("img_path", img_path)
          prev_action = action
          if img is not None:
            prev_img = img
          img = cv2.imread(img_path)
          if pass_num == 0:
            alset_state = AlsetState()
            cvu = CVAnalysisTools(alset_state)
            adj_img,mean_diff,rl = cvu.adjust_light(img_path)
            if rl is not None:
              rl_img = img.copy()
              mask = rl["LABEL"]==rl["LIGHT"]
              mask = mask.reshape((rl_img.shape[:2]))
              # print("mask",mask)
              rl_img[mask==rl["LIGHT"]] = [0,0,0]
              # adj_img = rl_img
              center = np.uint8(rl["CENTER"].copy())
              rl_copy = rl["LABEL"].copy()
              res    = center[rl_copy.flatten()]
              rl_img2  = res.reshape((img.shape[:2]))


            # stego(rl_img)
            stego_img, unique_color = stego(img, unique_color)
            cv2.imshow("stego orig input img", img)
            cv2.imshow("stego img", stego_img)
            # cv2.imshow("stego adj input img", adj_img)
            # cv2.imshow("stego input img", rl_img)
            # cv2.imshow("stego rl img", rl_img2)
            cv2.waitKey(0)

            img_paths[ds_num].append(img_path)
            # if unmoved_pix is None:
              # gray = cv2.cvtColor(curr_img.copy(), cv2.COLOR_BGR2GRAY)
              # unmoved_pix = gray.copy()
            if prev_img is not None:
              watershed2(img)
              actions[ds_num].append(action)
              unmoved_pix, slow_moved_pix, hough_lines, d_edge = unmoved_pixels(unmoved_pix, slow_moved_pix, action, prev_img, img)
              edge_copy.append(d_edge.copy())
              # add box_lines for img[ds_num][img_num]
              box_lines[ds_num].append(hough_lines)
            else:
              box_lines[ds_num].append([])
            # if img_num % img_copy_num == 0:
              # img_copy.append(img.copy())

            # adjusted_image, mean_dif, rl = cvu.adjust_light(img.copy(), add_to_mean=True)
            # rlimg = np.zeros_like(gray)
            # center = np.uint8(rl["CENTER"].copy())
            # rl_copy = rl["LABEL"].copy()
            # res    = center[rl_copy.flatten()]
            # rlimg  = res.reshape((rlimg.shape[:2]))
            # robot_light.append(rlimg.copy())

          elif pass_num == 1:
            if compute_gripper:
              compute_gripper = False
              num_ors = int(np.sqrt(len(edge_copy)))
              print("num edge_copy", num_ors, len(edge_copy), len(ds))
              running_ors = None
              running_ands = None
              do_or = 0
              for i2 in range(2, len(edge_copy)-2):
                if do_or == 0:
                  running_ors = cv2.bitwise_or(edge_copy[i2-1], edge_copy[i2])
                  # running_ors = cv2.bitwise_or(edge_copy[i2-1], robot_light[i2])
                  do_or = 2
                else:
                  running_ors = cv2.bitwise_or(running_ors, edge_copy[i2])
                  # running_ors = cv2.bitwise_or(running_ors, robot_light[i2])
                  do_or += 1
                if do_or == num_ors:
                  do_or = 0
                  if running_ands is None:
                    running_ands = running_ors
                  else:
                    running_ands = cv2.bitwise_and(running_ands, running_ors)
              gripper = running_ands
              cv2.imshow("running_gripper", running_ands)
              # cv2.waitKey(0)
            # overwrite previous attempt at getting gripper.
            box_lines[ds_num][img_num] = get_box_lines(img, gripper)
          elif pass_num == 2:
            analyze_box_lines(box_lines, actions, gripper, slow_moved_pix, img_paths)
            black_out_gripper(unmoved_pix, img)

