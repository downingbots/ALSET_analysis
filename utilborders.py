import cv2
from shapely.geometry import *
from config import *
import numpy as np

#########################
# Border manipulation
#########################
def add_border(img, bordersize):
    print("bordersize:", bordersize, img.shape[:2])
    height, width = img.shape[:2]
    bottom = img[height-2:width, 0:width]
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

def replace_border(img, desired_height, desired_width, offset_height, offset_width):
    shape, border = real_map_border(img)
    maxw, minw, maxh, minh = get_min_max_borders(border)
    # maxh, minh, maxw, minw = get_min_max_borders(border)
    print("maxh, minh, maxw, minw :", maxh, minh, maxw, minw, desired_height, desired_width, offset_height, offset_width)
    extract_img_rect = img[minh:maxh, minw:maxw]
    # extract_img_rect = img[minw:maxw, minh:maxh]
    extract_height, extract_width = extract_img_rect.shape[:2]
    insert_height = int(extract_height + 2*abs(offset_height))
    insert_width  = int(extract_width + 2*abs(offset_width))
    insert_img_rect = np.zeros((insert_height,insert_width,3),dtype="uint8")
    print("ext_h, ins_h, off_h:", extract_height, insert_height, offset_height)
    for eh in range(extract_height):
      for ew in range(extract_width):
        new_w = ew + abs(offset_width) + offset_width
        new_h = eh + abs(offset_height) + offset_height
        insert_img_rect[new_h, new_w] = extract_img_rect[eh,ew]
    border_top = int((desired_height - insert_height)/2)   
    border_bottom = desired_height - border_top - insert_height
    border_left = int((desired_width - insert_width)/2) 
    border_right = desired_width - border_left - insert_width 
    print("replace_border:",border_top, border_bottom, border_left, border_right, offset_height, offset_width)
    # replace_border: -2355 -2355 674 675 3014 0
    bordered_img = cv2.copyMakeBorder(
        insert_img_rect,
        top=border_top,
        bottom=border_bottom,
        left=border_left,
        right=border_right,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]  # black
    )
    return bordered_img

def radians_to_degrees(rad):
    return  rad * (180/np.pi)

# height/width to x/y coordinates (for using image vs. math APIs)
def hw_to_xy(A):
    return [A[1],A[0]]

# x/y to height/width coordinates (for using image vs. math APIs)
def xy_to_hw(A):
    return [A[1],A[0]]

# angle = int(math.atan((y1-y2)/(x2-x1))*180/math.pi)
# image is "new map" == the transformed "curr move" with border
# center is about 4xfinger pads -> robot_location
def rotate_about_robot(image, angle, robot_location):
    # Draw the text using cv2.putText()
    # Rotate the image using cv2.warpAffine()
 
    # getRotationMatrix2D angle is in degrees!
    
    rbt_loc = [robot_location[1], robot_location[0]]
    M = cv2.getRotationMatrix2D(rbt_loc, radians_to_degrees(angle), 1)
    print("angle,M:", angle, M, robot_location, image.shape)
    out = cv2.warpAffine(image, M, (image.shape[0], image.shape[1]))

    pt = [[int(robot_location[0]), int(robot_location[1])], [int(image.shape[1]/2), int(image.shape[0]/2)]] # [w,h]
    pt = np.float32(pt).reshape(-1,1,2)
    # ln2 = cv2.perspectiveTransform(pt,M)
    # print("3 robot loc pt", pt2[0][0])
    # pt2 = [int(robot_location[1]), int(robot_location[0])]
    # out = cv2.circle(out,pt2,3,(255,0,0),-1)
    # out = cv2.polylines(out,[np.int32(ln2)],True,255,3, cv2.LINE_AA)

    # Display the results
    # cv2.imshow('rotate about robot', out)
    # cv2.waitKey(0)
    return out

def middle_angle(pt1, angle_pt, pt2):
  # dist1 = np.sqrt((pt1[0] - angle_pt[0])**2 + (pt1[1] - angle_pt[1])**2)
  # dist2 = np.sqrt((pt2[0] - angle_pt[0])**2 + (pt2[1] - angle_pt[1])**2)
  # dist3 = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
  # rad_angle = np.arccos((dist1**2 + dist2**2 - dist3**2) / (2.0 * dist1 * dist2))
  # return rad_angle
  dx1 = pt1[0] - angle_pt[0]
  dy1 = pt1[1] - angle_pt[1]
  dx2 = pt2[0] - angle_pt[0]
  dy2 = pt2[1] - angle_pt[1]
  rad_angle2 = (dx1*dx2 + dy1*dy2)/np.sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10)

  # print("compare angles:", rad_angle, rad_angle2)
  return rad_angle2


def real_map_border(mapimg, ret_outside=True):
    # convert the stitched image to grayscale and threshold it
    # such that all pixels greater than zero are set to 255
    # (foreground) while all others remain 0 (background)
    shape, approximations = None, None
    try:
      gray = cv2.cvtColor(mapimg, cv2.COLOR_BGR2GRAY)
      thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
    except:
      thresh = mapimg
    # find all external contours in the threshold image then find
    # the *largest* contour which will be the contour/outline of
    # the stitched image
    if ret_outside:
      # imagecontours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
      imagecontours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
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
      for count in imagecontours:
        if len(imagecontours) > 1:
          area = cv2.contourArea(count)
          if area < 1000:
            continue
        epsilon = 0.01 * cv2.arcLength(count, True)
        approximations = cv2.approxPolyDP(count, epsilon, True)
        # e.g. [[[224 224]] [[252 372]] [[420 372]] [[447 224]]]
        #the name of the detected shapes are written on the image
        i, j = approximations[0][0] 
        if len(approximations) == 3:
          shape = "Triangle"
        elif len(approximations) == 4:
          shape = "Trapezoid"
          # area = Polygon(b2).area
          # if area > 100 &  cv2.isContourConvex(approximations):
#          if cv2.isContourConvex(approximations):
#            max_Cosine = -1000000000000000000
#            for j in range(2, 5):
#              cosine = abs(middle_angle(approximations[j%4], approximations[j-2], approximations[j-1]))
#              maxCosine = max(maxCosine, cosine)
#            if maxCosine < 0.3 and maxCos >= 0:
#              shape = "Square"
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
        return shape, approximations
      return shape, approximations

def get_min_max_borders(border):
    b = []
    for bdr in border:
      b.append(list(bdr[0]))
    poly = Polygon(b)
    minw, minh, maxw, maxh = poly.bounds
    return int(maxw), int(minw), int(maxh), int(minh)
    # minh, minw, maxh, maxw = poly.bounds
    # minx, miny, maxx, maxy
    # return int(maxh), int(minh), int(maxw), int(minw)

def border_to_polygon(border, bufzone=0):
    b = []
    for bdr in border:
      # b.append([int(bdr[0][0]), int(bdr[0][1])])
      b.append(list(bdr[0]))
    # print("brdr to poly:", b, bufzone)
    poly = Polygon(b)
    if bufzone > 0:
      b2 = []
      center = [poly.centroid.x,poly.centroid.y]
      for bpt in b:
        b2pt = bpt
        for i in range(2):
          if center[i] < bpt[i]-bufzone and center[i] < bpt[i]:
            b2pt[i] = bpt[i]-bufzone
          elif center[i] > bpt[i]+bufzone and center[i] > bpt[i]:
            b2pt[i] = bpt[i]+bufzone
          else:
            b2pt[i] = bpt[i]
        b2.append(b2pt)
      poly = Polygon(b2)
      # print("len b2, center", len(b2), center)
    return poly

def border_radius(border):
    poly = border_to_polygon(border)
    cent = poly.centroid
    minradius = INFINITE
    for bpt in border:
      b = bpt[0]
      if (abs(b[0] - cent.x) < minradius):
        minradius = int(abs(b[0] - cent.x))
      if (abs(b[1] - cent.y) < minradius):
        minradius = int(abs(b[1] - cent.y))
      # line = LineString([cent,b])
      # if line.length < minradius:
      # minradius = int(line.length)
    c = [round(cent.x), round(cent.y)]
    return c, minradius

def image_around_center(map, center, radius):
    side = int(radius*2-1)
    img = np.zeros((side, side), dtype = "float32")
    for h in range(side):
      for w in range(side):
        mh = center[0] - radius + h
        mw = center[1] - radius + w
        img[h,w] = map[mh, mw]
    return img


def intersect_borders(border1,border2):
    dbg = False
    poly1 = border_to_polygon(border1)
    poly2 = border_to_polygon(border2)
    intersect = poly1.intersection(poly2)
    max_area = 0
    if intersect.type=='MultiPolygon':
      # for x in geo.geom:
      # for x in intersect.geom:
      if dbg:
        print("intersect:", intersect)
      for x in intersect:
        b = []
        b2 = []
        coordslist = list(x.exterior.coords)
        for pt in coordslist[:]:
            if dbg:
              print("i.e.c2", pt)
            b.append([[int(pt[0]), int(pt[1])]])
            b2.append([int(pt[0]), int(pt[1])])

        for interior in x.interiors:
          for pt in interior.coords[:]:
            # if dbg:
            print("i.i.c3", pt)
            b.append([[int(pt[0]), int(pt[1])]])
            b2.append([int(pt[0]), int(pt[1])])

        print("coords",b)
        if len(b) == 0:
          if dbg:
            print("no intersection between borders:")
            print("map1 border", border1)
            print("map2 border", border2)
        else:
          area = Polygon(b2).area
          if dbg:
              print("intersect area", area)
          if area > max_area:
            max_area = area
            max_b = b.copy()
    else:
        if dbg:
          # print("intersection of maps", intersect.exterior.coords[:])
          print("intersection of maps", intersect.interiors[0].coords[:])
        b = []
        b2 = []
        for pt in intersect.exterior.coords[:]:
            # print("i.e.c", pt)
            b.append([[int(pt[0]), int(pt[1])]])
            b2.append([int(pt[0]), int(pt[1])])
        for interior in intersect.interiors:
          for pt in interior.coords[:]:
            print("i.i.c", pt)
            b.append([[int(pt[0]), int(pt[1])]])
            b2.append([int(pt[0]), int(pt[1])])
        print("i.e.c",b)
        if len(b) == 0:
          if dbg:
            print("no intersection between borders:")
            print("map1 border", border1)
            print("map2 border", border2)
        else:
          area = Polygon(b2).area
          if dbg:
            print("i.e.c. area", area)
        max_b = b
    return max_b

def line_intersect_border(poly, pt1, pt2, ignore_pt, border):
    dbg = False
    line = LineString([pt1, pt2])
    intersect = None
    for interior in poly.interiors:
      intersect = interior.intersection(line)
      print("INTERIOR intersection")
      break
    if intersect is None:
      intersect = poly.exterior.intersection(line)
    if dbg:
      print("Type:", intersect.geom_type)
    if intersect.is_empty:
      if dbg:
        print("shapes don't intersect", pt1, pt2)
      return None
    elif intersect.geom_type.startswith('Point'):
      if dbg:
        print("intercept,p1,p2", intersect.coords[0], pt1, pt2)
      return [round(intersect.coords[0][0]), round(intersect.coords[0][1])]
    elif intersect.geom_type.startswith('Line'):
      if len(intersect.coords) == 2:
        if ignore_pt[0] is not None and ignore_pt[0] == round(intersect.coords[0][0]):
          if dbg:
            print("Line: ignore/return,p1,p2 ", intersect.coords[0][0], intersect.coords[1], pt1, pt2)
          return [round(intersect.coords[1][0]),round(intersect.coords[1][1])]
        if ignore_pt[1] is not None and ignore_pt[1] == round(intersect.coords[0][1]):
          if dbg:
            print("Line: ignore/return,p1,p2 ", intersect.coords[0][1], intersect.coords[1],pt1,pt2)
          return [round(intersect.coords[1][0]),round(intersect.coords[1][1])]
        if dbg:
          print("Line: return ", intersect.coords[0])
        return [round(intersect.coords[0][0]), round(intersect.coords[0][1])]
      if dbg:
        print("len coords:", len(intersect.coords))
    elif intersect.geom_type.startswith('Multi') or intersect.geom_type == 'GeometryCollection':
      for shp in intersect:
        if dbg:
          print("shape:",shp)
        # gather together Points
        mpts = []
        if len(shp.coords) == 1:
          mpts.append([round(shp.coords[0][0]), round(shp.coords[0][1])])
        # elif shp.geop_type.startswith('Line'):
        elif len(shp.coords) == 2:
          mpts.append([round(shp.coords[0][0]), round(shp.coords[0][1])])
          mpts.append([round(shp.coords[1][0]), round(shp.coords[1][1])])

        best_pt = None
        for p in mpts:
          if ignore_pt[0] is not None and ignore_pt[0] == p[0]:
            if dbg:
              print("MultiPt: ignore ", p)
            continue
          elif ignore_pt[1] is not None and ignore_pt[1] == p[1]:
            if dbg:
              print("MultiPt: ignore ", p)
            continue
          if ((pt1[0] == pt2[0] and pt1[0] == p[0]) and
              (pt1[1] == ignore_pt[1] and pt1[1] < pt2[1])):
            # take Min Point
            if best_pt is None or best_pt[1] < p[1]:
              best_pt = p
          if ((pt1[0] == pt2[0] and pt1[0] == p[0]) and
              (pt1[1] == ignore_pt[1] and pt1[1] > pt2[1])):
            # take Max Point
            if best_pt is None or best_pt[1] > p[1]:
              best_pt = p
          if ((pt1[1] == pt2[1] and pt1[1] == p[1]) and
              (pt1[0] == ignore_pt[0] and pt1[0] < pt2[0])):
            # take Min Point
            if best_pt is None or best_pt[0] < p[0]:
              best_pt = p
          if ((pt1[1] == pt2[1] and pt1[1] == p[1]) and
              (pt1[0] == ignore_pt[0] and pt1[0] > pt2[0])):
            # take Max Point
            if best_pt is None or best_pt[0] > p[0]:
              best_pt = p
        return best_pt

    elif intersect.geom_type.startswith('Coord'):
      for shp in intersect:
        print("Coord shape:",shp.coords)
        if ignore_pt[0] is not None and ignore_pt[0] == shp.coords[0][0]:
          continue
        if ignore_pt[1] is not None and ignore_pt[1] == shp.coords[1][1]:
          continue
        return shp.coords
    else:
      print("Unknown intersect", intersect)
      return intersect.coords

def rectangle_in_border(border):
    dbg = False
    # dbg = True
    poly = border_to_polygon(border)
    try:
      minw, minh, maxw, maxh = poly.bounds
      print("rect in brdr: ",minw, minh, maxw, maxh)
    except:
      print("no poly bounds:", poly, border)
      return None, None, None, None

    max_area = 0
    for bpt in border:
      pt = bpt[0]
      pt1 = [pt[0]+1,pt[1]]
      # pt2 = [pt[0]+maxh,pt[1]]
      pt2 = [pt[0]+maxw,pt[1]]
      ignore_pt = [pt[0]+1, None]
      if dbg:
        print("Line Up", pt, pt1, pt2, ignore_pt)
      vert_pt = line_intersect_border(poly, pt1, pt2, ignore_pt, border)
      if vert_pt is None:
        if dbg:
          print("no intersect with vert_pt up")
        pt1 = [pt[0]-1,pt[1]]
        # pt2 = [pt[0]-maxh,pt[1]]
        pt2 = [pt[0]-maxw,pt[1]]
        ignore_pt = [pt[0]-1, None]
        if dbg:
          print("Line Down", pt, pt1, pt2, ignore_pt)
        vert_pt = line_intersect_border(poly, pt1, pt2, ignore_pt, border)
        if vert_pt is None:
          if dbg:
            print("no intersect with vert_pt down")
          continue

      pt1 = [pt[0],pt[1]-1]
      # pt2 = [pt[0],pt[1]-maxw]
      pt2 = [pt[0],pt[1]-maxh]
      ignore_pt = [None, pt[1]-1]
      if dbg:
        print("Line Left", pt, pt1, pt2, ignore_pt)
      horiz_pt = line_intersect_border(poly, pt1, pt2, ignore_pt, border)
      if horiz_pt is None:
        # ARD: pt1 is off by 1
        pt1 = [pt[0],pt[1]+1]
        # pt2 = [pt[0],pt[1]+maxw]
        pt2 = [pt[0],pt[1]+maxh]
        ignore_pt = [None, pt[1]+1]
        if dbg:
          print("Line Right", pt, pt1, pt2, ignore_pt)
        horiz_pt = line_intersect_border(poly, pt1, pt2, ignore_pt, border)
        if horiz_pt is None:
          continue

      pt1 = vert_pt
      pt2 = [vert_pt[0],horiz_pt[1]]
      ignore_pt = [None, vert_pt[1]]
      if dbg:
        # ARD: vert is off by 1
        print("vert:", vert_pt)
        print("horiz:", horiz_pt)
        print("Find Diag", pt1, pt2, ignore_pt)
      out_pt = line_intersect_border(poly, pt1, pt2, ignore_pt, border)
      if out_pt is None:
        diag_pt = pt2
      elif out_pt is not None:
        diag_pt = out_pt
        horiz_pt = [horiz_pt[0],diag_pt[1]]
        if dbg:
          print("diag:", diag_pt)
          print("horiz:", horiz_pt)
      pt1 = horiz_pt
      pt2 = diag_pt
      ignore_pt = [horiz_pt[0], None]
      if dbg:
        print("Tune/Verify Diag", pt1, pt2, ignore_pt)
      out_pt = line_intersect_border(poly, pt1, pt2, ignore_pt, border)
      if out_pt is not None:
        diag_pt = out_pt
        vert_pt = [diag_pt[0],vert_pt[1]]
        if dbg:
          print("diag:", diag_pt)
          print("vert:", vert_pt)

      rect = [pt,[pt[0],int(diag_pt[1])],[int(diag_pt[0]), int(diag_pt[1])],
              [int(diag_pt[0]), pt[1]]]
      area = Polygon(rect).area
      if area > max_area:
        max_area = area 
        # min_h = int(min(pt[0], diag_pt[0])) 
        # min_w = int(min(pt[1], diag_pt[1])) 
        # max_h = int(max(pt[0], diag_pt[0]))
        # max_w = int(max(pt[1], diag_pt[1]))
        min_w = int(min(pt[0], diag_pt[0])) # why does w&h seem reversed??
        # why is min_h off by 1?
        min_h = int(min(pt[1], diag_pt[1])) # ... gives the right results tho
        max_w = int(max(pt[0], diag_pt[0]))
        max_h = int(max(pt[1], diag_pt[1]))
    if max_area == 0:
      return None, None, None, None
    if dbg:
      print("Best area", area, min_h, min_w, max_h, max_w )
    return min_h, min_w, max_h, max_w 

def image_in_border(border, image):
    minh, minw, maxh, maxw = rectangle_in_border(border)
    if minh is None:
      print("Error: problem with border:", border)
      im = image.copy()
      bdr = []
      for bpt in border:
        im = cv2.circle(im,bpt[0],3,(255,0,0),-1)
      cv2.imshow("Image with no rectangle", im)
      # cv2.waitKey(0)
      return None
    print("minw, minh, maxw, maxh:", minw, minh, maxw, maxh)
    if minw is None:
      cv2.imshow("Image with None minh/minw", image)
      # cv2.waitKey(0)
    final_image = np.zeros((maxh-minh, maxw-minw), dtype="uint8")
    final_image[0:maxh-minh, 0:maxw-minw] = image[minh:maxh, minw:maxw]
    final_image_disp = final_image.copy()
    # cv2.rectangle(final_image_disp, (minh,minw), (maxh,maxw), 255, -1)
    # cv2.imshow("final_img", final_image_disp)
    # cv2.waitKey(0)
    return final_image

def point_in_border(border, pt, bufzone=10):
    # Create Point objects
    poly = border_to_polygon(border, bufzone)
    Pt = Point(pt)
    if Pt.within(poly):
      # print("pt within", Pt)
      return True
    else:
      # print("pt not within", Pt)
      return False

def line_in_border(border,pt0,pt1, bufzone=10):
    poly = border_to_polygon(border, bufzone)
    line_a = LineString([pt0,pt1])
    # print("line_in_border:", line_a.centroid, border)
    if line_a.centroid.within(poly):
      return True
    else:
      return False

def check_gripper_bounding_box(max_bb, test_bb):
    if len(test_bb) == 2:
      print("test_bb:", test_bb)
      if point_in_border(max_bb, test_bb[0][0], 0):
        if point_in_border(max_bb, test_bb[1][0], 0):
          return True
      return False
    elif len(test_bb) == 1:
      return point_in_border(max_bb, test_bb[0][0], 0)
    max_bb_p = border_to_polygon(max_bb)
    test_bb_p = border_to_polygon(test_bb)
    if test_bb_p.within(max_bb_p):
      return True
    return False
