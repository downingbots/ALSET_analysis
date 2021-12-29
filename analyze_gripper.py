import cv2
from cv_analysis_tools import *
from analyze_texture import *
from dataset_utils import *
# from cube import *

class AnalyzeGripper():

  def __init__(self, alset_state):
      self.stop_at_frame = 117
      self.gripper_img = None
      self.prev_frame_num = -1
      self.prev_action = ""
      self.background = None
      self.threshold = 10
      self.gripper_open_cnt = 0
      self.gripper_close_cnt = 0
      self.gripper_state = "UNKNOWN"
      self.gripper_state_values = None
      self.gripper_closed_image = 0
      self.gripper_open_image = 0
      self.fully_open_cnt = None
      self.fully_closed_cnt = None
      self.alset_state = alset_state
      self.add_edges = False   # didn't seem to add much, if anything
      self.cvu = CVAnalysisTools(self.alset_state)
      self.cfg = Config()
      self.dsu = None

  def detect_robot_body(self, image_path):
      # store non-gripper image fragments of the robot body
      # todo
      pass

  def detect_grabbed_object(self, image_path):
      # register non-gripper image fragments of the grabbed object
      # with other classes that collect images of interest.
      # todo
      pass
      # compare with closed gripper image

  def mask_gripped_object(self, img):
      pass

  def digital_servo(self, img, target, direction):
      if direction not in ["ABOVE", "ANY", "HORIZONTAL"]:
        pass
      if known_cluster(target):
        pass

  def detect_digital_servo(self):
      pass
      # arm movement centered around or above an object
      # aimed target -> learn the target, associate with the cluster

  # bound the images based upon movement
  # then get gripper contours from the area of movement
  def gripper_edges(self, image_path):
      try:
        image = cv2.imread(image_path)
      except:
        image = image_path
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      gray = image
      # Threshold the image
      ret, thresh = cv2.threshold(gray, 20, 255, 0)
      edges = cv2.Canny(gray, 50, 200, None, 3)

  # then get gripper contours from the area of movement
  def gripper_contours(self, image_path):
      try:
        # image = cv2.imread(image_path)
        image,mean_diff, rl = self.adjust_light(image_path)
      except:
        image = image_path
      try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      except:
        gray = image
      # Threshold the image
      ret, thresh = cv2.threshold(gray, 20, 255, 0)
      # Find contours 
      contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
      # print("contours:",contours)
      # print("hierarchy:",hierarchy)
      # Iterate through each contour and compute the approx contour
      for c in contours:
          # Calculate accuracy as a percent of the contour perimeter
          accuracy = 0.1 * cv2.arcLength(c, True)
          approx = cv2.approxPolyDP(c, accuracy, True)
          cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)
      # cv2.imshow("gripper_contours", image)
      # cv2.waitKey(0)

  # not really used
  def get_gripper_pos_info(self):
      if self.gripper_state == "UNKNOWN":
        return None
      if self.gripper_state_values is None:
        self.gripper_state_values = []
      while len(self.gripper_state_values) <= self.gripper_close_cnt:
        self.gripper_state_values.append([])
      while len(self.gripper_state_values[self.gripper_close_cnt]) <= self.gripper_open_cnt:
        self.gripper_state_values[self.gripper_close_cnt].append({})
      return self.gripper_state_values[self.gripper_close_cnt][self.gripper_open_cnt]

  # not really used
  def set_gripper_pos_info(self, key, value):
      if self.gripper_state == "UNKNOWN":
        print("set_gripper_pos_info called in UNKNOWN state")
        return 
      g_pos_info = self.get_gripper_pos_info()
      # print("g_pos_info:", g_pos_info)
      print("g_pos_info: close/open cnts", self.gripper_close_cnt,self.gripper_open_cnt)
      self.gripper_state_values[self.gripper_close_cnt][self.gripper_open_cnt][key] = value

  def get_object_bounding_box(self, lg_bb, rg_bb, img):
      obj_bounding_box = None
      return obj_bounding_box 

  def get_obj_name(self, curr_func_name):
      if curr_func_name.startswith("SEARCH_FOR_"):
        obj_nm = curr_func_name[len("SEARCH_FOR_"):]
      elif curr_func_name.startswith("GOTO_") and "_IN_" not in curr_func_name:
        obj_nm = curr_func_name[len("GOTO_"):]
      elif curr_func_name.startswith("PICK_UP_"):
        obj_nm = curr_func_name[len("PICK_UP_"):]
      elif curr_func_name.startswith("GOTO_") and "_IN_" in curr_func_name:
        idx = curr_func_name.find("_IN_")
        obj_nm = curr_func_name[idx + len("_IN_"):]
      elif curr_func_name.startswith("DROP_") and "_IN_" in curr_func_name:
        idx = curr_func_name.find("_IN_")
        obj_nm = curr_func_name[len("DROP_"):idx]
      return obj_nm

  def get_container_name(self, curr_func_name):
      if curr_func_name.startswith("SEARCH_FOR_") and "_WITH_" in curr_func_name:
        idx = curr_func_name.find("_WITH_")
        container_nm = curr_func_name[len("SEARCH_FOR_"):idx]
      elif curr_func_name.startswith("GOTO_") and "_WITH_" in curr_func_name:
        idx = curr_func_name.find("_WITH_")
        container_nm = curr_func_name[len("GOTO_"):idx]
      elif curr_func_name.startswith("DROP_") and "_IN_" in curr_func_name:
        idx = curr_func_name.find("_IN_")
        container_nm = curr_func_name[idx:]
      return container_nm

  def get_safe_ground(self, lg_bb, rg_bb, obj_bb, image_path):
      # only call if action is forward. Return image between gripper bbs.
      # typically arm should be parked.
      # left_bb = [[[0, miny]],[[0, y-1]], [[lmaxx, y-1]], [[lmaxx, miny]]]
      # right_bb = [[[minx, miny]],[[x-1, miny]], [[x-1, y-1]],[[minx, y-1]]]
#      minK = 3
#      maxK = 6
#      for K in range(minK, maxK):
#        pm_floor_clusters = self.cvu.color_quantification(image, K)
      image, mean_dif, rl = self.cvu.adjust_light(image_path)
      sg_minx = lg_bb[2][0][0]+1
      sg_maxx = rg_bb[0][0][0]-1 
      sg_miny = max(lg_bb[0][0][1], rg_bb[0][0][1])
      sg_maxy = lg_bb[1][0][1]
      # if sg_maxx - sg_minx < 10 or sg_maxy - sg_miny < 10:
      if sg_maxx - sg_minx < 50 or sg_maxy - sg_miny < 50:
        print("No Safe Ground:", sg_minx, sg_maxx, sg_miny, sg_maxy)
        return [None, None, None]
    
      safe_ground_bb = [[[sg_minx, sg_miny]],[[sg_minx, sg_maxy]], [[sg_maxx, sg_maxy]], [[sg_maxx, sg_miny]]]
 
      # safe_ground_img = np.zeros((sg_maxx-sg_minx, sg_maxy-sg_miny, 3), dtype="uint8")
      # safe_ground_img[0:sg_maxx-sg_minx, 0:sg_maxy-sg_miny] = image[sg_minx:sg_maxx, sg_miny:sg_maxy]
      safe_ground_img = np.zeros((sg_maxy-sg_miny, sg_maxx-sg_minx, 3), dtype="uint8")
      safe_ground_img[0:sg_maxy-sg_miny, 0:sg_maxx-sg_minx] = image[sg_miny:sg_maxy, sg_minx:sg_maxx]
      gray_sg_img = cv2.cvtColor(safe_ground_img, cv2.COLOR_BGR2GRAY)

      # print("Safe Ground BB:", safe_ground_bb)
      # cv2.imshow("Safe Ground", safe_ground_img)
      # cv2.waitKey(0)
      radius = min(sg_maxx-sg_minx, sg_maxy-sg_miny)
      num_radius = max(int((radius-1) / 6), 2)
      # safe_ground_lbp = []
      # for rad in range(1,(radius-1), num_radius):
        # safe_ground_lbp.append(LocalBinaryPattern(gray_sg_img, rad))
      # ARD: TODO: move lbp results to cluster analysis
      #            make a lbp catalog of safe ground
      self.alset_state.record_drivable_ground(safe_ground_bb, safe_ground_img)
      return [safe_ground_bb, safe_ground_img]

  def analyze(self, action, prev_img_path, curr_img_path, done=False, curr_func_name=None):
      frame_num = self.alset_state.get_frame_num()
      left_gripper_bounding_box, right_gripper_bounding_box, safe_ground_info = [],[],[None]
      if self.gripper_open_cnt == 0 and self.gripper_close_cnt == 0:
        init = True
      else:
        init = False
      opt_flw = self.cvu.optflow(prev_img_path, curr_img_path, add_edges=self.add_edges, thresh = 0.05) 
      ##########################
      # GRIPPER OPEN           #
      ##########################
      # if frame_num == self.prev_frame_num + 1 and action == "GRIPPER_OPEN" and action == self.prev_action:
      if frame_num == self.prev_frame_num + 1 and action == "GRIPPER_OPEN":
        print("1 prev_frame, frame, prev_action, action:", self.prev_frame_num, frame_num, self.prev_action, action)
        self.gripper_img = self.cvu.moved_pixels(prev_img_path, curr_img_path, init)
        if not opt_flw and (self.gripper_state == "UNKNOWN" or
                        self.gripper_state == "UNKNOWN_CLOSE"):
          self.gripper_state = "UNKNOWN_OPEN"
          self.fully_open_cnt = 0
          # 0,0 counts shouldn't happen
          # gripper open_cnt should go down when gripper_close_cnt goes up & vica-versa
          self.gripper_open_cnt = 0
          self.gripper_close_cnt = 0
        elif self.gripper_state == "UNKNOWN":
          pass
        elif not opt_flw and self.gripper_state == "UNKNOWN_OPEN":
          self.gripper_state = "FULLY_OPEN"
          self.fully_open_cnt = self.gripper_open_cnt
          self.gripper_close_cnt = 0
        elif self.gripper_state == "UNKNOWN_CLOSE":
          self.gripper_open_cnt += 1
          self.gripper_close_cnt = max(self.gripper_close_cnt-1, 0)
        elif self.gripper_state == "UNKNOWN_OPEN":
          self.gripper_state = "UNKNOWN"
          self.gripper_open_cnt += 1
          self.gripper_close_cnt = max(self.gripper_close_cnt-1, 0)
        else:
          # g_img = self.cvu.moved_pixels(prev_img_path, curr_img_path, init)
          left_gripper_bounding_box, right_gripper_bounding_box, g_img = self.cvu.moved_pixels(prev_img_path, curr_img_path, init)
          # cv2.imshow("Gripper open", g_img)
          # cv2.waitKey(0)
          if g_img is None:
            print("g_img is None")
          # ARD: opened up. No longer fully closed...
          if done:
            self.gripper_state = "FULLY_OPEN"
            self.fully_open_cnt = self.gripper_open_cnt
            self.gripper_close_cnt = 0
          elif self.gripper_state == "FULLY_OPEN" and self.fully_open_cnt <= self.gripper_open_cnt and self.gripper_close_cnt == 0:
            pass
          elif not opt_flw and self.gripper_state != "FULLY_CLOSED" and action == self.prev_action:
            # want opt_flow to fail 2x in a row
            self.gripper_state = "FULLY_OPEN"
            self.fully_open_cnt = self.gripper_open_cnt
            self.set_gripper_pos_info("FULLY_OPEN_IMG", g_img)
          else:
            # do OPT check if unchanged, then done
            self.gripper_open_cnt += 1
            self.gripper_close_cnt = max(self.gripper_close_cnt-1, 0)
            self.gripper_state = "PARTIALLY_OPEN"
        print("1 gripper state: ", self.prev_frame_num, self.gripper_open_cnt, self.gripper_close_cnt, self.gripper_state)
        if frame_num >= self.stop_at_frame:
            # cv2.imshow("Gripper FG", g_img)
            # cv2.waitKey(0)
            pass
      ##########################
      # GRIPPER CLOSE          #
      ##########################
      # elif frame_num == self.prev_frame_num + 1 and action == "GRIPPER_CLOSE" and action == self.prev_action:
      elif frame_num == self.prev_frame_num + 1 and action == "GRIPPER_CLOSE":
        print("2 prev_frame, frame, prev_action, action:", self.prev_frame_num, frame_num, self.prev_action, action)
        if not opt_flw and (self.gripper_state == "UNKNOWN" or
                            self.gripper_state == "UNKNOWN_OPEN"):
          self.gripper_state = "UNKNOWN_CLOSED"
          self.fully_open_cnt = 0
          # 0,0 counts shouldn't happen
          # gripper open_cnt should go down when gripper_close_cnt goes up & vica-versa
          self.gripper_open_cnt = 0
          self.gripper_close_cnt = 0
        elif self.gripper_state == "UNKNOWN":
          pass
        elif not opt_flw and self.gripper_state == "UNKNOWN_CLOSE":
          self.gripper_state = "FULLY_CLOSED"
          self.fully_close_cnt = self.gripper_close_cnt
          self.gripper_open_cnt = 0
        elif self.gripper_state == "UNKNOWN_OPEN":
          self.gripper_open_cnt = max(self.gripper_open_cnt-1, 0)
          self.gripper_close_cnt += 1
        elif self.gripper_state == "UNKNOWN_CLOSE":
          self.gripper_state = "UNKNOWN"
          self.gripper_open_cnt = max(self.gripper_open_cnt-1, 0)
          self.gripper_close_cnt += 1
        else:
          # g_img = self.cvu.moved_pixels_over_time(prev_img_path, curr_img_path, init)
          left_gripper_bounding_box, right_gripper_bounding_box, g_img = self.cvu.moved_pixels(prev_img_path, curr_img_path, init)
          # cv2.imshow("Gripper close", g_img)
          # cv2.waitKey(0)
          if done:
            # Problem: if picking up cube, done may not indicate a fully closed state
            self.gripper_state = "FULLY_CLOSED"
            self.set_gripper_pos_info("FULLY_CLOSED_IMG", g_img)
            print("FULLY_CLOSED: done")
            # ARD: should take mean
            self.fully_close_cnt = self.gripper_close_cnt
            self.gripper_open_cnt = 0
          elif self.gripper_state == "FULLY_CLOSED" and self.fully_close_cnt <= self.gripper_close_cnt and self.gripper_open_cnt == 0:
            pass
          elif not opt_flw and self.gripper_state != "FULLY_OPEN" and action == self.prev_action:
            # ARD: check if grabbing a cube
            # want opt_flow to fail 2x in a row
            self.set_gripper_pos_info("FULLY_CLOSED_IMG", g_img)
            self.gripper_state = "FULLY_CLOSED"
            self.fully_close_cnt = self.gripper_close_cnt
            self.gripper_open_cnt = 0
          else:
            self.gripper_open_cnt = max(self.gripper_open_cnt-1, 0)
            self.gripper_close_cnt += 1
            self.gripper_state = "PARTIALLY_CLOSED"
        print("2 gripper state: ", self.prev_frame_num, self.gripper_open_cnt, self.gripper_close_cnt, self.gripper_state, opt_flw, done)
        #
        # TODO: store gripper_state with image 
        #  ../GRIPPER/CxOy_#[...].img
        #  ../GRIPPER/CxOyCLz_#[...].img    # cluster
        #
        # else:
      ##########################
      # NON-GRIPPER ACTION     #
      ##########################
      elif frame_num == self.prev_frame_num + 1 and not self.gripper_state.startswith("UNKNOWN"):
          # Non-gripper movement. Robot parts will be part of "unmoved pixels".
          # if frame_num == self.prev_frame_num + 1 and action in ["LEFT","RIGHT","FORWARD","REVERSE"] and self.gripper_state == "FULLY_OPEN":
            g_pos_info = self.get_gripper_pos_info()
            try:
              g_edges = g_pos_info["EDGES"]
              g_mean_edges = g_pos_info["MEAN_EDGES"]
              g_mean_edges_img = g_pos_info["MEAN_EDGES_IMG"]
              g_edges_cnt = g_pos_info["EDGES_COUNT"]
            except Exception as e:
              print("exception g_pos_info:", e)
              g_edges = None
              g_mean_edges = None
              g_mean_edges_img = None
              g_edges_cnt = 0
              gripper_bounding_box = None 
            g_edges,lg_bb,rg_bb = self.cvu.unmoved_pixels(prev_img_path, curr_img_path, True, g_edges)
            g_edges_cnt += 1
            if g_edges_cnt > self.cfg.MIN_UNMOVED_PIX_COUNT:
              if g_mean_edges is None:
                g_mean_edges = np.zeros(g_edges.shape, dtype="float32")
              g_mean_edges = (g_edges + (g_edges_cnt - self.cfg.MIN_UNMOVED_PIX_COUNT-1) * g_mean_edges) / (g_edges_cnt - self.cfg.MIN_UNMOVED_PIX_COUNT)
              max_255 = np.zeros(g_edges.shape, dtype="float32")
              # max_255 = (255,255,255)
              max_255 = 255
              g_mean_edges = np.minimum(max_255, g_mean_edges)
              g_mean_edges = np.round(g_mean_edges)
              g_mean_edges_img = np.uint8(g_mean_edges.copy())
              # curr_img = cv2.imread(curr_img_path)
              curr_img, mean_dif, rl = self.cvu.adjust_light(curr_img_path)
              # contours, image = self.cvu.unmoved_pixel_contours(g_mean_edges_img, curr_img)
              left_gripper_bounding_box, right_gripper_bounding_box, image = self.cvu.get_gripper_bounding_box(g_mean_edges_img, curr_img)
              # cv2.imshow("mean_gripper_edges", g_mean_edges_img)
              # cv2.imshow("mean_contour", image)
            if curr_func_name in ["SEARCH_FOR_CUBE", "GOTO_CUBE", "PICK_UP_CUBE", "GOTO_BOX_WITH_CUBE", "DROP_CUBE_IN_BOX"]:
                obj_nm = self.get_obj_name(curr_func_name)
                # obj_bounding_box = self.get_object_bounding_box(left_gripper_bounding_box, right_gripper_bounding_box, curr_img)
                if obj_nm == "CUBE":
                  print("find cube")
                  curr_img, mean_dif, rl = self.cvu.adjust_light(curr_img_path)
                  # find_cube(curr_img)

            if action == "FORWARD":
                # prev_img is pre-FORWARD ; curr_img is post-FORWARD 
                safe_ground_info = self.get_safe_ground(left_gripper_bounding_box, right_gripper_bounding_box, None, prev_img_path)
                if safe_ground_info[0] is not None:
                  self.alset_state.record_bounding_box(["DRIVABLE_GROUND"], safe_ground_info[0]) 
            self.set_gripper_pos_info("MEAN_EDGES", g_mean_edges)
            self.set_gripper_pos_info("MEAN_EDGES_IMG", g_mean_edges_img)
            self.set_gripper_pos_info("EDGES", g_edges)
            self.set_gripper_pos_info("EDGES_COUNT", g_edges_cnt)
            # ret, thresh = cv2.threshold(g_mean_edges_img, 125, 255, 0)


            print("g_edge_cnt:", g_edges_cnt, self.cfg.MIN_UNMOVED_PIX_COUNT)
            print("5 gripper state: ", self.prev_frame_num, self.gripper_open_cnt, self.gripper_close_cnt, self.gripper_state)
            # cv2.imshow("gripper_edges", g_edges)
            # cv2.waitKey(0)
#      else:
#        print("4 gripper state: ", self.prev_frame_num, self.gripper_open_cnt, self.gripper_close_cnt, self.gripper_state)
#        print("4 prev_frame, frame, prev_action, action:", self.prev_frame_num, frame_num, self.prev_action, action)
        # self.gripper_open_cnt = 0
        # self.gripper_close_cnt = 0

      self.prev_frame_num = frame_num
      self.prev_action = action
      # gripper contour and edges do worse than pixel movements (averaged/overlapped).
      # self.gripper_contours(curr_img_path)
      # self.gripper_edges(curr_img_path)
      self.alset_state.record_gripper_state(self.gripper_state, self.gripper_open_cnt, self.gripper_close_cnt)

      # list of potential gripper labels:
      # "RIGHT_GRIPPER_OPEN", "RIGHT_GRIPPER_CLOSED", "LEFT_GRIPPER_OPEN",
      # "LEFT_GRIPPER_CLOSED", "RIGHT_GRIPPER_C_X_Y", "LEFT_GRIPPER",
      # "RIGHT_GRIPPER", "PARKED_GRIPPER", "GRIPPER_WITH_CUBE", "DROP_CUBE"
      if self.gripper_state in ["FULLY_OPEN"]:
        left_labels = ["LEFT_GRIPPER","LEFT_GRIPPER_OPEN"]
        right_labels = ["RIGHT_GRIPPER","RIGHT_GRIPPER_OPEN"]
      elif self.gripper_state in ["FULLY_CLOSED"]:
        left_labels = ["LEFT_GRIPPER","LEFT_GRIPPER_CLOSED"]
        right_labels = ["RIGHT_GRIPPER","RIGHT_GRIPPER_CLOSED"]
      elif self.gripper_state in ["PARTIALLY_OPEN"]:
        left_labels = ["LEFT_GRIPPER",f'LEFT_GRIPPER_O_{self.gripper_open_cnt}_{self.gripper_close_cnt}']
        right_labels = ["RIGHT_GRIPPER",f'RIGHT_GRIPPER_O_{self.gripper_open_cnt}_{self.gripper_close_cnt}']
      elif self.gripper_state in ["PARTIALLY_CLOSED"]:
        left_labels = ["LEFT_GRIPPER",f'LEFT_GRIPPER_C_{self.gripper_open_cnt}_{self.gripper_close_cnt}']
        right_labels = ["RIGHT_GRIPPER",f'RIGHT_GRIPPER_C_{self.gripper_open_cnt}_{self.gripper_close_cnt}']
      elif self.gripper_state.startswith("UNKNOWN"):
        left_labels = ["LEFT_GRIPPER"]
        right_labels = ["RIGHT_GRIPPER"]
        
      # Computed later using post-analysis;
      #   "PARKED_GRIPPER", "GRIPPER_WITH_CUBE", "DROP_CUBE"
      #   self.robot_state["KNOWN_STATE"] == "PARK_ARM_RETRACTED"
      if len(left_gripper_bounding_box) == 4:
        self.alset_state.record_bounding_box(left_labels, left_gripper_bounding_box) 
      if len(right_gripper_bounding_box) == 4:
        self.alset_state.record_bounding_box(right_labels, right_gripper_bounding_box) 
      if len(right_gripper_bounding_box) != 4 or len(right_gripper_bounding_box) != 4:
        print("left/right grip bb:", left_gripper_bounding_box, right_gripper_bounding_box)
      # note: safe_ground_info recorded by self.get_drivable_ground()
      return [left_gripper_bounding_box, right_gripper_bounding_box, safe_ground_info[0]]

  def check_cube_in_gripper(self, action, prev_img_path, curr_img_path, done):
      pass

  def label_gripper(self, action, prev_img_path, curr_img_path, done=False):

      ret_val = [[label, bounding_box],[label, bounding_box]]
      return ret_val

  def load_state(self):
      self.gripper_state = self.alset_state.gripper_state["GRIPPER_POSITION"]
      self.gripper_open_cnt = self.alset_state.gripper_state["GRIPPER_OPEN_COUNT"]
      self.gripper_close_cnt = self.alset_state.gripper_state["GRIPPER_CLOSED_COUNT"]

