from CPT import *
from config import *
from dataset_utils import *
from PIL import Image
from utilborders import *
from cv_analysis_tools import *
from analyze_keypoints import *
from analyze_texture import *
from cube import *
import pickle


class AlsetState():
  def __init__(self):
      # CPT-based move prediction
      # self.model  = CPT(5)
      self.data   = None
      self.target = None
      self.cfg    = Config()
      # err for unknown reason, used "magic numbers" instead :(
      self.IMG_HW  = self.cfg.IMG_H

      # Global VARIABLES
      self.location  = None
      self.state     = []
      self.app_name  = None
      self.dsu       = None
      self.func_state = None
      self.run_num   = -1
      self.func_num  = -1
      self.frame_num = -1
      self.restart_frame_num = self.last_frame_state("FRAME_NUM")
      print("RESTART FRAME:", self.restart_frame_num)
      if self.restart_frame_num is None:
        self.restart_frame_num = -1
      self.arm_state = None
      self.gripper_state = None
      self.robot_state = {}

      # Global state contains cross-run analysis
      # Details are stored on a per-run basis
      self.global_state = {}
      self.global_state["LEFT"]    = {}
      self.global_state["RIGHT"]   = {}
      self.global_state["FORWARD"] = {}
      self.global_state["REVERSE"] = {}
      self.global_state["AVG_BOUNDING_BOX"] = {}
      self.global_state["EVENT_DETAILS"] = {}

      self.cvu = CVAnalysisTools(self)
      self.lbp = LocalBinaryPattern()

      # TODO: Derived Physical Robot Attributes
      #       Could be put into a physics engine
      # self.battery_power         = None
      # self.gripper_height        = None
      # self.gripper_width         = None
      # self.gripper_offset_height = None
      # self.lower_arm_length      = None
      # self.upper_arm_length      = None

  #############
  # Set App, Frame
  #############
#  def set_frame(self, id=None, type=None):
#      if type not in ["PREVIOUS", "NEXT", "CURRENT", "FIRST", "LAST", "GOTO"]:
#        pass
#
#  def get_frame(id, type="CURRENT"):
#      pass
#
#  def set_function(frame_num, func_nm):
#      pass
#
#  def get_app_run(app_name, date, run_id):
#      return
#      return self.app_run_id

  def init_app_run(self, app_run_num=None, app_name=None, app_mode=None):
      if app_run_num is not None:
        run_num = app_run_num
      else:
        run_num = self.run_num
      if run_num == len(self.state):
        self.state.append({})
      elif run_num > len(self.state):
        print("unexpected run_num:", app_run_num, self.run_num, len(self.state))
        i = 1/0
      print("run_nums:", app_run_num, self.run_num, len(self.state))
      self.state[run_num]["APP_NAME"]        = app_name
      self.state[run_num]["APP_RUN_ID"]      = None
      self.state[run_num]["APP_RUN_NUM"]     = app_run_num
      self.state[run_num]["APP_MODE"]        = app_mode
      self.state[run_num]["FUNC_STATE"]      = {}
      self.state[run_num]["FRAME_STATE"]     = []
      # self.state[self.run_num]["GRIPPER_STATE"]   = {}
      self.state[run_num]["BOUNDING_BOXES"]  = {}
      self.state[run_num]["ACTIONS"]         = {}
      self.state[run_num]["ACTIVE_MAP"]      = None
      self.state[run_num]["FINAL_MAP"]       = None
      self.state[run_num]["MAP_ORIGIN"]      = None
      self.state[run_num]["MEAN_LIGHTING"]   = None
      self.state[run_num]["LOCATION_HISTORY"] = []
      self.state[run_num]["PICKUP"]          = {}
      self.state[run_num]["DROP"]            = {}
      self.state[run_num]["LOCAL_BINARY_PATTERNS"] = {}
      self.state[run_num]["BLACKBOARD"]      = {}
      self.robot_origin = None

      # initialize shortcut pointers
      self.run_state     = self.state[run_num]
      self.func_state    = self.state[run_num]["FUNC_STATE"]
      self.frame_state   = self.state[run_num]["FRAME_STATE"]
      self.bb            = self.state[run_num]["BOUNDING_BOXES"]
      self.actions       = self.state[run_num]["ACTIONS"]
      self.lbp_state     = self.state[run_num]["LOCAL_BINARY_PATTERNS"]
      self.blackboard    = self.state[run_num]["BLACKBOARD"]

      self.app_name = app_name
      self.app_mode = app_mode
      self.dsu = DatasetUtils(app_name, app_mode)
      self.func_state["FUNC_NAME"]        = []
      self.func_state["FUNC_FRAME_START"] = []
      self.func_state["FUNC_FRAME_END"]   = []

      # each label is a list to be stored in its own classification directory for training
      # one bb may have multiple labels. Store the label images once and references to the
      # image the other times. 
      #
      # frame img already stored with other index, store path
      # compute / store bb image at end of state save, no in-memory storage
      # bb_path self.bb[
      #
      # at run level, store key:[frame#_list] only
      # at frame level, store key:bb_idx
      # for get_next_bb("key")
      #
      # self.bb should associated with a frame. Multiple BB. 
      self.bb["RIGHT_GRIPPER_OPEN"]   = []
      self.bb["RIGHT_GRIPPER_CLOSED"] = []
      self.bb["LEFT_GRIPPER_OPEN"]    = []
      self.bb["LEFT_GRIPPER_CLOSED"]  = []
      self.bb["LEFT_GRIPPER"]         = []
      self.bb["RIGHT_GRIPPER"]        = []
      self.bb["PARKED_GRIPPER"]       = []
      self.bb["GRIPPER_WITH_CUBE"]    = []
      self.bb["CUBE"]                 = []
      self.bb["CUBE_IN_ROBOT_LIGHT"]  = []
      self.bb["BOX"]                  = []
      self.bb["DRIVABLE_GROUND"]      = []
      self.bb["DO_PICKUP"]            = []
      self.bb["DO_DROPOFF"]           = []
      self.bb["DROP_CUBE_IN_BOX"]     = []
      # self.bb["RIGHT_GRIPPER_C_X_Y"]  = []  # fill in each O/C, X,Y
      # self.bb["START_CUBE_GRAB"]      = []  # Done later

      # For lbp pattern matching
      # at run-level, global lbp pattern matching is supported
      # at frame level, store frame# for lbp
      # get_lbp_match(lbp)
      self.lbp_state["DRIVABLE_GROUND"]     = []
      self.lbp_state["BOX"]                 = []
      self.lbp_state["CUBE"]                = []
      self.lbp_state["CUBE_IN_ROBOT_LIGHT"] = []

      # need the following history in a list for simple statistical analysis
      self.actions["LEFT"]    = []
      self.actions["RIGHT"]   = []
      self.actions["FORWARD"] = []
      self.actions["REVERSE"] = []

      # pointer to current value.  History by traversing moves. 
      self.location = None
      self.run_id   = None

  def record_frame_info(self, app, mode, nn_name, action, img_nm, full_img_path):
      if self.frame_num != len(self.frame_state):
        print(self.frame_num, len(self.frame_state))
        i = 1/0
      # Prev Func State
      # Current Func State
      # print(self.func_state[-1]["APP_NAME"], self.func_state[-1]["APP_MODE"])
      # print("record_frame_info: ",len(self.func_state),self.func_state[-1]["APP_NAME"], self.func_state[-1]["APP_MODE"])
      if (len(self.func_state["FUNC_NAME"]) == 0 or
          self.func_state["FUNC_NAME"][-1] != nn_name):
        self.func_state["FUNC_NAME"].append(nn_name)
        self.func_state["FUNC_FRAME_START"].append(self.frame_num)
        self.func_state["FUNC_FRAME_END"].append(self.frame_num)
        print("len(self.func_state):", len(self.func_state["FUNC_NAME"]))
      func_num = len(self.func_state["FUNC_NAME"]) - 1
      self.func_state["FUNC_FRAME_END"][func_num] = self.frame_num

      # Action-related state
      self.frame_state.append({})
      print(self.frame_num, "record_frame_info: ",len(self.func_state), action)
      self.frame_state[-1]["ACTION"]         = action
      self.frame_state[-1]["FRAME_PATH"]     = full_img_path
      self.frame_state[-1]["FRAME_NUM"]      = self.frame_num
      self.frame_state[-1]["FUNCTION_NAME"]  = nn_name
      print("bb", len(self.bb))
      self.frame_state[-1]["MEAN_DIF_LIGHTING"] = None
      self.frame_state[-1]["ROBOT_LIGHT"]       = None
      self.frame_state[-1]["MOVE_ROTATION"]     = None
      self.frame_state[-1]["MOVE_STRAIGHT"]     = None
      # Robot-related state
      self.frame_state[-1]["LOCATION"]       = {}
      self.frame_state[-1]["ARM_STATE"]      = {}   # "O"/ "C", open/close count
      self.frame_state[-1]["GRIPPER_STATE"]  = {}
      self.frame_state[-1]["PREDICTED_MOVE"] = None
      self.frame_state[-1]["BOUNDING_BOX_LABELS"] = {}
      self.frame_state[-1]["BOUNDING_BOX"]        = []
      self.frame_state[-1]["BOUNDING_BOX_PATH"]   = []  # computed and added later
      self.frame_state[-1]["BOUNDING_BOX_IMAGE_SYMLINK"] = []
      self.frame_state[-1]["BOUNDING_BOX_IMAGE_SYMLINK"].append(None)
      self.location      = self.frame_state[-1]["LOCATION"]
      self.gripper_state = self.frame_state[-1]["GRIPPER_STATE"]
      self.arm_state     = self.frame_state[-1]["ARM_STATE"]

  # run_id based upon IDX id
  def record_run_id(self, func_idx, run_id):
      self.state[self.run_num]["FUNC_INDEX"] = func_idx
      self.state[self.run_num]["APP_RUN_ID"] = run_id
      self.run_id = run_id

  def get_bb(self, label, frame_num=None):
      if frame_num is None:
        frame_num = -1
      try:
        bb_idx = self.frame_state[frame_num]["BOUNDING_BOX_LABELS"][label]
        bb     = self.frame_state[frame_num]["BOUNDING_BOX"][bb_idx]
      except:
        print("labels:", self.frame_state[frame_num]["BOUNDING_BOX_LABELS"])
        return None
      return bb

  def get_bounding_box_image(self, orig_image_path, bb):
      orig_img,orig_mean_diff,orig_rl = self.cvu.adjust_light(orig_image_path)
      return get_bb_img(orig_img, bb)

  def record_bounding_box(self, label_list, bb, frame_num=None, use_img=None):
      if frame_num is None:
        frame_num = self.frame_num
      self.frame_state[frame_num]["BOUNDING_BOX"].append(copy.deepcopy(bb))
      bb_idx = len(self.frame_state[frame_num]["BOUNDING_BOX"]) - 1
      for i, label in enumerate(label_list):
        self.frame_state[frame_num]["BOUNDING_BOX_LABELS"][label] = bb_idx
        # create/get bb directory path 
        bb_path = self.dsu.bounding_box_path(label)
        bb_file = bb_path + self.dsu.bounding_box_file(self.run_id, frame_num, bb_idx)
        if i == 0:
          orig_img_path = self.frame_state[frame_num]["FRAME_PATH"]
          # print("orig_img_path:", orig_img_path)
          # print("frame_state[frame_num]:", self.frame_state[frame_num])
          if use_img is None:
            bb_img = self.get_bounding_box_image(orig_img_path, bb)
          else:
            bb_img = get_bb_img(use_img, bb)
          # if label == "CUBE":
          #   cv2.imshow("record cube: revised obj img", bb_img)
          #   cv2.waitKey(0)
          if bb_img is not None:
            retkey, encoded_image = cv2.imencode(".png", bb_img)
            with open(bb_file, 'wb') as f:
              f.write(encoded_image)
            self.frame_state[frame_num]["BOUNDING_BOX_PATH"].append(bb_file)
            sl_path = self.dsu.bounding_box_symlink_path(label)
            sl_file = sl_path+self.dsu.bounding_box_file(self.run_id,frame_num,bb_idx)
          else:
            print("WARNING: illegal bb", label_list, bb, frame_num)
            return
        else:
          print("*********************")
          print("mksymlink", bb_file, sl_file)
          self.dsu.mksymlinks([bb_file], sl_file)
          self.frame_state[frame_num]["BOUNDING_BOX_IMAGE_SYMLINK"] = bb_file

  def record_gripper_state(self, state, open_cnt, closed_cnt):
      self.gripper_state["GRIPPER_POSITION"]     = state
      self.gripper_state["GRIPPER_OPEN_COUNT"]   = open_cnt
      self.gripper_state["GRIPPER_CLOSED_COUNT"] = closed_cnt

  def get_arm_cnt(self):
      try:
        armcnt = self.arm_state["ARM_POSITION_COUNT"]
      except:
        return None

  def record_arm_state(self, state, arm_cnt, blackboard, frame_num=None):
      if frame_num is None:
        frame_num = self.frame_num
      # "ARM_POSITION" => "UNKNOWN", "ARM_RETRACTED", "ARM_RETRACTED_DELTA"
      arm_state     = self.frame_state[frame_num]["ARM_STATE"]
      arm_state["ARM_POSITION"] = state
      # "ARM_POSITION_COUNT" => ["<armaction>"] = cnt
      arm_state["ARM_POSITION_COUNT"] = arm_cnt.copy()
      # blackboard "ARM" => upper_arm_done, lower_arm_done]
      self.blackboard["ARM"] = blackboard.copy()
      print("record_arm_state:", arm_state["ARM_POSITION"], arm_state["ARM_POSITION_COUNT"])

  def copy_prev_arm_state(self, frame_num=None):
      if frame_num is None:
        frame_num = self.frame_num
      print("frame_num, len frame_state:", frame_num, len(self.frame_state))
      print("frame_state:", self.frame_state[frame_num])
      arm_state      = self.frame_state[frame_num]["ARM_STATE"]
      if len(arm_state) != 0:
        print("WARNING: copy_prev_arm_state called when state already exists")
        print("frame_num, arm_pos, armcnt:", frame_num, arm_state["ARM_POSITION"], arm_state["ARM_POSITION_COUNT"])
        return
      if frame_num is None or frame_num <= 0:
        print("copy: UNKNOWN arm state due to frame_num", frame_num)
        arm_state["ARM_POSITION"] = "UNKNOWN"
        arm_state["ARM_POSITION_COUNT"] = None
        self.blackboard["ARM"] = None
        return
      prev_arm_state = self.frame_state[frame_num-1]["ARM_STATE"]
      arm_state["ARM_POSITION"] = prev_arm_state["ARM_POSITION"]
      arm_state["ARM_POSITION_COUNT"] = prev_arm_state["ARM_POSITION_COUNT"]
      # blackboard is for active transient state at the Run level, so no need 
      # to copy.
      print("copy_prev_arm_state:", arm_state["ARM_POSITION"], arm_state["ARM_POSITION_COUNT"], self.blackboard["ARM"])

  # This is called by alset_analysis during restart.
  # Frame_num is maintained by alset_state but restart needs to replay the
  # parsing of the history files to find the right place.
  def get_frame_num(self):
      return self.frame_num

  def increment_frame(self):
      self.frame_num += 1
      print("increment_frame", self.frame_num)

  def reset_frame_num(self):
      self.frame_num = -1

  def restarting(self):
      if self.frame_num >= self.restart_frame_num:
        # last frame is run again in case it was only partially completed
        print("Not restarting: ", self.frame_num, self.restart_frame_num)
        return False
      print("Restarting: ", self.frame_num, self.restart_frame_num)
      return True

  def record_drivable_ground(self, bb, img):
      self.record_bounding_box(["DRIVABLE_GROUND"], bb)
      self.record_lbp("DRIVABLE_GROUND", img)

  def record_lbp(self, key, image, bb=None, radius=None, frame_num=None):
      if frame_num is None:
        frame_num = self.frame_num
      if radius is not None:
        lbp_radius = radius
      elif bb is not None:
        maxw, minw, maxh, minh = get_min_max_borders(bb)
        lbp_radius = min((maxw-minw), (maxh-minh)) * 3 / 8
      else:
        lbp_radius = None
      if len(image.shape) == 2:
        gray_img = image
      else:
        gray_img  = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
      locbinpat = self.lbp.register(key, gray_img, radius=lbp_radius, 
                            run_id=self.run_id, frame_num=frame_num)
      try:
        self.lbp_state[key].append([locbinpat, self.lbp.radius])
      except:
        self.lbp_state[key] = []
        self.lbp_state[key].append([locbinpat, self.lbp.radius])

  def record_lighting(self, mean_lighting, mean_dif, robot_light):
      if robot_light is not None:
        self.record_robot_light(robot_light.copy())
      if mean_dif is not None:
        self.frame_state[-1]["MEAN_DIF_LIGHTING"] = mean_dif
      # overwrite with most recent
      if mean_lighting is not None:
        self.state[self.run_num]["MEAN_LIGHTING"] = mean_lighting 

  def get_lighting(self, frame_num=None):
      if frame_num is None:
        frame_num = self.frame_num
      mean_dif = self.frame_state[frame_num]["MEAN_DIF_LIGHTING"]
      # overwrite with most recent
      mean_lighting = self.state[self.run_num]["MEAN_LIGHTING"]
      rl = self.get_robot_light(frame_num)
      mean_dif = self.frame_state[frame_num]["MEAN_DIF_LIGHTING"]
      # overwrite with most recent
      mean_lighting = self.state[self.run_num]["MEAN_LIGHTING"] 
      return mean_lighting, mean_dif, rl

  def record_map_origin(self, origin):
      self.state[self.run_num]["MAP_ORIGIN"] = origin.copy()  
      self.global_state["EVENT_DETAILS"] = {}

  def load_map(self):
      try:
        map_file = self.state[self.run_num]["ACTIVE_MAP"]
        img = cv2.imread(map_file)
        return img
      except:
        return None

  def record_map(self, map):
      # overwrite with latest map
      map_file = self.dsu.map_filename(self.app_name, "In_Progress")
      retkey, encoded_image = cv2.imencode(".jpg", map)
      self.state[self.run_num]["ACTIVE_MAP"] = map_file
      with open(map_file, 'wb') as f:
        f.write(encoded_image)

  def record_map_overlay(self, map):
      # overwrite with latest map
      map_file = self.dsu.map_filename(self.app_name, self.run_id) 
      retkey, encoded_image = cv2.imencode(".jpg", map)
      self.state[self.run_num]["FINAL_MAP"] = map_file
      with open(map_file, 'wb') as f:
        f.write(encoded_image)

  def record_movement(self, action, mv_pos,  mv_rot):
      if self.is_mappable_position(action):
        self.frame_state[-1]["MOVE_ROTATION"] = mv_rot
        self.frame_state[-1]["MOVE_STRAIGHT"] = mv_pos
      else:
        print("not in mappable position; arm state, action", self.frame_state[-1]["ARM_STATE"], action)
        exit()

  def post_run_bounding_box_analysis(self):
      try:
        g_s = self.global_state["AVG_BOUNDING_BOX"]
      except:
        self.global_state["AVG_BOUNDING_BOX"] = {}
        g_s = self.global_state["AVG_BOUNDING_BOX"]
      for fr_s in self.frame_state:
        for bbl,bb_idx in fr_s["BOUNDING_BOX_LABELS"].items():
          try:
            g_s_bbl = g_s[bbl]
          except:
            g_s[bbl] = {}
            g_s[bbl]["BOUNDING_BOX_COUNT"] = 0
            g_s[bbl]["BOUNDING_BOX"] = [[[0,0]],[[0,0]],[[0,0]],[[0,0]]]
          g_s_bb = g_s[bbl]["BOUNDING_BOX"]
          g_s_bb_cnt = g_s[bbl]["BOUNDING_BOX_COUNT"]

          bb = fr_s["BOUNDING_BOX"][bb_idx]
          for i in range(4):
            for xy in range(2):
              g_s_bb[i][0][xy] = (g_s_bb[i][0][xy] * g_s_bb_cnt + bb[i][0][xy]) / (g_s_bb_cnt+1)
          print("avgbb:", bbl, g_s_bb)
          g_s[bbl]["BOUNDING_BOX_COUNT"] += 1

  def get_gripper_pos_label(self,frame_num):
      gripper_state = self.frame_state[frame_num]["GRIPPER_STATE"]
      gripper_position = gripper_state["GRIPPER_POSITION"]
      gripper_open_cnt = gripper_state["GRIPPER_OPEN_COUNT"]
      gripper_close_cnt = gripper_state["GRIPPER_CLOSED_COUNT"]
      if gripper_position in ["FULLY_OPEN"]:
        left_label = "LEFT_GRIPPER_OPEN"
        right_label = "RIGHT_GRIPPER_OPEN"
      elif gripper_position in ["FULLY_CLOSED"]:
        left_label = "LEFT_GRIPPER_CLOSED"
        right_label = "RIGHT_GRIPPER_CLOSED"
      elif gripper_position in ["PARTIALLY_OPEN"]:
        left_label = f'LEFT_GRIPPER_O_{gripper_open_cnt}_{gripper_close_cnt}'
        right_label = f'RIGHT_GRIPPER_O_{gripper_open_cnt}_{gripper_close_cnt}'
      elif gripper_position in ["PARTIALLY_CLOSED"]:
        left_label = f'LEFT_GRIPPER_C_{gripper_open_cnt}_{gripper_close_cnt}'
        right_label = f'RIGHT_GRIPPER_C_{gripper_open_cnt}_{gripper_close_cnt}'
      elif gripper_position.startswith("UNKNOWN"):
        left_label = "LEFT_GRIPPER"
        right_label = "RIGHT_GRIPPER"
      return left_label, right_label

  def get_avg_bb(self,label):
      g_s = self.global_state["AVG_BOUNDING_BOX"][label]
      bb = copy.deepcopy(g_s["BOUNDING_BOX"])  # float
      for i in range(4):
        for xy in range(2):
          bb[i][0][xy] = round(bb[i][0][xy])   # int
      return bb

  def post_run_analysis(self):
      self.post_run_bounding_box_analysis()
      self.record_pickup_info()
      return False
      # self.record_drop_info()
      # self.record_gripper(arm_state, gripper_state, image, robot_location, cube_location)
      # self.analyze_state_for_gripper_labels()

      # self.track_cube() 
      # self.track_box()
      # self.analyze_drop_cube()
      # self.analyze_pickup_cube()
      # /home/ros/ALSET/alset_opencv/appsself.analyze_battery_level()

  def record_drop_info(self):
      self.bb["CUBE"]                 = []
      self.bb["GRIPPER_WITH_CUBE"]    = []
      self.bb["BOX"]                  = []
      self.bb["DROP_CUBE_IN_BOX"]     = []
      self.bb["DO_DROPOFF"]           = []
      # record_pickup_info(self, arm_state, gripper_state, image, robot_location, cube_location)
      # record_gripper(self, arm_state, gripper_state, image, robot_location, cube_location)


  def binary_search_for_obj(self, low_val, high_val, obj_bb, obj_rl, obj_img, full_rl, full_img, search_direction, lv_mse=None, mv_mse=None, hv_mse=None):
      if not search_direction.startswith("HORIZ") and not search_direction.startswith("VERT") and not search_direction.startswith("ROT"):
        print("Warning: unexpected search direction", search_direction)
      # get range of values
      if search_direction.startswith("HORIZ"):
        w_h = 0
      elif search_direction.startswith("VERT"):
        w_h = 1
      elif  not search_direction.startswith("ROT"):
        # not yet implemented
        pass
      # apply new filter to old cube img
      # obj_no_rl_img = self.filter_rl_from_bb_img(obj_bb, obj_img, rl)
      if obj_rl is not None:
        obj_mean_rl_img = self.filter_rl_from_bb_img(obj_bb, obj_img, obj_rl)
      else:
        obj_mean_rl_img = obj_img
      # cv2.imshow("1BB Robot Light filter", obj_mean_rl_img)
      min_mse = self.cfg.INFINITE
      best_obj_bb  = None
      best_obj_img = None
      new_obj_bb   = None
      val = [0,0,0,0,0]
      new_rl     = full_rl 
      for i in range(5):
        val[i] = low_val + (high_val - low_val) * i / 4
      val_mse = [lv_mse, None, mv_mse, None, hv_mse]
      if abs(low_val - high_val) >= 5:
        for i in range(5):
          if val_mse[i] is None:
            if w_h == 0:
              new_obj_bb  = center_bb(obj_bb, val[i], None)
            elif w_h == 1:
              new_obj_bb  = center_bb(obj_bb, None, val[i])
            # The Robot Light and the cube will overlap during the pickup/grab phase.
            # To avoid training on the Robot Light, the mse of the cube BB should skip 
            # the robot light BB.
            new_obj_img = get_bb_img(full_img, new_obj_bb)
            if new_rl is not None:
              new_obj_mean_rl_img = self.filter_rl_from_bb_img(new_obj_bb, new_obj_img, new_rl)
            else:
              new_obj_mean_rl_img = new_obj_img
            # apply new filter to new cube img
            # new_obj_no_rl_img = self.filter_rl_from_bb_img(new_obj_bb, new_obj_img, rl)
            # if self.target is not None:
            #   cv2.imshow("new_obj_img", new_obj_mean_rl_img)
            #   cv2.waitKey(0)
            # cv2.imshow("full_img", full_img)
            # print("new/prev obj img/bb2:", new_obj_img.shape, exclude_bb)
            # print("new :", new_obj_bb, new_obj_mean_rl_img)
            # print("prev:", obj_bb, obj_mean_rl_img)
            print("new :", new_obj_bb)
            print("prev:", obj_bb)
            # compare the newly computed "new_obj_img" with the desired "obj_img"
            # val_mse[i] = self.cvu.mean_sq_err(new_obj_no_rl_img, obj_no_rl_img)
            # val_mse[i] = self.cvu.mean_sq_err(new_obj_img, obj_mean_rl_img)
            val_mse[i] = self.cvu.mean_sq_err(new_obj_mean_rl_img, obj_mean_rl_img)
            if min_mse >  val_mse[i]:
              min_mse = val_mse[i]
              best_obj_bb  = copy.deepcopy(new_obj_bb)
              best_obj_img  = new_obj_img.copy()
              print("MSE BEST_OBJ_BB:", best_obj_bb, val_mse[i])
            # print("BOI", best_obj_img)
            # if best_obj_img is not None:
            #   cv2.imshow("BOI", best_obj_img)
            # cv2.waitKey(0)
        for i in range(5):
          # just do recursion at local minima
          b1_mse, b1_img, b1_mse = None, None, None
          if i == 0 and val_mse[0] < val_mse[1]:
            print(i, "REC:", val[i], val[i+1], val_mse[i], val_mse[i+1])
            b1_bb,b1_img,b1_mse = self.binary_search_for_obj(val[0],val[1],
                     obj_bb, obj_rl, obj_img, full_rl, full_img, 
                     search_direction, val_mse[0], None, val_mse[1])
          elif i == 4 and val_mse[4] < val_mse[3]:
            print(i, "REC:", val[i-1], val[i], val_mse[i-1], val_mse[i])
            b1_bb,b1_img,b1_mse = self.binary_search_for_obj(val[3],val[4],
                     obj_bb, obj_rl, obj_img, full_rl, full_img, 
                     search_direction, val_mse[3], None, val_mse[4])
          elif val_mse[i-1] > val_mse[i] and val_mse[i] < val_mse[i+1]:
            print(i,"REC:",val[i-1],val[i],val[i+1],val_mse[i-1],val_mse[i],val_mse[i+1])
            b1_bb,b1_img,b1_mse = self.binary_search_for_obj(val[i-1],val[i+1],
                     obj_bb, obj_rl, obj_img, full_rl, full_img, 
                     search_direction, val_mse[i-1], val_mse[i], val_mse[i+1])
          # too big of a gap. Drill down anyway.
          # elif i >= 1 and val[i-1] - val[i] > obj_img.shape[w_h] / 2 and val[i-1] - val[i] >= 5:
          elif i >= 1 and abs(val[i-1] - val[i]) >= 5:
            print(i,"REC GAP:",val[i-1],val[i],val_mse[i-1],val_mse[i])
            b1_bb,b1_img,b1_mse = self.binary_search_for_obj(val[i-1],val[i],
                     obj_bb, obj_rl, obj_img, full_rl, full_img, 
                     search_direction, val_mse[i-1], None, val_mse[i])
          else:
            print(i,"NO REC:",val[i-1],val[i],val_mse[i-1],val_mse[i], obj_img.shape[w_h])

          if b1_mse is not None and min_mse >  b1_mse:
            if lv_mse is None: # only top level of recursion
              print(i, "MSE MIN_B1_BB:", min_mse, b1_mse)
            min_mse = b1_mse
            if b1_bb is not None:
              # if None, then it was computed in the previous level of recursion
              best_obj_bb   = copy.deepcopy(b1_bb)
              best_obj_img  = b1_img.copy()
              if lv_mse is None: # only top level of recursion
                print("MSE B1_BB:", min_mse, best_obj_bb)
            elif lv_mse is None: # only top level of recursion
              print("MSE B1_BB:", min_mse)
          # cv2.imshow("best img", best_obj_img)
          # cv2.imshow("full img", full_img)
          # cv2.waitKey(0)
      if lv_mse is None: # only top level of recursion
        print("MSE FINAL BB:", best_obj_bb, min_mse)
      return best_obj_bb, best_obj_img, min_mse

  # horiz, vert, rot
  # problem with changes in lights
  # problem with changes in sizes (due to distance)
  # problem with compounding errors
  # start where previous obj was
  # sensitive to action?
  # sensitive to action?
  # 
  def find_obj_betw_grippers(self, obj_name, end_frame, start_frame, expected_obj_bb, expected_obj_img):
      prev_obj_bb = copy.deepcopy(expected_obj_bb)
      prev_obj_img = copy.deepcopy(expected_obj_img)
      prev_rl = self.get_robot_light(end_frame)
      expected_obj_width = prev_obj_img.shape[0]
      expected_obj_height = prev_obj_img.shape[1]
      for fr_num in range(end_frame, start_frame-1, -1):
        print("#########################################")
        print("process frame_num", fr_num)
        print("Last known location bb:", prev_obj_bb)
        curr_img_path = self.frame_state[fr_num]["FRAME_PATH"]
        curr_img,curr_mean_diff,curr_rl = self.cvu.adjust_light(curr_img_path,force_bb=True)
        # print("curr_rl_bb:", curr_rl_bb)
        lgbb= self.get_bb("LEFT_GRIPPER",fr_num)
        rgbb= self.get_bb("RIGHT_GRIPPER",fr_num)
        # adjust gripper width so that object is not included
        lg_maxw, lg_minw, lg_maxh, lg_minh = get_min_max_borders(lgbb)
        rg_maxw, rg_minw, rg_maxh, rg_minh = get_min_max_borders(rgbb)
        gripper_width = min(lg_maxw, curr_img.shape[0]-rg_minw)
        # should be between the grippers
        betw_gripper_top_left  = [int(gripper_width + expected_obj_width/2),int(lgbb[1][0][1] - expected_obj_height/2)]
        betw_gripper_top_right = [int(curr_img.shape[0] - gripper_width - expected_obj_width/2),int(lgbb[1][0][1] - expected_obj_height/2)]
        betw_gripper_bot_left  = [int(gripper_width + expected_obj_width/2),int(lgbb[0][0][1] + expected_obj_height/2)]
        betw_gripper_bot_right = [int(curr_img.shape[0] - gripper_width - expected_obj_width/2),int(rgbb[0][0][1] + expected_obj_height/2)]
        betw_gripper_bb = [[betw_gripper_bot_left],[betw_gripper_bot_right],
                           [betw_gripper_top_left],[betw_gripper_top_right]]
        w_h = 0
        low_val  = betw_gripper_top_left[w_h] + int(prev_obj_img.shape[w_h]*3/4)
        high_val = betw_gripper_top_right[w_h] - int(prev_obj_img.shape[w_h]*3/4)
        # print("curr_rl_bb:", curr_rl_bb)
        best_obj_bb, best_obj_img, best_mse = self.binary_search_for_obj(low_val, high_val, prev_obj_bb, prev_rl, prev_obj_img, curr_rl, curr_img, "HORIZ")
        print("FOBG: intermediate best obj bb", best_obj_bb)
        cv2.imshow("fobg: curr img", curr_img)
        cv2.imshow("fobg: prev obj img", prev_obj_img)
        cv2.imshow("fobg: intermediate obj img", best_obj_img)
        # cv2.waitKey(0)
        w_h = 1
        bo_maxw, bo_minw, bo_maxh, bo_minh = get_min_max_borders(best_obj_bb)
        low_val  = bo_minh - int(prev_obj_img.shape[w_h]*3/4)
        if low_val < int(prev_obj_img.shape[w_h]/2):
          low_val = int(prev_obj_img.shape[w_h]/2)
        high_val = bo_maxh + int(prev_obj_img.shape[w_h]*3/4)
        if high_val > 223 - int(prev_obj_img.shape[w_h]/2):
          high_val = 223 - int(prev_obj_img.shape[w_h]/2)
        # use the best_obj_bb's postition, but the prev_obj_img for img comparison
        # best_obj_bb, best_obj_img, best_mse = self.binary_search_for_obj(low_val, high_val, curr_rl, best_obj_bb, prev_obj_img, curr_img, "VERT")

        # Note: Assume RL is a factor when the grippers are trying to pick up.
        best_obj_bb, best_obj_img, best_mse = self.binary_search_for_obj(low_val, high_val, best_obj_bb, prev_rl, prev_obj_img, curr_rl, curr_img, "VERT")
        # Was contour analysis here - but didn't work well enough
        new_obj_img = best_obj_img
        new_obj_bb  = best_obj_bb
        cv2.imshow("fobg: revised obj img", new_obj_img)
        ##################
        # contour analysis
        ##################
        new_obj_contour_bb = get_contour_bb(curr_img, new_obj_bb, limit_width=True, left_gripper_bb=lgbb, right_gripper_bb=rgbb)
        print("new_obj_contour_img", new_obj_contour_bb)
        new_obj_contour_img = get_bb_img(curr_img, new_obj_contour_bb)
        cv2.imshow("new_obj_contour_bb", new_obj_contour_img)
        cv2.waitKey(0)
        # Not sure that contours are needed in state. If so, return in get_contour_bb
        # label = obj_name + "_CONTOURS"
        # event_state[label] = obj_contours.copy()
        obj_maxw, obj_minw, obj_maxh, obj_minh = get_min_max_borders(new_obj_bb)
        label = obj_name + "_WIDTH"
        self.frame_state[fr_num][label] = obj_maxw - obj_minw
        label = obj_name + "_HEIGHT"
        self.frame_state[fr_num][label] = obj_maxw - obj_minw
        label = obj_name
        print("fobg: record bb:", new_obj_bb, fr_num)
        self.record_bounding_box([label], new_obj_bb, fr_num)

        prev_obj_bb  = new_obj_bb
        prev_obj_img = new_obj_img
        prev_rl = curr_rl
        continue  # try the next frame
      return  best_obj_bb, best_obj_img

  # move to analyze_move.py?
  # This following is more analyzis than recording/loading of info 
  def record_pickup_info(self):
      print("state", len(self.state))
      print("run_state", len(self.run_state))
      print("func_state", len(self.func_state))
      print("bb", len(self.bb))
      print("lbp", len(self.lbp_state))
      if self.gripper_state is not None:
        print("gripper_state", len(self.gripper_state))
      if self.arm_state is not None:
        print("arm_state", len(self.arm_state))

      ########################################################################
      # (self, arm_state, gripper_state, image, robot_location, cube_location)
      # go back and find arm up frame after gripper close and gather info
      # get LBP. Get Robot Light info.
      do_pickup_bb = None
      start_cube_grab_fr_num = None
      end_cube_grab_fr_num = None
      for fn, func in enumerate(self.func_state["FUNC_NAME"]):
        print("func s/e:", func, self.func_state["FUNC_FRAME_START"][fn], self.func_state["FUNC_FRAME_END"][fn])
      for fn, func in enumerate(self.func_state["FUNC_NAME"]):
        print("Func s/e:", func, self.func_state["FUNC_FRAME_START"][fn], self.func_state["FUNC_FRAME_END"][fn])
        if func != "PICK_UP_CUBE":
          # may have multi-pickups
          continue
        # TODO: support multiple pickups
        start_pickup = self.func_state["FUNC_FRAME_START"][fn]
        end_pickup = self.func_state["FUNC_FRAME_END"][fn]

        try:
          self.global_state["EVENT_DETAILS"]["PICK_UP_CUBE"].append({})
          puc = self.global_state["EVENT_DETAILS"]["PICK_UP_CUBE"][-1]
        except:
          self.global_state["EVENT_DETAILS"]["PICK_UP_CUBE"] = [{}]
          puc = self.global_state["EVENT_DETAILS"]["PICK_UP_CUBE"][0]
        print("start/end func frame:", start_pickup, end_pickup)
        for fr_num in range(start_pickup, end_pickup):
          rl = None
          fr_state = self.frame_state[fr_num]
          action = fr_state["ACTION"]
          print("action:", action, fr_num)
          if action in ["FORWARD","REVERSE","LEFT","RIGHT"]:
            # reset location
            puc["ROBOT_ORIGIN"]      = fr_state["MAP_ORIGIN"].copy()
            puc["ROBOT_LOCATION"]    = fr_state["LOCATION"].copy()
            puc["ROBOT_ORIENTATION"] = fr_state["ORIENTATION"]
            start_cube_grab_fr_num = None
            end_cube_grab_fr_num = None
          elif action in ["UPPER_ARM_UP", "UPPER_ARM_DOWN", "LOWER_ARM_UP", "LOWER_ARM_DOWN"]:
           
            print("fs arm state",  fr_state["ARM_STATE"])
            print("arm_pos:", fr_state["ARM_STATE"]["ARM_POSITION"], fr_state["ARM_STATE"]["ARM_POSITION_COUNT"])
            start_cube_grab_fr_num = None
            end_cube_grab_fr_num = None
          elif action in ["GRIPPER_OPEN"]:
            start_cube_grab_fr_num = None
            end_cube_grab_fr_num = None
          elif action in ["GRIPPER_CLOSE"]:
            ########################################################################
            # Track closing grippers. Find frame where closed around object.
            ########################################################################
            lgbb= self.get_bb("LEFT_GRIPPER",fr_num)
            rgbb= self.get_bb("RIGHT_GRIPPER",fr_num)
            if lgbb is None or rgbb is None:
              print("that's weird")
              continue
            # gripper_w_cube_bb keeps get overwritten until closed
            gripper_w_cube_bb = [[lgbb[0][0].copy()], [rgbb[1][0].copy()],
                                 [rgbb[2][0].copy()], [lgbb[1][0].copy()]]
            print("gwc0", gripper_w_cube_bb)
            # only update if starting new close sequence
            if start_cube_grab_fr_num is None:
              start_cube_grab_fr_num = fr_num
              last_start_cube_grab_fr_num = start_cube_grab_fr_num # never gets reset
              start_cube_grab_bb = copy.deepcopy(gripper_w_cube_bb)
            end_cube_grab_fr_num = fr_num
            last_end_cube_grab_fr_num = end_cube_grab_fr_num

            # overwrite the following until gripper is closed around the cube
            puc["GRAB_CUBE_POSITION"]     = self.gripper_state["GRIPPER_POSITION"]
            puc["GRAB_CUBE_OPEN_COUNT"]   = self.gripper_state["GRIPPER_OPEN_COUNT"]
            puc["GRAB_CUBE_CLOSED_COUNT"] = self.gripper_state["GRIPPER_CLOSED_COUNT"]
            puc["MEAN_DIF_LIGHTING"] = self.frame_state[fr_num]["MEAN_DIF_LIGHTING"] 
            lgmaxw, lgminw, lgmaxh, lgminh = get_min_max_borders(lgbb)
            rgmaxw, rgminw, rgmaxh, rgminh = get_min_max_borders(rgbb)
            cube_size = rgminw - lgmaxw
            if cube_size > 5:
              puc["CUBE_SIZE"] = cube_size
            else:
              cube_size = puc["CUBE_SIZE"]
            cube_size2 = puc["CUBE_SIZE"]

            multiplier = .5
            cube_top_left  = [int(max(lgbb[2][0][0] - cube_size2*multiplier,0)),int(lgbb[1][0][1] + cube_size2*multiplier)]
            cube_top_right = [int(rgbb[0][0][0] + cube_size2*multiplier),int(lgbb[1][0][1] + cube_size2*multiplier)]
            cube_bot_left  = [int(max(lgbb[2][0][0] - cube_size2*multiplier,0)),int(max(lgbb[0][0][1] - cube_size2*multiplier,0))]
            cube_bot_right = [int(rgbb[0][0][0] + cube_size2*multiplier),int(max(rgbb[0][0][1] - cube_size2*multiplier,0))]

            if (cube_top_left[0] >= rgmaxw / 2 - 5):
              print("adj cube 1a")
              new_cube_lw = int(rgmaxw - cube_top_right[0])
              if (cube_top_left[0] > new_cube_lw):
                cube_top_left[0] = new_cube_lw
                cube_bot_left[0] = new_cube_lw
                print("adj cube 1")
            print("cube top right", cube_top_right[0] , rgmaxw / 2 + 5)
            if (cube_top_right[0] <= rgmaxw / 2 + 5):
              print("adj cube 2a")
              new_cube_rw = int(rgmaxw - cube_top_left[0])
              if (cube_top_right[0] < new_cube_rw):
                cube_top_right[0] = new_cube_lw 
                cube_bot_right[0] = new_cube_lw
                print("adj cube 2")
            if (cube_top_right[0] <= cube_top_left[0] + 5):
              print("adj cube 3a")
              new_cube_lw = int(rgmaxw*multiplier - cube_size*multiplier)
              new_cube_rw = int(rgmaxw*multiplier + cube_size*multiplier)
              if (cube_top_right[0] < new_cube_rw):
                cube_top_left[0]  = new_cube_lw
                cube_bot_right[0] = new_cube_rw
                cube_top_right[0] = new_cube_rw
                cube_bot_left[0]  = new_cube_lw
                print("adj cube 3")

            cube_bb        = [[cube_bot_left],[cube_bot_right],
                              [cube_top_left],[cube_top_right]]
            cube_bb = bound_bb(cube_bb)

            # make an immutable copy. This bb is further processed to translate
            # cube space into full_space.
            cube_betw_gripper_full_img_space = copy.deepcopy(cube_bb)
                              
            print("cube_bb:",cube_bb)
            print("cube_sz:",cube_size)

            curr_img_path = self.frame_state[fr_num]["FRAME_PATH"]
            prev_img_path = self.frame_state[fr_num-1]["FRAME_PATH"]
            cube_img = self.get_bounding_box_image(curr_img_path, cube_bb)

            # self.record_bounding_box(["CUBE"], cube_bb, fr_num)
            gripper_w_cube_img = self.get_bounding_box_image(curr_img_path, gripper_w_cube_bb)
            # self.record_bounding_box(["CUBE", "GRIPPER_WITH_CUBE"], gripper_w_cube_bb, fr_num)
            self.record_bounding_box(["GRIPPER_WITH_CUBE"], gripper_w_cube_bb, fr_num)
            cv2.imshow("cube bb1", cube_img)
            cv2.imshow("gripper_w_cube bb1", gripper_w_cube_img)
            # square = find_square(cube_img.copy())
            # cube = find_cube(cube_img.copy(), self)
            # print("cube bb, square, cube:", cube_bb, square, cube)
            # cv2.waitKey(0)

            # Only run self.process_object_in_robot_light for actual GRAB_CUBE_POSITION?
            obj_contours = self.process_final_grab("CUBE", curr_img_path, cube_bb, puc, fr_num)

            # opt_flw = self.cvu.optflow(prev_img_path, curr_img_path, thresh = 0.05)
            opt_flw = self.cvu.optflow(prev_img_path, curr_img_path, thresh = self.cfg.OPTFLOWTHRESH/10)
            if not opt_flw:
              puc["GRAB_CUBE_POSITION"] = ["GRIPPER_WITH_CUBE"]
              print("not opt flow:", fr_num, opt_flw)
              # continue
              # break

        ########################################################################
        # Found Frame Where Gripper Has Grabbed Cube. 
        # Do in-depth analysis, gather refined stats about cube and state.
        ########################################################################
        if "GRAB_CUBE_POSITION" in puc.keys() and puc["GRAB_CUBE_POSITION"] != ["GRIPPER_WITH_CUBE"]:
          print("forcing GRAB_CUBE_POSITION to be GRIPPER_WITH_CUBE. Was:", puc["GRAB_CUBE_POSITION"])
          puc["GRAB_CUBE_POSITION"] = ["GRIPPER_WITH_CUBE"]
        # we've now forward scanned the potential actions that resulted in the cube being
        # grabbed by the gripper.  Store the results for this and other 1-time events.
        print("PUC2:", puc)
        if "GRAB_CUBE_POSITION" in puc.keys() and puc["GRAB_CUBE_POSITION"] == ["GRIPPER_WITH_CUBE"]:
          puc["ARM_POSITION"] = fr_state["ARM_STATE"]["ARM_POSITION"]
          puc["ARM_POSITION_COUNT"] = fr_state["ARM_STATE"]["ARM_POSITION_COUNT"].copy()
          # finalize start/end cube grab frames
          start_cube_grab_fr_num = last_start_cube_grab_fr_num
          end_cube_grab_fr_num = last_end_cube_grab_fr_num
          puc["START_CUBE_GRAB_FRAME_NUM"] = start_cube_grab_fr_num
          puc["END_CUBE_GRAB_FRAME_NUM"] = end_cube_grab_fr_num
          self.record_bounding_box(["START_CUBE_GRAB"], start_cube_grab_bb, start_cube_grab_fr_num)

          # the grabbed cube is probably the best cube image we'll find during the run.
          # other cube images will be computed by doing a backward pass from this image.
          best_cube_bb = self.get_bb("FWDPASS_CUBE_BB",end_cube_grab_fr_num)
          print("BEST_CUBE_BB",best_cube_bb,end_cube_grab_fr_num)
          # This is the last "CUBE" bb that should be recorded for this frame!
          self.record_bounding_box(["BEST_CUBE_BB","CUBE"],best_cube_bb,end_cube_grab_fr_num)

          print("start_cube_grab_fr_num:", start_cube_grab_fr_num, 
                "end_", end_cube_grab_fr_num)
          start_cube_grab_img_path = self.frame_state[start_cube_grab_fr_num]["FRAME_PATH"]
          start_cube_grab_img = self.get_bounding_box_image(start_cube_grab_img_path, start_cube_grab_bb)
          cv2.imshow("start_cube_grab", start_cube_grab_img)

          # do CUBE BB checks
          curr_img_path = self.frame_state[end_cube_grab_fr_num]["FRAME_PATH"]
          if (puc["MEAN_DIF_LIGHTING"] is not None and 
              self.state[self.run_num]["MEAN_LIGHTING"] is not None and 
              abs(puc["MEAN_DIF_LIGHTING"]) > .1 * 
                                      abs(self.state[self.run_num]["MEAN_LIGHTING"])):
            print("GRIPPER_W_CUBE: big change in lighting")
            rl = self.get_robot_light(end_cube_grab_fr_num)
            rlimg = self.get_robot_light_img(curr_img_path, rl)
            cv2.imshow("robot light bb", rlimg)
            # cv2.waitKey(0)
            # square = find_square(rlimg)
            # print("cube bb, square:", cube_bb, square)
            cube_rlbb = self.get_bb("CUBE_IN_ROBOT_LIGHT",end_cube_grab_fr_num)
            cube_bb   = self.get_bb("CUBE", end_cube_grab_fr_num)
          else:
            print("WARNING: GRIPPER_W_CUBE: no big change in lighting")

          # self.record_bounding_box(["CUBE"], cube_bb, fr_num)
          cube_img_path = self.frame_state[end_cube_grab_fr_num]["FRAME_PATH"]
          cube_img = self.get_bounding_box_image(curr_img_path, cube_bb)
          self.record_bounding_box(["GRIPPER_WITH_CUBE"], gripper_w_cube_bb, fr_num)
          # self.record_bounding_box(["GRIPPER_WITH_CUBE","CUBE"], gripper_w_cube_bb, fr_num)
          gripper_w_cube_img = self.get_bounding_box_image(curr_img_path, gripper_w_cube_bb)
          # ARD: cube_img is more like left gripper...
          # cv2.imshow("cube bb2", cube_img)
          # cv2.imshow("gripper_w_cube bb2", gripper_w_cube_img)
          # cv2.waitKey(0)

        ########################################################################
        # Now, track backwards during cube pickup, estimate cube location,
        # find cube, store cube state
        ########################################################################

        ########################################################################
        # initialize pick_up_cube state: 
        puc_frame_num  = puc["END_CUBE_GRAB_FRAME_NUM"] 
        puc_frame_path = self.frame_state[puc_frame_num]["FRAME_PATH"]
        puc_cube_bb    = self.get_bb("BEST_CUBE_BB", puc_frame_num)
        puc_cube_img   = self.get_bounding_box_image(puc_frame_path, puc_cube_bb)
        # puc_cube_contours  = puc["CUBE_CONTOURS"]
        puc_cube_width  = puc["CUBE_WIDTH"]
        puc_cube_height  = puc["CUBE_HEIGHT"]
        # cv2.destroyAllWindows()
        mappable_cnt = 0

        ########################################################################
        # For the time where cube is known to be between the grippers
        # self.find_cube_in_img(cropped_cube_bb, next_cube_bb, next_cube_contours, "HORIZ")
        print("start,end:", start_cube_grab_fr_num, end_cube_grab_fr_num)
        if end_cube_grab_fr_num is None or start_cube_grab_fr_num is None:
          end_cube_grab_fr_num = -1
          start_cube_grab_fr_num = 0
 
        # next_cube_contours = puc_cube_contours
        next_cube_bb       = puc_cube_bb
        next_cube_img      = puc_cube_img
        next_cube_bb, next_cube_img = self.find_obj_betw_grippers("CUBE", end_cube_grab_fr_num-1, start_cube_grab_fr_num, next_cube_bb, next_cube_img)
        curr_cube_bb = next_cube_bb.copy()
        cv2.destroyAllWindows()

        ########################################################################
        # Now, track backwards to first time cube is seen based on robot 
        # movement and keypoints.
        for fr_num in range( start_cube_grab_fr_num-1, -1, -1):
          print("#########################################")
          print("process frame_num", fr_num)

          # Do robot movement comparison
          # if ARM movement, then images only change along vert axis
          curr_img_path = self.frame_state[fr_num]["FRAME_PATH"]
          curr_img,curr_mean_diff,curr_rl=self.cvu.adjust_light(curr_img_path)
          # next_* refers to values associated with "frame_num+1"
          # next_* state may have just been stored in prev iteration of loop
          next_img_path = self.frame_state[fr_num+1]["FRAME_PATH"]
          next_img,next_mean_diff,next_rl=self.cvu.adjust_light(next_img_path)
          next_cube_bb = self.get_bb("CUBE", frame_num=fr_num+1)
          next_cube_img = get_bb_img(next_img, next_cube_bb)
          if fr_num in [121]:
            print("Frame num:", fr_num)
            cv2.destroyAllWindows()
            # self.target = True
            # cv2.imshow("CURR IMG", curr_img)

          #################################
          # Estimate Movement via Keypoints
          #################################
          # find keypoints on "next" bb (working backwards) 
          # compare "next" keypoints to find "curr" keypoints
          curr_img_KP = Keypoints(curr_img)
          next_img_KP = Keypoints(next_img)

          good_matches, bad_matches = curr_img_KP.compare_kp(next_img_KP,include_dist=True)
          next_kp_list = []
          kp_dif_x = None
          kp_dif_y = None
          min_filter_direction_dif = self.cfg.INFINITE
          action = self.frame_state[fr_num]["ACTION"]
          for i,kpm in enumerate(good_matches):
            print("kpm", kpm)
            # m = ((i, j, dist1),(i, j, dist2), ratio)
            curr_kp = curr_img_KP.keypoints[kpm[0][0]]
            next_kp = next_img_KP.keypoints[kpm[0][1]]
            # (x,y) and (h,w) mismatch
            h,w = 1,0
            kp_dif_x = 0
            kp_dif_y = 0
            direction_kp_x = None
            direction_kp_y = None
            if point_in_border(curr_cube_bb,next_kp.pt, 0):
              next_kp_list.append([next_kp.pt, curr_kp.pt])
              kp_dif_x += (curr_kp.pt[0] - next_kp.pt[0])
              kp_dif_y += (curr_kp.pt[1] - next_kp.pt[1])
              print("suspected x/y diff:", kp_dif_x, kp_dif_y)
            if action in ["FORWARD","REVERSE","UPPER_ARM_UP","UPPER_ARM_DOWN", "LOWER_ARM_UP", "LOWER_ARM_DOWN"]:
              # only y direction should change. minimize x.
              dif_x = (curr_kp.pt[0] - next_kp.pt[0])
              if abs(dif_x) < min_filter_direction_dif:
                min_filter_direction_dif = abs(dif_x)
                dif_y = (curr_kp.pt[1] - next_kp.pt[1])
                direction_kp_y = dif_y
            if action in ["LEFT","RIGHT"]:
              dif_y = (curr_kp.pt[1] - next_kp.pt[1])
              if abs(dif_y) < min_filter_direction_dif:
                min_filter_direction_dif = abs(dif_y)
                dif_x = (curr_kp.pt[0] - next_kp.pt[0])
                direction_kp_x = dif_x
          if len(next_kp_list) > 0:
            kp_dif_x = round(kp_dif_x / len(next_kp_list))
            kp_dif_y = round(kp_dif_x / len(next_kp_list))
            print("kp_dif: ", kp_dif_x, kp_dif_y)
          print("directional best kp dif: ", direction_kp_x, direction_kp_y)

          #####################################################################
          # also compute delta based on estimated movement
          #####################################################################
          next_img_path = self.frame_state[fr_num+1]["FRAME_PATH"]
          mv_dif_x =  None
          mv_dif_y =  None
          print("action:", self.frame_state[fr_num]["ACTION"], fr_num, self.is_mappable_position(self.frame_state[fr_num]["ACTION"], fr_num))
          if self.is_mappable_position(self.frame_state[fr_num]["ACTION"], fr_num):
            # Reenable after hack is gone. For now, move down a little
            # if mappable_cnt == 0:
            #   cv2.destroyAllWindows()
            #   mappable_cnt = 1
            mv_rot = self.frame_state[fr_num]["MOVE_ROTATION"] 
            mv_pos = self.frame_state[fr_num]["MOVE_STRAIGHT"] 
            print("mv_rot/pos:", mv_rot, mv_pos)
            if mv_rot is None and mv_pos is None:
              print("frame_state:", self.frame_state[fr_num])
            elif mappable_cnt == 0:
              cv2.destroyAllWindows()
              mappable_cnt = 1

            # convert movement from birds_eye_view perspective back to camera perspective
            if mv_pos is not None:
              # following prob doesn't work if bb height is too large
              mv_dif_x =  0
              mv_dif_y =  mv_pos / (self.cfg.BIRDS_EYE_RATIO_H+1)
              print("MV_POS: ", mv_dif_x, mv_pos, (self.cfg.BIRDS_EYE_RATIO_H+1))
              w_h = 1
              next_cube_ctr  = bounding_box_center(next_cube_bb)
              # search_window = abs(mv_dif_y) / 2
              # low_val = next_cube_ctr[w_h] + mv_dif_y - search_window - int(next_cube_img.shape[w_h]*3/4)

              low_val = next_cube_ctr[w_h] + mv_pos
              print("val:", next_cube_ctr[w_h], mv_dif_y, int(next_cube_img.shape[w_h]*3/4))
              if low_val < int(next_cube_img.shape[w_h]/2):
                low_val = int(next_cube_img.shape[w_h]/2)
              # high_val = next_cube_ctr[w_h] + mv_dif_y + search_window + int(next_cube_img.shape[w_h]*3/4)
              high_val = next_cube_ctr[w_h] + mv_pos/2
              if high_val > 223 - int(next_cube_img.shape[w_h]/2):
                high_val = 223 - int(next_cube_img.shape[w_h]/2)
              high_val = int(high_val)
              low_val = int(low_val)
              # use the best_obj_bb's postition, but the next_cube_img for img comparison
              best_obj_bb, best_obj_img, best_mse = self.binary_search_for_obj(low_val, high_val, next_cube_bb, next_rl, next_cube_img, curr_rl, curr_img, "VERT")
              new_obj_img = best_obj_img
              new_obj_bb  = best_obj_bb
              print("low, high val:", low_val, high_val)
              cv2.imshow("curr img", curr_img)
              cv2.imshow("0cube: obj img", next_cube_img)
              cv2.imshow("1mv mse: obj img", new_obj_img)

              step_by = next_cube_img.shape[w_h]
              next_maxw, next_minw, next_maxh, next_minh = get_min_max_borders(next_cube_bb)
#              for c_min_h in range(low_val, high_val, step_by):
#              # for c_min_h in range(0, 223, step_by):
#                next_maxw, next_minw, next_maxh, next_minh = get_min_max_borders(next_cube_bb)
#                c_max_h = c_min_h + step_by
#                c_bb = make_bb(next_maxw, next_minw, c_max_h, c_min_h)
#                print("c_bb", c_bb)
#                c_img = get_bb_img(curr_img, c_bb)
#                cv2.imshow(str(c_min_h), c_img)
#                cv2.waitKey(0)
       
                
              ##################
              # contour analysis
              ##################
              new_obj_contour_bb = get_contour_bb(curr_img, new_obj_bb, limit_width=False, left_gripper_bb=None, right_gripper_bb=None)
              print("new_obj_contour_bb", new_obj_contour_bb)
              new_obj_contour_img = get_bb_img(curr_img, new_obj_contour_bb)
              cv2.imshow("2mv mse ctr:", new_obj_contour_img)
              cv2.waitKey(0)

              new_obj_contour_ctr = bounding_box_center(new_obj_contour_bb)
              next_cube_ctr  = bounding_box_center(next_cube_bb)
              mv_dif_x =  next_cube_ctr[0] - new_obj_contour_ctr[0]
              mv_dif_y =  next_cube_ctr[1] - new_obj_contour_ctr[1]
              print("final mv_dif_x, mv_dif_y:", mv_dif_x, mv_dif_y)
  
            elif mv_rot is not None:
              curr_cube_ctr = bounding_box_center(curr_cube_bb)
              robot_pos = [(-self.MAP_ROBOT_H_POSE / (self.cfg.BIRDS_EYE_RATIO_W)), int(next_img.shape[0]*multiplier) ]
              next_angle = rad_arctan2(curr_cube_ctr[0]-robot_pos[0], curr_cube_ctr[1]-robot_pos[1])
              curr_angle = next_angle + mv_rot
              dist = np.sqrt((curr_cube_ctr[0] - robot_pos[0])**2 + (curr_cube_ctr[1] - robot_pos[1])**2)

              mv_dif_x = round(dist * math.cos(curr_angle))
              mv_dif_y = round(dist * math.sin(curr_angle))
              print("mv_dif: ", mv_dif_x, mv_dif_y)
              cv2.imshow("curr img", curr_img)
              cv2.imshow("0cube: obj img", next_cube_img)
              cv2.imshow("1mv dif: obj img", new_obj_img)
              cv2.waitKey(0)

          #####################################################################
          # MSE-based analysis
          #####################################################################
          next_img_path = self.frame_state[fr_num+1]["FRAME_PATH"]
          mse_dif_x =  None
          mse_dif_y =  None
          action = self.frame_state[fr_num]["ACTION"]
          if (action.startswith("UPPER_ARM") or action.startswith("LOWER_ARM") 
              or action.startswith("GRIPPER")):
            # only up/down; next stands for "from frame_num + 1"
            next_maxw, next_minw, next_maxh, next_minh = get_min_max_borders(next_cube_bb)
            if action.startswith("UPPER_ARM") or action.startswith("LOWER_ARM"):
              print("binary_search_for_obj: vert")
              w_h = 1  # height
              v_multiplier = 3
              low_val  = next_minh - int(next_cube_img.shape[w_h]*v_multiplier)
              if low_val < int(next_cube_img.shape[w_h]/2):
                low_val = int(next_cube_img.shape[w_h]/2)
              high_val = next_maxh + int(next_cube_img.shape[w_h]/2)
              if high_val > 223 - int(next_cube_img.shape[w_h]/2):
                high_val = 223 - int(next_cube_img.shape[w_h]/2)
              print("V low, high val", low_val, high_val)
              search_dir = "VERT"
            elif action.startswith("GRIPPER"):
              print("binary_search_for_obj: horiz")
              w_h = 0  # width
              low_val  = next_minw - int(next_cube_img.shape[w_h]*3)
              if low_val < int(next_cube_img.shape[w_h]/2):
                low_val = int(next_cube_img.shape[w_h]/2)
              high_val = next_maxw + int(next_cube_img.shape[w_h]*3)
              if high_val > 223 - int(next_cube_img.shape[w_h]/2):
                high_val = 223 - int(next_cube_img.shape[w_h]/2)
              search_dir = "HORIZ"
            # The robot light makes a huge difference in results, but figuring out
            # when the robot light should (not) be considered is not trivial.
            # Run twice (with and without RL) and take lowest mse..
            rl_mse_best_obj_bb_vert, rl_mse_best_obj_img_vert, rl_best_mse = self.binary_search_for_obj(low_val, high_val, next_cube_bb, next_rl, next_cube_img, curr_rl, curr_img, search_dir)
            norl_mse_best_obj_bb_vert, norl_mse_best_obj_img_vert, norl_best_mse = self.binary_search_for_obj(low_val, high_val, next_cube_bb, None, next_cube_img, None, curr_img, search_dir)
            if norl_best_mse <= rl_best_mse:
              # lower is better
              mse_best_obj_bb_vert = norl_mse_best_obj_bb_vert
              mse_best_obj_img_vert = norl_mse_best_obj_img_vert
              filter_rl = False
            else:
              mse_best_obj_bb_vert = rl_mse_best_obj_bb_vert
              mse_best_obj_img_vert = rl_mse_best_obj_img_vert
              filter_rl = True
            print("V filter rl, norlmse, rlmse", filter_rl, norl_best_mse, rl_best_mse)

            # now tweak to allow some recentering
            RECENTER_PIX = 30  # check 40 pixels on each side
            print("next cube shape:", next_cube_img.shape)
            if action.startswith("UPPER_ARM") or action.startswith("LOWER_ARM"):
              print("binary_search_for_obj: horiz")
              w_h = 0  # width
              search_dir = "HORIZ"
              low_val  = next_minw - int(RECENTER_PIX)
              if low_val < int(next_cube_img.shape[w_h]/2):
                low_val = int(next_cube_img.shape[w_h]/2)
              high_val = next_maxw + int(RECENTER_PIX)
              if high_val > 223 - int(next_cube_img.shape[w_h]/2):
                high_val = 223 - int(next_cube_img.shape[w_h]/2)
              print("H low, high val", low_val, high_val)
            elif action.startswith("GRIPPER"):
              print("binary_search_for_obj: vert")
              w_h = 1  # height
              low_val  = next_minh - int(RECENTER_PIX)
              if low_val < int(next_cube_img.shape[w_h]/2):
                low_val = int(next_cube_img.shape[w_h]/2)
              high_val = next_maxh + int(RECENTER_PIX)
              if high_val > 223 - int(next_cube_img.shape[w_h]/2):
                high_val = 223 - int(next_cube_img.shape[w_h]/2)
              search_dir = "VERT"
            rl_mse_best_obj_bb, rl_mse_best_obj_img, rl_best_mse = self.binary_search_for_obj(low_val, high_val, mse_best_obj_bb_vert, next_rl, next_cube_img, curr_rl, curr_img, search_dir)
            norl_mse_best_obj_bb, norl_mse_best_obj_img, norl_best_mse = self.binary_search_for_obj(low_val, high_val, mse_best_obj_bb_vert, None, next_cube_img, None, curr_img, search_dir)
            if norl_best_mse <= rl_best_mse:
              mse_best_obj_bb = norl_mse_best_obj_bb
              mse_best_obj_img = norl_mse_best_obj_img
              filter_rl = False
            else:
              mse_best_obj_bb = rl_mse_best_obj_bb
              mse_best_obj_img = norl_mse_best_obj_img
              filter_rl = True
            print("H filter rl, norlmse, rlmse", filter_rl, norl_best_mse, rl_best_mse)


            ##################
            # contour analysis
            ##################
            if mappable_cnt == 0:
              lgbb= self.get_bb("LEFT_GRIPPER",fr_num)
              rgbb= self.get_bb("RIGHT_GRIPPER",fr_num)
              print("lgbb", lgbb)
              print("rgbb", rgbb)
            else:
              lgbb, rgbb = None, None
            # used to try to determine to filter Robot Light in following func:
            # mean_rl_img, fitler_rl = self.filter_rl_from_bb_img(mse_best_obj_bb, mse_best_obj_img, curr_rl)
            if filter_rl:
              mse_rl = curr_rl
            else:
              mse_rl = None
            # we may want to limit width if RL is significantly overlapping the cube
            # The robot light may cause the contours to be wrong/incomplete.
            # No RL being passed in.
            mse_best_obj_contour_bb = get_contour_bb(curr_img, mse_best_obj_bb, mse_rl, padding_pct=.25, left_gripper_bb=lgbb, right_gripper_bb=rgbb)
            print("mse_best_obj_contour_bb", mse_best_obj_contour_bb)
            if mse_best_obj_contour_bb is not None:
              mse_best_obj_contour_img = get_bb_img(curr_img, mse_best_obj_contour_bb)
              if fr_num == 160:
                cv2.destroyAllWindows()
              if True:
                print("Frame num:", fr_num)
                print("V best MSE", mse_best_obj_bb_vert)
                print("H best MSE", mse_best_obj_bb)
                print("contour_bb", mse_best_obj_contour_bb)
                cv2.imshow("next cube img", next_cube_img)
                cv2.imshow("V best MSE", mse_best_obj_img_vert)
                cv2.imshow("H best MSE", mse_best_obj_img)
                cv2.imshow("curr_img", curr_img)
                cv2.imshow("mse_best_obj_contour_bb", mse_best_obj_contour_img)
                cv2.waitKey(0)
              # Contour_BB seems better in general
              mse_best_obj_bb = mse_best_obj_contour_bb
              mse_best_obj_img = mse_best_obj_contour_img

            ###############
            # record MSE best cube and LBP
            # problem: LBP is probably not valid if overlaps with Robot Light
            self.record_bounding_box(["BEST_CUBE_BB", "CUBE"],
                                     mse_best_obj_bb, fr_num)
            self.record_lbp("CUBE", mse_best_obj_img, bb=mse_best_obj_bb, frame_num=fr_num)
            cv2.imshow("MSE CUBE", mse_best_obj_img)

            # compute mse_dif_x/mse_dif_y
            next_ctr_x, next_ctr_y = bounding_box_center(next_cube_bb)
            bo_ctr_x, bo_ctr_y = bounding_box_center(mse_best_obj_bb)
            # mse_dif_x = next_ctr_x - bo_ctr_x
            # mse_dif_y = next_ctr_y - bo_ctr_y
            mse_dif_x = bo_ctr_x - next_ctr_x
            mse_dif_y = bo_ctr_y - next_ctr_y

          ##########################################
          # Final Calculations:
          # Record the final cube bb, image, and lbp.
          # MSE computes and stores the cube bb, image, and lbp.
          # If no mse, then we need to compute final bb, images, lbp here.
          ##########################################
          print("mse dif:", mse_dif_x, mse_dif_y)
          print("mv  dif:", mv_dif_x, mv_dif_y)
          # print("arm dif:", arm_dif_x, arm_dif_y)
          print("kp  dif:", kp_dif_x, kp_dif_y)
          if mse_dif_y is not None:
            dx = mse_dif_x
            dy = mse_dif_y
            print("mse dif wins:", dx, dy)
          elif mv_dif_y is not None:
            dx = mv_dif_x
            dy = mv_dif_y
            print("mv dif wins:", dx, dy)
          # elif arm_dif_y is not None:
          #   dx = arm_dif_x
          #   dy = arm_dif_y
          #   print("arm dif wins:", dx, dy)
          elif kp_dif_y is not None:
            dx = kp_dif_x
            dy = kp_dif_y
            print("kp dif wins:", dx, dy)
          else:
            print("ERROR: no estimated cube location")
            # Do brute-force VERT binary search 
      
          if mse_dif_y is None:
            for i in range(5):
              # find last cube_bb
              final_cube_bb = self.get_bb("CUBE",fr_num+i)
              if final_cube_bb is not None:
                print("final cube bb:", fr_num+i, i, final_cube_bb)
                break
  
            final_img_path = self.frame_state[fr_num]["FRAME_PATH"]
            final_img,final_mean_dif,final_rl = self.cvu.adjust_light(curr_img_path)
            for i in range(4):
              dx = min(dx, -final_cube_bb[i][0][0] + curr_img.shape[xy2hw(0)] - 1)
              dx = max(dx, -final_cube_bb[i][0][0]) 
              dy = min(dy, -final_cube_bb[i][0][1] + curr_img.shape[xy2hw(1)] - 1)
              dy = max(dy, -final_cube_bb[i][0][1]) 
            for i in range(4):
              final_cube_bb[i][0][0] += dx
              final_cube_bb[i][0][1] += dy
              if final_cube_bb[i][0][0] >= final_img.shape[xy2hw(0)]:
                final_cube_bb[i][0][0] = final_img.shape[xy2hw(0)] - 1
              elif final_cube_bb[i][0][0] < 0:
                final_cube_bb[i][0][0] = 0
              if final_cube_bb[i][0][1] >= final_img.shape[xy2hw(1)]:
                final_cube_bb[i][0][1] = final_img.shape[xy2hw(1)] - 1
              elif final_cube_bb[i][0][1] < 0:
                final_cube_bb[i][0][1] = 0
  
            if final_cube_bb is None:
              cv2.imshow("final_img", curr_img)
              # cv2.waitKey(0)
              final_cube_img = None
              print("DIFF: cube bb:", final_cube_bb)
              break
            else:
              print("final cube bb:", final_cube_bb)
              final_cube_img = self.get_bounding_box_image(curr_img, final_cube_bb)
            print("DIFF: cube bb:", final_cube_bb)
            if final_cube_img is not None:
              cv2.imshow("final_img", curr_img)
              cv2.imshow("next_img", next_img)
              cv2.imshow("MV cube bb", final_cube_img)
              cv2.waitKey(0)
              # square = find_square(curr_cube_img)
              # print("DIFF: cube bb, square:", curr_cube_bb, square)
              gray_cube_img = cv2.cvtColor(final_cube_img, cv2.COLOR_BGR2GRAY)
              self.record_bounding_box(["CUBE"], final_cube_bb, fr_num)
              self.record_lbp("CUBE", gray_cube_img, bb=final_cube_bb, frame_num=fr_num)

#############################################################
#############################################################
#############################################################

  def track_box(self): 
      # start from DROPOFF and work backwards, based on reverse of movements
      # and keypoints and LBP until box is out of frame.
      self.bb["BOX"]                  = []


  def analyze_battery_level(self):
      # track how movements decrease over time as battery levels go down
      # min, max, stddev as you go. Plot. Get fit slope to results.
      pass

  #############
  # Post-Analysis Helper Functions (statistical)
  #############

  def is_light_on_cube(img):
      # check for overlapping BB
      return
      # possible_reply = ["ON", "LOWER_ARM_UP", "LOWER_ARM_DOWN", "UPPER_ARM_UP", "UPPER_ARM_DOWN", "LEFT",RIGHT", "OFF"]

      # return answer

  def is_cube_between_grippers(frame_num):
      # lighting <big change in lighting>
      # cube_size
      return


  def is_close_to_cube(img):
      return

  def is_close_to_box(img):
      return False

  def is_gripper_above_box(img):
      return True

  def process_final_grab(self, obj_name, img_path, obj_bb, event_state, frame_num=None):
      if frame_num is None:
        frame_num = self.frame_num
      ############################
      # initialize lighting state
      mean_lighting, mean_dif2, rl2 = self.get_lighting(frame_num)
      self.cvu.MeanLighting = mean_lighting
      self.cvu.MeanLightCnt = frame_num
      orig_img,mean_diff,rl = self.cvu.adjust_light(img_path, force_bb=True)
      gray_orig_img  = cv2.cvtColor(orig_img.copy(), cv2.COLOR_BGR2GRAY)
      # orig_img,mean_diff,rl = self.cvu.adjust_light(img_path)
      obj_img = self.get_bounding_box_image(img_path, obj_bb)

      # get lighting info about orig img
      # print("rl_bb orig,new", rl_bb2, rl_bb)
      if rl is None and rl2 is not None:
        rl = rl2
        
      ############################
      # get robot lighting bounding box, color, lbp
      if rl is not None:
        rl_img = orig_img.copy()
        mask = rl["LABEL"]==rl["LIGHT"]
        mask = mask.reshape((rl_img.shape[:2]))
        # print("mask",mask)
        rl_img[mask==rl["LIGHT"]] = [0,0,0]
        cv2.imshow("Robot Light", rl_img)
        print("obj_bb", obj_bb, rl_img.shape)
        cube_rl_img = get_bb_img(rl_img, obj_bb)
        cv2.imshow("Cube in Robot Light", cube_rl_img)
        # cv2.waitKey()
        cube_rl_label = obj_name+"_IN_ROBOT_LIGHT"
        self.record_bounding_box([cube_rl_label], obj_bb, frame_num=frame_num, use_img=cube_rl_img)


        ############################
        # Compute Radius for LBP 
        #
        # center of light is approx center of obj when gripper closed
        # maxw, minw, maxh, minh = get_min_max_borders(rl_bb)
        # center = rl["CENTER"]
        sum_x = 0.0
        sum_y = 0.0
        cnt = 0
        # print("ravel", rl["LABEL"].ravel())
        # print("zip", zip(gray_orig_img,rl["LABEL"].ravel()))
        # for (x, y), label in zip(np.int32(gray_orig_img), rl["LABEL"].ravel()):
        # IMG_HW = self.cfg.IMG_H
        IMG_HW = 224
        for x in range(IMG_HW):
          for y in range(IMG_HW):
            if not rl["LABEL"][x*IMG_HW+y]:
              # print(x,y,rl["LABEL"][x*IMG_HW+y])
              sum_x += x
              sum_y += y
              cnt += 1
        center = [int(sum_x / cnt), int(sum_y / cnt)]
        print("cnt", cnt, center)
        rl_radius = 0
        imshp = orig_img.shape
        rl_radius_img = np.zeros((imshp[0], imshp[1], 3), dtype = "uint8")
        for radius in range(1,int(min(imshp[0]/2, imshp[1]/2))):
          if rl_radius != 0:
            break
          side = int(radius*2-1)
          for s in range(1, side+1):
            # check all for sides:
            # Top:
            mh = int(center[0] - radius)
            mw = int(center[1] - radius + s)
            # print(radius,".0 mh,mw", mh, mw, mh*imshp[0]+mw, np.shape(rl["LABEL"]))
            if (mh > 223 or mh < 0 or mw > 223 or mh < 0 or
                rl["LABEL"][mh*imshp[0]+mw][0] == rl["LIGHT"]):
              rl_radius = radius - 1
              break
            rl_radius_img[mh,mw,:] = orig_img[mh, mw,:]
            # Bottom:
            mh = int(center[0] + radius)
            mw = int(center[1] - radius + s)
            # print(radius,".1 mh,mw", mh, mw, mh*imshp[0]+mw, np.shape(rl["LABEL"]))
            if (mh > 223 or mh < 0 or mw > 223 or mh < 0 or
                rl["LABEL"][mh*imshp[0]+mw][0] == rl["LIGHT"]):
              rl_radius = radius - 1
              break
            rl_radius_img[mh,mw,:] = orig_img[mh, mw, :]
            # Left:
            mh = int(center[0] - radius + s)
            mw = int(center[1] - radius)
            # print(radius,".2 mh,mw", mh, mw, mh*imshp[0]+mw, np.shape(rl["LABEL"]))
            if (mh > 223 or mh < 0 or mw > 223 or mh < 0 or
                rl["LABEL"][mh*imshp[0]+mw][0] == rl["LIGHT"]):
              rl_radius = radius - 1
              break
            rl_radius_img[mh,mw,:] = orig_img[mh, mw,:]
            # Right:
            mh = int(center[0] - radius + s)
            mw = int(center[1] + radius)
            # print(radius,".3 mh,mw", mh, mw, mh*imshp[0]+mw, np.shape(rl["LABEL"]))
            if (mh > 223 or mh < 0 or mw > 223 or mh < 0 or
                rl["LABEL"][mh*imshp[0]+mw][0] == rl["LIGHT"]):
              rl_radius = radius - 1
              break
            rl_radius_img[mh,mw,:] = orig_img[mh, mw,:]

        print("rl_radius", rl_radius)
        rl_img = orig_img.copy()
        mask = rl["LABEL"]==rl["LIGHT"]
        mask = mask.reshape((rl_img.shape[:2]))
        # print("mask",mask)
        rl_img[mask==rl["LIGHT"]] = [0,0,0]
        # cv2.imshow("Robot Light", rl_img)
        print("obj_bb", obj_bb, rl_img.shape)
        cube_rl_img = get_bb_img(rl_img, obj_bb)
        cv2.imshow("Cube in Robot Light", cube_rl_img)
        cube_rl_label = obj_name+"_IN_ROBOT_LIGHT"
        self.record_bounding_box([cube_rl_label], obj_bb, frame_num=frame_num, use_img=cube_rl_img)

        # bounding box image for robot light stored with frame state
        self.record_robot_light(rl, frame_num=frame_num)
        # store lbp state
        self.record_lbp(cube_rl_label, cube_rl_img.copy(), bb=obj_bb, frame_num=frame_num)

        # record mean colors
        robot_light_hsv  = cv2.cvtColor(cube_rl_img.copy(), cv2.COLOR_BGR2HSV)
        rl_h, rl_s, rl_v = cv2.split(robot_light_hsv)
        rl_mean_h = cv2.mean(rl_h)[0]
        rl_mean_s = cv2.mean(rl_s)[0]
        rl_mean_v = cv2.mean(rl_v)[0]
        event_state["ROBOT_LIGHT_MEAN_HSV"] = [rl_mean_h, rl_mean_s, rl_mean_v]

      ##################
      # contour analysis
      ##################
      fwdpass_obj_bb = obj_bb
      lgbb= self.get_bb("LEFT_GRIPPER",frame_num)
      rgbb= self.get_bb("RIGHT_GRIPPER",frame_num)
      fwdpass_obj_contour_bb = get_contour_bb(obj_img, obj_bb, limit_width=True, left_gripper_bb=lgbb, right_gripper_bb=rgbb)
      print("fwdpass_obj_bb", fwdpass_obj_bb)
      print("fwdpass_obj_contour_bb", fwdpass_obj_contour_bb)
      fwdpass_obj_img = get_bb_img(orig_img, fwdpass_obj_bb)
      fwdpass_obj_contour_img = get_bb_img(orig_img, fwdpass_obj_contour_bb)
      # cv2.imshow("1fwdpass_obj_bb", fwdpass_obj_img)
      # cv2.imshow("2fwdpass_obj_contour_bb", fwdpass_obj_contour_img)
      # cv2.waitKey(0)
      fwdpassobj_label = "FWDPASS_" + obj_name + "_BB"  # forward pass only
      self.record_bounding_box([fwdpassobj_label], fwdpass_obj_bb, frame_num=frame_num)

      if rl is not None:
        # bounding box image for robot light stored with frame state
        self.record_robot_light(rl, frame_num=frame_num)
        # store lbp state
        label = obj_name+"_IN_ROBOT_LIGHT"
        rl_radius = min(obj_img.shape[0],obj_img.shape[1]) * 3 / 8
        self.record_lbp(label, obj_img.copy(), radius=rl_radius, 
                        frame_num=frame_num)
        # record mean colors
        robot_light_hsv  = cv2.cvtColor(rl_img.copy(), cv2.COLOR_BGR2HSV)
        rl_h, rl_s, rl_v = cv2.split(robot_light_hsv)
        rl_mean_h = cv2.mean(rl_h)[0]
        rl_mean_s = cv2.mean(rl_s)[0]
        rl_mean_v = cv2.mean(rl_v)[0]
        event_state["ROBOT_LIGHT_MEAN_HSV"] = [rl_mean_h, rl_mean_s, rl_mean_v]

      lgbb= self.get_bb("LEFT_GRIPPER",frame_num)
      rgbb= self.get_bb("RIGHT_GRIPPER",frame_num)
      fwdpass_obj_bb = obj_bb
      fwdpass_obj_contour_bb = get_contour_bb(obj_img, obj_bb, left_gripper_bb=lgbb, right_gripper_bb=rgbb)
      print("3fwdpass_obj_bb", fwdpass_obj_bb,)
      print("3fwdpass_obj_contour_bb", fwdpass_obj_bb,)
      fwdpassobj_label = "FWDPASS_" + obj_name + "_BB"  # forward pass only
      self.record_bounding_box([fwdpassobj_label], fwdpass_obj_bb, frame_num=frame_num)
      obj_maxw, obj_minw, obj_maxh, obj_minh = get_min_max_borders(fwdpass_obj_bb)
      label = obj_name + "_WIDTH"
      event_state[label] = obj_maxw - obj_minw
      label = obj_name + "_HEIGHT"
      event_state[label] = obj_maxh - obj_minh
      cv2.waitKey()


  def found_object(img):
      # [cube distance [pixels]]
      # [cube size=
      # Start from light center
      # For every radius:
      # - Find light/dark separation of light
      # - Do LBP inside and outside of light
      # - Keep track of light, color means
      # - Compare full image's bottom middle to top right / top left 
      #   to narrow down object size
      # - store B&W Contour in full grasp (light will effect)
      # - record len(approximations) 

      return

  def found_box(img):
      # position = ["LEFT","CENTER","RIGHT"]
      # confidence = 
      # box_distance = [compute_pixels]
      # box_size =
      # lbp = 
      return

  def is_drivable_forward(img):
      # <no obstacle / cube / within expected distance
      # <safe lbp within expected distance
      return

  def is_cube_in_gripper(frame_num, action, direction):
      return False

  def get_robot_arm_state(self, frame_num=None):
      if frame_num is None:
        frame_num = -1
      arm_state      = self.frame_state[frame_num]["ARM_STATE"]
      if len(arm_state) > 0:
        print (frame_num,"arm state:", arm_state)
        return arm_state
#      # now we copy, used to have to search back.
#      # For backward compatability, should delete soon.
#      print("num frame states togo",len(self.frame_state)-frame_num)
#      frame_num = self.frame_num
#      for i in range(len(self.frame_state)-frame_num):
#        if len(self.frame_state[frame_num-i]["ARM_STATE"]) > 0:
#          print (frame_num-i,"arm state:", self.frame_state[frame_num-i]["ARM_STATE"], len(self.frame_state[frame_num-i]["ARM_STATE"]), len(self.frame_state)-frame_num)
#          return self.frame_state[frame_num-i]["ARM_STATE"]
#        else:
#          print (frame_num-i,"frame state:", self.frame_state[frame_num-i])
      print("no arm state")
      return {}

  def get_robot_gripper_position(self, frame_num=None):
      if frame_num is None:
        frame_num = -1
      gripper_state = self.frame_state[frame_num]["GRIPPER_STATE"]
      gripper_position = gripper_state["GRIPPER_POSITION"]
      return gripper_position

  def is_mappable_position(self, action, fr_num = None):
      # don't update map / location
      # combine history with optical flow estimates
      if action in ["FORWARD", "REVERSE", "LEFT", "RIGHT"]:
        print("action: ", action)
        if fr_num is None:
          fr_num = -1
        arm_state = self.get_robot_arm_state(fr_num)
        print("ARM STATE: ", arm_state)
        if len(arm_state) == 0:
          return False
        if arm_state["ARM_POSITION"] == "ARM_RETRACTED":
          return True
      arm_state = self.get_robot_arm_state(fr_num)
      print("unmappable action: ", action, arm_state["ARM_POSITION"])
      # return True  # Hack until regenerate state
      return False

  def record_metrics(self, frame_path, data):
      # KP, MSE, LBP
      # record Metrics for different rotation angles of a frame
      # record Line/KP rotation analysis
      pass

  def get_expected_movement(self, action):
      return self.move_dist[action], var

  def record_location(self, origin, location, orientation):
      # Note: stored as part of the frame_state
      self.location["MAP_ORIGIN"]  = origin.copy()
      self.location["LOCATION"]    = location.copy()
      self.location["ORIENTATION"] = orientation
      record_hist = False
      if len(self.state[self.run_num]["LOCATION_HISTORY"]) > 0:
        [prev_origin,prev_loc,prev_orient,frame] = self.state[self.run_num]["LOCATION_HISTORY"][-1]
        if ((prev_origin[0] != origin[0] or prev_origin[1] != origin[1])
           or (prev_loc[0] != location[0] or prev_loc[1] != location[1])
           or prev_orient != orientation):
          record_hist = True
      else:
          record_hist = True
      if record_hist:
        self.state[self.run_num]["LOCATION_HISTORY"].append([origin.copy(), location.copy(), orientation, self.frame_num])
      return

  ###
#  def train_predictions(self,move, done=False):
#      self.model.add_next_move(move, done)

#  def predict(self):
#      prediction = self.model.predict_move()
#      self.frame_state[self.frame_num]["PREDICTED_MOVE"] = prediction
#      return prediction

  # 
  # known state: keep track of arm movements since the arm was 
  # parked in a known position
  # 
  def set_known_robot_state(self, frame_num, known_state):
      # self.state_history.append([self.robot_state[:]])
      self.robot_state["ARM_STATE"] = known_state
      self.robot_state["DELTA"] = False
      for action in self.cfg.arm_actions_no_wrist:
        self.robot_state[action] = 0
      print("set known robot state:", frame_num, known_state)

  def get_robot_state(self):
      return [self.robot_state]

  def add_delta_from_known_state(self, action):
      if action in self.arm_actions_no_wrist:
        self.robot_state_delta[move] += 1
      self.robot_state["DELTA"] = True

  ################### 
  # save/load state
  ################### 
  def save_state(self):
      if self.dsu is not None: 
        filename = self.dsu.alset_state_filename(self.app_name)
        with open(filename, 'wb') as outp:  # Overwrites any existing file.
          pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

  def load_state_completed(self):
      self.restart_frame_num = self.last_frame_state("FRAME_NUM")
      print("restart frame:", self.restart_frame_num)

  def load_state(self, filename):
      """ Deserialize a file of pickled objects. """
      loaded = False
      print("alset_state.load_state", filename)
      try:
        with open(filename, "rb") as f:
          while True:
            try:
                obj = pickle.load(f)
                loaded = True
                print("pickle load", obj.app_name)
            except EOFError:
                # print("pickle EOF")
                return obj
      except Exception as e:
        print("load_state exception: ", e)
      # print("load: False")
      return self

  # "APP_NAME" "APP_RUN_NUM" "APP_RUN_ID" "APP_MODE" "FUNC_STATE" "FRAME_STATE" 
  # "GRIPPER_STATE" "BOUNDING_BOXES" "ACTIONS" "FINAL_MAP" "MAP_ORIGIN" 
  # "MEAN_LIGHTING" "ACTIVE_MAP"
  # "PICKUP" "DROP" "LOCAL_BINARY_PATTERNS" "FUNC_FRAME_START" "FUNC_FRAME_END"
  def last_app_state(self, key):
      # print(self.state[-1])
      return self.state[-1][key]

  # "RIGHT_GRIPPER_OPEN" "RIGHT_GRIPPER_CLOSED" "LEFT_GRIPPER_OPEN" "LEFT_GRIPPER_CLOSED"
  # "LEFT_GRIPPER" "RIGHT_GRIPPER" "PARKED_GRIPPER" "GRIPPER_WITH_CUBE" "CUBE"
  # "BOX" "DRIVABLE_GROUND" "ROBOT_LIGHT" "DO_PICKUP" "DO_DROPOFF" "DROP_CUBE_IN_BOX"
  def last_bb(self, key):
      return self.bb[key][-1]

  # "ACTION" "FRAME_PATH" "FRAME_NUM" "FUNCTION_NAME" "MEAN_DIF_LIGHTING"
  # "MOVE_ROTATION" "MOVE_STRAIGHT" "LOCATION" "ARM_STATE" "GRIPPER_STATE"
  # "PREDICTED_MOVE" "BOUNDING_BOX_LABELS" "BOUNDING_BOX"
  # "BOUNDING_BOX_PATH" "BOUNDING_BOX_IMAGE_SYMLINK" 
  def last_frame_state(self, key):
      try:
        state = self.frame_state[-1][key]
        return state
      except:
        return None

  # to reload predictions
  def get_action(self, fr_num):
      return self.frame_state[fr_num]["ACTION"]

  def get_blackboard(self, key):
      try:
        val = self.state[-1]["BLACKBOARD"][key]
      except:
        val = None
      return val

  def set_blackboard(self, key, value):
      self.state[-1]["BLACKBOARD"][key] = value

  def get_robot_light(self, frame_num=None):
      if frame_num is None:
        frame_num = self.frame_num
      rl = copy.deepcopy(self.frame_state[frame_num]["ROBOT_LIGHT"])
      return rl

  def record_robot_light(self, rl, frame_num=None):
      if frame_num is None:
        frame_num = self.frame_num
      self.frame_state[frame_num]["ROBOT_LIGHT"]  = copy.deepcopy(rl)

  def get_robot_light_img(self, full_img_path, rl):
      rlimg = cv2.imread(full_img_path)
      center = np.uint8(rl["CENTER"].copy())
      rl_copy = rl["LABEL"].copy()
      res    = center[rl_copy.flatten()]
      rlimg  = res.reshape((rlimg.shape[:2]))
      return rlimg

  def filter_rl_from_bb_img(self, bb, bbimg, rl):
      IMG_HW = 224
      # rl is full-image. bbimg is only for bb subset.  Use bb to get bb_mask 
      # and rl_img subset. 
      rl_img = bbimg.copy()
      mask = rl["LABEL"].copy()
      mask = mask.reshape((IMG_HW,IMG_HW))
      bb_maxw, bb_minw, bb_maxh, bb_minh = get_min_max_borders(bb)
      bb_mask = np.zeros((bb_maxh-bb_minh, bb_maxw-bb_minw), dtype="uint8")
      bb_mask[0:bb_maxh-bb_minh,0:bb_maxw-bb_minw]=mask[bb_minh:bb_maxh,bb_minw:bb_maxw]
      rl_img_hsv  = cv2.cvtColor(rl_img.copy(), cv2.COLOR_BGR2HSV)
      rl_h, rl_s, rl_v = cv2.split(rl_img_hsv)
      # rl_mean_h = cv2.mean(rl_h)[0]
      # rl_mean_s = cv2.mean(rl_s)[0]
      rl_mean_v = cv2.mean(rl_v)[0]

      nonlight = 1 - rl["LIGHT"]
      print("light cnt in bb:", nonlight, np.count_nonzero(bb_mask), len(bb_mask)*len(bb_mask[0]), bb)
      # (non)light_mean_v is the mean just for the bb subset.
      nonlight_mean_v = np.mean(rl_v[bb_mask==nonlight])
      light_mean_v = np.mean(rl_v[bb_mask==rl["LIGHT"]])
      print("3 nonlight mean v ", nonlight_mean_v)
      print("3b   light mean v ", light_mean_v)
      # if nonlight_mean_v / light_mean_v > .7:
      # # record the values, figure out the proper ratio
      # # gap not big enough; Robot Light might not be a factor.
      # # The means change over time. For example, when the dark floor starts to be
      # # seen as the robot gets closer to the edge of a white table, then the ratio 
      # # will go down. Interestingly, even when doing just the bb subset that doesn't
      # # have the dark floor, the bb light mean ration goes down.
      # # We need to stop filtering out the light at some point to prevent wrong H/V mse
      # # results.
      # # 
      # # New Approach: Run twice; with RL and without RL and choose the best mse.
      #  print("light mean ratios too big: turn of light filtering") 
      # return bbimg, False
      rl_v[bb_mask==rl["LIGHT"]] = nonlight_mean_v
      rl_img = cv2.merge([rl_h, rl_s, rl_v])
      # cv2.imshow("3MEANNONLIGHT", rl_img)
      # cv2.waitKey(0)

      return rl_img
      # return rl_img, True


