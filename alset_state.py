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

      # Global VARIABLES
      self.location  = None
      self.state     = []
      self.app_name  = None
      self.dsu       = None
      self.func_state = None
      self.run_num   = -1
      self.func_num  = -1
      self.frame_num = -1
      self.arm_state = None
      self.gripper_state = None

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
      self.state[run_num]["LOCATION_ORIGIN"] = None
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
      self.bb["BOX"]                  = []
      self.bb["DRIVABLE_GROUND"]      = []
      self.bb["ROBOT_LIGHT"]          = []
      self.bb["DO_PICKUP"]            = []
      self.bb["DO_DROPOFF"]           = []
      self.bb["DROP_CUBE_IN_BOX"]     = []
      # self.bb["RIGHT_GRIPPER_C_X_Y"]  = []  # fill in each O/C, X,Y
      # self.bb["START_CUBE_GRAB"]      = []  # Done later

      # For lbp pattern matching
      # at run-level, global lbp pattern matching is supported
      # at frame level, store frame# for lbp
      # get_lbp_match(lbp)
      self.lbp_state["DRIVABLE_GROUND"] = []
      self.lbp_state["BOX"]         = []
      self.lbp_state["CUBE"]        = []

      # need the following history in a list for simple statistical analysis
      self.actions["LEFT"]    = []
      self.actions["RIGHT"]   = []
      self.actions["FORWARD"] = []
      self.actions["REVERSE"] = []

      # pointer to current value.  History by traversing moves. 
      self.location = None
      self.run_id   = None

  def record_frame_info(self, app, mode, nn_name, action, img_nm, full_img_path):
      self.frame_num += 1
      # Prev Func State
      # Current Func State
      # print("record_frame_info: ",len(self.func_state))
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
      self.frame_state[-1]["ACTION"]         = action
      self.frame_state[-1]["FRAME_PATH"]     = full_img_path
      self.frame_state[-1]["FRAME_NUM"]      = self.frame_num
      self.frame_state[-1]["FUNCTION_NAME"]  = nn_name
      print("bb", len(self.bb))
      self.frame_state[-1]["MEAN_DIF_LIGHTING"] = nn_name
      self.frame_state[-1]["MOVE_ROTATION"] = None
      self.frame_state[-1]["MOVE_STRAIGHT"] = None
      # Robot-related state
      self.frame_state[-1]["LOCATION"]       = {}
      self.frame_state[-1]["ARM_STATE"]      = {}   # "O"/ "C", open/close count
      self.frame_state[-1]["GRIPPER_STATE"]  = {}
      self.frame_state[-1]["KNOWN_STATE"]    = None # "ARM_RETRACTED", "DELTA"
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
      orig_img,orig_mean_diff,orig_rl_bb = self.cvu.adjust_light(orig_image_path)
      print("bb:", bb)
      maxw, minw, maxh, minh = get_min_max_borders(bb)
      orig_maxh, orig_maxw, c = orig_img.shape
      maxh = min(maxh, orig_maxh)
      maxw = min(maxw, orig_maxw)
      # print("maxh-minh, maxw-minw:",maxh-minh, maxw-minw,  maxw, minw, maxh, minh )
      bb_img = np.zeros((maxh-minh, maxw-minw, 3), dtype="uint8")
      bb_img[0:maxh-minh, 0:maxw-minw, :] = orig_img[minh:maxh, minw:maxw, :]
      return bb_img

  def record_bounding_box(self, label_list, bb, frame_num=None):
      if frame_num is None:
        frame_num = self.frame_num
      self.frame_state[frame_num]["BOUNDING_BOX"].append(bb.copy())
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
          bb_img = self.get_bounding_box_image(orig_img_path, bb)
          retkey, encoded_image = cv2.imencode(".jpg", bb_img)
          with open(bb_file, 'wb') as f:
            f.write(encoded_image)
          self.frame_state[frame_num]["BOUNDING_BOX_PATH"].append(bb_file)
          sl_path = self.dsu.bounding_box_symlink_path(label)
          sl_file = sl_path+self.dsu.bounding_box_file(self.run_id,frame_num,bb_idx)
        else:
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

  def record_arm_state(self, state, arm_cnt, blackboard):
      self.arm_state["ARM_POSITION"] = state
      self.arm_state["ARM_POSITION_COUNT"] = arm_cnt.copy()
      self.blackboard["ARM"] = blackboard.copy()
      print("record_arm_state:", self.arm_state["ARM_POSITION"], self.arm_state["ARM_POSITION_COUNT"])

  def record_drivable_ground(self, bb, img):
      self.record_bounding_box(["DRIVABLE_GROUND"], bb)
      self.record_lbp("DRIVABLE_GROUND", img)

  def record_lbp(self, key, image, frame_num=None):
      locbinpat = self.lbp.register(key, image, run_id=self.run_id, frame_num=frame_num)
      self.lbp_state[key].append(locbinpat)

  def record_lighting(self, mean_lighting, mean_dif, robot_light_bb):
      if robot_light_bb is not None:
        self.record_bounding_box(["ROBOT_LIGHT"], robot_light_bb.copy())
      if mean_dif is not None:
        self.frame_state[-1]["MEAN_DIF_LIGHTING"] = mean_dif
      # overwrite with most recent
      if mean_lighting is not None:
        self.state[self.run_num]["MEAN_LIGHTING"] = mean_lighting 

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
        print("not in mappable position; known state, action", self.frame_state[-1]["KNOWN_STATE"], action)

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
      bb = g_s["BOUNDING_BOX"].copy()  # float
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

  # move to analyze_move.py?
  # This following is more analyzis than recording/loading of info 
  def record_pickup_info(self):
      print("state", len(self.state))
      print("run_state", len(self.run_state))
      print("func_state", len(self.func_state))
      print("bb", len(self.bb))
      print("lbp", len(self.lbp_state))
      print("gripper_state", len(self.gripper_state))
      print("arm_state", len(self.arm_state))

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
        # puc["ORIGIN"]      = fr_state["ORIGIN"].copy()
        # puc["LOCATION"]    = fr_state["LOCATION"].copy()
        # puc["ORIENTATION"] = fr_state["ORIENTATION"]
        # print("PUC:", puc)
        print("start/end func frame:", start_pickup, end_pickup)
        for fr_num in range(start_pickup, end_pickup):
          rlbb = None
          fr_state = self.frame_state[fr_num]
          action = fr_state["ACTION"]
          print("action:", action, fr_num)
          if action in ["FORWARD","REVERSE","LEFT","RIGHT"]:
            # reset location
            puc["ROBOT_ORIGIN"]      = fr_state["ORIGIN"].copy()
            puc["ROBOT_LOCATION"]    = fr_state["LOCATION"].copy()
            puc["ROBOT_ORIENTATION"] = fr_state["ORIENTATION"]
            start_cube_grab_fr_num = None
            end_cube_grab_fr_num = None
          elif action in ["UPPER_ARM_UP", "UPPER_ARM_DOWN", "LOWER_ARM_UP", "LOWER_ARM_DOWN"]:
           
            print("fs arm state",  fr_state["ARM_STATE"])
            print("arm_pos:", fr_state["ARM_STATE"]["ARM_POSITION"], fr_state["ARM_STATE"]["ARM_POSITION_COUNT"])
            puc["ARM_POSITION"] = fr_state["ARM_STATE"]["ARM_POSITION"]
            puc["ARM_POSITION_COUNT"] = fr_state["ARM_STATE"]["ARM_POSITION_COUNT"].copy()
            start_cube_grab_fr_num = None
            end_cube_grab_fr_num = None
          elif action in ["GRIPPER_OPEN"]:
            start_cube_grab_fr_num = None
            end_cube_grab_fr_num = None
          elif action in ["GRIPPER_CLOSE"]:
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
              start_cube_grab_bb = gripper_w_cube_bb.copy()
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
            # max_left_gripper_bounding_box = [[[0, int(y*.4)]], [[0, y-1]],
            #                 [[ int(x/2), y-1 ]], [[ int(x/2), int(y*.4)]]]
            # max_right_gripper_bounding_box =[[[int(x/2),int(y*.4)]],[[x-1,int(y*.4)]],
            #                               [[x-1, y-1]], [[int(x/2), y-1]]]
            # [[[49, 48]], [[174, 48]], [[49, 285]], [[174, 285]]]
            # 
            print("lgbb:",lgbb)
            print("rgbb:",rgbb)
            cube_size2 = puc["CUBE_SIZE"]

            cube_top_left  = [int(max(lgbb[2][0][0] - cube_size2/2,0)),int(lgbb[1][0][1] + cube_size2/2)]
            cube_top_right = [int(rgbb[0][0][0] + cube_size2/2),int(lgbb[1][0][1] + cube_size2/2)]
            cube_bot_left  = [int(max(lgbb[2][0][0] - cube_size2/2,0)),int(max(lgbb[0][0][1] - cube_size2/2,0))]
            cube_bot_right = [int(rgbb[0][0][0] + cube_size2/2),int(max(rgbb[0][0][1] - cube_size2/2,0))]

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
              new_cube_lw = int(rgmaxw/2 - cube_size/2)
              new_cube_rw = int(rgmaxw/2 + cube_size/2)
              if (cube_top_right[0] < new_cube_rw):
                cube_top_left[0]  = new_cube_lw
                cube_bot_right[0] = new_cube_rw
                cube_top_right[0] = new_cube_rw
                cube_bot_left[0]  = new_cube_lw
                print("adj cube 3")

            cube_bb        = [[cube_bot_left],[cube_bot_right],
                              [cube_top_left],[cube_top_right]]
            print("cube_bb:",cube_bb)
            print("cube_sz:",cube_size)

            curr_img_path = self.frame_state[fr_num]["FRAME_PATH"]
            prev_img_path = self.frame_state[fr_num-1]["FRAME_PATH"]
            cube_img = self.get_bounding_box_image(curr_img_path, cube_bb)

            self.record_bounding_box(["CUBE"], cube_bb, fr_num)
            gripper_w_cube_img = self.get_bounding_box_image(curr_img_path, gripper_w_cube_bb)
            self.record_bounding_box(["CUBE", "GRIPPER_WITH_CUBE"], gripper_w_cube_bb, fr_num)
            cv2.imshow("cube bb1", cube_img)
            cv2.imshow("gripper_w_cube bb1", gripper_w_cube_img)
            square = find_square(cube_img.copy())
            cube = find_cube(cube_img.copy(), self)
            print("cube bb, square, cube:", cube_bb, square, cube)
            cv2.waitKey(0)

            # opt_flw = self.cvu.optflow(prev_img_path, curr_img_path, thresh = 0.05)
            opt_flw = self.cvu.optflow(prev_img_path, curr_img_path, thresh = self.cfg.OPTFLOWTHRESH/10)
            if not opt_flw:
              puc["GRAB_CUBE_POSITION"] = ["GRIPPER_WITH_CUBE"]
              print("not opt flow:", fr_num, opt_flw)
              # continue
              # break
        if puc["GRAB_CUBE_POSITION"] != ["GRIPPER_WITH_CUBE"]:
          print("forcing GRAB_CUBE_POSITION to be GRIPPER_WITH_CUBE. Was:", puc["GRAB_CUBE_POSITION"])
          puc["GRAB_CUBE_POSITION"] = ["GRIPPER_WITH_CUBE"]
        # we've now scanned the potential actions that resulted in the cube being
        # grabbed by the gripper.  Store the results for this and other 1-time events.
        print("PUC2:", puc)
        if puc["GRAB_CUBE_POSITION"] == ["GRIPPER_WITH_CUBE"]:
          # finalize start/end cube grab frames
          start_cube_grab_fr_num = last_start_cube_grab_fr_num
          end_cube_grab_fr_num = last_end_cube_grab_fr_num
          puc["START_CUBE_GRAB_FRAME_NUM"] = start_cube_grab_fr_num
          puc["END_CUBE_GRAB_FRAME_NUM"] = end_cube_grab_fr_num
          self.record_bounding_box(["START_CUBE_GRAB"], start_cube_grab_bb, fr_num)

          print("start_cube_grab_fr_num:", start_cube_grab_fr_num)
          start_cube_grab_img_path = self.frame_state[start_cube_grab_fr_num]["FRAME_PATH"]
          start_cube_grab_img = self.get_bounding_box_image(start_cube_grab_img_path, start_cube_grab_bb)
          cv2.imshow("start_cube_grab", start_cube_grab_img)

          # do CUBE BB checks
          curr_img_path = self.frame_state[start_cube_grab_fr_num]["FRAME_PATH"]
          if abs(puc["MEAN_DIF_LIGHTING"]) > .1 * abs(self.state[self.run_num]["MEAN_LIGHTING"]):
            print("GRIPPER_W_CUBE: big change in lighting")
            rlbb = self.get_bb("ROBOT_LIGHT",start_cube_grab_fr_num)
            rlimg = self.get_bounding_box_image(curr_img_path, rlbb)
            cv2.imshow("robot light bb", rlimg)
            cv2.waitKey(0)
            square = find_square(rlimg)
            print("cube bb, square:", cube_bb, square)

            if check_gripper_bounding_box(rlbb, cube_bb):
              print("GRIPPER_W_CUBE: cube and robot light overlap", rlbb, cube_bb)
            else:
              print("WARNING: GRIPPER_W_CUBE: no cube and robot light overlap", rlbb, cube_bb)
          else:
            print("WARNING: GRIPPER_W_CUBE: no big change in lighting")

          self.record_bounding_box(["CUBE"], cube_bb, fr_num)
          cube_img_path = self.frame_state[end_cube_grab_fr_num]["FRAME_PATH"]
          cube_img = self.get_bounding_box_image(curr_img_path, cube_bb)
          self.record_bounding_box(["CUBE", "GRIPPER_WITH_CUBE"], gripper_w_cube_bb, fr_num)
          self.record_bounding_box(["CUBE"], cube_bb, fr_num)
          gripper_w_cube_img = self.get_bounding_box_image(curr_img_path, gripper_w_cube_bb)
          # ARD: cube_img is more like left gripper...
          cv2.imshow("cube bb2", cube_img)
          cv2.imshow("gripper_w_cube bb2", gripper_w_cube_img)
          cv2.waitKey(0)

        # Now, track backwards during cube pickup
        print("start,end:", start_cube_grab_fr_num, end_cube_grab_fr_num)
        if end_cube_grab_fr_num is None or start_cube_grab_fr_num is None:
          end_cube_grab_fr_num = -1
          start_cube_grab_fr_num = 0
 
        for fr_num in range(end_cube_grab_fr_num, start_cube_grab_fr_num, -1):
          lgbb= self.get_bb("LEFT_GRIPPER",fr_num)
          rgbb= self.get_bb("RIGHT_GRIPPER",fr_num)
          # should be between the grippers
          # cube_top_left  = [int(lgbb[3][0][0]),int(lgbb[3][0][1] - cube_size2/2)]
          # cube_top_right = [int(rgbb[0][0][0]),int(lgbb[3][0][1] - cube_size2/2)]
          # cube_bot_left  = [int(lgbb[3][0][0]),int(lgbb[3][0][1] + cube_size2/2)]
          # cube_bot_right = [int(rgbb[0][0][0]),int(rgbb[0][0][1] + cube_size2/2)]
          # cube_bb = [[cube_top_left],[cube_top_right], [cube_bot_left],[cube_bot_right]]
          cube_top_left  = [int(lgbb[2][0][0] - cube_size2/2),int(lgbb[1][0][1] + cube_size2/2)]
          cube_top_right = [int(rgbb[0][0][0] + cube_size2/2),int(lgbb[1][0][1] + cube_size2/2)]
          cube_bot_left  = [int(lgbb[2][0][0] - cube_size2/2),int(lgbb[0][0][1] - cube_size2/2)]
          cube_bot_right = [int(rgbb[0][0][0] + cube_size2/2),int(rgbb[0][0][1] - cube_size2/2)]
          cube_bb        = [[cube_bot_left],[cube_bot_right],
                            [cube_top_left],[cube_top_right]]


          cube_img = self.get_bounding_box_image(curr_img_path, cube_bb)

          # do square check, cube_size should be about same size
          if cube_img is not None:
            square = find_square(cube_img)
          print("cube bb, square:", cube_bb, square, cube_size)

          # do light check
          curr_img_path = self.frame_state[fr_num]["FRAME_PATH"]
          if abs(puc["MEAN_DIF_LIGHTING"]) > .1 * abs(self.state[self.run_num]["MEAN_LIGHTING"]):
            print("GRIPPER_W_CUBE: big change in lighting")
            rlbb = self.get_bb("ROBOT_LIGHT",fr_num)
            rlimg = self.get_bounding_box_image(curr_img_path, rlbb)
            cv2.imshow("robot light bb", rlimg)

            if check_gripper_bounding_box(rlbb, cube_bb):
              print("GRIPPER_W_CUBE: cube and robot light overlap", rlbb, cube_bb)
            else:
              print("WARNING: GRIPPER_W_CUBE: no cube and robot light overlap", rlbb, cube_bb)
          else:
            print("WARNING: GRIPPER_W_CUBE: no big change in lighting")

          cube_img_path = self.frame_state[fr_num]["FRAME_PATH"]
          cube_img = self.get_bounding_box_image(curr_img_path, cube_bb)
          gripper_w_cube_img = self.get_bounding_box_image(curr_img_path, gripper_w_cube_bb)

          # do LBP check
          gray_cube_img = cv2.cvtColor(cube_img, cv2.COLOR_BGR2GRAY)
          lbp_name, lbp_score = self.lbp.match(gray_cube_img)
          if lbp_name.startswith("CUBE"):
            print("pass LBP check", lbp_name, lbp_score)
          self.record_lbp("CUBE", gray_cube_img, frame_num=fr_num)

          # do square check
          square = find_square(cube_img)
          cube = find_cube(cube_img)
          print("cube bb, square, cube:", cube_bb, square, cube)
          cv2.imshow("cube bb", cube_img)
          cv2.imshow("gripper_w_cube bb", gripper_w_cube_img)
          cv2.waitKey(0)

        next_cube_bb = cube_bb.copy()
        next_cube_size = cube_size
        est_cube_bb = None
        # Now, track backwards using Light BB
        # for fr_num in range(start_pickup, 0, -1):
        for fr_num in range(start_cube_grab_fr_num, 0, -1):
          curr_img_path = self.frame_state[fr_num]["FRAME_PATH"]
          next_img_path = self.frame_state[fr_num+1]["FRAME_PATH"]
          curr_img,curr_mean_diff,curr_rl_bb = self.cvu.adjust_light(curr_img_path)
          next_img,next_mean_diff,next_rl_bb = self.cvu.adjust_light(next_img_path)

          if curr_rl_bb is not None and next_rl_bb is not None:
            curr_rl_ctr_x, curr_rl_ctr_y = bounding_box_center(curr_rl_bb)
            print("curr_rl_ctr:", curr_rl_ctr_x, curr_rl_ctr_y)
            print("next rl_ctr:", next_rl_ctr_x, next_rl_ctr_y)

            # estimated cube location
            est_cube_bb = curr_rl_bb.copy() # initialize shape
            for i in range(4):
              est_cube_bb[i][0] = next_cube_bb[i][0] + curr_rl_ctr_x - next_rl_ctr_x
              est_cube_bb[i][1] = next_cube_bb[i][1] + curr_rl_ctr_y - next_rl_ctr_y
            print("est cube bb:", est_cube_bb)

            # pad the bb to handle margin of error
            est_cube_bb[0][0] -= cube_size
            est_cube_bb[0][1] -= cube_size
            est_cube_bb[1][0] += cube_size
            est_cube_bb[1][1] -= cube_size
            est_cube_bb[2][0] -= cube_size
            est_cube_bb[2][1] += cube_size
            est_cube_bb[3][0] += cube_size
            est_cube_bb[3][1] += cube_size
            print("est padded cube bb:", est_cube_bb)

            est_cube_img = self.get_bounding_box_image(curr_img_path, est_cube_bb)
            # do square check
            square = find_square(est_cube_img)
            print("RL: cube bb, square:", est_cube_bb, square)
            cv2.imshow("est cube bb", est_cube_img)
            cv2.waitKey(0)

            # do LBP check
            gray_cube_img = cv2.cvtColor(cube_img, cv2.COLOR_BGR2GRAY)
            lbp_name, lbp_score = self.lbp.match(gray_cube_img)
            if lbp_name.startswith("CUBE"):
              print("pass LBP check", lbp_name, lbp_score)
              self.record_bounding_box(["CUBE"], est_cube_bb, fr_num)
              self.record_lbp("CUBE", gray_cube_img, frame_num=fr_num)
    
          else:
            print("light check fails")
            # cv2.imshow("curr_img", curr_img)
            # cv2.imshow("next_img", next_img)
            # cv2.waitKey(0)
            # continue
            # break

# ARD: todo: update size once smaller cube image has been figured out
        next_cube_size = cube_size 
        if est_cube_bb is not None:
          next_cube_bb = est_cube_bb.copy()
        else:
          frn = puc["START_CUBE_GRAB_FRAME_NUM"] 
          next_cube_bb = self.get_bb("GRIPPER_WITH_CUBE",frn)

        # Now, track backwards to first time cube is seen based on robot movement
        # and keypoints
        # for fr_num in range(start_pickup, 0, -1):
        for fr_num in range( start_cube_grab_fr_num, 0, -1):
          # find keypoints on "next" bb (working backwards) 
          # compare "next" keypoints to find "curr" keypoints

          # Do robot movement comparison
          # if ARM movement, then only goes up/down
          curr_img_path = self.frame_state[fr_num]["FRAME_PATH"]
          curr_image,mean_diff,rl_bb = self.cvu.adjust_light(curr_img_path)
          curr_img_KP = Keypoints(curr_image)
          next_img_path = self.frame_state[fr_num+1]["FRAME_PATH"]
          next_image,mean_diff,rl_bb = self.cvu.adjust_light(next_img_path)
          next_img_KP = Keypoints(next_image)

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
            if point_in_border(next_cube_bb,next_kp.pt, 0):
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
            kp_dif_x /= len(next_kp_list)
            kp_dif_y /= len(next_kp_list)
            print("kp_dif: ", kp_dif_x, kp_dif_y)
          print("directional best kp dif: ", direction_kp_x, direction_kp_y)

          # also compute delta based on estimated movement
          next_img_path = self.frame_state[fr_num+1]["FRAME_PATH"]
          mv_dif_x =  None
          mv_dif_y =  None
          print("action:", self.frame_state[fr_num]["ACTION"])
          if self.is_mappable_position(self.frame_state[fr_num]["ACTION"], fr_num):
            next_img,next_mean_dif,next_rl_bb = self.cvu.adjust_light(next_img_path)
            mv_rot = self.frame_state[-1]["MOVE_ROTATION"] 
            mv_pos = self.frame_state[-1]["MOVE_STRAIGHT"] 
            # convert movement from birds_eye_view perspective back to camera perspective
            if mv_rot is not None:
              mv_dif_x =  mv_pos / (self.BIRDS_EYE_RATIO_H+1)
              mv_dif_y =  0
            elif mv_rot is not None:
              curr_cube_ctr = bounding_box_center(curr_cube_bb)
              robot_pos = [(-self.MAP_ROBOT_H_POSE / (self.BIRDS_EYE_RATIO_H+1)), int(next_img.shape[1]/2) ]
              next_angle = rad_arctan2(curr_cube_ctr[0]-robot_pos[0], curr_cube_ctr[1]-robot_pos[1])
              curr_angle = next_angle + mv_rot
              dist = np.sqrt((curr_cube_ctr[0] - robot_pos[0])**2 + (curr_cube_ctr[1] - robot_pos[1])**2)

              mv_dif_x = round(dist * math.cos(curr_angle))
              mv_dif_y = round(dist * math.sin(curr_angle))
              print("mv_dif: ", mv_dif_x, mv_dif_y)
          else:
            if not(action.startswith("UPPER_ARM") or action.startswith("LOWER_ARM")):
              print("computing mse for action:", action)
            next_img,next_mean_dif,next_rl_bb = self.cvu.adjust_light(next_img_path)
            curr_img,curr_mean_dif,curr_rl_bb = self.cvu.adjust_light(curr_img_path)
            
            lgbb = self.get_bb("LEFT_GRIPPER",fr_num)
            rgbb = self.get_bb("RIGHT_GRIPPER",fr_num)
            if lgbb is None or rgbb is None:
              lg_label, rg_label = self.get_gripper_pos_label(fr_num)
            if lgbb is None:
              lgbb = self.get_avg_bb(lg_label)
            if rgbb is None:
              rgbb = self.get_avg_bb(rg_label)
            print("lgbb,rgbb", lgbb, rgbb)
            black = [0,0,0]
            for x in range(lgbb[0][0][0], lgbb[1][0][0]):
              for y in range(lgbb[0][0][1], lgbb[1][0][1]):
                next_img[x][y] = black
                curr_img[x][y] = black
            for x in range(rgbb[0][0][0], rgbb[1][0][0]):
              for y in range(rgbb[0][0][1], rgbb[1][0][1]):
                next_img[x][y] = black
                curr_img[x][y] = black
            arm_dif_x = None
            arm_dif_y = None
            minmse = self.cfg.INFINITE
            for ny in range(0, next_img.shape[0]-10):
              offtop = next_img.shape[0] - ny
              #(223,224,3) vs (224,223,3) 
              mse = np.sum((curr_img[:,ny:].astype("float") - next_img[:,:offtop].astype("float")) ** 2)
              mse /= np.sum((curr_img[:,ny:].astype("float")) ** 2)
              print("Arm Dif: ", arm_dif_y, mse, minmse)
              if minmse > mse:
                minmse = mse
                arm_dif_x = 0
                arm_dif_y = ny
                print("Arm Dif: ", arm_dif_y, minmse)
            print("arm_dif: ", arm_dif_x, arm_dif_y, fr_num)

          if mv_dif_y is not None:
            dx = mv_dif_x
            dy = mv_dif_y
            print("mv dif wins:", dx, dy)
          elif arm_dif_y is not None:
            dx = arm_dif_x
            dy = arm_dif_y
            print("arm dif wins:", dx, dy)
          elif kp_dif_y is not None:
            dx = kp_dif_x
            dy = kp_dif_y
            print("kp dif wins:", dx, dy)
          else:
            print("ERROR: no estimated cube location")
        
          for i in range(5):
            curr_cube_bb = self.get_bb("CUBE",fr_num+i)
            if curr_cube_bb is not None:
              print("prev cube bb:", fr_num+i, i, curr_cube_bb)

          curr_img_path = self.frame_state[fr_num]["FRAME_PATH"]
          curr_img,curr_mean_dif,next_rl_bb = self.cvu.adjust_light(curr_img_path)
          for i in range(4):
            curr_cube_bb[i][0][0] += dx
            curr_cube_bb[i][0][1] += dy
            if curr_cube_bb[i][0][0] >= curr_img.shape[0]:
              curr_cube_bb[i][0][0] = curr_img.shape[0] - 1
            if curr_cube_bb[i][0][1] >= curr_img.shape[1]:
              curr_cube_bb[i][0][1] = curr_img.shape[1] - 1
          if curr_cube_bb is None:
            cv2.imshow("curr_img", curr_img)
            cv2.waitKey(0)
            curr_cube_img = None
            print("DIFF: cube bb:", curr_cube_bb)
            break
          else:
            curr_cube_img = self.get_bounding_box_image(curr_img, curr_cube_bb)
          print("DIFF: cube bb:", curr_cube_bb)
          # do square check
          if curr_cube_img is not None:
            cv2.imshow("curr_img", curr_img)
            cv2.imshow("next_img", next_img)
            cv2.imshow("MV cube bb", curr_cube_img)
            cv2.waitKey(0)
            square = find_square(curr_cube_img)
            print("DIFF: cube bb, square:", curr_cube_bb, square)
            gray_cube_img = cv2.cvtColor(curr_cube_img, cv2.COLOR_BGR2GRAY)
            self.record_bounding_box(["CUBE"], curr_cube_bb, fr_num)
            self.record_lbp("CUBE", gray_cube_img, frame_num=fr_num)

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

  def preprocess_object(img):
      img,mean_diff,rl_bb = cvu.adjust_light(img_path)
      orig_img = img.copy()
      img = cv2.Canny(img, 50, 200, None, 3)
      # thresh = 10
      thresh = 20
      img = cv2.GaussianBlur(img, (5, 5), 0)
      # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      gray_img = cv2.dilate(img,None,iterations = 2)
      gray_img = cv2.threshold(gray_img, thresh, 255, cv2.THRESH_BINARY_INV)[1]
      # gray_img = cv2.bitwise_not(gray_img)
      imagecontours, hierarchy = cv2.findContours(gray_img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
      if len(imagecontours) > 1:
        # print("hierarchy:", hierarchy)
        for i, c  in enumerate(imagecontours):
          area = cv2.contourArea(c)
          M = cv2.moments(c)
          print(i, "area, moment:", area, M, len(c))
          print(i, "area:", area, len(c))
      for count in imagecontours:
        # epsilon = 0.01 * cv2.arcLength(count, True)
        epsilon = 0.01 * cv2.arcLength(count, True)
        approximations = cv2.approxPolyDP(count, epsilon, True)
        # e.g. [[[224 224]] [[252 372]] [[420 372]] [[447 224]]]
        #the name of the detected shapes are written on the image
        if len(approximations) == 3:
        shape = "Trapezoid"
        area = cv2.contourArea(approximations)
        sqimg2 = sqimg.copy()
        cv2.drawContours(sqimg2, imagecontours, -1, (0,255,0), 3)
        cv2.imshow("contours", sqimg2)
        cv2.waitKey()




      return gray_img, orig_img

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

  def get_robot_arm_position(self):
      return self.arm_state["ARM_STATE"], self.arm_state["ARM_POSITION_COUNT"]

  def get_robot_gripper_position(self):
      return

  def is_mappable_position(self, action, fr_num = None):
      # don't update map / location
      # combine history with optical flow estimates
      if action in ["FORWARD", "REVERSE", "LEFT", "RIGHT"]:
        if fr_num is None:
          fr_num = -1
        if self.frame_state[fr_num]["KNOWN_STATE"] == "ARM_RETRACTED":
          return True
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
      self.location["ORIGIN"]      = origin.copy()
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
  def set_known_robot_state(self, frame_num, new_known_state):
      self.state_history.append([self.robot_state[:]])
      self.robot_state["KNOWN_STATE"] = new_known_state
      self.robot_state["DELTA"] = False
      for action in self.arm_actions_no_wrist:
        self.robot_state[action] = 0

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
      except:
        return self
      # print("load: False")
      return self

  # "APP_NAME" "APP_RUN_NUM" "APP_RUN_ID" "APP_MODE" "FUNC_STATE" "FRAME_STATE" 
  # "GRIPPER_STATE" "BOUNDING_BOXES" "ACTIONS" "FINAL_MAP" "LOCATION_ORIGIN" 
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
  # "KNOWN_STATE" "PREDICTED_MOVE" "BOUNDING_BOX_LABELS" "BOUNDING_BOX"
  # "BOUNDING_BOX_PATH" "BOUNDING_BOX_IMAGE_SYMLINK" 
  def last_frame_state(self, key):
      return self.frame_state[-1][key]

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
