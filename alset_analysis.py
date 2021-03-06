from config import *
from PIL import Image
from dataset_utils import *
from func_app import *
from analyze_gripper import *
from analyze_arm import *
from alset_state import *
from analyze_map import *
from analyze_move import *
from analyze_clusters import *
from alset_ratslam import *
from alset_stego import *
import gc

class DSAnalysis():
  def __init__(self, app_name="TT", app_type="APP"):
      self.dsu = DatasetUtils(app_name, "ANALYZE")
      self.app_name = app_name
      self.app_type = app_type
      self.run_id   = None
      self.run_num  = 0
      self.func_app = None
      self.start_with_func_index = None
      self.cfg = Config()
      self.alset_state = AlsetState()
      self.line_analysis = AnalyzeLines()
      self.stego = Stego()
      self.stego_cv_img = None 
      self.stego_img = None
      self.prev_stego_img = None
      self.stego_plot_img = None
      ## find potential clusters including floor, keypoints, potential COIs, obstacles, BG movements
      ## runs YOLO9000, segmentation
      ## detailed analysis of cube
      # self.cube_analysis = AnalyzeCube()
      ## detailed analysis of box
      # self.box_analysis = AnalyzeBox
      ## detailed analysis of box
      # self.face_analysis = AnalyzeFace
      self.MAX_MOVES_REWARD = None
      self.alset_state = self.alset_state.load_state(self.dsu.alset_state_filename(app_name))
      if self.alset_state is None:
        print("self.alset_state is None")
      state_loaded = False
      if self.alset_state.app_name is not None:
        # reinitialize and load state to current frame to resume processing
        print("alset_analyzis.load_state")
        self.load_state()
        self.app_ds_idx = self.dsu.dataset_indices(mode="ANALYZE", nn_name=self.app_name, position="START_FROM", start_from_idx=self.start_with_func_index)
        self.map_analysis = AnalyzeMap(self.alset_state)
        self.map_analysis.load_state()
        self.move_analysis = AnalyzeMove(self.alset_state)
        self.move_analysis.load_state()
        self.gripper_analysis = AnalyzeGripper(self.alset_state)
        self.gripper_analysis.load_state()
        self.arm_analysis = AnalyzeArm(self.alset_state)
        self.arm_analysis.load_state()
        self.cvu = CVAnalysisTools(self.alset_state)
        self.cvu.load_state() # note: does load automatically
        self.alset_state.load_state_completed()
        self.line_analysis = AnalyzeLines()
        # ARD TODO: (re)store line analysis state
        self.line_analysis.load_state()
        state_loaded = True
      while True:
        ## predict next move
        self.analyze_clusters = AnalyzeClusters(self.alset_state)
        restart_alset_ratslam()

        ## Analyze the next app dataset
        if not state_loaded:
          # just reinitialize
          self.cvu = CVAnalysisTools(self.alset_state)
          self.map_analysis = AnalyzeMap(self.alset_state)
          self.move_analysis = AnalyzeMove(self.alset_state)
          self.gripper_analysis = AnalyzeGripper(self.alset_state)
          self.arm_analysis = AnalyzeArm(self.alset_state)
          self.alset_state.init_app_run(app_run_num=self.run_num, app_name=self.app_name, app_mode=self.app_type)
        res = self.parse_app_dataset()
        self.alset_state.reset_frame_num()
        res2 = self.alset_state.post_run_analysis()
        if res is None or not res2:
          break
        self.run_num += 1
        self.alset_state.reset_restart()
        self.alset_state.init_app_run(app_run_num=self.run_num, app_name=self.app_name, app_mode=self.app_type)

  # TODO: largely cloned from alset_dqn.py.  Should cleanup/parameterize a function callback.
  def parse_func_dataset(self, NN_name, app_mode="FUNC"):

      print("PFD: >>>>> parse_func_dataset")

      ###################################################
      # iterate through NNs and fill in the info
      while True:
          func_index = self.dsu.dataset_indices(mode="ANALYZE",nn_name=NN_name,position="NEXT")
          if func_index is None:
            print("PFD: parse_func_dataset: done")
            break
          print("Parsing FUNC idx", func_index)
          self.run_id = app_index[-13:-4]
          self.alset_state.record_run_id(func_index, self.run_id)
          run_complete = False
          line = None
          next_action = None
          next_line = None
          self.curr_phase = 0
          nn_filehandle = open(func_index, 'r')
          line = None
          while True: # iterate through Img frames in nn
            # read a single line
            next_line = nn_filehandle.readline()
            if not next_line:
                run_complete = True
                print("PFD: Function Index completed:", self.get_frame_num, NN_name, next_action)
                break
            # get action & next_action
            [tm, app, mode, next_nn_name, next_action, img_name, next_state] = self.dsu.get_dataset_info(next_line, mode="FUNC")
            if line is not None:
              [tm, app, mode, nn_name, action, img_name, state] = self.dsu.get_dataset_info(line, mode="FUNC")
              if action == "NOOP":
                line = next_line
                continue
              elif action == "REWARD1":
                print("PFD: Goto next NN; NOOP Reward, curr_NN", action, nn_name)
                line = next_line
                continue

              else:
                self.alset_state.increment_frame()
                if not self.alset_state.restarting():
                  self.alset_state.record_frame_info(app, mode, nn_name, action, img_name, state, self.stego_plot_img)
                  self.dispatch(action, state, next_state, done)
                else:
                  print("ratslam_replay")
                  self.analyze_map.alset_ratslam_replay()
                print("PFD: dispatch:", action, reward, done, next_action)
              if next_action == "REWARD1":
                # self.alset_state.increment_frame()
                if not self.alset_state.restarting():
                  self.alset_state.record_frame_info(app, mode, nn_name, action, img_name, state, self.stego_plot_img)
                  self.dispatch(action, state, next_state, done)
                done = True  # end of func is done in this mode
                print("PFD: completed REWARD phase2", self.get_frame_num(), next_action, reward, done)
              elif next_action in ["PENALTY1", "PENALTY2"]:
                # self.alset_state.increment_frame()
                if not self.alset_state.restarting():
                  prev_stego = self.stego_plot_img
                  self.dispatch(action, state, next_state, done)
                done = True  # end of func is done in this mode
                print("PFD: assessed run-ending PENALTY", next_action, reward, done)
              # add dummy 0 q_val for now. Compute q_val at end of run.
              q_val = 0
            if next_action not in ["REWARD1", "PENALTY1", "PENALTY2"]:
              line = next_line
          # close the pointer to that file
          nn_filehandle.close()

  # TODO: largely cloned from alset_dqn.py.  Should cleanup/parameterize a function callback.
  def parse_app_dataset(self, app_mode="APP"):
        print("PAD: >>>>> parse_app_dataset")
        # app_dsu = DatasetUtils(self.app_name, "ANALYZE")
        self.alset_state.reset_frame_num()
        reward = []
        # val = self.cfg.get_value(self.cfg.app_registry, self.app_name)
        # func_nn_list = val[1]
        self.func_app = FunctionalApp(app_name=self.app_name, app_type=self.app_type, alset_state=self.alset_state)
        done = False
        self.parse_app_ds_details = []

        ###################################################
        # iterate through NNs and fill in the active buffer
        app_index = self.dsu.dataset_indices(mode="ANALYZE",nn_name=None,position="NEXT")
        if app_index is None:
          print("PAD: parse_app_dataset: unknown NEXT index or DONE")
          return None
        # get run_id from APP_IDX (e.g., TT_APP_IDX_21_06_09b.txt)
        self.run_id = app_index[-13:-4]
        self.alset_state.record_run_id(app_index, self.run_id)
        print("PAD: Parsing APP idx", app_index, self.run_id)
        app_filehandle = open(app_index, 'r')
        run_complete = False
        line = None
        next_action = None
        next_line = None
        self.curr_phase = 0
        func_flow_reward = None
        first_time_through = True
        while True:  
          [func_flow_nn_name, func_flow_reward] = self.func_app.eval_func_flow_model(reward_penalty="REWARD1", init=first_time_through)
          first_time_through = False
          print("PAD: FUNC_FLOW_REWARD1:", func_flow_reward)
          # get the next function index 
          nn_idx = app_filehandle.readline()
          if not nn_idx:
            if func_flow_nn_name is None:
              if func_flow_reward == "REWARD1":
                print("PAD: FUNC_FLOW_REWARD2:", func_flow_reward)
                if next_action != "REWARD1":
                    print("PAD: Soft Error: last action of run is expected to be a reward:", func_flow_action, next_action)
                if func_flow_nn_name is None and func_flow_reward == "REWARD1":
                    print("PAD: FUNC_FLOW_REWARD2:", func_flow_reward)
                    # End of Func Flow. Append reward.
                    # add dummy 0 q_val for now. Compute q_val at end of run.
                    done = True
                    # local_frame_num += 1  # now maintained by alset_state
                    if not self.alset_state.restarting():
                      prev_stego = self.stego_plot_img
                      self.dispatch(action, state, next_state, done)
              run_complete = True
              print("PAD: Function flow complete", state, action, reward, next_state, done, q_val)
              break
            else:
              # don't train DQN with incomplete TT results; continue with next idx
              # ARD: TODO: differentiate APP_FOR_DQN and native DQN datasets
              # TT_APP_IDX_PROCESSED.txt vs. TT_APP_IDX_PROCESSED_BY_DQN.txt 
              if self.MAX_MOVES_REWARD is not None:
                if self.REWARD1_REWARD is None or self.REWARD1_REWARD == 0:
                   # then REWARD/PENALTY is not required; compute rewards
                   print("PAD: No final reward required (MAX_MOVE_REWARD only)")
                   run_complete = True
                   break
              print("PAD: Function Flow expected:", func_flow_nn_name, "but app index ended: ", app_index)
              # don't train DQN with incomplete TT results; continue with next idx
              # ARD: TODO: differentiate APP_FOR_DQN and native DQN datasets
              # TT_APP_IDX_PROCESSED.txt vs. TT_APP_IDX_PROCESSED_BY_DQN.txt
              self.dsu.save_dataset_idx_processed(mode = app_mode)
              return "INCOMPLETE_APP_RUN"
            break
          # print("last char:", nn_idx[-1:], "DONE")
          if nn_idx[-1:] == '\n':
            nn_idx = nn_idx[0:-1]
          # print("PAD: remove trailing newline1", nn_idx)
          # file_name = nn_idx[0:-1]   # remove carriage return
          # NN_name = app_dsu.dataset_idx_to_func(file_name)
          NN_name = self.dsu.get_func_name_from_idx(nn_idx)
          if NN_name != func_flow_nn_name:
            print("PAD: Func flow / index inconsistency:", NN_name, func_flow_nn_name)
          print("PAD: Parsing NN idx", nn_idx)
          # Parsing NN idx ./apps/FUNC/PARK_ARM_RETRACTED_WITH_CUBE/dataset_indexes/FUNC_PARK_ARM_RETRACTED_WITH_CUBE_21_05_15a.txt
          nn_filehandle = open(nn_idx, 'r')
          line = None
          compute_reward_penalty = False
          while True: # iterate through Img frames in nn
            # read a single line
            next_line = nn_filehandle.readline()
            print("next_line:", next_line)
            if not next_line:
                run_complete = True
                print("PAD: Function Index completed:", NN_name, next_action)

                break
            # get action & next_action (nn_name is None)
            [tm, app, mode, next_nn_name, next_action, img_name, next_state] = self.dsu.get_dataset_info(next_line, mode="FUNC")
            if line is not None:
              [tm, app, mode, NN_name, action, img_name, state] = self.dsu.get_dataset_info(line, mode="FUNC")
              if action == "NOOP":
                # Too common in initial test dataset. Print warning later?
                # print("NOOP action, NN:", action, NN_name)
                line = next_line
                continue
              elif action == "REWARD1":
                print("PAD: Goto next NN; NOOP Reward, curr_NN", action, NN_name)
                line = next_line
                continue
              else:
                if not self.alset_state.restarting():
                  print("amnais:", self.app_name, mode, NN_name, action, img_name, state)
                  self.alset_state.increment_frame()
                  self.alset_state.record_frame_info(self.app_name, mode, NN_name, action, img_name, state, self.stego_plot_img)
                  # the final compute reward sets done to True (see above)
                  prev_stego = self.stego_plot_img
                  self.dispatch(action, state, next_state, done)
                  print("PAD: dispatch:", self.get_frame_num(), action, reward, done)
                else:
                  self.alset_state.increment_frame()
              if next_action == "REWARD1" and func_flow_reward == "REWARD1":
                if not self.alset_state.restarting():
                  print("PAD: FUNC_FLOW_REWARD4:", func_flow_reward)
                  self.alset_state.increment_frame()
                  self.alset_state.record_frame_info(self.app_name, mode, NN_name, action, img_name, state, self.stego_plot_img)
                  done = True  
                  prev_stego = self.stego_plot_img
                  self.dispatch(action, state, next_state, done)
                  print("PAD: completed REWARD phase", self.get_frame_num(), next_action, reward, done)
                else:
                  self.alset_state.increment_frame()
                if func_flow_nn_name is None:
                  done = True  
                print("PAD: granted REWARD", self.get_frame_num(), next_action, reward, done)
              elif next_action in ["PENALTY1", "PENALTY2"]:
                self.alset_state.increment_frame()
                if not self.alset_state.restarting():
                  self.alset_state.record_frame_info(self.app_name, mode, NN_name, action, img_name, state, self.stego_plot_img)
                  prev_stego = self.stego_plot_img
                  self.dispatch(action, state, next_state, done)
                if func_flow_nn_name is None:
                  done = True  
                  print("PAD: assessed run-ending PENALTY", self.get_frame_num(), next_action, reward, done)
              # local_frame_num += 1  # obsolete
              # frame_num = self.alset_state.frame_num
              # add dummy 0 q_val for now. Compute q_val at end of run.
              q_val = 0
            if next_action not in ["REWARD1", "PENALTY1", "PENALTY2"]:
              line = next_line
          # close the pointer to that file
          nn_filehandle.close()
        app_filehandle.close()
        self.dsu.save_dataset_idx_processed(mode = app_mode, ds_idx=app_index)
        return "PROCESSED_APP_RUN"

  def dispatch(self, action, prev_img, curr_img, done=False):
      print("curr_func_name: ", self.func_app.curr_func_name)
      print("CPT: predicted, actual:", self.move_analysis.predict(), action)
      gc.collect()
      self.move_analysis.train_predictions(action)
      # add_to_mean should only be True in dispatch()
      adjusted_image, mean_dif, rl = self.cvu.adjust_light(curr_img, add_to_mean=True)
      # self.prev_stego_img = copy.deepcopy(self.stego_img)
      # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      # img_pil = Image.fromarray(curr_img)
      # self.stego_cv_img, self.stego_img = self.stego.run(img_pil)
      # print(curr_img)
      self.stego_img = self.stego.run(adjusted_image)
      # self.stego_cv_img, self.stego_img = self.stego.run(curr_img)

      try:
        print("record lighting :", self.cvu.MeanLight, mean_dif, rl["COMPACTNESS"], rl["CENTER"])
        self.alset_state.record_lighting(self.cvu.MeanLight, mean_dif, rl)
      except Exception as e:
        # ugly hack for last REWARD; why empty frame state?
        print("record lighting error:", e)

      # if action in ["GRIPPER_OPEN", "GRIPPER_CLOSE"]:
      # All frames have a gripper bounding box and need analysis
      lg_bb, rg_bb, safe_ground_bb = self.gripper_analysis.analyze(action, prev_img, adjusted_image, done)
      if len(lg_bb) == 4 and len(rg_bb) == 4:
        stego_gripper, stego_safe_ground = self.stego.analyze_bb(lg_bb, rg_bb, safe_ground_bb, self.stego_img)
        if stego_gripper is not None:
          cv2.imshow('stego_gripper', stego_gripper)
          cv2.waitKey(0)
        if stego_safe_ground is not None:
          cv2.imshow('stego_safe_ground', stego_safe_ground)
          cv2.waitKey(0)
      else:
        stego_gripper = None
        stego_safe_ground = None

      robot_movement = None
      arm_movement = None
      if action.startswith("UPPER_ARM") or action.startswith("LOWER_ARM"):
        prev_adjusted_image, prev_mean_dif, prev_rl = self.cvu.adjust_light(prev_img, add_to_mean=False)
        arm_movement = self.map_analysis.get_arm_moved_pixels(action, prev_adjusted_image, adjusted_image, done)
        self.arm_analysis.analyze(action, arm_movement, prev_img, curr_img, self.prev_stego_img, self.stego_img, done, self.func_app.curr_func_name)
      else:
        self.alset_state.copy_prev_arm_state()
      if action in ["FORWARD","REVERSE","LEFT","RIGHT"]:
        prev_adjusted_image, prev_mean_dif, prev_rl = self.cvu.adjust_light(prev_img, add_to_mean=False)
        robot_movement = self.map_analysis.analyze(action, prev_adjusted_image, adjusted_image, done, self.arm_analysis.arm_nav)

      curr_plot_img = self.stego.convert_cv_to_plot(adjusted_image)
      self.line_analysis.analyze(self.run_num, self.get_frame_num(), curr_plot_img, adjusted_image, prev_img, action, robot_movement, arm_movement, stego_gripper)

      self.alset_state.save_state()
      return done 

# integration todo:
#  - line analysis
#    - handle rotations, with tracking
#  - stego tracking objects (small polygons)
#  - deep sort (lines, bounding boxes), estimated movement
#  - table analysis
#  - box analysis (goto box)
#  - stego with cube (in gripper, on table)
#  - add new state to alset_state
#  - factor in get_arm_moved_pixels into position of arm (arm_nav.py)
#    - looks like gripper is filtered in brute-force way (make pixel-level?)

  def get_frame_num(self):
      return self.alset_state.get_frame_num()

  def load_state(self):
      self.app_name  = self.alset_state.last_app_state("APP_NAME")
      self.app_type  = self.alset_state.last_app_state("APP_MODE")
      self.run_id    = self.alset_state.last_app_state("APP_RUN_ID")
      self.run_num   = self.alset_state.last_app_state("APP_RUN_NUM")
      self.start_with_func_index = self.alset_state.last_app_state("FUNC_INDEX")
      print("state:", self.app_name, self.app_type, self.run_id, self.run_num, self.start_with_func_index, self.alset_state.restart_frame_num)

dsa = DSAnalysis()
