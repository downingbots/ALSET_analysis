from config import *
import dataset_utils as ds_utils

# Note: a partial clone of ALSET/functional_app.py to minimalize includes
# the functions used here to parse a func app should be factored out of functional_app.py
class FunctionalApp():
  def __init__(self, app_name, app_type):
      # print("TableTop: 8 Functional NNs")
      self.NN = []
      self.dsu = ds_utils.DatasetUtils(app_name, app_type)
      self.app_name = app_name
      self.app_type = app_type
      self.app_ds_idx = self.dsu.dataset_indices(mode="ANALYZE", nn_name=self.app_name, position="NEW")
      self.cfg = Config()
      self.ff_nn_num = None
      self.curr_func_name = None
      self.func_classifier_outputs = []
      self.func_classifier_flow = []
      if app_type not in ["FUNC", "APP", "DQN"]:
        print("App Type must be in [nn, func_app, dqn]. Unknown value:", mode)
        exit()
      self.app_type = app_type
      outputs = self.cfg.full_action_set
      val = self.cfg.get_value(self.cfg.app_registry, self.app_name)
      print("app registry val:", val)
      if val is not None:
        [self.NN, self.app_flow_model] = val
        self.is_composite_app = True
      else:
        print("No such app defined: ", self.app_name)
        exit()

  #############
  # Go through the app func flow model.
  # Starting with ff_nn_num=None and passing in [REWARD1/2, PENALTY1/2] joystick action, 
  # return the next func_name and the output reward type.
  # If func_name is None, then the process flow is completed.
  #############
  def eval_func_flow_model(self, reward_penalty, init=False):
    rew_pen = None
    if init:
       self.ff_nn_num = None
       # self.curr_phase = 0
    if self.ff_nn_num is not None:
      NN_name = self.NN[self.ff_nn_num]
    for [it0, it1] in self.app_flow_model:
      print("AFM:", reward_penalty, self.ff_nn_num, it0, it1)
      # AFM: ending:  None
      output_rew_pen = None
      if len(it0) == 0 and self.ff_nn_num is None:
         # starting point
         print("AFM: starting")
         self.ff_nn_num = it1[1]
         break
      elif ((type(it0) == str and it0 == "ALL") or 
          (type(it0) == list and self.ff_nn_num in it0)):
          if it1[0] == "IF":
            print("AFM: IF")
            if reward_penalty == it1[1]:
              print("AFM: matching reward")
              if type(it1[2])==list:
                print("AFM: list compare")
                self.ff_nn_num = it1[2][1]
                if len(it1[2]) == 2 and it1[2][0] == "GOTO_WITH_REWARD1":
                  output_rew_pen = "REWARD1"
                elif len(it1[2]) == 2 and it1[2][0] == "GOTO_WITH_REWARD2":
                  output_rew_pen = "REWARD2"
                elif len(it1[2]) == 2 and it1[2][0] == "GOTO_WITH_PENALTY1":
                  output_rew_pen = "PENALTY1"
                elif len(it1[2]) == 2 and it1[2][0] == "GOTO_WITH:PENALTY2":
                  output_rew_pen = "PENALTY2"
                break
              elif it1[2] == "NEXT":
                print("AFM: NEXT")
                self.ff_nn_num += 1
                break
              elif it1[2] in ["NEXT_WITH_REWARD1"]:
                print("AFM: NEXT WITH REW1")
                output_rew_pen = "REWARD1"
                self.ff_nn_num += 1
                break
              elif it1[2] == "STOP":
                print("AFM: STOP")
                self.ff_nn_num = None
                break
              elif it1[2] in ["STOP_WITH_REWARD1", "STOP_WITH_REWARD2", "STOP_WITH_PENALTY1", "STOP_WITH_PENALTY2"]:
                print("AFM: STOP w REW/PEN")
                self.ff_nn_num = None
                if it1[2] == "STOP_WITH_REWARD1":
                  output_rew_pen = "REWARD1"
                elif it1[2] == "STOP_WITH_REWARD2":
                  output_rew_pen = "REWARD2"
                elif it1[2] == "STOP_WITH_PENALTY1":
                  output_rew_pen = "PENALTY1"
                elif it1[2] == "STOP_WITH_PENALTY2":
                  output_rew_pen = "PENALTY2"
                break
            else:
                output_rew_pen = reward_penalty
                print("AFT: rew_pen ", output_rew_pen)
    if self.ff_nn_num is None:
      print("AFM: ending: ", output_rew_pen)
      return [None, output_rew_pen]
    print("AFM: eval ", self.ff_nn_num, self.NN[self.ff_nn_num], output_rew_pen)
    self.curr_func_name = self.NN[self.ff_nn_num]
    return [self.NN[self.ff_nn_num], output_rew_pen]

