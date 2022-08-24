# MIT License.  Derived from:
#
# Decision Transformer
# Lili Chen\*, Kevin Lu\*, Aravind Rajeswaran, Kimin Lee, Aditya Grover, 
# Michael Laskin, Pieter Abbeel, Aravind Srinivas†, and Igor Mordatch†
#
#\*equal contribution, †equal advising
#
# A link to the paper can be found on [arXiv](https://arxiv.org/abs/2106.01345).
#
# Based on the officieal
# codebase for [Decision Transformer: Reinforcement Learning via Sequence Mode
#ling](https://sites.google.com/berkeley.edu/decision-transformer).
# which contains scripts to reproduce experiments.
#
# And:
# https://github.com/higgsfield/RL-Adventure/blob/master/2.double%20dqn.ipynb
#
# And:
# alset_dqn.py
# 
# Takes the ALSET datasets for functional apps and DQN on a real robot, 
# and converts them into a ATARI-compatible dataset appropriate to be 
# loaded into a GPT model for reinforcement learning.

import csv
# import sys
import logging
# make deterministic
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
# import torch.nn.functional as F
# import torch.ops
from torch.nn import functional as F
from torch.utils.data import Dataset
from mingpt.model_atari import GPT, GPTConfig
from mingpt.trainer_atari import Trainer, TrainerConfig
from mingpt.utils import sample
from mingpt.utils import set_seed
from collections import deque
import random
import torch
import pickle
#ARD import blosc
import argparse
#ARD from fixed_replay_buffer import FixedReplayBuffer
# from alset_gpt import *
from alset_ddqn import *
from robot import *

import math, random
# from builtins import bytes
import codecs
from PIL import Image
import cv2

# import gym
import numpy as np
import collections, itertools
from collections import Counter

# import torch.ops.image
# from torch import read_file, decode_image

import pickle
from dataset_utils import *
from functional_app import *

# import alset_image 

# from IPython.display import clear_output
# import matplotlib.pyplot as plt
# %matplotlib inline

from collections import deque

##############
# Initially defined for atari env
##############
import torchvision 
import torchvision.io
# import torchvision.io.image
# from torchvision.io.image import read_image
# from torchvision.io import read_image
# import image
# from .alset_image import read_image
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

import os
import os.path as osp
import importlib.machinery
from config import *
from dataset_utils import *

# derived from class ALSET_DDQN(nn.Module):
class ALSET_GPT():

    def __init__(self, alset_robot=None, app_name=None, app_type="GPT", capac = None):
        self.app_name = app_name
        self.app_type = app_type
        self.robot = alset_robot
        if self.robot is None:
          # args = ["alset_train.py", sys.argv[1], sys.argv[2]]
          args = ["alset_train.py", "--gpt", app_name]
          # self.robot = Robot(args)
          # self.robot.stop_all(True)
          self.robot = None
        self.cfg = Config()
        self.dsu = DatasetUtils(self.app_name, self.app_type)
        self.parse_app_ds_details = []
        if app_type == "GPT" or app_type == "APP":
          # self.BEST_MODEL_PATH = self.dsu.best_model(mode="GPT")
          self.GPT_PATH_PREFIX = self.dsu.dataset_path()
          self.REPLAY_BUFFER_PATH = self.dsu.dqn_replay_buffer()
          if app_type == "GPT":
            func_restrict = self.cfg.get_func_value(self.app_name, "MOVEMENT_RESTRICTIONS")
            print("func_restrict:", self.app_name, func_restrict)
            self.GPT_DS_IDX_PROCESSED = self.dsu.dataset_idx_processed(mode="GPT",nn_name=self.app_name)
            self.GPT_DS_IDX = self.dsu.dataset_indices(mode="GPT",nn_name=self.app_name,position="NEW")
            self.dqn_ds_idx_nm = self.dsu.get_filename_from_full_path(self.GPT_DS_IDX)
            self.dqn_ds_idx_nm = self.dqn_ds_idx_nm[0:-len(".txt")]
            self.GPT_DS_PATH = self.dsu.dataset_path(mode="GPT", dqn_idx_name=self.dqn_ds_idx_nm)
            print("processed,idx,pth:", self.GPT_DS_IDX_PROCESSED, self.GPT_DS_IDX, self.GPT_DS_PATH)
          else:
            self.GPT_DS_PATH = self.dsu.dataset_path(mode="APP")

          # Reward computation constants from config file attributes
          dqn_registry           = self.cfg.get_value(self.cfg.DQN_registry, self.app_name)
          # print("dqn_registry:", dqn_registry)
          # print("GPT_registry:", self.cfg.GPT_registry)
          # print("app_name:", self.app_name)
          DQN_Policy = dqn_registry[0]
        else:
          # self.BEST_MODEL_PATH = self.dsu.best_model(mode="FUNC", nn_name=app_name )
          self.GPT_PATH_PREFIX = self.dsu.dataset_path()
          # print("alset_ddqn: ", self.BEST_MODEL_PATH, self.GPT_PATH_PREFIX)
          self.REPLAY_BUFFER_PATH = self.dsu.gpt_replay_buffer()
          self.GPT_DS_PATH = self.dsu.dataset_path(mode="FUNC", nn_name=app_name)
          DQN_Policy = self.cfg.FUNC_policy

        self.REPLAY_INITIAL    = self.cfg.get_value(DQN_Policy, "REPLAY_BUFFER_CAPACITY")
        if self.REPLAY_INITIAL is None or self.REPLAY_INITIAL == 0:
          self.REPLAY_INITIAL = 100000  # from create_dataset
          # self.REPLAY_INITIAL  = 20000

        # GPT just uses the DQN Policy
        self.REPLAY_PADDING    = self.cfg.get_value(DQN_Policy, "REPLAY_BUFFER_PADDING")
        self.BATCH_SIZE        = self.cfg.get_value(DQN_Policy, "BATCH_SIZE")
        self.GAMMA             = self.cfg.get_value(DQN_Policy, "GAMMA")
        self.QVAL_THRESHOLD    = self.cfg.get_value(DQN_Policy, "QVAL_THRESHOLD")
        print("qvthresh:",self.QVAL_THRESHOLD)
        self.LEARNING_RATE     = self.cfg.get_value(DQN_Policy, "LEARNING_RATE")
        print("lr:",self.LEARNING_RATE)
        self.ERROR_CLIP        = self.cfg.get_value(DQN_Policy, "ERROR_CLIP")
        self.ERROR_LINEAR_CLIP = self.cfg.get_value(DQN_Policy, "ERROR_LINEAR_CLIP")
        self.DQN_REWARD_PHASES = self.cfg.get_value(DQN_Policy, "DQN_REWARD_PHASES")
        self.REWARD1_REWARD    = self.cfg.get_value(DQN_Policy, "REWARD1")
        self.REWARD2_REWARD    = self.cfg.get_value(DQN_Policy, "REWARD2")
        self.PENALTY1_PENALTY  = self.cfg.get_value(DQN_Policy, "PENALTY1")
        self.PENALTY2_PENALTY  = self.cfg.get_value(DQN_Policy, "PENALTY2")
        self.DQN_MOVE_BONUS    = self.cfg.get_value(DQN_Policy, "DQN_MOVE_BONUS")
        self.PER_MOVE_PENALTY  = self.cfg.get_value(DQN_Policy, "PER_MOVE_PENALTY")
        self.PER_MOVE_REWARD   = self.cfg.get_value(DQN_Policy, "PER_MOVE_REWARD")
        self.PER_FORWARD_REWARD   = self.cfg.get_value(DQN_Policy, "PER_FORWARD_REWARD")
        self.MAX_MOVES         = self.cfg.get_value(DQN_Policy, "MAX_MOVES")
        self.MAX_MOVES_EXCEEDED_PENALTY = self.cfg.get_value(DQN_Policy, "MAX_MOVES_EXCEEDED_PENALTY")
        self.MAX_MOVES_REWARD  = self.cfg.get_value(DQN_Policy, "MAX_MOVES_REWARD")
        self.ESTIMATED_VARIANCE         = self.cfg.get_value(DQN_Policy, "ESTIMATED_VARIANCE")
        print("est var:", self.ESTIMATED_VARIANCE)
        if self.LEARNING_RATE is None:
          self.LEARNING_RATE = 0.00005

        # reward variables
        self.standard_mean      = 0.0
        self.standard_variance  = 1.0
        self.estimated_mean     = 0.0
        self.curr_phase   = 0
        self.max_phase   = len(self.DQN_REWARD_PHASES)
        self.total_reward = 0
        self.frame_num    = 0
        self.action       = None
        self.prev_action  = None
        self.state        = None
        self.prev_state   = None
        self.all_rewards  = []
        self.clip_max_reward  = -1
        self.clip_min_reward  = 1

        ############
        # GPT variables
        ############
        if capac is None:
          capac = self.REPLAY_INITIAL + self.REPLAY_PADDING
        if not self.load_replay_buffer():
          print("create new replay_buffer")
          self.replay_buffer  = ReplayBuffer(capacity=capac, name="replay")
        self.active_buffer  = ReplayBuffer(capacity=capac, name="active")  
        
        # self.epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)
        
        # plt.plot([self.epsilon_by_frame(i) for i in range(1000000)])
        self.num_frames = self.REPLAY_INITIAL

        self.USE_CUDA = torch.cuda.is_available()
        self.Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if self.USE_CUDA else autograd.Variable(*args, **kwargs)
        
        # self.losses = []

        ############
        # FINE TUNE PRETRAINED MODEL USING IMITATION LEARNING
        # add to history
        
        self.mean_cnt = 0
        self.mean_i = 0
        self.num_actions = len(self.cfg.full_action_set)
        # self.current_model = None
        # self.target_model = None
        self.init_model = False
        # self.train_model = do_train_model
        self.all_loss = []
        self.all_act_rank = []
        self.all_allowed_act_rank = []
        # print("DQN initialization: ",self.init_model, self.train_model)
        
        if self.app_type == "GPT":
            ds_path = self.GPT_DS_PATH + "/"
            ds_dirs = [self.GPT_DS_PATH]
            # for act in self.cfg.full_action_set:
            #   ds_dirs.append(ds_path + act)
            self.dsu.mkdirs(ds_dirs)
            # self.robot.gather_data.set_ds_idx(self.DQN_DS_IDX)

        # self.frame_num = len(self.active_buffer)
        self.frame_num = 0
        # target_model = copy.deepcopy(current_model)
        # robot DQN can return self.robot_actions
        # joystick + robot can return self.actions
        # return False, self.cfg.full_action_set

    def transform_image(self, img, mode='val'):
        # data transforms, for pre-processing the input image before feeding into the net
        # Data augmentation and normalization for training

        data_transforms = transforms.Compose([
            # transforms.Resize((224,224)),  # resize to 224x224
            transforms.Resize((84,84)),  # resize to 224x224
            transforms.ToTensor(),         # tensor format
            transforms.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])  ])
        # pre-process the input
        transformed_img = data_transforms(img)
        # transformed_img = transforms(transformed_img.to("cuda")) # done in Variable decl
        return transformed_img

    def transform_image_from_path(self, image_filepath):
        for batch_item_num, path in enumerate(image_filepath):
            img = Image.open(path)
            img = self.transform_image(img)
            # TypeError: float() argument must be a string or a number, not 'set'
            img = self.Variable(torch.FloatTensor(np.float32(img)))
            img = torch.unsqueeze(img, 0)
            if batch_item_num == 0:
                state_image = img
            else:
                state_image = torch.cat((state_image,img),0)
            # print("transformed_img shape", img.shape)
        return state_image.numpy()
        # return tensor.numpy(state_image)

    # import json
    # json.dump(self.replay_buffer, filehandle)
    # json.load(filehandle)
    def save_replay_buffer(self):
        with open(self.REPLAY_BUFFER_PATH, 'wb+') as filehandle:
          # store the data as binary data stream
          pickle.dump(self.replay_buffer, filehandle)
        filehandle.close()

    def load_replay_buffer(self):
        try:
          with open(self.REPLAY_BUFFER_PATH, 'rb') as filehandle:
            self.replay_buffer = pickle.load(filehandle)
          filehandle.close()
        except:
            print("load_replay_buffer: open failed")
            return False
        print("loaded replay_buffer. Len = ", self.replay_buffer.entry_len())
        if self.replay_buffer.entry_len() == 5:
            # compatibility with earlier version of buffer format
            # state, action, reward, next_state, done
            print("adding q_values to replay_buffer")
            self.replay_buffer.compute_real_q_values(gamma=self.GAMMA)
            self.save_replay_buffer()
        static_vars.name.append("replay")
        static_vars.sample_start.append(0)
        return True

    def train_DQN_qvalue(self):
        loss = 0
        # while loss is not None:
          # loss = self.compute_td_loss(batch_size = self.BATCH_SIZE, mode = "IMITATION_TRAINING")
        # self.current_model.save_state(self.BEST_MODEL_PATH)
        # print("Initial model trained by imitation learning")
        return loss

    def shape(self, tensor):
        # s = tensor.get_shape()
        s = tensor.size()
        print("s: ", s)
        return tuple([s[i].value for i in range(0, len(s))])

    def compute_ranking_stats(self, current_q_values, action_index):
          try:
            # determine rankings
            curr_q_vals = current_q_values.detach().cpu().numpy()
            act_rank = []
            allowed_act_rank = []
            if self.cfg.nn_disallowed_actions is None:
              disallowed = []
              feasible_act = list(self.cfg.full_action_set)
            else:
              disallowed = []
              for a in self.cfg.nn_disallowed_actions:
                try:
                  disallowed.append(list(self.cfg.full_action_set).index(a))
                except:
                  pass
              feasible_act = list(Counter(list(self.cfg.full_action_set)) - Counter(self.cfg.nn_disallowed_actions))
    
            for i, curr_q_v in enumerate(curr_q_vals):
              act_srt = np.argsort(curr_q_v)
              act_srt = act_srt[::-1]
              # print("act_srt1",act_srt, disallowed)
              act_srt = list(Counter(act_srt.tolist()) - Counter(disallowed))
              # print("act_srt2",act_srt)
              # act_srt = list(Counter(act_srt.tolist()) - (Counter(list(self.cfg.full_action_set) - Counter(allowed_actions))))
              # a_r = act_srt.index(action_index[frame_num])
              try:
                a_r = act_srt.index(action_index[i])
                act_rank.append(a_r)
                cqv = curr_q_vals[i]
                print("real action,rank:", self.cfg.full_action_set[action_index[i]], a_r, qval_idx[i], cqv[act_srt[a_r]])
                self.all_act_rank.append(a_r)
              except:
                print("unranked top action:",self.cfg.full_action_set[action_index[i]])
                pass
              ranks = []
              for i2, a2 in enumerate(act_srt):
                  cqv = curr_q_vals[i]
                  ranks.append([i2, self.cfg.full_action_set[a2], cqv[a2]])
              print("top action rank: ", ranks)
    
              # print("ar",a_r)
              aa_r = []
              for ff_nn, fn, allowed_actions in self.parse_app_ds_details:
                if fn > self.frame_num:
                  # print("PARSE_APP_DS: ", ff_nn, fn, self.frame_num, allowed_actions)
                  break
                aa_r = []
                for aa in allowed_actions:
                  # aa_i = self.cfg.full_action_set.index(aa)  # Human readable to integer index
                  try:
                    aa_i_s = act_srt.index(aa)
                    aa_r.append(aa_i_s)
                    allowed_act_rank.append(aa_i_s)
                    self.all_allowed_act_rank.append(aa_i_s)
                  except:
                    # should be a WRIST_ROTATE_LEFT/RIGHT, which is really not an allowed action
                    # print("unranked allowed action:",self.cfg.full_action_set[aa])
                    pass
              var_aa_r = np.var(aa_r)
              mean_aa_r = np.mean(aa_r)
              # print("action ranking", a_r, mean_aa_r, var_aa_r, len(feasible_act))
              self.frame_num += 1
    
            mean_act_rank = np.mean(act_rank)
            var_act_rank = np.var(act_rank)
            mean_all_act_rank = np.mean(self.all_act_rank)
            var_all_act_rank = np.var(self.all_act_rank)
            mean_allowed_act_rank = np.mean(allowed_act_rank)
            var_allowed_act_rank = np.var(allowed_act_rank)
            mean_all_allowed_act_rank = np.mean(self.all_allowed_act_rank)
            var_all_allowed_act_rank = np.var(self.all_allowed_act_rank)
            # This should evaluate how good of the last run of the NN was  
            print("FINAL ACTION RANKING:" 
                    "mean", mean_all_act_rank, mean_all_allowed_act_rank,
                    #       mean_act_rank, mean_allowed_act_rank, 
                    "var", var_all_act_rank, var_all_allowed_act_rank,
                    #       var_act_rank, var_allowed_act_rank, 
                    "numact", len(feasible_act), len(allowed_actions))
          except Exception as e:
            print("Error during statistics:", e)
            pass

    # ARD: GPT doesn't use. Delete.
    def compute_td_loss(self, batch_size=32, mode="REAL_Q_VALUES", compute_stats=True):
        # self.current_model.train()  # Set model to training mode
        if mode == "IMITATION_TRAINING":
          # Train based on composite app runs
          state_path, action, rewards, next_state_path, done_val, q_val = self.replay_buffer.get_next_sample(batch_size)
        elif mode == "TARGET_Q_VALUES":
          # Train based on runs of DQN datasets
          state_path, action, rewards, next_state_path, done_val, q_val = self.active_buffer.get_next_sample(batch_size=batch_size, name="active")
        elif mode == "REAL_Q_VALUES":
          # Train based on runs of DQN datasets
          state_path, action, rewards, next_state_path, done_val, q_val = self.active_buffer.get_next_sample(batch_size=batch_size, name="active")
        elif mode == "EXPERIENCE_REPLAY":
          # Part of DQN algorithm.
          # 
          # The learning phase is then logically separate from gaining experience, and based on 
          # taking random samples from the buffer. 
          #
          # Advantages: More efficient use of previous experience, by learning with it multiple times.
          # This is key when gaining real-world experience is costly, you can get full use of it.
          # The Q-learning updates are incremental and do not converge quickly, so multiple passes 
          # with the same data is beneficial, especially when there is low variance in immediate 
          # outcomes (reward, next state) given the same state, action pair.
          #
          # Disadvantage: It is harder to use multi-step learning algorithms
          state_path, action, rewards, next_state_path, done_val, q_val = self.replay_buffer.random_sample(batch_size)
        else:
          print("Unknown mode for computed td_loss:", mode)
          exit()
        try:
          if state_path is None:
            print("Completed replay buffer training.")
            return None
        except:
          pass


        # 4-D tensors are for batches of images, 3-D tensors for individual images.
        # image_batch is a tensor of the shape (32, 180, 180, 3).  
        # transform state paths to images.  already a Tensor.
        #
        # compute_real_q_values has actual state, not state_path.
        if type(state_path) == np.ndarray and type(state_path[0]) is not np.str_: 
          print("state already converted from path.", type(state_path), type(state_path[0]))
          state = state_path
        else:
          state = self.transform_image_from_path(state_path)
        if type(next_state_path) == np.ndarray and type(next_state_path[0]) is not np.str_: 
          print("next_state already converted from path.", type(next_state_path), type(next_state_path[0]))
          next_state = next_state_path
        else:
          next_state = self.transform_image_from_path(next_state_path)

        # transform action string to number
        action_idx = []
        qval_idx = []
        for i3,a3 in enumerate(action):
          # action_idx.append(self.robot_actions.index(a))  # Human readable to integer index
          try:
            action_idx.append(self.cfg.full_action_set.index(a3))  # Human readable to integer index
            qval_idx.append(q_val[i3])
          except:
              print("error with:", a3, i3)
              exit()
        action_index = tuple(action_idx)
        # print("action_idx:",action_idx)
        # print("max action_idx:",len(self.cfg.full_action_set))
        # print("q_val:", q_val)

        action_idx = self.Variable(torch.LongTensor(action_index))

        # the real q value is precomputed in q_val
        # real q value computed from done end-pt
        q_val      = self.Variable(torch.FloatTensor(q_val))  

        # Computed q-values from Alexnet
        # current_q_values = self.current_model.alexnet_model(state)
        # q_value = current_q_values.gather(1, action_idx.unsqueeze(1)).squeeze(1)
        # print("curr_q_val:", current_q_values)
        # print("q_value   :", q_value)

        # if compute_stats and False:
        #     self.compute_ranking_stats(current_q_values, action_index)

        if mode == "IMITATION_TRAINING":
          # loss = (q_value - q_val).pow(2).mean()
          # print("IMITATION_TRAINING:")
          pass
        elif mode == "TARGET_Q_VALUES":
          print("TARGET_Q_VALUES:")
          # For DDQN, using target_model to predict q_val (derived from rladvddqn.py).
          # //github.com/higgsfield/RL-Adventure/blob/master/2.double%20dqn.ipynb
          # current_next_q_values  = self.current_model.alexnet_model(next_state)
          # target_next_q_values  = self.target_model.alexnet_model(next_state)
          # next_q_value = target_next_q_values.gather(1, torch.max(current_next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
          reward     = self.Variable(torch.FloatTensor(rewards))
          done       = self.Variable(torch.FloatTensor(done_val))
          expected_q_value = reward + self.GAMMA * next_q_value * (1 - done)

          if self.ERROR_CLIP is not None and self.ERROR_CLIP >= 0:
            difference = torch.abs(q_value - self.Variable(expected_q_value.data))
            print("qval diff: ",  difference)
            quadratic_part = torch.clamp(difference, 0.0, self.ERROR_CLIP)
            linear_part = difference - quadratic_part
            if self.ERROR_LINEAR_CLIP:
              loss = (0.5 * quadratic_part.pow(2) + (self.ERROR_CLIP * linear_part)).mean()
            else:
              loss = (quadratic_part.pow(2)).mean()
          else:
            loss = (q_value - self.Variable(expected_q_value.data)).pow(2).mean()
          print("TARGET_Q_VALUES:")
          print("qval loss: ",  loss)
        elif mode == "REAL_Q_VALUES":
          # nondeterministic err: device-side assert triggered: nonzero_finite_vals
          # print("len q_value, q_val:", len(q_value), len(q_val))
          difference = torch.abs(q_value - q_val)
          if self.ERROR_CLIP is not None and self.ERROR_CLIP >= 0:
            quadratic_part = torch.clamp(difference, 0.0, self.ERROR_CLIP)
            linear_part = difference - quadratic_part
            if self.ERROR_LINEAR_CLIP:
              loss = (0.5 * quadratic_part.pow(2) + (self.ERROR_CLIP * linear_part)).mean()
            else:
              loss = (0.5 * quadratic_part.pow(2)).mean()
          else:
            loss = (0.5 * difference.pow(2)).mean()
            # loss = (q_value - q_val).pow(2).mean()
          # print("REAL_Q_VALUES:")
          # print("qval diff: ",  difference)
          # print("qval loss: ",  loss)
        elif mode == "EXPERIENCE_REPLAY":
          loss = (q_value - q_val).pow(2).mean()
          print("EXPERIENCE_REPLAY FROM TARGET:")

        if loss is not None:
          # Note that following the first .backward call, a second call is only possible after you have performed another forward pass.
          # print("loss training")
          # self.current_model.train()  # Set model to training mode
          # self.current_model.alexnet_optimizer.zero_grad()
          # loss.backward()
          # self.current_model.alexnet_optimizer.step()
          # self.current_model.eval()  # Set model to eval mode
          # curr_q_vals = current_q_values.detach().cpu().numpy()
          # self.all_loss.append(loss.item())
          pass
        else:
          print("None loss")
        return loss
    
#    def plot(self, frame_num, rewards, losses):
#        clear_output(True)
#        plt.figure(figsize=(20,5))
#        plt.subplot(131)
#        plt.title('frame %s. reward: %s' % (frame_num, np.mean(rewards[-10:])))
#        plt.plot(rewards)
#        plt.subplot(132)
#        plt.title('loss')
#        plt.plot(losses)
#        plt.show()

    def compute_reward(self, frame_num, action):
                         # phase 0: to first DQN reward; 50 award & 300 allocated moves
                         # phase 1: to second DQN reward; 100 award & 400 allocated moves
                         # more phases allowed
        # ["DQN_REWARD_PHASES", [[50,    300],   [100,   400]]],
        if self.DQN_REWARD_PHASES is not None:
          if self.curr_phase < len(self.DQN_REWARD_PHASES):
            PHASE_ALLOCATED_MOVES = self.DQN_REWARD_PHASES[self.curr_phase][1]
            PHASE_REWARD = self.DQN_REWARD_PHASES[self.curr_phase][0]
            print("curr phase does not exceed DQN_REWARD_PHASES.", self.curr_phase, frame_num, action, len(self.DQN_REWARD_PHASES), self.DQN_REWARD_PHASES)
          else:
            PHASE_ALLOCATED_MOVES = self.DQN_REWARD_PHASES[-1][1]
            PHASE_REWARD = self.DQN_REWARD_PHASES[-1][0]
            print("WARN: curr phase exceeds DQN_REWARD_PHASES.", self.curr_phase, frame_num, action, len(self.DQN_REWARD_PHASES), self.DQN_REWARD_PHASES)
          # print("COMPUTE_REWARD: action, phase, self.DQN_REWARD_PHASES:",  action, frame_num, self.curr_phase, self.DQN_REWARD_PHASES, len(self.DQN_REWARD_PHASES))
        reward = 0
        if frame_num > self.MAX_MOVES:
            return 0, True
        if frame_num == self.MAX_MOVES:
          if self.MAX_MOVES_EXCEEDED_PENALTY is not None:
            return (self.MAX_MOVES_EXCEEDED_REWARD / self.ESTIMATED_VARIANCE), True
          elif self.MAX_MOVES_REWARD is not None:
            return (self.MAX_MOVES_REWARD / self.ESTIMATED_VARIANCE), True
        elif action == "REWARD1":
          done = False
          if self.DQN_REWARD_PHASES is not None:
            reward = PHASE_REWARD + max((PHASE_ALLOCATED_MOVES - frame_num),0)*self.DQN_MOVE_BONUS
            self.curr_phase += 1
            print("reward, phase, self.DQN_REWARD_PHASES:",  reward, self.curr_phase, self.DQN_REWARD_PHASES, len(self.DQN_REWARD_PHASES), frame_num)
          # reward, phase, self.DQN_REWARD_PHASES: 52.0 1 [[50, 300], [100, 400]] 2 292
            if self.curr_phase >= len(self.DQN_REWARD_PHASES):
              done = True
            return (reward / self.ESTIMATED_VARIANCE), done
          elif self.REWARD1_REWARD is not None:
            return (self.REWARD1_REWARD / self.ESTIMATED_VARIANCE), done
          else:
            return 0, done
        elif action == "REWARD2":
          if self.REWARD2_REWARD is not None:
            return (self.REWARD2_REWARD / self.ESTIMATED_VARIANCE), True
          else:
            return 0, True
        elif action in ["PENALTY1","PENALTY2"]:
          if self.PENALTY1_PENALTY is not None and action == "PENALTY1":
            return (self.PENALTY1_PENALTY / self.ESTIMATED_VARIANCE), True
          elif self.PENALTY2_PENALTY is not None and action == "PENALTY2":
            return (self.PENALTY2_PENALTY / self.ESTIMATED_VARIANCE), True
          else:
            return 0, True
        # elif self.DQN_REWARD_PHASES is not None and self.PER_MOVE_PENALTY is not None:
        #   if self.curr_phase < len(self.DQN_REWARD_PHASES):
        #     return (self.PER_MOVE_PENALTY / self.ESTIMATED_VARIANCE), False
        if (self.PER_MOVE_REWARD is not None or self.PER_FORWARD_REWARD is not None 
            or self.PER_MOVE_PENALTY is not None):
          rew = 0
          if self.PER_MOVE_REWARD is not None:
              rew += self.PER_MOVE_REWARD
          if self.PER_MOVE_PENALTY is not None: 
              rew += self.PER_MOVE_PENALTY
          if self.PER_FORWARD_REWARD is not None and action in ["FORWARD"]:
              rew += self.PER_FORWARD_REWARD
          return (rew / self.ESTIMATED_VARIANCE), False
        elif action not in self.cfg.full_action_set:
          print("unknown action: ",  action)
          exit()
    
    #
    #  Can improve scoring over time as collect more output (mean, stddev).
    #    total # moves, # moves to pick up cube, # moves to drop in box,
    #    End game event: too many moves, success, robot off table, cube off table
    #                    auto            success, penalty,         pause (lowerarm  up)
    #                                       left  right            go    (lowerarm down)
    #
    #  Human Labeled events:
    #    pick up cube, drop cube in box, drop cube over edge, off table
    #  Computer labeled events:
    #    each move

    def set_dqn_action(self, action):
        # set by joystick: REWARD1, PENALTY1, REWARD2, PENALTY2
        self.dqn_action = action

    def get_dqn_action(self):
        # set by joystick: REWARD1, PENALTY1, REWARD2, PENALTY2
        return self.dqn_action
    
    # Func: need to factor out common functionality with parse_app_dataset
    #       Currently, just a minor modified clone
    def parse_func_dataset(self, NN_name, init=False, app_mode="FUNC"):
        print("PFD: >>>>> parse_func_dataset", init)
        app_dsu = DatasetUtils(self.app_name, "FUNC")
        if init:
          # start at the beginning
          # e.g., clear TTT_APP_IDX_PROCESSED_BY_DQN.txt
          app_dsu.save_dataset_idx_processed(mode = app_mode, clear = True )

        frame_num = 0
        # parse FUNC ds doesn't use reward phases
        reward = []
        ###################################################
        # iterate through NNs and fill in the active buffer
        while True:
          func_index = app_dsu.dataset_indices(mode=app_mode,nn_name=NN_name,position="NEXT")
          if func_index is None:
            print("PFD: parse_func_dataset: done")
            break
          print("Parsing FUNC idx", func_index)
          frame_num = 0
          run_complete = False
          line = None
          next_action = None
          next_line = None
          self.active_buffer.clear()
          self.curr_phase = 0
          nn_filehandle = open(func_index, 'r')
          line = None
          while True: # iterate through Img frames in nn
            # read a single line
            next_line = nn_filehandle.readline()
            if not next_line:
                run_complete = True
                print("PFD: Function Index completed:", frame_num, NN_name, next_action)
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
                reward, done = self.compute_reward(frame_num, action)
                print("PFD: compute_reward:", frame_num, action, reward, done, next_action)
              if next_action == "REWARD1":
                reward, done = self.compute_reward(frame_num, next_action)
                done = True  # end of func is done in this mode
                print("PFD: completed REWARD phase2", frame_num, next_action, reward, done)
              elif next_action in ["PENALTY1", "PENALTY2"]:
                reward, done = self.compute_reward(frame_num, next_action)
                done = True  # end of func is done in this mode
                print("PFD: assessed run-ending PENALTY", frame_num, next_action, reward, done)
              frame_num += 1
              # add dummy 0 q_val for now. Compute q_val at end of run.
              q_val = 0
              self.active_buffer.push(state, action, reward, next_state, done, q_val)
            if next_action not in ["REWARD1", "PENALTY1", "PENALTY2"]:
              line = next_line
          # close the pointer to that file
          nn_filehandle.close()
          #################################################
          ## NOT DONE FOR IMMITATION LEARNING
          # if len(self.replay_buffer) > self.REPLAY_INITIAL:
          #   loss = self.compute_td_loss(batch_size, app_path_prefix)
          # if frame_num % 1000 == 0 or done:
          #   self.update_target(self.current_model, self.target_model)
          #################################################
          if run_complete:
              print("PFD: SAVING STATE; DO NOT STOP!!!")
              self.active_buffer.compute_real_q_values(gamma=self.GAMMA,  name="active", done_required=done_required)
              self.active_buffer.reset_sample(name="active", start=0)
              self.frame_num = 0
              for i in range(self.cfg.NUM_DQN_EPOCHS):
                loss = 0
                while loss is not None:
                  loss = self.compute_td_loss(batch_size=self.BATCH_SIZE, mode="REAL_Q_VALUES")
                  print("PFD: real q values loss: ", i, loss)
              # print("PFD: loss: ",loss)
              # print("PFD: ACTIVE BUFFER:", self.active_buffer)
              try:
                self.replay_buffer.concat(self.active_buffer)
              except:
                if self.replay_buffer is None:
                    print("PFD: self.replay_buffer is None")
                if self.active_buffer is None:
                    print("PFD: self.active_buffer is None")
              self.save_replay_buffer()
              # print(self.BEST_MODEL_PATH)
              # self.current_model.save_state(self.BEST_MODEL_PATH)
              # torch.save(model.state_dict(), self.BEST_MODEL_PATH)
              # self.update_target(self.current_model, self.target_model)
              app_dsu.save_dataset_idx_processed(mode = app_mode, nn_name=nn_name, ds_idx=func_index)
              print("PFD: STATE SAVED")
        return "PROCESSED_FUNC_RUN"

    # for training DQN by processing app dataset (series of functions/NNs)
    def parse_app_dataset(self, init=False, app_mode="APP"):
        print("PAD: >>>>> parse_app_dataset:", init)
        app_dsu = DatasetUtils(self.app_name, "APP")
        if init:
          # start at the beginning
          # e.g., clear TTT_APP_IDX_PROCESSED_BY_DQN.txt
          app_dsu.save_dataset_idx_processed(mode = app_mode, clear = True )
          # app_dsu.save_dataset_idx_processed(mode = "GPT", clear = True )

        frame_num = 0
        reward = []
        val = self.cfg.get_value(self.cfg.app_registry, self.app_name)
        func_nn_list = val[1]
        func_app = FunctionalApp(alset_robot=self.robot, app_name=self.app_name, app_type=self.app_type)
        # func_app = FunctionalApp(alset_robot=self.robot, app_name=self.app_name, app_type=self.app_type)
        # func_app = FunctionalApp(alset_robot=self.robot, app_name=self.app_name, app_type="APP")

        # func_app.nn_init()
        self.parse_app_ds_details = []

        ###################################################
        # iterate through NNs and fill in the active buffer
        app_index = app_dsu.dataset_indices(mode=app_mode,nn_name=None,position="NEXT")
        if app_index is None:
          print("PAD: parse_app_dataset: unknown NEXT index or DONE")
          return None
        print("PAD: Parsing APP idx", app_index)
        app_filehandle = open(app_index, 'r')
        run_complete = False
        line = None
        next_action = None
        next_line = None
        self.active_buffer.clear()
        self.curr_phase = 0
        func_flow_reward = None
        first_time_through = True
        self.clip_max_reward  = -1
        self.clip_min_reward  = 1
        while True:  
          [func_flow_nn_name, func_flow_reward] = func_app.eval_func_flow_model(reward_penalty="REWARD1", init=first_time_through)
          print("PAD: FUNC_FLOW_REWARD1:", func_flow_reward)
          # find out what the func flow model expects
          if first_time_through:
            first_time_through = False
            if self.cfg.nn_disallowed_actions is None:
              disallowed = []
              feasible_act = list(self.cfg.full_action_set)
            else:
              disallowed = []
              for a in self.cfg.nn_disallowed_actions:
                try:
                  disallowed.append(list(self.cfg.full_action_set).index(a))  
                except:
                  pass
              feasible_act = list(Counter(list(self.cfg.full_action_set)) - Counter(disallowed))
          # allowed_actions is different for each nn
          allowed_actions = None
          for [func, func_allowed_actions] in self.cfg.func_movement_restrictions:
            if func_flow_nn_name == func:
               allowed_actions = []
               for a in func_allowed_actions:
                 allowed_actions.append(list(self.cfg.full_action_set).index(a))
               break
          if allowed_actions is None:
            allowed_actions = feasible_act
          self.parse_app_ds_details.append([func_flow_nn_name, frame_num, allowed_actions])

          # get the next function index 
          nn_idx = app_filehandle.readline()
          print("PAD: allowed_actions, nn_idx:", allowed_actions, nn_idx, func_flow_nn_name)
          if not nn_idx:
            if func_flow_nn_name is None:
              if func_flow_reward == "REWARD1":
                print("PAD: FUNC_FLOW_REWARD2:", func_flow_reward)
                if next_action != "REWARD1":
                    print("PAD: Soft Error: last action of run is expected to be a reward:", func_flow_action, next_action)
                if func_flow_nn_name is None and func_flow_reward == "REWARD1":
                    print("PAD: FUNC_FLOW_REWARD2:", func_flow_reward)
                    # End of Func Flow. Append reward.
                    reward, done = self.compute_reward(frame_num, "REWARD1")
                    if reward > self.clip_max_reward:
                       self.clip_max_reward = reward
                    if reward < self.clip_min_reward:
                       self.clip_min_reward = reward
                    frame_num += 1
                    # add dummy 0 q_val for now. Compute q_val at end of run.
                    q_val = 0
                    done = True
                    self.active_buffer.push(state,"REWARD1", reward, next_state, done, q_val)
                    print("PAD: Appending default REWARD1 at end of buffer", func_flow_nn_name)
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
              app_dsu.save_dataset_idx_processed(mode = app_mode)
# ARD: BUG
              return "INCOMPLETE_APP_RUN"
            break
          if nn_idx[-1] != 't':
            nn_idx = nn_idx[0:-1]
          # print("PAD: remove trailing newline1", nn_idx)
          NN_name = app_dsu.get_func_name_from_idx(nn_idx)
          if NN_name != func_flow_nn_name:
            print("PAD: Func flow / index inconsistency:", NN_name, func_flow_nn_name)
          nn_filehandle = open(nn_idx, 'r')
          line = None
          compute_reward_penalty = False
          while True: # iterate through Img frames in nn
            # read a single line
            next_line = nn_filehandle.readline()
            if not next_line:
                run_complete = True
                print("PAD: Function Index completed:", frame_num, NN_name, next_action)
                break
            # get action & next_action (nn_name is None)
            [tm, app, mode, next_nn_name, next_action, img_name, next_state] = self.dsu.get_dataset_info(next_line, mode="FUNC")
            if line is not None:
              [tm, app, mode, nn_name, action, img_name, state] = self.dsu.get_dataset_info(line, mode="FUNC")
              if action == "NOOP":
                # Too common in initial test dataset. Print warning later?
                # print("NOOP action, NN:", action, nn_name)
                line = next_line
                continue
              elif action == "REWARD1":
                print("PAD: Goto next NN; NOOP Reward, curr_NN", action, nn_name)
                line = next_line
                continue
              else:
                # the final compute reward sets done to True (see above)
                reward, done = self.compute_reward(frame_num, action)
                print("PAD: compute_reward:", frame_num, action, reward, done)
              if next_action == "REWARD1" and func_flow_reward == "REWARD1":
                print("PAD: FUNC_FLOW_REWARD4:", func_flow_reward)
                reward, done = self.compute_reward(frame_num, next_action)
                print("PAD: completed REWARD phase", frame_num, next_action, reward, done)
                if func_flow_nn_name is None:
                  done = True  
                print("PAD: granted REWARD", frame_num, next_action, reward, done)
              elif next_action in ["PENALTY1", "PENALTY2"]:
                reward, done = self.compute_reward(frame_num, next_action)
                if func_flow_nn_name is None:
                  done = True  
                print("PAD: assessed run-ending PENALTY", frame_num, next_action, reward, done)
              frame_num += 1
              # add dummy 0 q_val for now. Compute q_val at end of run.
              q_val = 0
              self.active_buffer.push(state, action, reward, next_state, done, q_val)

            if next_action not in ["REWARD1", "PENALTY1", "PENALTY2"]:
              line = next_line
          # close the pointer to that file
          nn_filehandle.close()
        app_filehandle.close()
        #################################################
        ## NOT DONE FOR IMMITATION LEARNING
        # if len(self.replay_buffer) > self.REPLAY_INITIAL:
        #   loss = self.compute_td_loss(batch_size, app_path_prefix)
        # if frame_num % 1000 == 0 or done:
        #   self.update_target(self.current_model, self.target_model)
        #################################################
        if run_complete:
            print("PAD: SAVING STATE; DO NOT STOP!!!")
            done_required = True
            if self.MAX_MOVES_REWARD is not None:
              if self.REWARD1_REWARD is None or self.REWARD1_REWARD == 0:
                 done_required = False
            print("PAD: done_required: ", done_required, self.MAX_MOVES_REWARD, self.REWARD1_REWARD)
            self.active_buffer.compute_real_q_values(gamma=self.GAMMA,  name="active", done_required=done_required)
            self.active_buffer.reset_sample(name="active", start=0)
            print("PAD: parse_app_ds:", self.parse_app_ds_details)
            # for i in range(self.cfg.NUM_DQN_EPOCHS):
              # loss = 0
              # while loss is not None:
                # loss = self.compute_td_loss(batch_size=self.BATCH_SIZE, mode="REAL_Q_VALUES")
                # mean_loss = np.mean(self.all_loss)
                # if len(self.all_loss) > 0:
                  # print("PAD: real q values loss: ", mean_loss, i, self.all_loss[-1])
            # print("loss: ",loss)
            print("ACTIVE BUFFER:", self.active_buffer)
            self.replay_buffer.concat(self.active_buffer)
            self.save_replay_buffer()
            # print(self.BEST_MODEL_PATH)
            # self.current_model.save_state(self.BEST_MODEL_PATH)
            # torch.save(model.state_dict(), self.BEST_MODEL_PATH)
            # self.update_target(self.current_model, self.target_model)
            app_dsu.save_dataset_idx_processed(mode = app_mode, ds_idx=app_index)
            print("PAD: STATE SAVED")
        return "PROCESSED_APP_RUN"

    def parse_unprocessed_app_datasets(self, init=False):
        # nn_apps sets self.init_model to True in following line:
        #         self.app_instance = ALSET_DDQN(True, False, alset_app_name, alset_app_type)
        # Why? set to False? 
        # Alternatie taken: don't use self.init_model here...
        # init_ds = init
        init_ds = False
        while self.parse_app_dataset(init_ds) in ["PROCESSED_APP_RUN", "INCOMPLETE_APP_RUN"]:
            init_ds = False
            print("process another app dataset")

    def parse_unprocessed_rand_datasets(self, init=False):
        # nn_apps sets self.init_model to True in following line:
        #         self.app_instance = ALSET_DDQN(True, False, alset_app_name, alset_app_type)
        # Why? set to False?
        # Alternatie taken: don't use self.init_model here...
        # init_ds = init
        print(">>>>> parse_rand_dataset")
        init_ds = False
        while self.parse_app_dataset(init=init_ds, app_mode="RAND") in ["PROCESSED_APP_RUN", "INCOMPLETE_APP_RUN"]:
            init_ds = False
            print("process another rand dataset")


    def parse_unprocessed_dqn_datasets(self, init=False):
        init_ds = False
        while self.parse_dqn_dataset(init_ds) in ["PROCESSED_APP_RUN", "INCOMPLETE_APP_RUN"]:
            init_ds = False
            print("process another dqn dataset")

    # for training DQN by processing native DQN dataset
    def parse_dqn_dataset(self, init=False):
        print("PDQNDS: >>>>> parse_app_dataset:", init)
        app_dsu = DatasetUtils(self.app_name, "GPT")
        if init:
          # start at the beginning
          # e.g., clear TTT_APP_IDX_PROCESSED_BY_DQN.txt
          app_dsu.save_dataset_idx_processed(mode = "GPT", clear = True )

        frame_num = 0
        reward = []

        ###################################################
        # iterate through NNs and fill in the active buffer
        dqn_index = self.dsu.dataset_indices(mode="GPT",nn_name=None,position="NEXT")
        if dqn_index is None:
          # print("parse_func_dataset: unknown NEXT index")
          print("PDQNDS: parse_dqn_dataset: no more indexes")
          return None
        print("PDQNDS: Parsing DQN idx", dqn_index)

        dqn_filehandle = open(dqn_index, 'r')
        run_complete = False
        line = None
        next_action = None
        next_line = None
        self.active_buffer.clear()
        self.curr_phase = 0
        self.clip_max_reward  = -1
        self.clip_min_reward  = 1
        while True:  
            next_line = dqn_filehandle.readline()
            if not next_line:
                run_complete = True
                print("PDQNDS: Function Index completed:", frame_num, next_action)
                break
            # get action & next_action
            [tm, app, mode, next_nn_name, next_action, img_name, next_state] = self.dsu.get_dataset_info(next_line, mode="GPT")
            if line is not None:
              [tm, app, mode, nn_name, action, img_name, state] = self.dsu.get_dataset_info(line, mode="GPT")
              if action == "NOOP":
                # Too common in initial test dataset. Print warning later?
                # print("NOOP action, NN:", action, nn_name)
                line = next_line
                continue
              elif action == "REWARD1":
                print("PDQNDS: Goto next NN; NOOP Reward, curr_NN", action, nn_name)
                line = next_line
                continue
              else:
                # the final compute reward sets done to True (see above)
                reward, done = self.compute_reward(frame_num, action)
                print("PDQNDS: compute_reward:", frame_num, action, reward, done)
              if next_action == "REWARD1":
                reward, done = self.compute_reward(frame_num, next_action)
                print("PDQNDS: granted REWARD", frame_num, next_action, reward, done)
              elif next_action in ["PENALTY1", "PENALTY2"]:
                reward, done = self.compute_reward(frame_num, next_action)
                done = True  
                print("PDQNDS: assessed run-ending PENALTY", frame_num, next_action, reward, done)
              frame_num += 1
              # add dummy 0 q_val for now. Compute q_val at end of run.
              q_val = 0
              self.active_buffer.push(state, action, reward, next_state, done, q_val)

            if next_action not in ["REWARD1", "PENALTY1", "PENALTY2"]:
              line = next_line
        dqn_filehandle.close()

        if run_complete:
            print("PDQNDS: SAVING STATE; DO NOT STOP!!!")
            done_required = True
            if self.MAX_MOVES_REWARD is not None:
              if self.REWARD1_REWARD is None or self.REWARD1_REWARD == 0:
                 done_required = False
            print("PDQNDS: done_required: ", done_required, self.MAX_MOVES_REWARD, self.REWARD1_REWARD)
            self.active_buffer.compute_real_q_values(gamma=self.GAMMA,  name="active", done_required=done_required)
            self.active_buffer.reset_sample(name="active", start=0)
            print("PDQNDS: parse_app_ds:", self.parse_app_ds_details)
            # for i in range(self.cfg.NUM_DQN_EPOCHS):
              # loss = 0
              # while loss is not None:
                # loss = self.compute_td_loss(batch_size=self.BATCH_SIZE, mode="TARGET_Q_VALUES", compute_stats=False)
                # mean_loss = np.mean(self.all_loss)
                # if len(self.all_loss) > 0:
                  # print("PDQNDS: target q values loss: ", mean_loss, i, self.all_loss[-1])
            # print("loss: ",loss)
            print("PDQNDS: ACTIVE BUFFER:", self.active_buffer)
            # self.replay_buffer.concat(self.active_buffer)
            self.save_replay_buffer()
            # print(self.BEST_MODEL_PATH)
            # self.current_model.save_state(self.BEST_MODEL_PATH)
            # self.update_target(self.current_model, self.target_model)
            app_dsu.save_dataset_idx_processed(mode = "GPT", ds_idx=dqn_index)
            print("PDQNDS: STATE SAVED")
        return "PROCESSED_APP_RUN"

    def nn_process_image(self, NN_num, next_state, reward_penalty = None):
        # NN_num is unused by ddqn, but keeping nn_apps API

        print("DQN process_image")
        if len(self.replay_buffer) > self.REPLAY_INITIAL:
            # assume immitation learning initialization reduces need for pure random actions
            epsilon_start = .3 * self.REPLAY_INITIAL / len(self.replay_buffer)
        else:
            epsilon_start = 1.0
        epsilon_final = 0.01
        epsilon_decay = 30000   # make a config parameter?
        epsilon_by_frame = lambda frame_idx : epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)
        fr_num = self.frame_num + len(self.replay_buffer)
        if fr_num < self.REPLAY_INITIAL:
            print("fr_num", fr_num, self.REPLAY_INITIAL + self.frame_num)
            fr_num = self.REPLAY_INITIAL + self.frame_num
        epsilon = epsilon_by_frame(fr_num)
        if epsilon > 0.1:
            epsilon = 0.1
        rand_num = random.random()
        next_action = None
        sal = []

        if reward_penalty in ["REWARD1", "PENALTY1", "PENALTY2", "REWARD2"]:
            next_action = reward_penalty
            print("reward/penalty: ", next_action)
        # ARD: for debugging supervised learning, minimize random actions
        elif self.app_type == "GPT" and rand_num < epsilon:
            # ARD: We start with supervised learning. The original algorithm starts
            # with pure random actions.  We want to be more focused and add randomness
            # based upon a results of the DQN NN weights.
            func_restrict = self.cfg.get_func_value(self.app_name, "MOVEMENT_RESTRICTIONS")
            while True:
              if func_restrict is not None:
                next_action_num = random.randrange(len(func_restrict))
                next_action = func_restrict[next_action_num]
              else:
                next_action_num = random.randrange(self.num_actions)
                next_action = list(self.cfg.full_action_set)[next_action_num]
              # next_action = list(self.robot_actions)[next_action_num]
              if next_action not in self.cfg.nn_disallowed_actions:
                    print("random action: ", next_action, epsilon, rand_num, fr_num, self.frame_num)
                    break
        else:
            # print("next_state:",next_state)
            # next_state = self.transform_image_from_path(next_state)
            # next_state = self.transform_image(next_state)
            # next_state = torch.unsqueeze(next_state, 0)

            # (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)), weights=((64, 3, 11, 11), (64,)), parameters=23296
            #  weight 64 3 11 11, but got 3-dimensional input of size [224, 224, 3] instead
            # RuntimeError: Given groups=1, weight of size 64 3 11 11, expected input[1, 224, 224, 3] to have 3 channels, but got 224 channels instead

            # print("B4 DQN NN exec")
            next_state = next_state.transpose((2, 0, 1))
            next_state_tensor = self.Variable(torch.FloatTensor(np.float32(next_state)))
            next_state_tensor = torch.unsqueeze(next_state_tensor, 0)
            # print(next_state_tensor)
            # batch_item_num = 0 
            ## Doesn't use CUDA. Use self.Variable()
            # next_state_tensor = torchvision.transforms.ToTensor()(next_state).unsqueeze(batch_item_num) 
            # next_state_tensor = torch.unsqueeze(next_state_tensor, 0)
            # torch.transpose(next_state_tensor, 0, 1)
            overfit = True
            while overfit:
              # sorted_actions, q_value = self.current_model.act(next_state_tensor)
              # top-level DQN movement_restrictions
              func_restrict = self.cfg.get_func_value(self.app_name, "MOVEMENT_RESTRICTIONS")
              sorted_action_order = sorted_actions.tolist()
              # print("sao:",sorted_action_order)
              sal = []
              tot = 0
              q_val = q_value[0].tolist()
              # print("QV:", q_val)
              act_list = []
              restrict_q_list = []
              for i, s_a in enumerate(sorted_action_order):
                  a = self.cfg.full_action_set[s_a]
                  if a in self.cfg.nn_disallowed_actions:
                     continue
                  if func_restrict is None or a in func_restrict:
                    q_val = q_value[0].tolist()
                    sal.append([i, a, q_val[s_a]])
                    restrict_q_list.append(q_val[s_a])
                    if q_val[s_a] > 0:
                      tot += q_val[s_a]
                    if len(act_list) < 3:
                      act_list.append([a, round(q_val[s_a],2)])
              print("DQN top actions: ", act_list)
              min_qv = round(min(restrict_q_list),2)
              max_qv = round(max(restrict_q_list),2)
              mean_qv = round(np.mean(restrict_q_list),2)
              overfit = False
              if max_qv > -self.QVAL_THRESHOLD and max_qv < self.QVAL_THRESHOLD:
                  print("DQN qval results:              MAX ", max_qv, end='')
              else:
                  print("DQN qval results:              max ", max_qv, end='')
                  overfit = True
              if mean_qv > -self.QVAL_THRESHOLD and mean_qv < self.QVAL_THRESHOLD:
                  print("; MEAN ", mean_qv, end='')
              else:
                  print("; mean ", mean_qv, end='')
                  overfit = True
              if min_qv > -self.QVAL_THRESHOLD and min_qv < self.QVAL_THRESHOLD:
                  print("; MIN ", min_qv)
              else:
                  print("; min ", min_qv)
                  overfit = True
              if overfit:
                  # time.sleep(3)
                  # ARD: resort to teleop
                  # nn_apps: return self._driver.gather_data.action_name
                  return None

            # print("all actions:",self.cfg.full_action_set)
            # print("disallowed:", self.cfg.nn_disallowed_actions)
            # print("restrictions:",self.app_name, func_restrict) 

            # next_action = list(self.robot_actions)[next_action_num]
            # SUM_1_16 = 136
            # ARD: hack
            # epsilon = .1
            if rand_num < epsilon:
              # ARD: Random selection weighted by positive q_val
              # q_val = q_value.data[0]
              # find max random number (sum of positive allowed q_vals)
              # tot = 0
              # for act in sorted_actions:
                  # next_action = list(self.cfg.full_action_set)[act]
                  # if q_val[act] <= 0:
                  #    break
                  # if next_action not in self.cfg.nn_disallowed_actions:
                     # tot += q_val[act]
              # rand_num = random.random() * tot
              # while True:
              if True:
                # rand_num = random.random() * SUM_1_16
                rand_num = random.random() * tot
                # find random selection
                tot2 = 0
                # for i, act in enumerate(sorted_actions):
                for i, act in enumerate(sal):
                  # next_action = list(self.cfg.full_action_set)[act]
                  # if q_val[act] <= 0:
                  #    print("ERR: rand select: ", act, next_action, rand_num, tot2, tot)
                  #    break
                  # tot2 += 16 - i
                  if act[1] in self.cfg.nn_disallowed_actions:
                     continue
                  next_action = act[1]
                  if act[2] > 0:   # problem if all negative 
                    tot2 += act[2]
                  else:
                    next_action = sal[0][1]
                    print("chose highest qval: ", sal[0] )
                    break
                  if rand_num <= tot2:
                        self.mean_cnt += 1
                        self.mean_i += act[0]
                        print("rand select: ", act, rand_num, tot2, tot, (self.mean_i/self.mean_cnt))
                        # rand select:  tensor(0, device='cuda:0') FORWARD 47.237562023512126 55 0
                        break
                # if next_action in func_restrict:
                #     print("allowed1")
                #     break
                # elif next_action not in self.cfg.nn_disallowed_actions:
                #    print("allowed2")
                #    break
                if func_restrict is None or next_action in func_restrict:
                    print("allowed:", next_action)
            else:
              for next_action_num in sorted_actions:
                  next_action = list(self.cfg.full_action_set)[next_action_num]
                  if func_restrict is None or next_action in func_restrict:
                    print("allowed1:", next_action)
                    break
                  # elif next_action not in self.cfg.nn_disallowed_actions:
                  #     print("next_action:", next_action)
                  #     break
                  # print("action disallowed:", next_action)
        print("sel_act:",  next_action, sal)
        if self.frame_num == 0 and self.state == None:
          self.frame_num += 1
          self.state  = next_state
          self.action = next_action
          print("NN3: ", next_action)
          return next_action

        # having figured out our next_action based upon the state next_state,
        # evaluate reward for the preceeding move (self.action) based upon self.state / next_state.
        # Note that events like REWARD/PENALTY aren't a real robot move, so these events 
        # are ignored and the previous real move is used (self.prev_action, self.prev_state)
        phase = self.curr_phase     # changes upon reward
        action_reward, action_done = self.compute_reward(self.frame_num, self.action)
        print("reward: ", self.frame_num, self.action, action_reward, action_done)
        self.total_reward += action_reward

        if self.action not in ["REWARD1","PENALTY1", "PENALTY2"]:
          # self.replay_buffer or active_buffer????  
          # add dummy 0 q_val for now. Compute q_val at end of run.
          self.active_buffer.push(self.state, self.action, action_reward, next_state, action_done, 0)
        else:
          # self.replay_buffer or active_buffer????  
          # add dummy 0 q_val for now. Compute q_val at end of run.
          self.active_buffer.push(self.prev_state, self.prev_action, action_reward, self.state, action_done, 0)

        if len(self.replay_buffer) > self.REPLAY_INITIAL:
          # loss = self.compute_td_loss(batch_size=self.BATCH_SIZE, mode="EXPERIENCE_REPLAY")
          # print("experience replay loss: ",loss)

        # if frame_num % 10000 == 0:
        #    plot(frame_num, self.all_rewards, self.losses)
          pass
    
        if action_done:
            self.active_buffer.compute_real_q_values(gamma=self.GAMMA,  name="active")
            self.active_buffer.reset_sample(name="active", start=0)
            loss = 0
            # while loss is not None:
              # loss = self.compute_td_loss(batch_size=self.BATCH_SIZE, mode="REAL_Q_VALUES")
              # print("real q values loss: ",loss)
            # print("loss: ",loss)
            self.replay_buffer.concat(self.active_buffer)
            self.save_replay_buffer()
            # self.current_model.save_state(self.BEST_MODEL_PATH)
            # torch.save(model.state_dict(), self.BEST_MODEL_PATH)
            # self.update_target(self.current_model, self.target_model)
            self.prev_state = None
            self.state = None
            self.curr_phase = 0
            # self.all_rewards.append(self.total_reward)
            self.frame_num = 0
            print("Episode is done; reset robot and cube")
            return "DONE"
        else:
            self.frame_num += 1
            self.prev_state = self.state 
            self.state = next_state 
            self.prev_action = self.action 
            self.action = next_action 
            
        print("NN4: ", next_action)
        return next_action

    # APP training is done automatically at start of DQN.  Just running DQN will pick up
    # any new datasets since it left off.  
    #
    # FUNC training based on a single end-to-end run of a random selection of FUNC/NN runs.
    # FUNC is done by calling:
    #   ./copy_to_python ; python3 ./alset_jetbot_train.py --dqn TTT
    def train(self, number_of_random_func_datasets = 30):
        # Check if any app training left undone:
        print("Checking if any Functional APP training to do...")
        self.parse_unprocessed_app_datasets(init=self.init_model)
        # TODO: RL training of DQN
        # print("Checking if any DQN training to do...")
        # self.parse_unprocessed_dqn_datasets(init=self.init_model)
        print("Train DQN based upon random selection of Functional Runs...")
        self.parse_unprocessed_rand_datasets(init=self.init_model)
        # for i in range(number_of_random_func_datasets):
        #   self.parse_func_datasets(init=self.init_model)
        print("Train on DQN-specific dataset...")
        self.parse_unprocessed_dqn_datasets(init=self.init_model)

    def create_dataset(self, num_buffers, num_steps, game, data_dir_prefix, trajectories_per_buffer):
        # -- load data from memory (make more efficient)
        obss = []
        actions = []
        returns = [0]
        done_idxs = []
        stepwise_returns = []
    
        transitions_per_buffer = np.zeros(50, dtype=int)
        num_trajectories = 0
        init1 = True
        init2 = True
        if True:
            print("Train on APP dataset...", len(obss), num_steps)
            self.init_model = init1
            init1 = False
            self.parse_unprocessed_app_datasets(init=self.init_model)
            print("Train on DQN-specific dataset...")
            self.init_model = init2
            init2 = False
            self.parse_unprocessed_dqn_datasets(init=self.init_model)

            buffer_num = np.random.choice(np.arange(50 - num_buffers, 50), 1)[0]
            i = transitions_per_buffer[buffer_num]
            print('loading from buffer %d which has %d already loaded' % (buffer_num, i))
            # get the next Replay buffer of a TT Pick/Drop run. Put in memory.
    #        frb = FixedReplayBuffer(
    #            data_dir=data_dir_prefix + game + '/1/replay_logs',
    #            replay_suffix=buffer_num,
    #            observation_shape=(84, 84),
    #            stack_size=4,
    #            update_horizon=1,
    #            gamma=0.99,
    #            observation_dtype=np.uint8,
    #            batch_size=32,
    #            replay_capacity=100000)
    #        if frb._loaded_buffers:
            # replay_capacity=100000
            batch_size = 90
            batch_sz = 1 
            # self.replay_buffer  = ReplayBuffer(capacity=replay_capacity, name="replay")
    
            # if True:
            done = False
            curr_num_transitions = len(obss)
            trajectories_to_load = trajectories_per_buffer
            replay_state_num = 0 
            replay_run_num = 0 
            all_done = False
            last_obss = 1000000
            # while len(obss) < num_steps:
            while not all_done:
                print("num obss, steps", len(obss), num_steps)
                if len(obss) == last_obss:
                  break
                last_obss = len(obss) 
                while not done:
                    state_path, action, rewards, next_state_path, done_val, q_val = self.replay_buffer.get_next_sample(batch_sz)
                    # states, ac, ret, next_states, next_action, next_reward, terminal, indices = frb.sample_transition_batch(batch_size=1, indices=[i])
                    if state_path is None:
                      print("state:",state_path, action, rewards, next_state_path, done_val, q_val)
                      all_done = True
                      break
                    states = self.transform_image_from_path(state_path)
                    states = states.transpose((0, 3, 1, 2))[0] # (1, 84, 84, 4) --> (4, 84, 84)


        #  state_embeddings = self.state_encoder(states.reshape(-1, 4, 84, 84).type(torch.float32).contiguous()) # (batch * block_size, n_embd)
        # state_embeddings = state_embeddings.reshape(states.shape[0], states.shape[1], self.config.n_embd) # (batch, block_size, n_embd)
        # RuntimeError: shape '[128, 30, 128]' invalid input of size 368640
        # RuntimeError: shape '[128, 30, 96]' invalid input of size 276480
        # RuntimeError: shape '[128, 30, 72]' invalid input of size 207360
        # RuntimeError: shape '[96, 30, 128]' invalid input of size 276480
        # RuntimeError: shape '[128, 30, 136]' invalid input of size 391680
        # RuntimeError: shape '[90, 32, 136]' invalid input of size 276480
        # RuntimeError: shape '[90, 30, 128]' invalid input of size 259200

        #
        # --block_size 90 --epochs 5 --model_type 'reward_conditioned' 
        # --num_steps 500000 --num_buffers 50 --game 'Breakout' 
        # --batch_size 128 --data_dir_prefix [DIRECTORY_NAME]
        #
        # batch = 90, block_size = 32, n_embd = 128
        # 368640 = 128 * 32 * 90
        # 368640 = 128 * 30 * 96
        # 368640 = 128 * 45 * 64
        # 28224  = 84*84*4
        #
        # states: (batch, block_size, 4*84*84)
        # actions: (batch, block_size, 1)
        # targets: (batch, block_size, 1)
        # rtgs: (batch, block_size, 1)
        # timesteps: (batch, 1, 1)

                    obss += [states]
                    ac = self.cfg.full_action_set.index(action[0])
                    # actions += [action[0]]
                    actions += [ac]
                    stepwise_returns += [rewards[0]]
                    if done_val[0]:
                        done_idxs += [len(obss)]
                        returns += [0]
                        if trajectories_to_load == 0:
                            done = True
                        else:
                            trajectories_to_load -= 1
                    replay_state_num += 1 
                    if done_val[0]:
                      print("replay sample:", replay_run_num, replay_state_num, action[0], done_val[0], trajectories_to_load)
                      replay_state_num = 0 
                      replay_run_num += 1 
                       
                    returns[-1] += rewards[0]
                    i += 1
                    if i >= 100000:
                        obss = obss[:curr_num_transitions]
                        actions = actions[:curr_num_transitions]
                        stepwise_returns = stepwise_returns[:curr_num_transitions]
                        returns[-1] = 0
                        i = transitions_per_buffer[buffer_num]
                        done = True
                num_trajectories += (trajectories_per_buffer - trajectories_to_load)
                transitions_per_buffer[buffer_num] = i
            print('this buffer has %d loaded transitions and there are now %d transitions total divided into %d trajectories' % (i, len(obss), num_trajectories))
    
        actions = np.array(actions)
        returns = np.array(returns)
        stepwise_returns = np.array(stepwise_returns)
        done_idxs = np.array(done_idxs)
    
        # -- create reward-to-go dataset
        start_index = 0
        rtg = np.zeros_like(stepwise_returns)
        for i in done_idxs:
            i = int(i)
            curr_traj_returns = stepwise_returns[start_index:i]
            for j in range(i-1, start_index-1, -1): # start from i-1
                rtg_j = curr_traj_returns[j-start_index:i-start_index]
                rtg[j] = sum(rtg_j)
            start_index = i
        print('max rtg is %d' % max(rtg))
    
        # -- create timestep dataset
        start_index = 0
        timesteps = np.zeros(len(actions)+1, dtype=int)
        for i in done_idxs:
            i = int(i)
            timesteps[start_index:i+1] = np.arange(i+1 - start_index)
            start_index = i+1
        print('max timestep is %d' % max(timesteps))
    
        return obss, actions, returns, done_idxs, rtg, timesteps
    
##################################################

class StateActionReturnDataset(Dataset):

    def __init__(self, data, block_size, actions, done_idxs, rtgs, timesteps):        
        self.block_size = block_size
        self.vocab_size = max(actions) + 1
        self.data = data
        self.actions = actions
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps
    
    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        block_size = self.block_size // 3
        done_idx = idx + block_size
        for i in self.done_idxs:
            if i > idx: # first done_idx greater than idx
                done_idx = min(int(i), done_idx)
                break
        idx = done_idx - block_size
        states = torch.tensor(np.array(self.data[idx:done_idx]), dtype=torch.float32).reshape(block_size, -1) # (block_size, 4*84*84)
        states = states / 255.
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(1) # (block_size, 1)
        rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:idx+1], dtype=torch.int64).unsqueeze(1)

        return states, actions, rtgs, timesteps

##################################################
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123)
# parser.add_argument('--context_length', type=int, default=29)
# parser.add_argument('--context_length', type=int, default=28) # block_size / 3
parser.add_argument('--context_length', type=int, default=8) # block_size / 3
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--model_type', type=str, default='reward_conditioned')
parser.add_argument('--num_steps', type=int, default=500000)
parser.add_argument('--num_buffers', type=int, default=8)
# parser.add_argument('--num_buffers', type=int, default=28)
parser.add_argument('--game', type=str, default='Breakout')
parser.add_argument('--batch_size', type=int, default=8) # total size of data % batch_size = 0
# parser.add_argument('--batch_size', type=int, default=56) # total size of data % batch_size = 0
# parser.add_argument('--batch_size', type=int, default=96)
#
# parser.add_argument('--trajectories_per_buffer', type=int, default=28, help='Number of trajectories to sample from each of the buffers.')
parser.add_argument('--trajectories_per_buffer', type=int, default=1, help='Number of trajectories to sample from each of the buffers.')
parser.add_argument('--data_dir_prefix', type=str, default='./dqn_replay/')
args = parser.parse_args()

set_seed(args.seed)

##################################################
# Main script
##################################################
alset_gpt = ALSET_GPT(app_name = "TT", capac = 100000)
#                                                                             num_buffers, num_steps, game, data_dir_prefix, trajectories_per_buffer):
# obss, actions, returns, done_idxs, rtgs, timesteps = alset_gpt.create_dataset(50, 500000, "TT", "./apps/ALSET_GPT_DATASET/", 128)
# obss, actions, returns, done_idxs, rtgs, timesteps = alset_gpt.create_dataset(56, 500000, "TT", "./apps/ALSET_GPT_DATASET/", 56)
obss, actions, returns, done_idxs, rtgs, timesteps = alset_gpt.create_dataset(8, 500000, "TT", "./apps/ALSET_GPT_DATASET/", 56)

# set up logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

#                                        data, block_size, actions, done_idxs, rtgs, timesteps):        
train_dataset = StateActionReturnDataset(obss, args.context_length*3, actions, done_idxs, rtgs, timesteps)
# train_dataset = StateActionReturnDataset(obss, args.context_length, actions, done_idxs, rtgs, timesteps)

# mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
#                   n_layer=6, n_head=8, n_embd=128, model_type=args.model_type, max_timestep=max(timesteps))
# mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
#                 n_layer=6, n_head=8, n_embd=96, model_type=args.model_type, max_timestep=max(timesteps))
#                 n_layer=6, n_head=8, n_embd=72, model_type=args.model_type, max_timestep=max(timesteps))
#                 n_layer=6, n_head=8, n_embd=56, model_type=args.model_type, max_timestep=max(timesteps))
mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                  n_layer=6, n_head=8, n_embd=8, model_type=args.model_type, max_timestep=max(timesteps))
model = GPT(mconf)
print("model block_size:", model.get_block_size)

# initialize a trainer instance and kick off training
epochs = args.epochs
tconf = TrainerConfig(max_epochs=epochs, batch_size=args.batch_size, learning_rate=6e-4,
                      lr_decay=True, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*args.context_length*3,
                      num_workers=4, seed=args.seed, model_type=args.model_type,game=args.game, max_timestep=max(timesteps))
trainer = Trainer(model, train_dataset, None, tconf)

trainer.train()

# Looks like we need many more runs:
# Atari had 500K timesteps played in paper
#    - we have 17923 loaded transitions
#    - we have same complex 84x84x4 input
# Gym had 1M timesteps, 25-50K timesteps and we're much more complex than reacher.
#    - input is much simpler
#    - output is more complex than reacher, but less than mobility gyms (quad)
#
# We have only 56 runs, but there are 125million parameters to tune
# Max trajectory was 435timesteps.
#
# We need a big context length. 50 for Pong, 30 for others.
# In our case, cause and effect are even farther part (between pickup/dropoff)
#
# Maybe after doing the phased auto-learning, we'll have enough to build upon.
