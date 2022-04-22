# https://github.com/renatopp/ratslam-python
# Please make a copy of the following directory and its contents:
#   https://github.com/renatopp/ratslam-python/tree/master/ratslam
# into your local directory.
#
# Note: ported this "main.py" file to Python3 with slight modifications for ALSET
#
# =============================================================================
# Federal University of Rio Grande do Sul (UFRGS)
# Connectionist Artificial Intelligence Laboratory (LIAC)
# Renato de Pontes Pereira - rppereira@inf.ufrgs.br
# =============================================================================
# Copyright (c) 2013 Renato de Pontes Pereira, renato.ppontes at gmail dot com
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy 
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights 
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
# copies of the Software, and to permit persons to whom the Software is 
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================

'''
This is a full Ratslam implementation in python. This implementation is based 
on Milford's original implementation [1]_ in matlab, and Christine Lee's python 
implementation [2]_. The original data movies can also be found in [1]_.

This file is the only dependent of OpenCV, which is used to open and convert 
the movie files. Thus, you can change only this file to use other media API.

.. [1] https://wiki.qut.edu.au/display/cyphy/RatSLAM+MATLAB
.. [2] https://github.com/coxlab/ratslam-python
'''

import cv2
import numpy as np
from matplotlib import pyplot as plot
from matplotlib import collections as mc
import mpl_toolkits.mplot3d.axes3d as p3
import math
import ratslam 
from config import *

global ALSET_SLAM 
global prm_road_map 
ALSET_SLAM = ratslam.Ratslam()
from scipy.spatial import KDTree


# RUN A RATSLAM ITERATION ==================================
def alset_ratslam(frame, vtrans=None, vrot=None, overlay=None, armcnt=None):
        print("ratslam frame", vtrans, vrot)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        slam = ALSET_SLAM
        if frame is not None and vtrans is None and vrot is None:
          slam.digest(img)
        elif vtrans is not None and vrot is not None:
          slam.digest(img, vtrans, vrot)
        # ==========================================================

        # PLOT THE CURRENT RESULTS =================================
        b, g, r = cv2.split(frame)
        rgb_frame = cv2.merge([r, g, b])

        # clear current figure
        # plot.clf()

        # RAW IMAGE -------------------
        ax = plot.subplot(3, 2, 1)
        plot.title('RAW IMAGE')
        plot.imshow(rgb_frame)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        # -----------------------------

        # RAW ODOMETRY ----------------
        plot.subplot(3, 2, 2)
        plot.title('RAW ODOMETRY')
        plot.plot(slam.odometry[0], slam.odometry[1])
        plot.plot(slam.odometry[0][-1], slam.odometry[1][-1], color='black', marker='o')
        # plot.plot(slam.odometry[0][-1], slam.odometry[1][-1], 'ko')
        #------------------------------

        # POSE CELL ACTIVATION --------
        ax = plot.subplot(3, 2, 3, projection='3d')
        plot.title('POSE CELL ACTIVATION')
        x, y, th = slam.pc
        ax.plot(x, y, 'x')
        ax.plot3D([0, 60], [y[-1], y[-1]], [th[-1], th[-1]], 'black')
        ax.plot3D([x[-1], x[-1]], [0, 60], [th[-1], th[-1]], 'black')
        ax.plot3D([x[-1], x[-1]], [y[-1], y[-1]], [0, 36], 'black')
        ax.plot3D([x[-1]], [y[-1]], [th[-1]], color='magenta', marker='o')
        # ax.plot3D([x[-1]], [y[-1]], [th[-1]], 'mo')
        ax.grid()
        ax.axis([0, 60, 0, 60]);
        ax.set_zlim(0, 36)
        # -----------------------------

        # EXPERIENCE MAP --------------
        plot.subplot(3, 2, 4)
        plot.title('EXPERIENCE MAP')
        xs = []
        ys = []
        for exp in slam.experience_map.exps:
            xs.append(exp.x_m)
            ys.append(exp.y_m)

        plot.plot(xs, ys, 'bo')
        plot.plot(slam.experience_map.current_exp.x_m,
                  slam.experience_map.current_exp.y_m, 'ko')

        # MAP OVERLAY IMAGE -------------------
        ax = plot.subplot(3, 2, 5)
        plot.title('MAP OVERLAY')
        if overlay is not None:
          plot.imshow(overlay)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        ########################
        # ARD: IGNORE THIS SECTION.  
        # TODO.  The arm representation is not correct. In the process of
        # being replaced by arm_nav.py.
        #
        # ARM VALUES -------------------
        ax = plot.subplot(3, 2, 6)
        plot.title('ARM POSITION')
        # arm_actions = ["UPPER_ARM_UP", "UPPER_ARM_DOWN", "LOWER_ARM_UP", "LOWER_ARM_DOWN"]
        
        #
        UPPER_ARM_LENGTH = 1
        TEN_DEG = .174 - np.pi / 2
        HORIZ_DEG = math.pi * 4 / 3 - TEN_DEG
        UPPER_ARM_UP_ANGLE = TEN_DEG
        UPPER_ARM_DOWN_ANGLE = TEN_DEG
        LOWER_ARM_LENGTH = 1
        LOWER_ARM_UP_ANGLE = TEN_DEG
        LOWER_ARM_DOWN_ANGLE = TEN_DEG
        base = (0,0)
        # arm_actions = ["UPPER_ARM_UP", "UPPER_ARM_DOWN", "LOWER_ARM_UP", "LOWER_ARM_DOWN"]
        if armcnt is None or len(armcnt) == 0:
          angle1 = TEN_DEG
          angle2 = HORIZ_DEG
        else:
          angle1 = - armcnt[0] * (UPPER_ARM_UP_ANGLE+1) - armcnt[1] * UPPER_ARM_DOWN_ANGLE 
          angle1 = max(angle1, -TEN_DEG)
          angle2 = armcnt[2] * LOWER_ARM_UP_ANGLE + armcnt[3] * LOWER_ARM_DOWN_ANGLE 
          angle2 = min(angle2, HORIZ_DEG)

        x1 = math.cos(angle1) * UPPER_ARM_LENGTH
        y1 = math.sin(angle1) * UPPER_ARM_LENGTH
        upper_arm = (x1,y1)
        x2 = x1 + math.cos(angle2) * LOWER_ARM_LENGTH
        y2 = y1 + math.sin(angle2) * LOWER_ARM_LENGTH
        lower_arm = (x2,y2)
        if armcnt is not None:
          print("upper arm angle, x, y", angle1, x1, y1, armcnt[0], armcnt[1])
          print("lower arm angle, x, y", angle2, x2, y2, armcnt[2], armcnt[3])

        lines = [[base, upper_arm], [upper_arm, lower_arm]]
        lc = mc.LineCollection(lines, color='black', linewidths=4)
        ax.add_collection(lc)
        ax.autoscale()
        ax.margins(0.1)
        ########################
        # ARD: END IGNORED SECTION.  
        ########################

        # -----------------------------

        plot.tight_layout()
        # plot.savefig('C:\\Users\\Renato\\Desktop\\results\\forgif\\' + '%04d.jpg'%loop)
        plot.pause(0.1)
        # ==========================================================

        print('n_ templates:', len(slam.view_cells.cells))
        print('n_ experiences:', len(slam.experience_map.exps))



def restart_alset_ratslam():
    ALSET_SLAM = ratslam.Ratslam()

"""
Derived from https://github.com/AtsushiSakai/PythonRobotics
Run probabilistic road map planning
:param start_x: start x position
:param start_y: start y position
:param goal_x: goal x position
:param goal_y: goal y position
:param obstacle_x_list: obstacle x positions (if init_map)
:param obstacle_y_list: obstacle y positions (if init_map)
:param init_map: boolean
:return: planned x/y move
"""
def alset_ratslam_prm_plan(start_x, start_y, goal_x, goal_y, 
                               obstacle_x_list=None, obstacle_y_list=None, init_map=False):
    slam = ALSET_SLAM
    exp_x = [exp.x_m for exp in slam.experience_map.exps]
    exp_y = [exp.y_m for exp in slam.experience_map.exps]
    if init_map: 
      # if slam map is precomputed and no new obstacles, don't recompute roadmap
      cfg = Config()
      robot_radius = np.sqrt(cfg.ROBOT_WIDTH**2 + cfg.ROBOT_LENGTH_W_TREAD**2) / 2
      obstacle_kd_tree = KDTree(np.vstack((obstacle_x_list, obstacle_y_list)).T)
      if prm_show_animation:
        plt.plot(exp_x, exp_y, ".b")
      # road_map is a global
      prm_road_map = prm_generate_road_map(exp_x, exp_y, robot_radius, obstacle_kd_tree)
    rx, ry = prm_dijkstra_planning( start_x, start_y, goal_x, goal_y, 
                                    prm_road_map, exp_x, exp_y)
    return rx, ry

