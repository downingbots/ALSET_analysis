"""
Based on: https://github.com/AtsushiSakai/PythonRobotics
Note: started with n_joint_arm_to_point_control.py, but 
  simple approach purely based upon jacobian inverses didn't 
  handle joint limits or the base (obstacle).  Added here.

Obstacle navigation using A* on a toroidal grid

Author: Daniel Ingram (daniel-s-ingram)
        Tullio Facchinetti (tullio.facchinetti@unipv.it)
"""
from math import pi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import from_levels_and_colors
import sys
from config import *

plt.ion()


class ArmNavigation(object):

    def __init__(self):
        # Simulation parameters
        # circle.obstacles = [[1.75, 0.75, 0.6], [0.55, 1.5, 0.5], [0, -1, 0.7]]
        self.circle_obstacles = []
        self.square_obstacles = []
        # Arm geometry in the working space
        self.link_length, self.link_angle_limit, self.init_link_angle, self.base = self.alset_arm()
        # there are three "spaces" here:
        # - actual size (in inches, from config.py)
        # - plot space (Normalized along length of arm to 2)
        # - grid space (2 plot_space == 200 grid_space)

        # convert to horiz axis
        # for i in range(len(self.init_link_angle)):
        #   self.init_link_angle[i] += np.pi
 
        # actual size
        total_link_len = 0
        for i in range(len(self.link_length)):
          total_link_len += self.link_length[i]
      
        
        self.plot_size = 2   # to the negative and positive
        self.normalize_factor = self.plot_size/total_link_len
        self.M = 200
        self.grid_factor = self.M / (self.plot_size*2)
        # convert
        for i in range(len(self.link_length)):
          self.link_length[i] = self.link_length[i] * self.normalize_factor
        for i in range(len(self.base)):
          # base is a series of line segments defined by points
          self.base[i][0][0] = self.base[i][0][0] * self.normalize_factor
          self.base[i][0][1] = self.base[i][0][1] * self.normalize_factor
          self.base[i][1][0] = self.base[i][1][0] * self.normalize_factor
          self.base[i][1][1] = self.base[i][1][1] * self.normalize_factor
        self.arm = NLinkArm(self.link_length, self.init_link_angle, self.base)
        self.n_links = len(self.link_length)
        self.link_angle = self.init_link_angle.copy()
        # (x, y) co-ordinates in the joint space [cell]
        curr_pos = self.forward_kinematics() 
        # start_node = [(curr_pos[0]+self.plot_size)*self.grid_factor, (curr_pos[1]+self.plot_size)*self.grid_factor]
        self.start_node = [(curr_pos[1]+self.plot_size)*self.grid_factor, (curr_pos[0]+self.plot_size)*self.grid_factor]
        print("start_node:", curr_pos, self.start_node)

        self.goal = self.get_random_goal()  # in plot space (as a float)
        # goal_node = [int((self.goal[0]+self.plot_size)*self.grid_factor), int((self.goal[1]+self.plot_size)*self.grid_factor)]
        goal_node = [int((self.goal[1]+self.plot_size)*self.grid_factor), int((self.goal[0]+self.plot_size)*self.grid_factor)]
        print("INIT goal:", self.goal, goal_node)
        self.grid = self.get_occupancy_grid(self.arm, self.circle_obstacles)
        self.route = self.astar_torus(self.grid, self.start_node, goal_node)
        if len(self.route) >= 0:
            self.animate(self.grid, self.arm, self.route, self.circle_obstacles, self.goal)
    
    def press(self, event):
        """Exit from the simulation."""
        if event.key == 'q' or event.key == 'Q':
            print('Quitting upon request.')
            sys.exit(0)
    
    def get_random_goal(self):
        from random import random

        ground_x1 = self.base[2][1][0]
        ground_y  = self.base[2][1][1]
        base_x1   = self.base[1][1][0]
        print("ground_x1, ground_y, base_x1:", ground_x1, ground_y, base_x1)
        SAREA = ground_x1 - base_x1
        return [(SAREA * random() + base_x1), (ground_y)]
    
    def alset_arm(self):
        cfg = Config()
        return cfg.ROBOT_ARM_LENGTHS, cfg.ROBOT_ARM_ANGLE_LIMITS, cfg.ROBOT_ARM_INIT_ANGLES, cfg.ROBOT_BASE
    
    def camera_angle_delta(self):
        camera_angle = self.ang_diff(np.sum(self.link_angle[:]), np.pi/2)
        curr_pos = self.forward_kinematics(self.link_angle)
        goal_angle = self.ang_diff(np.arctan2((self.goal[0]-curr_pos[0]), (self.goal[1]-curr_pos[1])), 0)
        delta = abs(self.ang_diff(camera_angle, goal_angle))
        print("cam,goal angle: ", camera_angle, goal_angle, delta)
        return delta
    
    def forward_kinematics(self, link_length = None, link_angle = None):
        if link_length is None:
          link_length = self.link_length
        if link_angle is None:
          link_angle = self.link_angle
        x = y = 0
        for i in range(1, len(link_length) + 1):
            x += link_length[i - 1] * np.cos(np.sum(link_angle[:i]))
            y += link_length[i - 1] * np.sin(np.sum(link_angle[:i]))
        return np.array([x, y]).T
    
    def get_move_delta_angle(self):
        # should be based on alset stats for current position's angle changes
        # est_delta_angle = [["UPPER_ARM_UP", 1/32],["UPPER_ARM_DOWN", -1/32],
        #                    ["LOWER_ARM_UP", 1/32],["LOWER_ARM_DOWN", -1/32]]
        est_delta_angle = [["UPPER_ARM_UP", 1],["UPPER_ARM_DOWN", -1],
                           ["LOWER_ARM_UP", 1],["LOWER_ARM_DOWN", -1]]
        return est_delta_angle

    def ang_diff(self, theta1, theta2):
        """
        Returns the difference between two angles in the range -pi to +pi
        """
        return (theta1 - theta2 + np.pi) % (2 * np.pi) - np.pi
    
    def animate(self, grid, arm, route, obstacles, goal):
        fig, axs = plt.subplots(1, 2)
        fig.canvas.mpl_connect('key_press_event', self.press)
        colors = ['white', 'black', 'red', 'pink', 'yellow', 'green', 'orange']
        levels = [0, 1, 2, 3, 4, 5, 6, 7]
        cmap, norm = from_levels_and_colors(levels, colors)
        self.link_angle = self.init_link_angle.copy()
        for i, node in enumerate(route):
            plt.subplot(1, 2, 1)
            grid[node] = 6
            plt.cla()
            plt.imshow(grid, cmap=cmap, norm=norm, interpolation=None)
            self.get_arm_theta(node, goal)
            plt.subplot(1, 2, 2)
            arm.plot_arm(plt, goal, circle_obstacles=obstacles)
            plt.xlim(-2.0, 2.0)
            plt.ylim(-3.0, 3.0)
            plt.show()
            # Uncomment here to save the sequence of frames
            # plt.savefig('frame{:04d}.png'.format(i))
            plt.pause(.5)
    
    
    def detect_collision_with_base_or_ground(self, line_segment):
        # note: doing collision detection based on link space, not grid space
        # curr_pos = self.forward_kinematics(self.link_length, arm.joint_angles)
        for i in range(2):
          if (line_segment[i][0] >= self.base[1][0][0] and line_segment[i][0] <= self.base[1][1][0] and line_segment[i][1] <= self.base[0][0][1]):
            # print("line_seg, base", line_segment, self.base)
            return True
          # assumes flat ground
          if (line_segment[i][1] < self.base[2][0][1]):
            # print("line_seg, base", line_segment, self.base[2][0][1])
            return True
        return False
    
    def detect_collision(self, line_seg, circle):
        """
        Determines whether a line segment (arm link) is in contact
        with a circle (obstacle).
        Credit to: http://doswa.com/2009/07/13/circle-segment-intersectioncollision.html
        Args:
            line_seg: List of coordinates of line segment endpoints e.g. [[1, 1], [2, 2]]
            circle: List of circle coordinates and radius e.g. [0, 0, 0.5] is a circle centered
                    at the origin with radius 0.5
    
        Returns:
            True if the line segment is in contact with the circle
            False otherwise
        """
        a_vec = np.array([line_seg[0][0], line_seg[0][1]])
        b_vec = np.array([line_seg[1][0], line_seg[1][1]])
        c_vec = np.array([circle[0], circle[1]])
        radius = circle[2]
        line_vec = b_vec - a_vec
        line_mag = np.linalg.norm(line_vec)
        circle_vec = c_vec - a_vec
        proj = circle_vec.dot(line_vec / line_mag)
        if proj <= 0:
            closest_point = a_vec
        elif proj >= line_mag:
            closest_point = b_vec
        else:
            closest_point = a_vec + line_vec * proj / line_mag
        if np.linalg.norm(closest_point - c_vec) > radius:
            return False
        return True
    
    def get_arm_theta(self, grid_node, goal):
        def get_hypot(len1, len2, angle):
            # hypot: c^2 = a^2 + b^2 - 2ab*cos(C)
            hypot = len1**2 + len2**2 + 2*len1*len2*np.cos(angle)
            return hypot

        curr_pos = self.forward_kinematics(self.link_angle)
        curr_node = [(curr_pos[1]+self.plot_size)*self.grid_factor, (curr_pos[0]+self.plot_size)*self.grid_factor]
        # translate grid coords to link coords
        grid_pos = [0.0, 0.0]
        grid_pos[0] = grid_node[1] / self.grid_factor - self.plot_size
        grid_pos[1] = grid_node[0] / self.grid_factor - self.plot_size
        n_links = self.n_links
        # consider link_angle_limits and link lengths, have the camera point at goal_node
        #
        # in Model S, camera-to-gripper angle is fixed, so only 1 feasible position
        # if need to have camera pointing at goal.
        #
        # we also want to be consistent with init_link_angle 

        fixed_angle = [-1 for i in range(self.n_links)]
        fixed_link  = [False for i in range(self.n_links)]
        # check for always fixed angle 
        for i in range(self.n_links):
          if self.link_angle_limit[i][0] == self.link_angle_limit[i][1]:
            fixed_link[i]  = True
            fixed_angle[i] = self.link_angle_limit[i][0]

        calc_jacobian = True
        while calc_jacobian:
          # initialize known start/end points
          fixed_angle_cnt = 0
          for tf in fixed_link:
            if tf:
              fixed_angle_cnt += 1
          jacobian_n_links = self.n_links - fixed_angle_cnt
          jacobian_angle = [0 for i in range(jacobian_n_links)]
          jacobian_link_length = [-1 for i in range(jacobian_n_links)]

          jacobian_i = 0
          for i in range(self.n_links):
            if fixed_link[i]:
              # Combine the angles. Change the lengths.
              if i == 0 or jacobian_angle == 0:
                jacobian_angle[0] +=  fixed_angle[i]
                if jacobian_link_length[0] == -1:
                  jacobian_link_length[0] = self.link_length[i]
                else:
                  jacobian_link_length[0] = get_hypot(jacobian_link_length[0], 
                                            self.link_length[i], jacobian_angle[0])
                continue    # do not increment jacobian_i
              else:
                jacobian_angle[jacobian_i-1] += jacobian_angle[jacobian_i-1] + fixed_angle[i]
                jacobian_link_length[jacobian_i-1] = get_hypot(
                               jacobian_link_length[jacobian_i-1], self.link_length[i],
                               jacobian_angle[0])
                continue    # do not increment jacobian_i
            else:
              print("jacob_i, len_jacob, i:", jacobian_i, len(jacobian_link_length), i) 
              jacobian_angle[jacobian_i] += self.link_angle[i]
              if jacobian_link_length[jacobian_i] == -1:
                jacobian_link_length[jacobian_i] = self.link_length[i]
              else:
                print("Unexpected jacobian link length")
              jacobian_i += 1
            continue    # calc_jacobian:

          # compute jacobian inverse 
          J = self.jacobian_inverse(jacobian_link_length, jacobian_angle)
          errors, distance = self.distance_to_goal(curr_pos, grid_pos)
          print("jacob angle", jacobian_angle, jacobian_link_length, errors)
          jacobian_angle = jacobian_angle + np.matmul(J, errors)
        
          # transform back the changed jacobian angles and the non-fixed angles
          link_angle = self.link_angle.copy()
          jacobian_i = 0
          new_fixed_angle = False
          for i in range(self.n_links):
            if fixed_link[i]:
              self.link_angles[i] = fixed_angle[i]
              jacobian_angle[jacobian_i] -= fixed_angle[i]
              continue # do not increment jacobian_i
            else:
              self.link_angle[i] = jacobian_angle[jacobian_i]
              jacobian_i += 1
          
            # print(i, "j_a1", jacobian_angles[i], j_a[i])
            if self.link_angle[i] > self.joint_angle_limits[i][0]:
              fixed_link[i]  = True
              fixed_angle[i] = self.link_angle_limit[i][0]
              self.link_angle[i] = self.link_angle_limit[i][0]
              new_fixed_angle = True
            elif jacobian_angles[i] < joint_angle_limits[i][1]:
              fixed_link[i]  = True
              fixed_angle[i] = self.link_angle_limit[i][1]
              self.link_angle[i] = self.link_angle_limit[i][1]
              new_fixed_angle = True
            if new_fixed_angle:
              print("joint angle limit prevents reaching grid_node ",
                    grid_node, i, joint_angle_limits)
          if new_fixed_angle:
            calc_jacobian = True
          else:
            calc_jacobian = False
          continue

        # Transform completed. Check if results are as expected.
        test_pos = self.forward_kinematics(self.link_angle)
        test_node = [(test_pos[1]+self.plot_size)*self.grid_factor, (test_pos[0]+self.plot_size)*self.grid_factor]
        if (test_node[0] == grid_node[0] and test_node[1] == grid_node[1]):
          print("Error: jacobian inverse did not reach  grid_node", 
                 test_node, grid_node, test_pos, grid_pos)
          link_angle = [None for i in range(self.n_links)]
          return False
        arm.update_joints(link_angles)
        return True

    def get_occupancy_grid(self, arm, obstacles):
        """
        Discretizes joint space into M values from -pi to +pi
        and determines whether a given coordinate in joint space
        would result in a collision between a robot arm and obstacles
        in its environment.
    
        Args:
            arm: An instance of NLinkArm
            obstacles: A list of obstacles, with each obstacle defined as a list
                       of xy coordinates and a radius. 
    
        Returns:
            Occupancy grid in joint space
        """
        grid = [[0 for _ in range(self.M)] for _ in range(self.M)]
        theta_list = [2 * i * pi / self.M for i in range(-self.M // 2, self.M // 2 + 1)]
        for i in range(self.M):
            for j in range(self.M):
                # only handles 2 joints
                arm.update_joints([theta_list[i], theta_list[j]])
                points = arm.points
                collision_detected = False
                collision_detected2 = False
                for k in range(len(points) - 1):
                    line_seg = [points[k], points[k + 1]]
                    for obstacle in obstacles:
                        collision_detected = self.detect_collision(line_seg, obstacle)
                        if collision_detected or collision_detected2:
                            break
                    collision_detected2 = self.detect_collision_with_base_or_ground(line_seg)
                    if collision_detected or collision_detected2:
                        break
                grid[i][j] = int(collision_detected)
        return np.array(grid)
    
    
    def astar_torus(self, grid, start_node, goal_node):
        """
        Finds a path between an initial and goal joint configuration using
        the A* Algorithm on a tororiadal grid.
    
        Args:
            grid: An occupancy grid (ndarray)
            start_node: Initial joint configuation (tuple)
            goal_node: Goal joint configuration (tuple)
    
        Returns:
            Obstacle-free route in joint space from start_node to goal_node
        """
        def Grid(grid, lst, num=None):
            if num is not None:
              grid[int(lst[0])][int(lst[1])] = num
            return grid[int(lst[0])][int(lst[1])]

        colors = ['white', 'black', 'red', 'pink', 'yellow', 'green', 'orange']
        levels = [0, 1, 2, 3, 4, 5, 6, 7]
        cmap, norm = from_levels_and_colors(levels, colors)
    
        Grid(grid, start_node, 4)
        Grid(grid, goal_node, 5)
    
        parent_map = [[() for _ in range(self.M)] for _ in range(self.M)]
    
        heuristic_map = self.calc_heuristic_map(self.M, goal_node)
    
        explored_heuristic_map = np.full((self.M, self.M), np.inf)
        distance_map = np.full((self.M, self.M), np.inf)
        Grid(explored_heuristic_map, start_node, Grid(heuristic_map,start_node))
        Grid(distance_map, start_node, 0)
        while True:
            Grid(self.grid, start_node, 4)
            Grid(self.grid, goal_node, 5)
    
            current_node = np.unravel_index(
                np.argmin(explored_heuristic_map, axis=None), explored_heuristic_map.shape)
            min_distance = np.min(explored_heuristic_map)
            print("curr, goal", current_node, goal_node, min_distance)
            # if (current_node[0]==int(goal_node[0]) and current_node[1]==int(goal_node[1])) or np.isinf(min_distance):
            if (current_node[0]==int(goal_node[0]) and current_node[1]==int(goal_node[1])) or np.isinf(min_distance):
                print("Found goal node:", current_node, min_distance)
                break
    
            Grid(self.grid, current_node, 2)
            Grid(explored_heuristic_map, current_node, np.inf)
    
            i, j = current_node[0], current_node[1]
    
            neighbors = self.find_neighbors(i, j)
    
            for neighbor in neighbors:
                # START_NODE = 4 ; GOAL_NODE = 5 ; UNOBSTRUCTED = 0 ; VISITED = 2 ; 
                # CURRENT NEIGHBOR = 3 ; OBSTACLE = 1
                if Grid(self.grid, neighbor) == 0 or Grid(self.grid, neighbor) == 5:
                    Grid(distance_map,neighbor, (Grid(distance_map,current_node) + 1))
                    Grid(explored_heuristic_map,neighbor, Grid(heuristic_map,neighbor))
                    parent_map[neighbor[0]][neighbor[1]] = current_node
                    Grid(self.grid, neighbor, 3)
    
        if np.isinf(Grid(explored_heuristic_map,goal_node)):
            route = []
            print("No route found.")
        else:
            route = [goal_node]
            while parent_map[int(route[0][0])][int(route[0][1])] != ():
                route.insert(0, parent_map[int(route[0][0])][int(route[0][1])])

            print("route", route)
            print("The route found covers %d grid cells." % len(route))
            for i in range(1, len(route)):
                Grid(grid, route[i], 6)
                plt.cla()
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                        lambda event: [exit(0) if event.key == 'escape' else None])
                plt.imshow(grid, cmap=cmap, norm=norm, interpolation=None)
                plt.show()
                plt.pause(1e-2)
                # plt.pause(20)
    
        return route
    
    
    def find_neighbors(self, i, j):
        neighbors = []
        if i - 1 >= 0:
            neighbors.append((i - 1, j))
        else:
            neighbors.append((self.M - 1, j))
    
        if i + 1 < self.M:
            neighbors.append((i + 1, j))
        else:
            neighbors.append((0, j))
    
        if j - 1 >= 0:
            neighbors.append((i, j - 1))
        else:
            neighbors.append((i, self.M - 1))
    
        if j + 1 < self.M:
            neighbors.append((i, j + 1))
        else:
            neighbors.append((i, 0))
    
        return neighbors
    
    
    def calc_heuristic_map(self, M, goal_node):
        X, Y = np.meshgrid([i for i in range(M)], [i for i in range(M)])
        # prioritize those closest to goal
        heuristic_map = np.abs(X - goal_node[1]) + np.abs(Y - goal_node[0])

        camera_angle_dif = self.camera_angle_delta()
        print("camera_angle_dif:",camera_angle_dif)

        for i in range(heuristic_map.shape[0]):
            for j in range(heuristic_map.shape[1]):
                heuristic_map[i, j] = min(heuristic_map[i, j],
                                          i + 1 + heuristic_map[M - 1, j],
                                          M - i + heuristic_map[0, j],
                                          j + 1 + heuristic_map[i, M - 1],
                                          M - j + heuristic_map[i, 0]
                                          ) * camera_angle_dif
    
        return heuristic_map

    def distance_to_goal(self, current_pos, goal_pos):
        x_diff = goal_pos[0] - current_pos[0]
        y_diff = goal_pos[1] - current_pos[1]
        return np.array([x_diff, y_diff]).T, np.hypot(x_diff, y_diff)
    
    def jacobian_inverse(self, link_lengths, joint_angles):
        J = np.zeros((2, self.n_links))
        n_links = len(link_lengths)
        for i in range(n_links):
            J[0, i] = 0
            J[1, i] = 0
            for j in range(i, n_links):
                J[0, i] -= link_lengths[j] * np.sin(np.sum(joint_angles[:j]))
                J[1, i] += link_lengths[j] * np.cos(np.sum(joint_angles[:j]))
        return np.linalg.pinv(J)


class NLinkArm(object):
    """
    Class for controlling and plotting a planar arm with an arbitrary number of links.
    """

    def __init__(self, link_lengths, joint_angles, base):
        self.n_links = len(link_lengths)
        
        if self.n_links != len(joint_angles):
            raise ValueError()

        self.link_lengths = np.array(link_lengths)
        self.joint_angles = np.array(joint_angles)
        self.base = base
        self.points = [[0, 0] for _ in range(self.n_links + 1)]

        self.lim = sum(link_lengths)
        self.update_points()
        print("INIT points", self.points)

    def update_joints(self, joint_angles):
        self.joint_angles = joint_angles
        self.update_points()

    def update_points(self):
        for i in range(1, self.n_links + 1):
            self.points[i][0] = self.points[i - 1][0] + \
                self.link_lengths[i - 1] * \
                np.cos(np.sum(self.joint_angles[:i]))
            self.points[i][1] = self.points[i - 1][1] + \
                self.link_lengths[i - 1] * \
                np.sin(np.sum(self.joint_angles[:i]))
        self.end_effector = np.array(self.points[self.n_links]).T

    def plot_arm(self, myplt, goal, circle_obstacles=[], square_obstacles=[]):  # pragma: no cover
        myplt.cla()

        for obstacle in circle_obstacles:
            circle = myplt.Circle(
                (obstacle[0], obstacle[1]), radius=0.5 * obstacle[2], fc='k')
            myplt.gca().add_patch(circle)

        for obstacle in square_obstacles:
            square = myplt.Rectangle(
                (obstacle[0], obstacle[1]), 0.5*obstacle[2], 0.5*side*obstacle[2], fc='k', ec='k')
            myplt.gca().add_patch(square)

        myplt.plot(goal[0], goal[1], 'go')
        print("goal: ", goal)
        print("points: ", self.points)

        # plot fixed base, ground in black
        for [pt1,pt2] in self.base:
          myplt.plot(pt1[0], pt1[1], 'ko')
          myplt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'k-')
          myplt.plot(pt2[0], pt2[1], 'ko')

        for i in range(self.n_links + 1):
            if i is not self.n_links:
                myplt.plot([self.points[i][0], self.points[i + 1][0]],
                           [self.points[i][1], self.points[i + 1][1]], 'r-')
            myplt.plot(self.points[i][0], self.points[i][1], 'k.')

        myplt.xlim([-self.lim, self.lim])
        myplt.ylim([-self.lim, self.lim])
        myplt.draw()
        # myplt.pause(1e-5)

if __name__ == '__main__':
    arm_nav = ArmNavigation()
