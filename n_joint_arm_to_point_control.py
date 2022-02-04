"""
Based on: https://github.com/AtsushiSakai/PythonRobotics
Inverse kinematics for an n-link arm using the Jacobian inverse method

Author: Daniel Ingram (daniel-s-ingram)
        Atsushi Sakai (@Atsushi_twi)
"""

import numpy as np
import sys
from utilradians import *
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

# from ArmNavigation.n_joint_arm_to_point_control.NLinkArm import NLinkArm
from NLinkArm import NLinkArm
from config import *

# Simulation parameters
Kp = 2
dt = 0.1
N_LINKS = 3
N_ITERATIONS = 10000

# States
WAIT_FOR_NEW_GOAL = 1
MOVING_TO_GOAL = 2

show_animation = True

def main():  # pragma: no cover
    """
    Creates an arm using the NLinkArm class and uses its inverse kinematics
    to move it to the desired position.
    """
    link_lengths, joint_angle_limits, joint_angles, base = alset_arm()
    N_LINKS = len(self.link_lengths)
    goal_pos = [N_LINKS, 0]
    arm = NLinkArm(link_lengths, joint_angles, goal_pos, show_animation)

    state = WAIT_FOR_NEW_GOAL
    solution_found = False
    while True:
        old_goal = np.array(goal_pos)
        goal_pos = np.array(arm.goal)
        end_effector = arm.end_effector
        errors, distance = distance_to_goal(end_effector, goal_pos)

        # State machine to allow changing of goal before current goal has been reached
        if state is WAIT_FOR_NEW_GOAL:
            if distance > 0.1 and not solution_found:
                goal_joint_angles, solution_found = inverse_kinematics(
                    link_lengths, joint_angles, joint_angle_limits, goal_pos)
                if not solution_found:
                    state = WAIT_FOR_NEW_GOAL
                    arm.goal = end_effector
                elif solution_found:
                    state = MOVING_TO_GOAL
        elif state is MOVING_TO_GOAL:
            if distance > 0.1 and all(old_goal == goal_pos):
                joint_angles = joint_angles + Kp * \
                    ang_diff(goal_joint_angles, joint_angles) * dt
            else:
                state = WAIT_FOR_NEW_GOAL
                solution_found = False

        arm.update_joints(joint_angles)

def alset_arm():
    cfg = Config()
    return cfg.ROBOT_ARM_LENGTHS, cfg.ROBOT_ARM_ANGLE_LIMITS, cfg.ROBOT_ARM_INIT_ANGLES, cfg.ROBOT_BASE 

# input: action, img_offset
def get_arm_joint_angle(action, link_lengths, joint_angles, img_offset):
    pass

def inverse_kinematics(link_lengths, joint_angles, joint_angle_limits, goal_pos):
    """
    Calculates the inverse kinematics using the Jacobian inverse method.
    """
    for iteration in range(N_ITERATIONS):
        current_pos = forward_kinematics(link_lengths, joint_angles)
        errors, distance = distance_to_goal(current_pos, goal_pos)
        if distance < 0.1:
            print("Solution found in %d iterations." % iteration)
            return joint_angles, True
        J = jacobian_inverse(link_lengths, joint_angles)
        joint_angles = joint_angles + np.matmul(J, errors)
        for i in range(len(joint_angles)):
          joint_angles[i] = min(joint_angle_limits[i][0], max(joint_angle_limits[i][1], joint_angles[i]))
    return joint_angles, False

def best_move(link_lengths, joint_angles, joint_angle_limits, goal_joint_angles, goal_pos):
    """
    Chooses the single move that puts the goal closest to the center of camera focus 
            while reducing distance to goal
    """
    mv_delta_angles = get_move_delta_angle(joint_angles)
    min_angle_dif = 10000000
    priority = 0
    best_mv = None
    # Kalman filter based on inverse function:
    # joint_angles = joint_angles + Kp * ang_diff(goal_joint_angles, joint_angles) * dt
    #
    # Our Variation:
    # find the move that gets a joint_angle closer to its goal_joint_angle
    # while keeping the center focus of the camera close to goal
    goal_joint_angles = np.array(goal_joint_angles)
    joint_angles = np.array(joint_angles)
    curr_j_dif = ang_diff(goal_joint_angles, joint_angles) 
    for mv,delta in mv_delta_angles:
      j_a = joint_angles.copy()
      if mv.startswith("UPPER"):
        if ((delta < 0 and curr_j_dif[0] < 0 and curr_j_dif[0] < delta) or
            (delta > 0 and curr_j_dif[0] > 0 and curr_j_dif[0] > delta)):
          print("delta, curr_j_dif", delta, curr_j_dif[1]) 
          j_a[0] = joint_angles[0] + delta
          j_a[0] = min(joint_angle_limits[0][0], max(joint_angle_limits[0][1], j_a[0]))
          if j_a[0] == joint_angles[0]:
            continue
        else:
          continue
      elif mv.startswith("LOWER"):
        if ((delta < 0 and curr_j_dif[1] < 0 and curr_j_dif[1] < delta) or
            (delta > 0 and curr_j_dif[1] > 0 and curr_j_dif[1] > delta)):
          print("delta, curr_j_dif", delta, curr_j_dif[1]) 
          j_a[1] = joint_angles[1] + delta
          j_a[1] = min(joint_angle_limits[1][0], max(joint_angle_limits[1][1], j_a[1]))
          if j_a[1] == joint_angles[1]:
            continue
        else:
          continue
      else:
          continue
      camera_angle_dif = camera_angle_delta(link_lengths, j_a, goal_pos)
      if min_angle_dif > camera_angle_dif:
        min_angle_dif = camera_angle_dif
        best_delta = delta
        best_mv = mv
    if best_mv is None:
      return None, joint_angles
    elif best_mv.startswith("UPPER"):
      joint_angles[0] += best_delta
      joint_angles[0] = min(joint_angle_limits[0][0], max(joint_angle_limits[0][1], joint_angles[0]))
    elif best_mv.startswith("LOWER"):
      joint_angles[1] += best_delta
      joint_angles[1] = min(joint_angle_limits[1][0], max(joint_angle_limits[1][1], joint_angles[1]))
    print("best:", best_mv, joint_angles)
    return best_mv, joint_angles

def camera_angle_delta(link_lengths, joint_angles, goal_pos):
    camera_angle = np.sum(joint_angles[:])
    curr_pos = forward_kinematics(link_lengths, joint_angles)
    goal_angle = rad_arctan2((goal_pos[0]-curr_pos[0]), (goal_pos[1]-curr_pos[1]))
    return rad_dif(camera_angle, goal_angle)

def get_move_delta_angle(joint_angles):
    # should be based on alset stats for current position's angle changes
    est_delta_angle = [["UPPER_ARM_UP", 1/32],["UPPER_ARM_DOWN", -1/32],
                       ["LOWER_ARM_UP", 1/32],["LOWER_ARM_DOWN", -1/32]]
    return est_delta_angle

def get_random_goal(base):
    from random import random

    ground_x1 = base[2][1][0]
    ground_y  = base[2][1][1]
    base_x1   = base[1][1][0]
    SAREA = ground_x1 - base_x1
    return [SAREA * random() + base_x1, ground_y]

def animation():
    link_lengths, joint_angle_limits, init_joint_angles, base = alset_arm()
    N_LINKS = len(link_lengths)
    goal_pos = get_random_goal(base)
    arm = NLinkArm(link_lengths, init_joint_angles, base, goal_pos, show_animation)
    state = WAIT_FOR_NEW_GOAL
    solution_found = False
    i_goal = 0
    move = ""
    while True:
        old_goal = np.array(goal_pos)
        goal_pos = np.array(arm.goal)
        end_effector = arm.end_effector  # last link pt (fixed)
        errors, distance = distance_to_goal(end_effector, goal_pos)

        # State machine to allow changing of goal before current goal has been reached
        if state is WAIT_FOR_NEW_GOAL:
            if distance >= .1 and not solution_found:
                goal_joint_angles, solution_found = inverse_kinematics(
                    link_lengths, init_joint_angles, joint_angle_limits, goal_pos)
                if not solution_found:
                    print("Solution could not be found.")
                    state = WAIT_FOR_NEW_GOAL
                    arm.goal = get_random_goal(base)
                elif solution_found:
                    state = MOVING_TO_GOAL
                joint_angles = init_joint_angles.copy()
        elif state is MOVING_TO_GOAL:
            if abs(distance) >= .1 and all(old_goal == goal_pos):
              move, joint_angles = best_move(
                    link_lengths, joint_angles, joint_angle_limits, goal_joint_angles, goal_pos)
              if move is None:
                state = WAIT_FOR_NEW_GOAL
                solution_found = False
                arm.goal = get_random_goal(base)
            else:
                state = WAIT_FOR_NEW_GOAL
                solution_found = False
                arm.goal = get_random_goal(base)
        arm.update_joints(joint_angles)


def forward_kinematics(link_lengths, joint_angles):
    x = y = 0
    for i in range(1, N_LINKS + 1):
        x += link_lengths[i - 1] * np.cos(np.sum(joint_angles[:i]))
        y += link_lengths[i - 1] * np.sin(np.sum(joint_angles[:i]))
    return np.array([x, y]).T


def jacobian_inverse(link_lengths, joint_angles):
    J = np.zeros((2, N_LINKS))
    for i in range(N_LINKS):
        J[0, i] = 0
        J[1, i] = 0
        for j in range(i, N_LINKS):
            J[0, i] -= link_lengths[j] * np.sin(np.sum(joint_angles[:j]))
            J[1, i] += link_lengths[j] * np.cos(np.sum(joint_angles[:j]))

    return np.linalg.pinv(J)


def distance_to_goal(current_pos, goal_pos):
    x_diff = goal_pos[0] - current_pos[0]
    y_diff = goal_pos[1] - current_pos[1]
    return np.array([x_diff, y_diff]).T, np.hypot(x_diff, y_diff)


def ang_diff(theta1, theta2):
    """
    Returns the difference between two angles in the range -pi to +pi
    """
    return (theta1 - theta2 + np.pi) % (2 * np.pi) - np.pi


if __name__ == '__main__':
    # main()
    animation()
