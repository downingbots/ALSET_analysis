<p align="center">
  <img src="https://raw.githubusercontent.com/downingbots/ALSET/master/ReadMeImages/alset5.png" width="600" alt="accessibility text">
</p> 

# ALSET ANALYSIS for ALSET Autonomous Vehicles 
## Inexpensive Autonomous RC Vehicles with Manipulators

I think machine vision is the key to robotics. After years of working on large ROS-based robots (since C-Turtle days), I've changed my focus to the supporting following:

1. Many inexpensive low-end robots working together to provide high functionality. The robot is a mobile manipulator, where the manipulator could be an arm, a shovel, a payload delivery system, etc.

2. Provide a high-level description of the behavior you want to achieve, demo it a few times and have the robot take over from there.  The robots continue to learn and improve over time with minimal supervision and guidance.

To accomplish #1, low-end RC toys and vehicles like robot toys with usable arms and high-quality construction RC vehicles like excavators, dump-trucks, fork-lifts, working fire engines can be easily retrofitted to become automated.  The RC vehicles essentially have a camera and a jetson and a huge SD card.  There's an interface between the Jetson and the RC control to enable automated control. In addition, you can use your own joystick to provide initial demonstrations and for overriding commands or assigning rewards / penalties.  The only sensor is a camera.  There's no odometry other than the visual odometry.

The software is written in python.  So far, the prototype software integrates:
 - visual odometry and mapping
 - rat-slam (based purely upon monocular camera)
 - discrete open-cv processing of actions while taking auto-labeled pictures.
 - storing of per-move state for post-run and cross-run analysis.
 - simple high-level behavior-based specification
 - post-run analysis for labeling bounding boxes and training the NNs
 - arm navigation planning
 - teleoperation
 - integrated Yolo training and execution
 - simple pretrained NNs (e.g., Alexnet) that can be tuned to provide jetbot-like functionality such as line following, table-top obstacle avoidance, facial recognition, etc.)
 - ddqn reinforcement learning
 
The software and hardware are still evolving. The training of complex behaviors requires way too many runs (I feel sorry for the PhD students doing reinforcement learning!).  I am trying to set up support so that generic OpenCV-based algorithms can limp through a more controlled environment until the robot can self-train YOLO object recognition (of an object to pick up, of a location, of a container drop-off, of a sign, etc.). Next, yolo-based identification will automatically replace any opencv algorithms to drive more generic functionality (search for object, goto object, pick-up object, goto destination, dropoff object, etc.) The yolo-centric functionality is then used to gather more data for even higher-level NN functionality and end-to-end reinforcement learning on real (non-simulated) robots.

This is a tough problem to crack. Approaches like open-cv programming are way too brittle and limited, while reinforcement learning requires an obscene amount of data for training. The goal here is to take advantage of the strengths of each approach to iteratively build up more complex funcionality. A side-goal is to hugely reduce the cost of entry.  For example, I started with an RC robot with an arm on clearance ($20 each!), added a $100 Jetson and a pi-camera, a 500 Gig SD card, and an inexpensive I/O control board, and I've got an amazing robot that has the functionality of a robot that costs thousands.  Essentially the same setup was done to automate an incredible $130 RC Excavator. The integrated software system will not require anything close to the complexity to understand/use/maintain ROS.

This is very much a work-in-progress.  More details of the project can be found at:

https://github.com/downingbots/ALSET

https://github.com/downingbots/ALSET_analysis

The first repository provides the runtime functionality. Run the Jetbot-like functions via teleop to get the dataset to train and then run a NN.  In theory, you can gather data and run DDQN end-to-end reinforcement learning for more advanced functionality.  In reality, I don't believe anybody would have the patience to gather a big enough training set to get things to work.  That's where the ALSET_analysis repository kicks in.

In the ALSET_analysis repository, you find human-programmed functionality (as opposed to NNs) that could be used with a single monocular camera to analyze runs done by the software in the ALSET repository. The goal is to do a quick demo of the desired functionality in a more controlled environment and have the robot take over to gather more and more data as discussed above. Hopefully, we can build more and more functionality over time that can transfer across different robots.  This project is a work in progress, and borders upon being a research project.


Thanks,

Alan 

DowningBots at gmail.com

