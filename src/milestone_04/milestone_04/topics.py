#!/usr/bin/env python3
"""
Topics and message layouts of the Milestone 04 sorting system (std_msgs/Float64MultiArray).

  m4/state         sim -> planner, controller, evaluation   [t, q1..q5, qd1..qd5, gripper_q, epoch]
  m4/reference     planner -> controller, evaluation         [t, q_ref(5), qd_ref(5), qdd_ref(5), gripper_ref, phase]
  m4/actuator_cmd  controller -> sim, evaluation             [t, u1..u5, gripper_ref, kind]
                   u = joint position targets (kind 0, servo) or joint torques (kind 1, torque controllers);
                   the simulator switches its actuator mode with `kind`, so a controller change is atomic
  m4/boxes         sim -> evaluation                         [t, (x, y, z, yaw, in_correct_bin) per colour]
  m4/boxes_initial sim -> planner (transient local)          same layout, sent once at start
  m4/status        planner -> all                            std_msgs/String  "<colour>: <phase>"
  m4/done          planner -> all                            std_msgs/Bool    plan finished
  m4/command       panel -> planner                          std_msgs/String  JSON motion command
  m4/sim_control   panel -> sim                              std_msgs/String  JSON {pause|resume|reset|speed}
  m4/controller    panel -> controller                       std_msgs/String  servo | pd_gravity | computed_torque
  m4/controller_info controller -> panel (transient local)   std_msgs/String  JSON description of the active controller
  m4/monitor       controller -> panel (10 Hz)               [t, phase, max joint error rad, TCP error mm]
  m4/validation    controller -> panel                       std_msgs/String  JSON result of every finished segment
  m4/plan_path     planner -> view node (transient local)    [n, (x, y, z, type) * n]  planned tool-point path; type 0 joint-space,
                                                              1 task-space, 2 manual; n = 0 clears it
  m4/view_state    sim -> view node                          [t, qpos...] state of everything, for rendering
  m4/image         view node -> panel                        sensor_msgs/CompressedImage  rendered view (JPEG)
  m4/camera        panel -> view node                        std_msgs/String  JSON camera / size control

`epoch` increases every time the simulation is reset, so the planner can drop its old plan.

The system runs in lock-step: the simulator publishes the state at time t and does not advance
until the controller answers with the command stamped t, so results do not depend on message
timing or CPU load.
"""
from std_msgs.msg import Float64MultiArray

STATE = "low_cost_robot_arm/m4/state"
REFERENCE = "low_cost_robot_arm/m4/reference"
ACTUATOR_CMD = "low_cost_robot_arm/m4/actuator_cmd"
BOXES = "low_cost_robot_arm/m4/boxes"
BOXES_INITIAL = "low_cost_robot_arm/m4/boxes_initial"
STATUS = "low_cost_robot_arm/m4/status"
DONE = "low_cost_robot_arm/m4/done"
COMMAND = "low_cost_robot_arm/m4/command"
SIM_CONTROL = "low_cost_robot_arm/m4/sim_control"
VIEW_STATE = "low_cost_robot_arm/m4/view_state"
IMAGE = "low_cost_robot_arm/m4/image"
CONTROLLER = "low_cost_robot_arm/m4/controller"
CONTROLLER_INFO = "low_cost_robot_arm/m4/controller_info"
MONITOR = "low_cost_robot_arm/m4/monitor"
VALIDATION = "low_cost_robot_arm/m4/validation"
PLAN_PATH = "low_cost_robot_arm/m4/plan_path"
CAMERA = "low_cost_robot_arm/m4/camera"
JOINT_STATE = "low_cost_robot_arm/joint_state"

PHASES = ["approach", "descend", "grasp", "lift", "transfer", "lower", "release", "retreat", "home", "done", "manual"]
QOS_DEPTH = 200


def pack(values):
    msg = Float64MultiArray()
    msg.data = [float(v) for v in values]
    return msg
