#!/usr/bin/env python3
"""
Topic names and message layouts shared by the Milestone 03 nodes.

All kinematic data travel as std_msgs/Float64MultiArray with a fixed layout:

  joint_kinematics      [t, q1..q5, qd1..qd5, qdd1..qdd5]      (16)  simulator state
  joint_kinematics_cmd  same layout                            (16)  kinematic-mode input
  ee/sim                [t, px,py,pz, vx,vy,vz, ax,ay,az]      (10)  sensed by MuJoCo
  ee/equations          same layout                            (10)  computed with J, Jdot
  ee/desired            [t, p(3), v(3), a(3), gripper]         (11)  Cartesian reference

`t` is the simulation / trajectory time and is echoed unchanged along the pipeline so the
validation node can pair samples exactly.
"""
from std_msgs.msg import Float64MultiArray

JOINT_COMMANDS = "low_cost_robot_arm/joint_commands"          # 6 joint positions (existing topic)
JOINT_STATE = "low_cost_robot_arm/joint_state"                # sensor_msgs/JointState (existing)
JOINT_KINEMATICS = "low_cost_robot_arm/joint_kinematics"
JOINT_KINEMATICS_CMD = "low_cost_robot_arm/joint_kinematics_cmd"
EE_SIM = "low_cost_robot_arm/ee/sim"
EE_EQ = "low_cost_robot_arm/ee/equations"
EE_DESIRED = "low_cost_robot_arm/ee/desired"
TRAJECTORY_DONE = "low_cost_robot_arm/trajectory_done"       # std_msgs/Bool

QOS_DEPTH = 100


def pack(values):
    msg = Float64MultiArray()
    msg.data = [float(v) for v in values]
    return msg


def pack_ee(t, p, v, a, *extra):
    return pack([t, *p, *v, *a, *extra])


def unpack_joint_kinematics(data):
    """-> (t, q, qd, qdd) from a joint_kinematics[_cmd] message payload."""
    return data[0], list(data[1:6]), list(data[6:11]), list(data[11:16])


def unpack_ee(data):
    """-> (t, p, v, a) from an ee/* message payload."""
    return data[0], list(data[1:4]), list(data[4:7]), list(data[7:10])
