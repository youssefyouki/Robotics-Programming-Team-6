#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np

"""
In the setup.py, change the entry point to include this file's name without the brackets

    entry_points={
        'console_scripts': [
            'kinematics = panda_kinematics.<file_name>:main',
        ],
    },
"""
# ==========================================================
# KINEMATICS IMPLEMENTATION — EDIT ONLY THIS SECTION
# ==========================================================

# Standard DH parameters for links 1 through 7.
#
# Format: one row per link, columns are [alpha, a, d].
# theta is the joint variable and is not stored here.
#
# Derive these from panda.xml. See the guide, Section 6.
DH_PARAMS = [
    # [alpha, a, d],   # link 1
    # [alpha, a, d],   # link 2
    # [alpha, a, d],   # link 3
    # [alpha, a, d],   # link 4
    # [alpha, a, d],   # link 5
    # [alpha, a, d],   # link 6
    # [alpha, a, d],   # link 7
]

# Fixed transform from link7 to the 'hand' body.
# 4x4 numpy array.
#
# Derive this from the <body name="hand"> tag in panda.xml.
# See the guide, Section 6.
T_L7_HAND = None

# Joint limits for joints 1 through 7, in radians.
# numpy arrays of shape (7,).
#
# Read them from the <joint> tags in panda.xml.
# See the guide, Section 6.
Q_MIN = None
Q_MAX = None

# The target pose the node will try to reach.
# 4x4 numpy array in the 'link0' frame.
#
# Pick a joint configuration inside the limits, run it through
# your forward_kinematics, and use the result. See the guide,
# Section 8.
TARGET_POSE = None


def dh_transform(alpha, a, d, theta):
    """
    Single-link transformation matrix for standard DH.

    alpha: twist angle (rad)
    a:     link length (m)
    d:     link offset (m)
    theta: joint angle (rad)

    Returns a 4x4 numpy array.
    """
    pass


def forward_kinematics(q):
    """
    Forward kinematics for the Panda arm.

    q: numpy array of 7 joint angles (rad).

    Returns a 4x4 homogeneous transform of the 'hand' frame
    relative to 'link0'.
    """
    pass


def inverse_kinematics(target_pose, q0):
    """
    Inverse kinematics for the Panda arm.

    Any correct method is acceptable — numerical, analytical,
    or a mix of both. See the guide, Section 8.

    target_pose: 4x4 desired pose of the 'hand' frame in 'link0'.
    q0:          numpy array of 7 joint angles to seed the solve.

    Returns a tuple (q_solution, converged_flag), where
    q_solution is a numpy array of 7 joint angles and
    converged_flag is a bool.
    """
    pass


# =================================================================
# DO NOT MODIFY BELOW (OR MODIFY, IF YOU KNOW WHAT YOU ARE DOING)
# =================================================================

CMD_SIZE = 9
GRIPPER_CTRL = 255.0


class PandaKinematicsNode(Node):
    def __init__(self):
        super().__init__('panda_kinematics')
        self.q_current = np.zeros(7)

        self.create_subscription(
            JointState, 'joint_state', self.on_joint_state, 10)
        self.cmd_pub = self.create_publisher(
            Float64MultiArray, 'joint_commands', 10)

        self.create_timer(0.05, self.tick)

        if TARGET_POSE is not None:
            self.get_logger().info(
                f"Target pose:\n{np.round(TARGET_POSE, 4)}")
        else:
            self.get_logger().warn(
                "TARGET_POSE is None — set it before running.")

    def on_joint_state(self, msg):
        self.q_current = np.array(msg.position[:7])

    def tick(self):
        if TARGET_POSE is None:
            return

        result = inverse_kinematics(TARGET_POSE, self.q_current)
        q_sol, ok = result

        if not ok:
            self.get_logger().warn(
                "IK did not converge.",
                throttle_duration_sec=2.0)

        cmd = Float64MultiArray()
        cmd.data = q_sol.tolist() + [GRIPPER_CTRL, 0.0]
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = PandaKinematicsNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
