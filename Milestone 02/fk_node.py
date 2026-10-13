#!/usr/bin/env python3
"""
FK node for Koch v1.1 — Milestone 02

Implements Standard (Craig) Denavit-Hartenberg forward kinematics
derived from the student's hand-written DH frame assignment.

DH Table:
  Link | θᵢ  | dᵢ  | aᵢ  | αᵢ
   1   |  q1  |  l1 |  0  | -π/2   base yaw   — rotates about world z
   2   |  q2  |  0  |  l2 |   0    shoulder   — rotates about z1 (horizontal)
   3   |  q3  |  0  |  l3 |   0    elbow      — rotates about z2
   4   |  q4  |  0  |  0  | +π/2   wrist pitch— introduces the roll axis
   5   |  q5  |  l5 |  0  |   0    wrist roll — EE offset along z4

Link lengths (measured from MuJoCo simulation):
  l1 = 0.049 m  — base to shoulder height (along world z)
  l2 = 0.110 m  — upper arm  (shoulder → elbow)
  l3 = 0.101 m  — forearm    (elbow → wrist)
  l5 = 0.047 m  — wrist to EE offset (along z4)

ros2 run milestone_02 fk_node
  Subscribes : low_cost_robot_arm/joint_state  (sensor_msgs/JointState)
  Publishes  : low_cost_robot_arm/ee_pose       (geometry_msgs/PoseStamped)
"""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped


# ──────────────────────────────────────────────────────────────────────────────
# DH link parameters (from the student's hand-derived DH table)
# ──────────────────────────────────────────────────────────────────────────────

_L1 = 0.049   # d1: base → shoulder height along world z  [m]
_L2 = 0.110   # a2: upper-arm length, shoulder → elbow    [m]
_L3 = 0.101   # a3: forearm length,   elbow → wrist       [m]
_L5 = 0.047   # d5: wrist → EE offset along z4            [m]

# Each row is (d, a, alpha); theta = q_i is added at runtime.
# Order matches the joint_state topic: base_rotation, pitch, elbow, wrist_pitch, wrist_roll
_DH_PARAMS = [
    (_L1,  0.0,  -np.pi / 2),  # Link 1: d=l1, a=0,  α=−90°
    ( 0.0, _L2,   0.0      ),  # Link 2: d=0,  a=l2, α=  0°
    ( 0.0, _L3,   0.0      ),  # Link 3: d=0,  a=l3, α=  0°
    ( 0.0,  0.0,  np.pi / 2),  # Link 4: d=0,  a=0,  α=+90°
    (_L5,   0.0,  0.0      ),  # Link 5: d=l5, a=0,  α=  0°
]


# ──────────────────────────────────────────────────────────────────────────────
# Kinematics functions
# ──────────────────────────────────────────────────────────────────────────────

def _dh_matrix(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """
    Build one standard DH homogeneous transformation matrix.

    Implements:  T = Rot_z(θ) · Trans_z(d) · Trans_x(a) · Rot_x(α)

    Expands to:
        ┌ cθ  -sθ·cα   sθ·sα   a·cθ ┐
        │ sθ   cθ·cα  -cθ·sα   a·sθ │
        │  0     sα      cα      d   │
        └  0      0       0      1   ┘

    Parameters
    ----------
    theta : joint angle [rad]
    d     : link offset along previous z-axis [m]
    a     : link length along common normal   [m]
    alpha : twist angle about common normal   [rad]
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa,  a * ct],
        [st,  ct * ca, -ct * sa,  a * st],
        [ 0,       sa,       ca,       d],
        [ 0,        0,        0,       1],
    ])


def forward_kinematics(q: list) -> np.ndarray:
    """
    Compute the end-effector homogeneous transform via DH chain multiplication.

        T_EE = ⁰T₁(q1) · ¹T₂(q2) · ²T₃(q3) · ³T₄(q4) · ⁴T₅(q5)

    Parameters
    ----------
    q : list of 5 joint angles [rad]
        [base_rotation, pitch, elbow, wrist_pitch, wrist_roll]

    Returns
    -------
    numpy 4×4 array.
        T[:3, 3]  → EE position [m]
        T[:3, :3] → EE rotation matrix
    """
    T = np.eye(4)                                    # start from world frame
    for (d, a, alpha), qi in zip(_DH_PARAMS, q):
        T = T @ _dh_matrix(qi, d, a, alpha)          # chain each link transform
    return T


def _rot_to_quat(R: np.ndarray):
    """
    Convert a 3×3 rotation matrix to a unit quaternion (x, y, z, w).

    Uses the numerically stable Shepperd branching method to avoid
    division by near-zero values depending on which diagonal element is largest.
    """
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w, x = 0.25 / s,               (R[2, 1] - R[1, 2]) * s
        y, z = (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w, x = (R[2, 1] - R[1, 2]) / s, 0.25 * s
        y, z = (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w, x = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s
        y, z = 0.25 * s,               (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w, x = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s
        y, z = (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return float(x), float(y), float(z), float(w)


# ──────────────────────────────────────────────────────────────────────────────
# ROS2 node
# ──────────────────────────────────────────────────────────────────────────────

class FKNode(Node):
    """
    ROS2 node: reads joint states → runs DH FK → publishes EE pose.

    Callback flow on each joint_state message:
      1. Extract q1..q5 in kinematic order from the message.
      2. Compute T_EE = ⁰T₁·¹T₂·²T₃·³T₄·⁴T₅ using DH matrices.
      3. Convert rotation matrix to quaternion.
      4. Publish PoseStamped to ee_pose and print position to terminal.

    Subscribe  : low_cost_robot_arm/joint_state  (sensor_msgs/JointState)
    Publish    : low_cost_robot_arm/ee_pose       (geometry_msgs/PoseStamped)
    """

    # Joint names must match the MJCF model exactly, in kinematic order
    _ORDER = ['base_rotation', 'pitch', 'elbow', 'wrist_pitch', 'wrist_roll']

    def __init__(self):
        super().__init__('fk_node')

        # Publisher: broadcasts the computed EE pose at the joint_state rate
        self.pub = self.create_publisher(
            PoseStamped, 'low_cost_robot_arm/ee_pose', 1)

        # Subscriber: each incoming joint_state triggers one FK computation
        self.sub = self.create_subscription(
            JointState, 'low_cost_robot_arm/joint_state', self._cb, 1)

        self.get_logger().info(
            'FK node started — using standard DH kinematics.')

    def _cb(self, msg: JointState):
        """
        Joint-state callback.

        Extracts angles, runs forward kinematics, builds and publishes
        a PoseStamped, then logs the EE position to the terminal.
        """
        # Build the joint-angle vector in kinematic order;
        # any joint missing from the message defaults to 0.
        q = [msg.position[msg.name.index(n)] if n in msg.name else 0.0
             for n in self._ORDER]

        # ── Forward kinematics ─────────────────────────────────────────────
        # T_EE = ⁰T₁(q1) · ¹T₂(q2) · ²T₃(q3) · ³T₄(q4) · ⁴T₅(q5)
        T = forward_kinematics(q)

        # ── Build PoseStamped ──────────────────────────────────────────────
        ps = PoseStamped()
        ps.header = msg.header           # inherit timestamp and frame_id

        # Position: extracted from the last column of T
        ps.pose.position.x = float(T[0, 3])
        ps.pose.position.y = float(T[1, 3])
        ps.pose.position.z = float(T[2, 3])

        # Orientation: rotation matrix → unit quaternion (x, y, z, w)
        qx, qy, qz, qw = _rot_to_quat(T[:3, :3])
        ps.pose.orientation.x = qx
        ps.pose.orientation.y = qy
        ps.pose.orientation.z = qz
        ps.pose.orientation.w = qw

        self.pub.publish(ps)

        # ── Terminal readout ───────────────────────────────────────────────
        self.get_logger().info(
            f'EE pos  x={T[0,3]:+.4f}  y={T[1,3]:+.4f}  z={T[2,3]:+.4f} m  |  '
            f'q [deg] {[f"{np.degrees(v):+.1f}" for v in q]}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = FKNode()
    try:
        rclpy.spin(node)          # block, processing callbacks until Ctrl-C
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
