#!/usr/bin/env python3
"""
IK node for Koch v1.1 — Milestone 02

Geometric (decoupled) inverse kinematics solver.

The arm extends in the −X world direction when all joints are at zero.
The solver works in the arm's vertical plane after decoupling the base yaw q1.

Derivation summary
──────────────────
Given target (x, y, z):

  1. Base yaw:
       q1 = atan2(y, −x)
       (the minus sign because the arm points in −X at q1=0)

  2. Arm-plane coordinates (r measured outward from J2 along the arm direction,
     h measured upward from J2):
       r_EE = √(x²+y²) − J1_X·x/√(x²+y²)   # correct for J1 x-offset
       h_EE = z − D2                           # correct for J2 height

  3. Wrist-centre back-projection (strip off the EE offset along approach axis):
       r_WC = r_EE − L_EE·cos(approach_pitch)
       h_WC = h_EE − L_EE·sin(approach_pitch)

  4. Two-link cosine rule (upper arm L2, forearm L3):
       cos(elbow_angle) = (r²+h² − L2² − L3²) / (2·L2·L3)
       q3 = elbow_angle − Δβ        where Δβ = β2 − β1

  5. Shoulder angle:
       γ = atan2(r_WC, h_WC)
       δ = atan2(L3·sin(elbow_angle), L2 + L3·cos(elbow_angle))
       q2 = β1 − (γ − δ)

  6. Wrist pitch to maintain desired EE approach:
       q4 = approach_pitch − q2 − q3

  7. Wrist roll (direct assignment):
       q5 = wrist_roll

ros2 run milestone_02 ik_node
  Subscribes : low_cost_robot_arm/ik_target     (geometry_msgs/PointStamped)
  Publishes  : low_cost_robot_arm/joint_commands (std_msgs/Float64MultiArray)
  Parameters : approach_pitch [rad, default 0.0]  — EE pitch from horizontal
               wrist_roll     [rad, default 0.0]

Send a target from the terminal:
  ros2 topic pub --once /low_cost_robot_arm/ik_target \\
    geometry_msgs/PointStamped "{point: {x: -0.15, y: 0.0, z: 0.08}}"
"""

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float64MultiArray

# Forward kinematics is imported to verify the IK solution after solving
from milestone_02.fk_node import forward_kinematics


# ──────────────────────────────────────────────────────────────────────────────
# Physical constants extracted from the MJCF model
# ──────────────────────────────────────────────────────────────────────────────

# X-offset of Joint 1 (base_rotation) from the world origin [m]
_J1_X = 0.012

# Height of Joint 2 (shoulder/pitch) above the ground [m]
# = base_rotation z-offset (0.0409) + pitch body z-offset (0.0154)
_D2 = 0.0563

# Upper-arm link length: distance from J2 (shoulder) to J3 (elbow) [m]
# Computed as the norm of the elbow body's offset vector in the MJCF
_L2 = float(np.linalg.norm([-0.0148, 0.0065, 0.1083]))   # ≈ 0.1095 m

# Forearm link length: distance from J3 (elbow) to J4 (wrist pitch) [m]
_L3 = float(np.linalg.norm([-0.10048, 5e-5, 0.0027]))    # ≈ 0.1005 m

# Wrist-to-EE offset: distance from J4 (wrist pitch) to J5 (wrist roll) [m]
_LEE = float(np.linalg.norm([-0.045, 0.013097, 0.0]))    # ≈ 0.0469 m

# ── Initial link angles from vertical at the home configuration ────────────
# These arise because the MJCF link bodies are not perfectly vertical at q=0.
# β1: angle of the upper arm from +z when q2=0  (r-component = 0.0148 m)
_B1 = np.arctan2(0.0148,  0.1083)    # ≈ 0.136 rad  (+7.77°)
# β2: angle of the forearm from +z  when q3=0  (r-component = 0.10048 m)
_B2 = np.arctan2(0.10048, 0.0027)    # ≈ 1.544 rad  (+88.5°)
# Natural elbow bend — the difference in resting angles between the two links
_DB = _B2 - _B1                      # ≈ 1.408 rad  (80.7°)

# ── Joint limits (from MJCF <joint range=...>) ─────────────────────────────
_LIMITS = [
    (-2.2,   2.2  ),   # q1: base rotation
    (-1.57,  0.6  ),   # q2: shoulder pitch
    (-1.57,  1.45 ),   # q3: elbow
    (-2.0,   2.0  ),   # q4: wrist pitch
    (-3.14,  3.14 ),   # q5: wrist roll
]


# ──────────────────────────────────────────────────────────────────────────────
# Solver
# ──────────────────────────────────────────────────────────────────────────────

def inverse_kinematics(target: list,
                       approach_pitch: float = 0.0,
                       wrist_roll: float = 0.0):
    """
    Compute joint angles that place the EE at target with the given approach.

    Parameters
    ----------
    target         : [x, y, z] desired EE position in world frame [m]
    approach_pitch : desired EE pitch angle from horizontal [rad]
                     0 = level, −π/2 = pointing straight down
    wrist_roll     : desired wrist roll angle [rad]

    Returns
    -------
    List of 5 joint angles [q1, q2, q3, q4, q5] in radians,
    or None if the target is out of reach.
    Joint values outside limits are clamped with a warning printed.
    """
    x, y, z = target

    # Horizontal distance from world z-axis to the target point
    r_xy = np.sqrt(x**2 + y**2)
    if r_xy < 1e-6:
        return None   # directly above/below the base — geometrically degenerate

    # ── Step 1: Base yaw ───────────────────────────────────────────────────
    # At q1=0 the arm extends in the −X direction, so the sign of x is flipped
    q1 = np.arctan2(y, -x)

    # ── Step 2: Arm-plane coordinates of the EE ────────────────────────────
    # r_EE: radial distance from J2 projected along the arm direction
    # The J1 x-offset shifts the effective origin; J2 y-offset cancels out
    r_ee = r_xy - _J1_X * x / r_xy
    # h_EE: height of the EE above J2
    h_ee = z - _D2

    # ── Step 3: Back-project to the wrist centre (WC) ──────────────────────
    # Strip the EE-to-WC link along the approach direction
    r_wc = r_ee - _LEE * np.cos(approach_pitch)
    h_wc = h_ee - _LEE * np.sin(approach_pitch)

    # ── Step 4: Two-link cosine rule ───────────────────────────────────────
    D2 = r_wc**2 + h_wc**2          # squared distance from J2 to WC
    D  = np.sqrt(D2)                 # actual distance

    if D > _L2 + _L3:
        return None   # WC too far — arm fully extended cannot reach
    if D < abs(_L2 - _L3):
        return None   # WC too close — arm folded cannot reach

    # Angle between the two links (elbow opening angle)
    cos_e = np.clip((D2 - _L2**2 - _L3**2) / (2.0 * _L2 * _L3), -1.0, 1.0)
    elbow_angle = np.arccos(cos_e)   # always in [0, π]

    # Convert elbow opening angle to the MJCF joint coordinate
    q3 = elbow_angle - _DB           # q3 = elbow_angle − (β2 − β1)

    # ── Step 5: Shoulder angle ─────────────────────────────────────────────
    # γ: angle from +z to the WC vector in the arm plane
    gamma = np.arctan2(r_wc, h_wc)
    # δ: offset due to the forearm triangle (law of sines)
    delta  = np.arctan2(_L3 * np.sin(elbow_angle),
                        _L2 + _L3 * np.cos(elbow_angle))
    # θ1: absolute angle of the upper arm from +z (arm-plane)
    theta1 = gamma - delta
    # Convert to MJCF joint coordinate using the resting angle β1
    q2 = _B1 - theta1

    # ── Step 6: Wrist pitch — maintains the desired EE approach direction ──
    # The sum q2+q3+q4 equals the total EE pitch from horizontal
    q4 = approach_pitch - q2 - q3

    # ── Step 7: Wrist roll — direct assignment ─────────────────────────────
    q5 = wrist_roll

    q = [q1, q2, q3, q4, q5]

    # Clamp any joint that exceeds its physical limit and warn the caller
    for i, (lo, hi) in enumerate(_LIMITS):
        if not (lo <= q[i] <= hi):
            print(f'  [WARN] joint {i+1} = {np.degrees(q[i]):.1f}° out of '
                  f'[{np.degrees(lo):.0f}°, {np.degrees(hi):.0f}°] — clamped')
            q[i] = float(np.clip(q[i], lo, hi))

    return q


# ──────────────────────────────────────────────────────────────────────────────
# ROS2 node
# ──────────────────────────────────────────────────────────────────────────────

class IKNode(Node):
    """
    ROS2 node: reads a 3D target point → solves IK → publishes joint commands.

    On each ik_target message it:
      1. Reads approach_pitch and wrist_roll from ROS parameters.
      2. Calls the geometric IK solver.
      3. Verifies the solution with forward kinematics and logs the error.
      4. Publishes the 6-element joint command (5 joints + gripper=0).

    Subscribe  : low_cost_robot_arm/ik_target     (geometry_msgs/PointStamped)
    Publish    : low_cost_robot_arm/joint_commands (std_msgs/Float64MultiArray)
    Parameters : approach_pitch [rad]  — EE pitch from horizontal
                 wrist_roll     [rad]  — wrist roll angle
    """

    def __init__(self):
        super().__init__('ik_node')

        # ROS parameters let the user tune the approach angle without restarting
        self.declare_parameter('approach_pitch', 0.0)   # 0 = EE level
        self.declare_parameter('wrist_roll',     0.0)   # 0 = no roll

        # Subscriber: each incoming target triggers one IK solve
        self.sub = self.create_subscription(
            PointStamped,
            'low_cost_robot_arm/ik_target',
            self._cb, 1)

        # Publisher: sends joint position commands to the simulation
        self.pub = self.create_publisher(
            Float64MultiArray,
            'low_cost_robot_arm/joint_commands', 1)

        self.get_logger().info(
            'IK node started. '
            'Send targets to /low_cost_robot_arm/ik_target '
            '(geometry_msgs/PointStamped).')

    def _cb(self, msg: PointStamped):
        """
        IK target callback.

        Reads the desired EE position, solves the IK, verifies the solution
        via FK, and publishes the resulting joint command.
        """
        # Read tuning parameters at call time (can be updated via ros2 param set)
        pitch = self.get_parameter('approach_pitch').value
        roll  = self.get_parameter('wrist_roll').value

        target = [msg.point.x, msg.point.y, msg.point.z]

        # ── Solve IK ───────────────────────────────────────────────────────
        q = inverse_kinematics(target, approach_pitch=pitch, wrist_roll=roll)

        if q is None:
            # Target is geometrically unreachable (too far or too close)
            self.get_logger().warn(
                f'Target {[f"{v:.3f}" for v in target]} m is unreachable.')
            return

        # ── Verify solution with FK ────────────────────────────────────────
        # Compute the EE position predicted by FK for the solved joint angles
        T   = forward_kinematics(q)
        err = np.linalg.norm(T[:3, 3] - np.array(target)) * 1000  # in mm

        # ── Publish joint command ──────────────────────────────────────────
        # The simulation expects 6 values: 5 arm joints + gripper (held at 0)
        out = Float64MultiArray()
        out.data = [float(v) for v in q] + [0.0]   # gripper = 0 (no change)
        self.pub.publish(out)

        # Log the result: joint angles in degrees and FK position error
        self.get_logger().info(
            f'IK solved  q = {[f"{np.degrees(v):+.1f}°" for v in q]}'
            f'  FK err = {err:.1f} mm')


def main(args=None):
    rclpy.init(args=args)
    node = IKNode()
    try:
        rclpy.spin(node)          # block, processing callbacks until Ctrl-C
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
