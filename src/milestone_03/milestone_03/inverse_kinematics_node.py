#!/usr/bin/env python3
"""
Inverse position / velocity / acceleration kinematics node (Milestone 03).

For every desired Cartesian state (p_d, v_d, a_d) it solves

    q   = IK(p_d)                         (Newton / damped least squares, warm-started)
    qd  = J+ v_d                          (damped pseudo-inverse, 3x5 Jacobian)
    qdd = J+ (a_d - Jv_dot qd)

and publishes the result for the simulator.

  Subscribes : low_cost_robot_arm/ee/desired               [t, p, v, a, gripper]
  Publishes  : low_cost_robot_arm/joint_commands           6 positions (dynamic mode)
               low_cost_robot_arm/joint_kinematics_cmd     [t, q, qd, qdd] (kinematic mode)
  Parameters : damping (float) - damped least-squares factor
"""
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from milestone_03 import kinematics as K
from milestone_03 import topics as T


class InverseKinematicsNode(Node):
    def __init__(self):
        super().__init__("inverse_kinematics_node")
        self.declare_parameter("damping", 1e-3)
        self.q_prev = None                            # warm start for the next IK solve

        self.pub_cmd = self.create_publisher(Float64MultiArray, T.JOINT_COMMANDS, T.QOS_DEPTH)
        self.pub_kin = self.create_publisher(Float64MultiArray, T.JOINT_KINEMATICS_CMD, T.QOS_DEPTH)
        self.create_subscription(Float64MultiArray, T.EE_DESIRED, self._cb, T.QOS_DEPTH)
        self.get_logger().info("Inverse kinematics node started; waiting for ee/desired.")

    def _cb(self, msg):
        t = msg.data[0]
        p_d, v_d, a_d = (np.array(msg.data[i:i + 3]) for i in (1, 4, 7))
        gripper = msg.data[10]
        damping = self.get_parameter("damping").value

        q, err = K.inverse_position(p_d, q0=self.q_prev, damping=damping)
        if err > 1e-4:
            self.get_logger().warn(f"Target {np.round(p_d, 3)} unreachable (residual {err:.1e} m).",
                                   throttle_duration_sec=2.0)
            return
        self.q_prev = q

        qd = K.inverse_velocity(q, v_d, damping=damping)
        qdd = K.inverse_acceleration(q, qd, a_d, damping=damping)

        # position command for the actuators (dynamic mode); gripper is passed straight through
        self.pub_cmd.publish(T.pack([*q, gripper]))
        # full kinematic state (kinematic mode)
        self.pub_kin.publish(T.pack([t, *q, *qd, *qdd]))

        self.get_logger().info(
            f"t={t:7.3f}  q={np.round(q, 3)}  qd={np.round(qd, 3)}  qdd={np.round(qdd, 2)}",
            throttle_duration_sec=1.0)


def main(args=None):
    rclpy.init(args=args)
    node = InverseKinematicsNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
