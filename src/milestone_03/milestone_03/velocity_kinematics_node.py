#!/usr/bin/env python3
"""
Forward position / velocity / acceleration kinematics node (Milestone 03).

For every simulator state it evaluates the derived equations

    p = FK(q)
    v = Jv(q) qd
    a = Jv(q) qdd + Jv_dot(q, qd) qd

and publishes them next to what MuJoCo senses (ee/sim), so the two can be compared.

  Subscribes : low_cost_robot_arm/joint_kinematics   [t, q, qd, qdd]
  Publishes  : low_cost_robot_arm/ee/equations       [t, p, v, a]
"""
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from milestone_03 import kinematics as K
from milestone_03 import topics as T


class VelocityKinematicsNode(Node):
    def __init__(self):
        super().__init__("velocity_kinematics_node")
        self.pub = self.create_publisher(Float64MultiArray, T.EE_EQ, T.QOS_DEPTH)
        self.create_subscription(Float64MultiArray, T.JOINT_KINEMATICS, self._cb, T.QOS_DEPTH)
        self.get_logger().info("Forward velocity/acceleration kinematics node started.")

    def _cb(self, msg):
        t, q, qd, qdd = T.unpack_joint_kinematics(msg.data)
        q, qd, qdd = np.array(q), np.array(qd), np.array(qdd)

        p = K.ee_position(q)                          # FK
        v, _ = K.forward_velocity(q, qd)              # Jv qd
        a, _ = K.forward_acceleration(q, qd, qdd)     # Jv qdd + Jv_dot qd
        self.pub.publish(T.pack_ee(t, p, v, a))

        self.get_logger().info(
            f"t={t:7.3f}  p={np.round(p, 4)} m  v={np.round(v, 4)} m/s  a={np.round(a, 3)} m/s^2",
            throttle_duration_sec=1.0)


def main(args=None):
    rclpy.init(args=args)
    node = VelocityKinematicsNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
