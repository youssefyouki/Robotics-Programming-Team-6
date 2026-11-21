#!/usr/bin/env python3
"""
Motor-level joint controller node (Milestone 04).

Pairs the simulator state and the planner reference with the same time stamp, evaluates the
selected control law (see control.py) and answers with the actuator command for that stamp.

  Subscribes : m4/state, m4/reference
  Publishes  : m4/actuator_cmd [t, u(5), gripper_ref]
  Parameters : controller  'servo' | 'pd_gravity' | 'computed_torque'
"""
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from milestone_04 import control, topics as T


class ControllerNode(Node):
    def __init__(self):
        super().__init__("controller_node")
        self.declare_parameter("controller", "computed_torque")
        name = self.get_parameter("controller").value
        self.ctrl = control.make_controller(name)
        self.state, self.ref = {}, {}
        self.pub = self.create_publisher(Float64MultiArray, T.ACTUATOR_CMD, T.QOS_DEPTH)
        self.create_subscription(Float64MultiArray, T.STATE, self._on_state, T.QOS_DEPTH)
        self.create_subscription(Float64MultiArray, T.REFERENCE, self._on_ref, T.QOS_DEPTH)
        self.get_logger().info(f"Controller '{name}' ({self.ctrl.kind} commands) ready.")

    def _on_state(self, msg):
        self.state[msg.data[0]] = msg.data
        self._try(msg.data[0])

    def _on_ref(self, msg):
        self.ref[msg.data[0]] = msg.data
        self._try(msg.data[0])

    def _try(self, t):
        if t not in self.state or t not in self.ref:
            return
        s, r = self.state.pop(t), self.ref.pop(t)
        for old in [k for k in self.state if k < t - 1.0]:
            del self.state[old]
        for old in [k for k in self.ref if k < t - 1.0]:
            del self.ref[old]
        q, qd, grip_q = np.array(s[1:6]), np.array(s[6:11]), s[11]
        q_ref, qd_ref, qdd_ref, grip_ref = (np.array(r[1:6]), np.array(r[6:11]),
                                            np.array(r[11:16]), r[16])
        u = self.ctrl(q, qd, q_ref, qd_ref, qdd_ref, grip_q=grip_q)
        self.pub.publish(T.pack([t, *u, grip_ref]))


def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    except RuntimeError:
        if rclpy.ok():                      # a real error; during shutdown it is just a teardown race
            raise
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
