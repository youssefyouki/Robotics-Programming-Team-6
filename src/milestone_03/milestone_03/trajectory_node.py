#!/usr/bin/env python3
"""
Cartesian trajectory generator node (Milestone 03).

Publishes the desired end-effector position, velocity and acceleration (analytic,
minimum-jerk based) at a fixed rate. Trajectory time advances one tick per message,
independent of wall-clock jitter, so samples are exactly reproducible.

  Publishes  : low_cost_robot_arm/ee/desired        [t, p, v, a, gripper]
               low_cost_robot_arm/trajectory_done   std_msgs/Bool (once, when finished)
  Parameters : trajectory   'circle' | 'pick_place'
               rate         publish rate [Hz]
               start_delay  wall-clock seconds to wait for the other nodes to connect
               radius, period, cycles   circle parameters
               loop         repeat the pick-and-place cycle forever
"""
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float64MultiArray

from milestone_03 import kinematics as K
from milestone_03 import topics as T
from milestone_03.trajectories import CircleTrajectory, PickPlaceTrajectory


class TrajectoryNode(Node):
    def __init__(self):
        super().__init__("trajectory_node")
        self.declare_parameter("trajectory", "circle")
        self.declare_parameter("rate", 100.0)
        self.declare_parameter("start_delay", 3.0)
        self.declare_parameter("radius", 0.03)
        self.declare_parameter("period", 3.0)
        self.declare_parameter("cycles", 2)
        self.declare_parameter("loop", False)

        kind = self.get_parameter("trajectory").value
        self.rate = self.get_parameter("rate").value
        start = K.ee_position(np.zeros(5))            # simulator starts at q = 0
        if kind == "circle":
            self.traj = CircleTrajectory(
                start, radius=self.get_parameter("radius").value,
                period=self.get_parameter("period").value,
                cycles=self.get_parameter("cycles").value)
        elif kind == "pick_place":
            self.traj = PickPlaceTrajectory(start, loop=self.get_parameter("loop").value)
        else:
            raise ValueError(f"unknown trajectory '{kind}' (use 'circle' or 'pick_place')")

        self.pub = self.create_publisher(Float64MultiArray, T.EE_DESIRED, T.QOS_DEPTH)
        self.pub_done = self.create_publisher(Bool, T.TRAJECTORY_DONE, 10)
        self.k = 0
        self.finished = False
        self.timer = None
        self.delay_timer = self.create_timer(self.get_parameter("start_delay").value, self._start)
        self.get_logger().info(f"'{kind}' trajectory ready, duration "
                               f"{'inf' if self.traj.loops else f'{self.traj.duration:.1f}'} s.")

    def _start(self):
        self.delay_timer.cancel()
        self.timer = self.create_timer(1.0 / self.rate, self._tick)
        self.get_logger().info("Trajectory started.")

    def _tick(self):
        t = self.k / self.rate
        p, v, a, grip = self.traj.sample(t)
        self.pub.publish(T.pack([t, *p, *v, *a, grip]))
        self.k += 1
        if not self.traj.loops and t >= self.traj.duration and not self.finished:
            self.finished = True
            self.timer.cancel()
            self.pub_done.publish(Bool(data=True))
            self.get_logger().info("Trajectory finished.")


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
