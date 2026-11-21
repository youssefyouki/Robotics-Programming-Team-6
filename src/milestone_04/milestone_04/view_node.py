#!/usr/bin/env python3
"""
View node: renders the simulation for the control panel (Milestone 04).

Keeps its own copy of the model, takes the joint positions the simulator publishes, renders the
scene offscreen at the size of the panel's view (anti-aliased) and streams it as JPEG. Running in
its own process, rendering never competes with the simulation or the controller.

  Subscribes : m4/view_state [t, qpos...], m4/camera (JSON: rotate | pan | zoom | reset | resize)
  Publishes  : m4/image (sensor_msgs/CompressedImage, JPEG)
  Parameters : image_fps, image_width, image_height (initial), max_image_width, max_image_height
"""
import json
import mujoco
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64MultiArray, String

from milestone_04 import cell as C, topics as T
from milestone_04.view_stream import ViewStreamer


class ViewNode(Node):
    def __init__(self):
        super().__init__("view_node")
        self.declare_parameter("image_fps", 30.0)
        self.declare_parameter("image_width", 1280)
        self.declare_parameter("image_height", 720)
        self.declare_parameter("max_image_width", 2560)
        self.declare_parameter("max_image_height", 1440)
        max_w = self.get_parameter("max_image_width").value
        max_h = self.get_parameter("max_image_height").value

        self.model, data = C.load_cell(actuation="position", offscreen=(max_w, max_h))
        C.reset_to_home(self.model, data)
        self.pub = self.create_publisher(CompressedImage, T.IMAGE, qos_profile_sensor_data)
        self.streamer = ViewStreamer(self.model, self._publish_jpeg,
                                     self.get_parameter("image_width").value,
                                     self.get_parameter("image_height").value, max_w, max_h,
                                     fps=self.get_parameter("image_fps").value)
        self.streamer.snapshot(data.qpos, 0.0)             # show the cell before the simulation starts
        self.streamer.start()
        self.create_subscription(Float64MultiArray, T.VIEW_STATE, self._on_state, qos_profile_sensor_data)
        self.create_subscription(String, T.CAMERA, self._on_camera, 50)
        self.get_logger().info("View node ready.")

    def _on_state(self, msg):
        self.streamer.snapshot(msg.data[1:], msg.data[0])

    def _on_camera(self, msg):
        try:
            self.streamer.control(json.loads(msg.data))
        except (ValueError, KeyError):
            pass

    def _publish_jpeg(self, jpeg):
        out = CompressedImage()
        out.header.stamp = self.get_clock().now().to_msg()
        out.format = "jpeg"
        out.data = jpeg
        self.pub.publish(out)

    def close(self):
        self.streamer.stop()


def main(args=None):
    rclpy.init(args=args)
    node = ViewNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    except RuntimeError:
        if rclpy.ok():
            raise
    finally:
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
