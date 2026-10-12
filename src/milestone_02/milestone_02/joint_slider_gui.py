#!/usr/bin/env python3
"""
Joint Slider GUI for Koch v1.1 — Milestone 02

ros2 run milestone_02 joint_slider_gui
  Publishes  : low_cost_robot_arm/joint_commands (std_msgs/Float64MultiArray)
  Subscribes : low_cost_robot_arm/joint_state    (sensor_msgs/JointState)

Requires POSITION control mode in the launch file.
"""
import sys
import threading
import tkinter as tk

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

JOINTS = [
    ("base_rotation", -2.2,   2.2  ),
    ("pitch",         -1.57,  0.6  ),
    ("elbow",         -1.57,  1.45 ),
    ("wrist_pitch",   -2.0,   2.0  ),
    ("wrist_roll",    -3.14,  3.14 ),
    ("gripper",       -1.60,  0.032),
]


class SliderNode(Node):
    def __init__(self):
        super().__init__('joint_slider_gui')
        self.pub = self.create_publisher(
            Float64MultiArray, 'low_cost_robot_arm/joint_commands', 1)
        self.sim_positions = [0.0] * len(JOINTS)
        self.create_subscription(
            JointState, 'low_cost_robot_arm/joint_state',
            self._joint_state_cb, 1)

    def _joint_state_cb(self, msg: JointState):
        for i, (name, _, _) in enumerate(JOINTS):
            if name in msg.name:
                idx = msg.name.index(name)
                self.sim_positions[i] = msg.position[idx]

    def send(self, positions):
        msg = Float64MultiArray()
        msg.data = [float(v) for v in positions]
        self.pub.publish(msg)


class GUI:
    def __init__(self, node: SliderNode):
        self.node = node

        self.root = tk.Tk()
        self.root.title("Koch v1.1  |  Joint Control  (POSITION mode)")
        self.root.configure(bg='#1e1e2e')

        tk.Label(self.root, text="Koch v1.1 — Joint Sliders",
                 font=('Helvetica', 14, 'bold'),
                 bg='#1e1e2e', fg='#cdd6f4').grid(
            row=0, column=0, columnspan=3, pady=(12, 6))

        self.sliders    = []
        self.val_labels = []

        for row, (name, lo, hi) in enumerate(JOINTS, start=1):
            tk.Label(self.root, text=name, width=16, anchor='w',
                     font=('Helvetica', 10), bg='#1e1e2e', fg='#cdd6f4'
                     ).grid(row=row, column=0, padx=12, pady=3)

            s = tk.Scale(
                self.root,
                from_=lo, to=hi, resolution=0.001,
                orient=tk.HORIZONTAL, length=420,
                bg='#313244', fg='#cdd6f4',
                troughcolor='#45475a', highlightthickness=0,
                command=lambda _, r=row: self._on_change(),
            )
            s.grid(row=row, column=1, padx=8, pady=3)
            self.sliders.append(s)

            lbl = tk.Label(self.root, text='0.000 rad', width=12, anchor='e',
                           font=('Courier', 10), bg='#1e1e2e', fg='#a6e3a1')
            lbl.grid(row=row, column=2, padx=12, pady=3)
            self.val_labels.append(lbl)

        btn_frame = tk.Frame(self.root, bg='#1e1e2e')
        btn_frame.grid(row=len(JOINTS)+1, column=0, columnspan=3, pady=10)

        tk.Button(btn_frame, text='Home (all 0)', width=14,
                  bg='#3a7bd5', fg='white', relief='flat',
                  command=self._home).pack(side=tk.LEFT, padx=6)

        tk.Button(btn_frame, text='Sync from sim', width=14,
                  bg='#585b70', fg='white', relief='flat',
                  command=self._sync).pack(side=tk.LEFT, padx=6)

        tk.Button(btn_frame, text='Close gripper', width=14,
                  bg='#e06c75', fg='white', relief='flat',
                  command=lambda: self._set_joint(5, -1.50)).pack(side=tk.LEFT, padx=6)

        tk.Button(btn_frame, text='Open gripper', width=14,
                  bg='#98c379', fg='white', relief='flat',
                  command=lambda: self._set_joint(5, 0.0)).pack(side=tk.LEFT, padx=6)

        self.root.after(100, self._update_labels)

    def _on_change(self):
        self.node.send([s.get() for s in self.sliders])

    def _home(self):
        for s in self.sliders:
            s.set(0.0)
        self._on_change()

    def _sync(self):
        for s, pos in zip(self.sliders, self.node.sim_positions):
            s.set(round(pos, 3))

    def _set_joint(self, idx, value):
        self.sliders[idx].set(value)
        self._on_change()

    def _update_labels(self):
        for s, lbl in zip(self.sliders, self.val_labels):
            lbl.config(text=f'{s.get():+.3f} rad')
        self.root.after(100, self._update_labels)

    def run(self):
        self.root.mainloop()


def main(args=None):
    rclpy.init(args=args)
    node = SliderNode()
    gui  = GUI(node)

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    gui.run()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
