#!/usr/bin/env python3
"""
Milestone 04 - render the sorting cycle to an MP4 (offscreen, no window, no ROS).

Run:  MUJOCO_GL=egl .venv/bin/python "Milestone 04/code/make_video.py" [controller] [out.mp4]
"""
import os
import sys
import cv2
import numpy as np
import mujoco

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")
sys.path.insert(0, os.path.join(ROOT, "milestone_03"))
sys.path.insert(0, os.path.join(ROOT, "milestone_04"))
from milestone_04 import offline

controller = sys.argv[1] if len(sys.argv) > 1 else "computed_torque"
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                         "results", f"sorting_{controller}.mp4")
os.makedirs(os.path.dirname(out), exist_ok=True)
FPS, EVERY, W, H = 30, 8, 640, 480           # one frame per 8 control ticks (32 ms)

state = {}
def render(model, data, k, phase):
    if k % EVERY:
        return
    if "r" not in state:
        state["r"] = mujoco.Renderer(model, H, W)
        state["w"] = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [-0.01, 0.0, 0.04]; cam.distance, cam.azimuth, cam.elevation = 0.6, 215, -35
        state["cam"] = cam
    state["r"].update_scene(data, state["cam"])
    frame = cv2.cvtColor(state["r"].render(), cv2.COLOR_RGB2BGR)
    cv2.putText(frame, f"{controller}  t={data.time:5.1f}s  {phase}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    state["w"].write(frame)

plan, log, in_bin, poses, _ = offline.simulate(controller, callback=render, verbose=True)
state["w"].release()
print(f"\nboxes sorted: {in_bin}\nvideo saved to {os.path.abspath(out)}")
