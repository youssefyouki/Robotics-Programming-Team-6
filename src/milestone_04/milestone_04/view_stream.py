#!/usr/bin/env python3
"""
Offscreen view of the simulation, rendered in its own thread (Milestone 04).

Fed with the generalized positions (qpos) of the simulation; the thread rebuilds the scene from
the latest ones, renders it at the size requested by the panel (multisample anti-aliasing),
JPEG-encodes it and hands it to `publish(jpeg_bytes)`. It only ever re-renders when something
changed (new state, camera or size), and the frame rate adapts to the render cost.
"""
import threading
import time

import cv2
import mujoco
import numpy as np


class ViewStreamer:
    def __init__(self, model, publish, width, height, max_width, max_height, fps=30.0, quality=90):
        self.model, self.publish = model, publish
        self.w, self.h, self.max_w, self.max_h = width, height, max_width, max_height
        self.fps, self.quality = fps, quality
        self.cam = mujoco.MjvCamera()
        self.scene = mujoco.MjvScene(model, maxgeom=1000)      # needed by mjv_moveCamera in MuJoCo 3.x
        self.reset_camera()
        self.lock = threading.Lock()
        self.snap = None
        self.version = 0
        self.running = False
        self.thread = None

    def snapshot(self, qpos, t):
        with self.lock:
            self.snap = (np.array(qpos, float), float(t))

    def reset_camera(self):
        self.cam.lookat[:] = [-0.01, 0.0, 0.04]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 0.6, 215, -35

    def control(self, c):
        """Camera / size command from the panel: {"action": rotate|pan|zoom|reset|resize, ...}."""
        action = c.get("action")
        with self.lock:
            if action == "reset":
                self.reset_camera()
            elif action == "fps":                         # frame-rate the panel can actually display
                self.fps = float(np.clip(c["fps"], 2.0, 30.0))
                return
            elif action == "resize":
                self.w = int(np.clip(c["w"], 320, self.max_w)) // 2 * 2
                self.h = int(np.clip(c["h"], 240, self.max_h)) // 2 * 2
            elif action in ("rotate", "pan", "zoom"):
                mouse = {"rotate": mujoco.mjtMouse.mjMOUSE_ROTATE_V, "pan": mujoco.mjtMouse.mjMOUSE_MOVE_V,
                         "zoom": mujoco.mjtMouse.mjMOUSE_ZOOM}[action]
                dx, dy = float(c.get("dx", 0)), float(c.get("dy", 0))
                try:                                                  # MuJoCo >= 4 signature
                    mujoco.mjv_moveCamera(self.model, int(mouse), dx, dy, self.cam)
                except TypeError:                                     # MuJoCo 3.x signature
                    mujoco.mjv_moveCamera(self.model, mouse, dx, dy, self.scene, self.cam)
            self.version += 1

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)

    # ---------------------------------------------------------------- render thread
    def _loop(self):
        data = mujoco.MjData(self.model)
        cam = mujoco.MjvCamera()
        renderer, size, last_key, cost, last = None, None, None, 0.03, 0.0
        try:
            while self.running:
                time.sleep(0.003)
                if time.time() - last < max(1.0 / self.fps, 1.05 * cost):
                    continue
                with self.lock:
                    snap, w, h, version = self.snap, self.w, self.h, self.version
                    cam.lookat[:] = self.cam.lookat
                    cam.distance, cam.azimuth, cam.elevation = self.cam.distance, self.cam.azimuth, self.cam.elevation
                if snap is None:
                    continue
                key = (snap[1], version, w, h)
                if key == last_key:                    # nothing changed (e.g. paused): do not re-render
                    continue
                t0 = time.time()
                if renderer is None or size != (w, h):
                    if renderer is not None:
                        renderer.close()
                    renderer, size = mujoco.Renderer(self.model, h, w), (w, h)
                data.qpos[:], data.time = snap
                mujoco.mj_forward(self.model, data)
                renderer.update_scene(data, cam)
                rgb = renderer.render()
                ok, buf = cv2.imencode(".jpg", rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                if ok:
                    self.publish(buf.tobytes())
                last_key = key
                last = t0                              # pace from the frame start, not its end
                cost = 0.7 * cost + 0.3 * (time.time() - t0)
        finally:
            if renderer is not None:
                renderer.close()
