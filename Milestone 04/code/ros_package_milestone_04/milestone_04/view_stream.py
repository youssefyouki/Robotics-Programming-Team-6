#!/usr/bin/env python3
"""
Offscreen view of the simulation, rendered in its own thread (Milestone 04).

Fed with the generalized positions (qpos) of the simulation; the thread rebuilds the scene from
the latest ones, renders it at the size requested by the panel (multisample anti-aliasing),
JPEG-encodes it and hands it to `publish(jpeg_bytes)`. It can overlay the planned tool-point path
(joint-space moves blue, task-space lines orange, manual moves purple) and a trail of the path the
tool point actually followed (green). It only ever re-renders when something
changed (new state, camera or size), and the frame rate adapts to the render cost.
"""
import threading
import time
from collections import deque

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
        self.path = None                              # (n, 4) array: x, y, z, type
        self.show_path, self.show_trail = True, True
        self.clear_trail = False

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
            elif action == "overlay":                     # show / hide the path and the trail
                self.show_path = bool(c.get("path", self.show_path))
                self.show_trail = bool(c.get("trail", self.show_trail))
                self.clear_trail = self.clear_trail or bool(c.get("clear_trail", False))
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

    def set_path(self, arr):
        """Planned path [(x, y, z, type), ...]; an empty array clears it (and the trail)."""
        with self.lock:
            self.path = np.asarray(arr, float).reshape(-1, 4) if len(arr) else None
            if self.path is None:
                self.clear_trail = True
            self.version += 1

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)

    @staticmethod
    def _line(scn, p0, p1, rgba, radius):
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                            np.eye(3).flatten(), np.array(rgba, dtype=np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, np.asarray(p0, float), np.asarray(p1, float))
        scn.ngeom += 1

    # ---------------------------------------------------------------- render thread
    def _loop(self):
        data = mujoco.MjData(self.model)
        cam = mujoco.MjvCamera()
        renderer, size, last_key, cost, last = None, None, None, 0.03, 0.0
        tcp_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        trail = deque(maxlen=1500)
        colours = {0: (0.35, 0.75, 1.0, 1.0), 1: (1.0, 0.55, 0.1, 1.0), 2: (0.8, 0.45, 1.0, 1.0)}
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
                with self.lock:
                    path, show_path, show_trail = self.path, self.show_path, self.show_trail
                    if self.clear_trail:
                        trail.clear()
                        self.clear_trail = False
                p_tcp = data.site_xpos[tcp_id].copy()
                if not trail or np.linalg.norm(p_tcp - trail[-1]) > 0.0015:
                    trail.append(p_tcp)
                scn = renderer.scene
                if show_path and path is not None:
                    for a, b in zip(path[:-1], path[1:]):
                        kind = int(b[3])
                        if kind in colours and np.linalg.norm(b[:3] - a[:3]) > 1e-6:
                            self._line(scn, a[:3], b[:3], colours[kind], 0.0013)
                if show_trail and len(trail) > 1:
                    for a, b in zip(list(trail)[:-1], list(trail)[1:]):
                        self._line(scn, a, b, (0.35, 1.0, 0.45, 1.0), 0.0009)
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
