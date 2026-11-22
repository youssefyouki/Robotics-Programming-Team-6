#!/usr/bin/env python3
"""
Live tracking monitor and per-segment validation (Milestone 04).

Fed with the measured joints q and the reference q_ref (plus the planner's phase) at every control
tick. It reports the current tracking errors and, when a segment ends, validates it:

  task-space segments (descend / lift / lower / retreat): largest deviation of the measured tool
      point from the commanded straight line [mm], and the error at the end of the segment
  joint-space segments (approach / transfer / home): largest joint tracking error [mrad]

A segment passes when the value is below the tolerance.
"""
import numpy as np

from milestone_04 import tcp_kinematics as TK, topics as T

TASK = ("descend", "lift", "lower", "retreat")
JOINT = ("approach", "transfer", "home")


class LiveMonitor:
    def __init__(self, tol_task_mm=3.0, tol_joint_rad=0.03):
        self.tol_task, self.tol_joint = tol_task_mm, tol_joint_rad
        self.phase, self.start_t = None, 0.0
        self.pts, self.ref, self.jerr = [], [], []

    def update(self, t, q, q_ref, phase_idx):
        """-> ((max joint error [rad], TCP error [mm]), finished-segment result or None)"""
        name = T.PHASES[int(phase_idx)]
        e = q_ref - q
        tcp, tcp_ref = TK.tcp_position(q), TK.tcp_position(q_ref)
        live = (float(np.abs(e).max()), float(np.linalg.norm(tcp - tcp_ref) * 1000))
        event = None
        if name != self.phase:
            event = self._finish()
            self.phase, self.start_t = name, t
            self.pts, self.ref, self.jerr = [], [], []
        if name in TASK:
            self.pts.append(tcp)
            self.ref.append(tcp_ref)
        elif name in JOINT:
            self.jerr.append(live[0])
        return live, event

    def reset(self):
        self.phase, self.pts, self.ref, self.jerr = None, [], [], []

    def _finish(self):
        if self.phase in TASK and len(self.pts) > 2:
            pts, ref = np.array(self.pts), np.array(self.ref)
            p0, p1 = ref[0], ref[-1]
            length = np.linalg.norm(p1 - p0)
            if length < 1e-6:
                return None
            u = (p1 - p0) / length
            rel = pts - p0
            dev = float(np.linalg.norm(rel - np.outer(rel @ u, u), axis=1).max() * 1000)
            end = float(np.linalg.norm(pts[-1] - ref[-1]) * 1000)
            return {"phase": self.phase, "space": "task", "value": dev, "end": end, "unit": "mm",
                    "tol": self.tol_task, "ok": dev < self.tol_task, "t": self.start_t}
        if self.phase in JOINT and self.jerr:
            value = max(self.jerr) * 1000
            return {"phase": self.phase, "space": "joint", "value": float(value), "end": float(self.jerr[-1] * 1000),
                    "unit": "mrad", "tol": self.tol_joint * 1000, "ok": value < self.tol_joint * 1000,
                    "t": self.start_t}
        return None
