#!/usr/bin/env python3
"""
Cartesian end-effector trajectories with analytic position, velocity and acceleration.

Every trajectory is a pure function of time: sample(t) -> (p, v, a, gripper).
Segments use minimum-jerk profiles, so velocity and acceleration are continuous and
start/end at rest.
"""
import numpy as np

GRIPPER_OPEN, GRIPPER_CLOSED = -1.2, 0.0

# (name, target EE position, segment duration [s], gripper command during the segment)
PICK_PLACE_WAYPOINTS = [
    ("above belt", [-0.16, 0.00, 0.10],  2.0, GRIPPER_OPEN),
    ("pick",       [-0.16, 0.00, 0.065], 1.5, GRIPPER_OPEN),
    ("grasp",      [-0.16, 0.00, 0.065], 0.8, GRIPPER_CLOSED),
    ("lift",       [-0.16, 0.00, 0.10],  1.2, GRIPPER_CLOSED),
    ("above bin",  [0.00, 0.15, 0.10],   2.5, GRIPPER_CLOSED),
    ("place",      [0.00, 0.15, 0.075],  1.2, GRIPPER_CLOSED),
    ("release",    [0.00, 0.15, 0.075],  0.8, GRIPPER_OPEN),
    ("retreat",    [0.00, 0.15, 0.10],   1.2, GRIPPER_OPEN),
    ("return",     [-0.16, 0.00, 0.10],  2.5, GRIPPER_OPEN),
]


def _min_jerk_scalar(tau, T):
    """Minimum-jerk blend s(tau) and its first two time derivatives."""
    tau = np.clip(tau, 0.0, 1.0)
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    ds = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / T
    dds = (60 * tau - 180 * tau**2 + 120 * tau**3) / T**2
    return s, ds, dds


def min_jerk(p0, p1, T, t):
    """Minimum-jerk position, velocity, acceleration from p0 to p1 over duration T."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    s, ds, dds = _min_jerk_scalar(t / T, T)
    d = p1 - p0
    return p0 + s * d, ds * d, dds * d


class CircleTrajectory:
    """Approach the circle start, then draw `cycles` circles in the y-z plane.

    The angle follows a minimum-jerk profile per lap (zero velocity at each lap start/end).
    """

    def __init__(self, start, center=(-0.15, 0.0, 0.12), radius=0.03, period=3.0,
                 cycles=2, approach_time=2.0):
        self.start = np.asarray(start, float)
        self.center = np.asarray(center, float)
        self.radius, self.period, self.cycles, self.approach = radius, period, cycles, approach_time
        self.circle_start = self.center + self.radius * np.array([0.0, 1.0, 0.0])
        self.duration = approach_time + cycles * period
        self.loops = False

    def sample(self, t):
        if t < self.approach:
            p, v, a = min_jerk(self.start, self.circle_start, self.approach, t)
            return p, v, a, GRIPPER_CLOSED
        tl = min(t - self.approach, self.cycles * self.period)
        tau = (tl % self.period) / self.period
        if tl >= self.cycles * self.period:
            tau = 1.0
        s, ds, dds = _min_jerk_scalar(tau, self.period)
        th, w, al = 2 * np.pi * s, 2 * np.pi * ds, 2 * np.pi * dds
        c, sn = np.cos(th), np.sin(th)
        r = self.radius
        p = self.center + r * np.array([0.0, c, sn])
        v = r * w * np.array([0.0, -sn, c])
        a = r * (al * np.array([0.0, -sn, c]) - w**2 * np.array([0.0, c, sn]))
        return p, v, a, GRIPPER_CLOSED


class PickPlaceTrajectory:
    """Conveyor -> bin pick-and-place cycle through PICK_PLACE_WAYPOINTS (optionally looping)."""

    def __init__(self, start, loop=False):
        wp = PICK_PLACE_WAYPOINTS
        pts = [np.asarray(start, float)] + [np.asarray(w[1], float) for w in wp]
        self._first = [(pts[i], pts[i + 1], w[2], w[3]) for i, w in enumerate(wp)]
        last = pts[-1]
        loop_pts = [last] + pts[2:]          # restart from the 'return' point
        self._loop = [(loop_pts[i], loop_pts[i + 1], w[2], w[3])
                      for i, w in enumerate(wp[1:])]
        self.d_first = sum(s[2] for s in self._first)
        self.d_loop = sum(s[2] for s in self._loop)
        self.loops = loop
        self.duration = self.d_first

    @staticmethod
    def _lookup(segments, t):
        for p0, p1, T, grip in segments:
            if t < T:
                p, v, a = min_jerk(p0, p1, T, t)
                return p, v, a, grip
            t -= T
        p0, p1, T, grip = segments[-1]
        p, v, a = min_jerk(p0, p1, T, T)
        return p, v, a, grip

    def sample(self, t):
        if t < self.d_first or not self.loops:
            return self._lookup(self._first, t)
        return self._lookup(self._loop, (t - self.d_first) % self.d_loop)
