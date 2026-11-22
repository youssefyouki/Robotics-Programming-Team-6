#!/usr/bin/env python3
"""
Motion planning for the colour-sorting cycle (Milestone 04).

Two kinds of segments, both sampled at DT (the simulator time step):

  joint-space  : quintic polynomial q(t) between two joint vectors with zero velocity and
                 acceleration at both ends (long free moves, no path constraint).
  task-space   : straight line of the TCP (minimum-jerk speed profile) with the tool kept
                 vertical; every sample is converted to joint angles by inverse kinematics,
                 qd and qdd come from the analytic path derivatives (J^-1 form) and are
                 cross-checked against differentiation (short moves where the path matters:
                 going straight down onto a box / up out of a bin).

A sorting cycle for one box:
  1 joint-space  home/previous -> 4 cm above the box (gripper opens)
  2 task-space   straight down to the box
  3 dwell        close the gripper
  4 task-space   straight up (lift)
  5 joint-space  swing to 4 cm above the bin of the same colour
  6 task-space   straight down into the bin
  7 dwell        open the gripper (box drops)
  8 task-space   straight up
Finally a joint-space move back home.
"""
from dataclasses import dataclass, field
import numpy as np

from milestone_03 import kinematics as K
from milestone_04 import tcp_kinematics as TK
from milestone_04 import cell as C

DT = 0.002                       # plan sample period [s] = simulator time step
GRIPPER_OPEN = -0.6              # jaws ~35 mm apart
GRIPPER_CLOSED = 0.0             # target while holding (jaws stop on the box)
APPROACH_HEIGHT = 0.04           # pre-grasp / pre-release height above the target [m]
RELEASE_Z = 0.045                # TCP height when dropping a box in a bin [m]
V_JOINT_MAX = 1.2                # peak joint speed allowed on free moves [rad/s]
HOME_TCP = np.array([-0.103, 0.0, 0.085])   # above the (emptied) belt, tool pointing down


class PlanError(RuntimeError):
    pass


@dataclass
class Plan:
    t: np.ndarray
    q: np.ndarray
    qd: np.ndarray
    qdd: np.ndarray
    grip: np.ndarray
    segments: list = field(default_factory=list)      # (name, first_index, last_index, colour)

    def __len__(self):
        return len(self.t)

    @property
    def duration(self):
        return float(self.t[-1])

    def sample(self, t):
        """Reference at time t (nearest sample, held after the end): q, qd, qdd, grip."""
        k = int(np.clip(round(t / DT), 0, len(self.t) - 1))
        return self.q[k], self.qd[k], self.qdd[k], float(self.grip[k])

    def segment_at(self, t):
        k = int(np.clip(round(t / DT), 0, len(self.t) - 1))
        for name, i0, i1, color in self.segments:
            if i0 <= k <= i1:
                return name, color
        return "done", None


# ---------------------------------------------------------------------------------- profiles
def quintic_coefficients(q0, v0, a0, q1, v1, a1, T):
    """Coefficients a0..a5 of q(t) = sum a_k t^k with the given boundary conditions."""
    A = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 2, 0, 0, 0],
                  [1, T, T**2, T**3, T**4, T**5],
                  [0, 1, 2 * T, 3 * T**2, 4 * T**3, 5 * T**4],
                  [0, 0, 2, 6 * T, 12 * T**2, 20 * T**3]], float)
    b = np.array([q0, v0, a0, q1, v1, a1], float)
    return np.linalg.solve(A, b)


def quintic_eval(c, t):
    p = np.array([t**k for k in range(6)])
    v = np.array([0] + [k * t**(k - 1) for k in range(1, 6)])
    a = np.array([0, 0] + [k * (k - 1) * t**(k - 2) for k in range(2, 6)])
    return c @ p, c @ v, c @ a


def _times(T):
    n = max(1, int(round(T / DT)))
    return np.arange(1, n + 1) * DT, n


def joint_segment(q0, q1, T=None, v0=None, a0=None):
    """Quintic joint-space move (optionally starting with velocity v0 / acceleration a0, ending at
    rest); returns q, qd, qdd arrays (start sample excluded)."""
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    v0 = np.zeros(5) if v0 is None else np.asarray(v0, float)
    a0 = np.zeros(5) if a0 is None else np.asarray(a0, float)
    if T is None:
        T = max(1.5, 1.875 * np.max(np.abs(q1 - q0)) / V_JOINT_MAX)
    T = max(1, int(round(T / DT))) * DT          # whole number of time steps: the last sample is exactly t = T
    ts, n = _times(T)
    Q, Qd, Qdd = np.zeros((n, 5)), np.zeros((n, 5)), np.zeros((n, 5))
    for j in range(5):
        c = quintic_coefficients(q0[j], v0[j], a0[j], q1[j], 0, 0, T)
        for k, t in enumerate(ts):
            Q[k, j], Qd[k, j], Qdd[k, j] = quintic_eval(c, t)
    return Q, Qd, Qdd


def _min_jerk_s(tau):
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def _min_jerk_derivs(tau, T):
    """s(tau) and its first two time derivatives for the minimum-jerk blend over duration T."""
    s = _min_jerk_s(tau)
    ds = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / T
    dds = (60 * tau - 180 * tau**2 + 120 * tau**3) / T**2
    return s, ds, dds


def task_segment(q_start, p0, p1, T, roll, pitch=-np.pi / 2):
    """Straight TCP line p0 -> p1 with the tool at `pitch`. Returns q, qd, qdd (start excluded).

    Path:  p(t) = p0 + s(t) (p1 - p0)  with the minimum-jerk blend s(t)  (analytic p', p'').
    Joints: q from inverse kinematics at every sample (first four joints, roll held).
    Task Jacobian J4 = d[p; pitch]/dq (4x4):
        qd  = J4^-1 [p'; 0]
        qdd = J4^-1 ([p''; 0] - J4' qd)        (J4' = directional derivative of J4 along qd)
    """
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    T = max(1, int(round(T / DT))) * DT          # whole number of time steps
    ts, n = _times(T)
    taus = np.clip(np.concatenate([[0.0], ts / T]), 0.0, 1.0)
    s, ds, dds = _min_jerk_derivs(taus, T)
    delta = p1 - p0
    P = p0 + s[:, None] * delta
    Q = np.zeros((n + 1, 5))
    Q[0] = q_start
    for k in range(1, n + 1):
        q, ok = TK.inverse_pose(P[k], pitch=pitch, roll=roll, q0=Q[k - 1])
        if not ok:
            raise PlanError(f"TCP {np.round(P[k], 3)} not reachable with the tool pointing down")
        Q[k] = q
    if np.any(Q < K.JOINT_LIMITS[:, 0] - 1e-9) or np.any(Q > K.JOINT_LIMITS[:, 1] + 1e-9):
        raise PlanError("task-space segment violates a joint limit")
    Qd, Qdd = np.zeros_like(Q), np.zeros_like(Q)
    eps = 1e-6
    for k in range(n + 1):
        J = TK.task_jacobian4(Q[k])
        qd4 = np.linalg.solve(J, np.hstack([ds[k] * delta, 0.0]))
        qd5 = np.append(qd4, 0.0)
        Jdot = (TK.task_jacobian4(Q[k] + eps * qd5) - TK.task_jacobian4(Q[k] - eps * qd5)) / (2 * eps)
        Qd[k, :4] = qd4
        Qdd[k, :4] = np.linalg.solve(J, np.hstack([dds[k] * delta, 0.0]) - Jdot @ qd4)
    return Q[1:], Qd[1:], Qdd[1:]


# ---------------------------------------------------------------------------------- poses
def _wrap_quarter(x):
    return (x + np.pi / 4) % (np.pi / 2) - np.pi / 4


def grasp_pose(p_tcp, box_yaw, q_seed):
    """Joint vector putting the TCP at p_tcp, tool pointing down, jaws aligned with the box faces."""
    roll = 0.0
    q = None
    for _ in range(5):
        q, ok = TK.inverse_pose(p_tcp, roll=roll, q0=q_seed)
        if not ok:
            raise PlanError(f"TCP {np.round(p_tcp, 3)} not reachable with the tool pointing down")
        c = TK.jaw_axis(q)
        psi = np.arctan2(c[1], c[0])
        d = _wrap_quarter(box_yaw - psi)
        best = None
        for cand in (roll + d, roll - d):
            qt = q.copy()
            qt[4] = cand
            c2 = TK.jaw_axis(qt)
            err = abs(_wrap_quarter(box_yaw - np.arctan2(c2[1], c2[0])))
            if best is None or err < best[0]:
                best = (err, cand)
        if abs(best[1] - roll) < 1e-10:
            break
        roll = best[1]
    return q


def home_pose(q_seed):
    q, ok = TK.inverse_pose(HOME_TCP, roll=0.0, q0=q_seed)
    if not ok:
        raise PlanError("home pose not reachable")
    return q


# ---------------------------------------------------------------------------------- plan
class _Builder:
    def __init__(self, q_start, grip_start, qd0=None, qdd0=None):
        self.q = [np.asarray(q_start, float)[None]]
        self.qd = [np.zeros((1, 5)) if qd0 is None else np.asarray(qd0, float)[None]]
        self.qdd = [np.zeros((1, 5)) if qdd0 is None else np.asarray(qdd0, float)[None]]
        self.grip = [np.array([grip_start])]
        self.segments = []
        self.n = 1
        self.q_now = np.asarray(q_start, float)
        self.g_now = grip_start

    def add(self, name, color, Q, Qd, Qdd, grip_to=None, ramp=None):
        n = len(Q)
        g = np.full(n, self.g_now)
        if grip_to is not None:
            m = n if ramp is None else min(n, int(round(ramp / DT)))
            s = _min_jerk_s(np.arange(1, m + 1) / m)
            g[:m] = self.g_now + s * (grip_to - self.g_now)
            g[m:] = grip_to
            self.g_now = grip_to
        self.segments.append((name, self.n, self.n + n - 1, color))
        self.q.append(Q); self.qd.append(Qd); self.qdd.append(Qdd); self.grip.append(g)
        self.n += n
        self.q_now = Q[-1]

    def finish(self):
        q = np.vstack(self.q)
        return Plan(t=np.arange(len(q)) * DT, q=q, qd=np.vstack(self.qd), qdd=np.vstack(self.qdd),
                    grip=np.concatenate(self.grip), segments=self.segments)


def build_sorting_plan(q_start, boxes, grip_start=0.0, order=None):
    """Full plan for sorting all boxes. boxes: {colour: (position, yaw)} read from the simulator."""
    b = _Builder(q_start, grip_start)
    up = np.array([0.0, 0.0, APPROACH_HEIGHT])
    for color in (order or C.COLORS):
        pos, yaw = boxes[color]
        pos = np.asarray(pos, float)

        # --- pick
        q_pre = grasp_pose(pos + up, yaw, b.q_now)
        b.add("approach", color, *joint_segment(b.q_now, q_pre), grip_to=GRIPPER_OPEN)
        roll = q_pre[4]
        b.add("descend", color, *task_segment(b.q_now, pos + up, pos, 1.0, roll))
        hold = np.repeat(b.q_now[None], int(round(1.2 / DT)), axis=0)
        b.add("grasp", color, hold, hold * 0, hold * 0, grip_to=GRIPPER_CLOSED, ramp=0.8)
        b.add("lift", color, *task_segment(b.q_now, pos, pos + up, 1.0, roll))

        # --- place
        bc = C.bin_center(color)
        above = np.array([bc[0], bc[1], RELEASE_Z + APPROACH_HEIGHT])
        drop = np.array([bc[0], bc[1], RELEASE_Z])
        q_bin = grasp_pose(above, 0.0, b.q_now)
        b.add("transfer", color, *joint_segment(b.q_now, q_bin))
        roll_b = q_bin[4]
        b.add("lower", color, *task_segment(b.q_now, above, drop, 0.8, roll_b))
        hold = np.repeat(b.q_now[None], int(round(0.9 / DT)), axis=0)
        b.add("release", color, hold, hold * 0, hold * 0, grip_to=GRIPPER_OPEN, ramp=0.6)
        b.add("retreat", color, *task_segment(b.q_now, drop, above, 0.8, roll_b))

    q_home = home_pose(b.q_now)
    b.add("home", None, *joint_segment(b.q_now, q_home))
    return b.finish()


# ---------------------------------------------------------------------------------- manual moves
GRIPPER_LIMITS = (-1.6, 0.032)


def _clip_limits(q):
    return np.clip(q, K.JOINT_LIMITS[:, 0], K.JOINT_LIMITS[:, 1])


MIN_TCP_Z = 0.012            # lowest tool-point height accepted for manual targets [m]


def _check_height(p_tcp):
    if p_tcp[2] < MIN_TCP_Z:
        raise PlanError(f"target tool point is {p_tcp[2] * 100:.1f} cm high - below the work surface "
                        f"(minimum {MIN_TCP_Z * 100:.1f} cm)")


def plan_joint_goto(q0, v0, a0, q1, g0, g1, T=None):
    """Smooth move to joint targets q1 (and gripper g1) starting from the current reference state."""
    q1 = _clip_limits(np.asarray(q1, float))
    _check_height(TK.tcp_position(q1))
    g1 = float(np.clip(g1, *GRIPPER_LIMITS))
    if T is None and np.max(np.abs(q1 - q0)) < 1e-6:
        T = 0.8                                   # gripper-only command
    b = _Builder(q0, g0, v0, a0)
    b.add("manual", None, *joint_segment(q0, q1, T, v0, a0), grip_to=g1)
    return b.finish()


def plan_tcp_goto(q0, v0, a0, p, pitch, roll, g0, g1, T=None, linear=False):
    """Move the TCP to p with the tool at `pitch` (rad above horizontal; -pi/2 = down) and roll.

    linear=True follows a straight line with the current tool pitch and roll (from rest);
    otherwise a joint-space quintic to the IK solution (works from any motion state).
    """
    g1 = float(np.clip(g1, *GRIPPER_LIMITS))
    _check_height(np.asarray(p, float))
    b = _Builder(q0, g0, v0, a0)
    if linear:
        p0 = TK.tcp_position(q0)
        T = T or max(1.0, 1.875 * np.linalg.norm(np.asarray(p) - p0) / 0.08)
        b.add("manual", None, *task_segment(q0, p0, p, T, q0[4], TK.tool_pitch(q0)), grip_to=g1)
    else:
        q1, ok = TK.inverse_pose(p, pitch=pitch, roll=roll, q0=q0)
        if not ok:
            raise PlanError(f"TCP {np.round(p, 3)} m is not reachable with tool pitch {np.degrees(pitch):.0f} deg")
        b.add("manual", None, *joint_segment(q0, q1, T, v0, a0), grip_to=g1)
    return b.finish()


def plan_stop(q0, v0, a0, g0, T=0.4):
    """Decelerate smoothly to rest."""
    q1 = _clip_limits(np.asarray(q0, float) + np.asarray(v0, float) * T / 2)
    b = _Builder(q0, g0, v0, a0)
    b.add("manual", None, *joint_segment(q0, q1, T, v0, a0))
    return b.finish()


SIGNAL_KINDS = ("constant", "step", "sine", "square", "triangle", "chirp")


def _ramp(t, duration):
    """Smooth 0 -> 1 blend (minimum jerk) over `duration` seconds, 1 afterwards."""
    return _min_jerk_s(np.clip(t / duration, 0.0, 1.0))


def signal_samples(kind, t, amp, freq):
    """Builder signals s(t) (radians), all starting at 0 smoothly:
    constant  slow move to `amp` and hold          step     fast (0.15 s) move to `amp` and hold
    sine      amp sin(2 pi f t)                    square   smoothed square wave (tanh edges)
    triangle  band-limited triangle wave           chirp    sine sweeping 0.05 Hz -> `freq` over the run
    Periodic signals fade in over max(0.5 s, one period) so the reference starts at rest."""
    t = np.asarray(t, float)
    w = 2 * np.pi * freq
    fade = _ramp(t, max(0.5, 1.0 / max(freq, 1e-3)))
    if kind == "constant":
        return amp * _ramp(t, 1.0)
    if kind == "step":
        return amp * _ramp(t, 0.15)
    if kind == "sine":
        return amp * fade * np.sin(w * t)
    if kind == "square":
        return amp * fade * np.tanh(4 * np.sin(w * t)) / np.tanh(4)
    if kind == "triangle":
        series = sum((-1) ** k * np.sin((2 * k + 1) * w * t) / (2 * k + 1) ** 2 for k in range(6))
        return amp * fade * 8 / np.pi**2 * series
    if kind == "chirp":
        T = max(t[-1], 1e-3)
        f0 = 0.05
        phase = 2 * np.pi * (f0 * t + (freq - f0) * t**2 / (2 * T))
        return amp * fade * np.sin(phase)
    raise PlanError(f"unknown signal '{kind}' (use {', '.join(SIGNAL_KINDS)})")


def plan_signal(q0, v0, a0, joint, kind, amp, freq, duration, g0):
    """Drive one joint with a builder signal around its current angle, then return to it smoothly."""
    q0 = np.asarray(q0, float)
    joint = int(joint)
    if not 0 <= joint < 5:
        raise PlanError("signal joint must be 1..5")
    duration = float(np.clip(duration, 1.0, 60.0))
    n = int(round(duration / DT))
    t = np.arange(1, n + 1) * DT
    sig = signal_samples(kind, t, float(amp), float(freq))
    back = sig[-1] * (1.0 - _ramp(np.arange(1, int(round(1.0 / DT)) + 1) * DT, 1.0))        # 1 s return to the start angle
    sig = np.concatenate([sig, back])

    lo, hi = K.JOINT_LIMITS[joint]
    room_up, room_down = hi - q0[joint], q0[joint] - lo
    scale = 1.0
    if sig.max() > room_up:
        scale = min(scale, room_up / sig.max())
    if sig.min() < -room_down:
        scale = min(scale, room_down / -sig.min())
    if scale < 1.0:
        sig = sig * max(scale, 0.0)

    Q = np.tile(q0, (len(sig), 1))
    Q[:, joint] += sig
    for k in range(0, len(Q), 25):                       # never drive the tool into the work surface
        _check_height(TK.tcp_position(Q[k]))
    Qd = np.gradient(np.vstack([q0, Q]), DT, axis=0, edge_order=2)[1:]
    Qdd = np.gradient(Qd, DT, axis=0, edge_order=2)
    b = _Builder(q0, g0, v0, a0)
    b.add("manual", None, Q, Qd, Qdd)
    plan = b.finish()
    plan.scale = scale                                    # < 1 when the amplitude had to be reduced for the joint limits
    return plan


class Executor:
    """Plays back the current plan against simulator time; holds the last pose when it ends."""

    def __init__(self, q, grip):
        self.plan = None
        self.t0 = 0.0
        self.q_hold = np.asarray(q, float)
        self.grip_hold = float(grip)
        self.last = (self.q_hold, np.zeros(5), np.zeros(5), self.grip_hold)

    def set_plan(self, plan, t0):
        self.plan, self.t0 = plan, t0

    def active(self, t):
        return self.plan is not None and (t - self.t0) < self.plan.duration

    def sample(self, t):
        """-> q, qd, qdd, grip, phase name, colour"""
        if self.plan is not None:
            tau = t - self.t0
            if tau < self.plan.duration:
                q, qd, qdd, g = self.plan.sample(tau)
                name, color = self.plan.segment_at(tau)
                self.last = (q, qd, qdd, g)
                return q, qd, qdd, g, name, color
            self.q_hold, self.grip_hold = self.plan.q[-1].copy(), float(self.plan.grip[-1])
            self.plan = None
        self.last = (self.q_hold, np.zeros(5), np.zeros(5), self.grip_hold)
        return self.q_hold, np.zeros(5), np.zeros(5), self.grip_hold, "done", None
