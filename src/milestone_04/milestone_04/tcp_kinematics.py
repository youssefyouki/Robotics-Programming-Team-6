#!/usr/bin/env python3
"""
Tool-centre-point (TCP) kinematics for the Koch v1.1 gripper (Milestone 04).

The TCP is the point between the two gripper pads where a 20 mm box is held. It is fixed in
the wrist_roll frame, so it moves with the roll joint. The tool axis points along -x of the
wrist_roll frame (from the roll origin towards the pads), and the gripper closes along y.

Pose IK: the first four joints place the TCP (3 constraints) and set the tool pitch (1 constraint,
angle of the tool axis above the horizontal, measured in the arm's vertical plane). The roll joint
q5 is chosen separately to align the jaws with the box.
"""
import numpy as np
from milestone_03 import kinematics as K

# box centre when approaching with open jaws (the moving jaw pushes the box ~3.5 mm towards the
# fixed jaw on closing), in the wrist_roll frame
TCP_LOCAL = np.array([-0.0647, -0.0050, 0.0006])
BOX_SIZE = 0.02


def tcp_position(q):
    _, _, p, R = K._geometry(q)
    return p + R @ TCP_LOCAL


def tcp_jacobian(q):
    """3x5 linear Jacobian of the TCP."""
    origins, axes, p_ee, R = K._geometry(q)
    p = p_ee + R @ TCP_LOCAL
    J = np.zeros((3, K.N))
    for i in range(K.N):
        J[:, i] = np.cross(axes[i], p - origins[i])
    return J


def tool_axis(q):
    """Unit vector from the wrist towards the gripper pads, in the world frame."""
    _, _, _, R = K._geometry(q)
    return R @ np.array([-1.0, 0.0, 0.0])


def jaw_axis(q):
    """Unit vector along which the jaws close (wrist_roll y axis), in the world frame."""
    _, _, _, R = K._geometry(q)
    return R[:, 1]


def tool_pitch(q):
    """Elevation of the tool axis above the horizontal, in the arm plane (-pi/2 = pointing down)."""
    a = tool_axis(q)
    r_hat = np.array([-np.cos(q[0]), np.sin(q[0]), 0.0])
    return float(np.arctan2(a[2], a @ r_hat))


def pitch_gradient(q):
    """Analytic d(tool_pitch)/dq (5 joints). With a the tool axis, r the arm direction (-cos q1, sin q1, 0)
    and pitch = atan2(a_z, a.r):  d pitch = (a_r da_z - a_z da_r) / (a_r^2 + a_z^2),  da/dq_i = w_i x a."""
    _, axes, _, R = K._geometry(q)
    a = R @ np.array([-1.0, 0.0, 0.0])
    r = np.array([-np.cos(q[0]), np.sin(q[0]), 0.0])
    dr = np.array([np.sin(q[0]), np.cos(q[0]), 0.0])
    ar, az = a @ r, a[2]
    g = np.zeros(K.N)
    for i in range(K.N):
        da = np.cross(axes[i], a)
        dar = da @ r + (a @ dr if i == 0 else 0.0)
        g[i] = (ar * da[2] - az * dar) / (ar * ar + az * az)
    return g


def task_jacobian4(q):
    """4x4 Jacobian of [TCP position; tool pitch] with respect to the first four joints."""
    J = np.zeros((4, 4))
    J[:3] = tcp_jacobian(q)[:, :4]
    J[3] = pitch_gradient(q)[:4]
    return J


def _wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def _solve(p_tcp, pitch, q, tol, max_iter, damping):
    lo, hi = K.JOINT_LIMITS[:, 0], K.JOINT_LIMITS[:, 1]
    for _ in range(max_iter):
        e = np.hstack([p_tcp - tcp_position(q), _wrap(pitch - tool_pitch(q))])
        if np.linalg.norm(e) < tol:
            return q, True
        J = task_jacobian4(q)
        step = np.linalg.solve(J.T @ J + damping**2 * np.eye(4), J.T @ e)
        q = q.copy()
        q[:4] = np.clip(q[:4] + step, lo[:4], hi[:4])
    e = np.hstack([p_tcp - tcp_position(q), _wrap(pitch - tool_pitch(q))])
    return q, bool(np.linalg.norm(e) < 1e-6)


def inverse_pose(p_tcp, pitch=-np.pi / 2, roll=0.0, q0=None, tol=1e-10, max_iter=100, damping=1e-4):
    """Solve q1..q4 for TCP position + tool pitch, with q5 = roll. Returns (q, ok).

    The given seed q0 is tried first (keeps the solution continuous along a path); if Newton does
    not converge from it, fresh seeds pointing the base at the target are tried.
    """
    p_tcp = np.asarray(p_tcp, float)
    yaw = np.arctan2(p_tcp[1], -p_tcp[0])
    seeds = []
    if q0 is not None:
        seeds.append(np.array(q0, float))
    for q2, q3, q4 in [(-0.5, 0.8, -1.0), (-0.2, 0.2, -1.3), (-0.6, -0.3, -1.5), (0.2, 0.5, -1.4)]:
        seeds.append(np.array([yaw, q2, q3, q4, roll]))
    best = None
    for q in seeds:
        q[4] = roll
        q, ok = _solve(p_tcp, pitch, q, tol, max_iter, damping)
        if ok:
            return q, True
        best = q if best is None else best
    return best, False
