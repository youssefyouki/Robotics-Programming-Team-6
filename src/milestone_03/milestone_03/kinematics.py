#!/usr/bin/env python3
"""
Milestone 03 - velocity and acceleration kinematics of the Koch v1.1 arm (5 DoF + gripper).

The kinematic chain is taken from the MJCF model (same data the simulator uses), so the
equations here can be validated directly against MuJoCo.

End-effector point = origin of the wrist_roll frame (same point as Milestone 02 FK).
Task space = end-effector linear position (3), joint space = 5 arm joints (the 3x5
Jacobian is redundant, so inverse solutions use a damped pseudo-inverse).

Forward:   v = Jv(q) qd                      w = Jw(q) qd
           a = Jv(q) qdd + Jv_dot(q,qd) qd   alpha = Jw(q) qdd + Jw_dot(q,qd) qd
Inverse:   qd  = J+ v
           qdd = J+ (a - Jv_dot qd)
"""
import numpy as np

# (body offset in parent frame, joint axis) taken from low_cost_robot_arm.xml
_CHAIN = [
    (np.array([0.012,    0.0,      0.0409]), np.array([0.0,  0.0, -1.0])),   # base_rotation
    (np.array([0.0,     -0.0209,   0.0154]), np.array([0.0,  1.0,  0.0])),   # pitch
    (np.array([-0.0148,  0.0065,   0.1083]), np.array([0.0, -1.0,  0.0])),   # elbow
    (np.array([-0.10048, 5e-5,     0.0026999]), np.array([0.0,  1.0,  0.0])),   # wrist_pitch
    (np.array([-0.045,   0.013097, 0.0]),    np.array([1.0,  0.0,  0.0])),   # wrist_roll
]

JOINT_NAMES = ["base_rotation", "pitch", "elbow", "wrist_pitch", "wrist_roll"]

JOINT_LIMITS = np.array([
    [-2.2,  2.2 ],
    [-1.57, 0.6 ],
    [-1.57, 1.45],
    [-2.0,  2.0 ],
    [-3.14, 3.14],
])

N = len(_CHAIN)


def _rot(n, theta):
    """Rodrigues rotation matrix about unit axis n."""
    c, s, t = np.cos(theta), np.sin(theta), 1.0 - np.cos(theta)
    x, y, z = n
    return np.array([
        [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c  ],
    ])


def _geometry(q):
    """World-frame joint origins o_i, joint axes w_i, EE position and EE rotation."""
    R = np.eye(3)
    o = np.zeros(3)
    origins, axes = [], []
    for (p, n), qi in zip(_CHAIN, q):
        o = o + R @ p            # joint i sits at the origin of body i
        origins.append(o.copy())
        axes.append(R @ n)       # axis expressed in the world frame (before rotating by qi)
        R = R @ _rot(n, qi)
    return origins, axes, o, R


# ---------------------------------------------------------------- position kinematics
def forward_kinematics(q):
    """4x4 homogeneous transform of the end effector in the world frame."""
    _, _, p, R = _geometry(q)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def ee_position(q):
    return forward_kinematics(q)[:3, 3]


# ---------------------------------------------------------------- Jacobians
def jacobian(q):
    """Geometric Jacobian: returns (Jv, Jw), each 3x5, in the world frame."""
    origins, axes, p_ee, _ = _geometry(q)
    Jv = np.zeros((3, N))
    Jw = np.zeros((3, N))
    for i in range(N):
        Jv[:, i] = np.cross(axes[i], p_ee - origins[i])
        Jw[:, i] = axes[i]
    return Jv, Jw


def jacobian_dot(q, qd):
    """Time derivative of the geometric Jacobian: returns (Jv_dot, Jw_dot), each 3x5."""
    origins, axes, p_ee, _ = _geometry(q)

    # velocity of every joint origin and angular velocity of every parent frame
    omega = np.zeros(3)                 # angular velocity of the frame before joint i
    o_dot, w_dot = [], []
    for i in range(N):
        v_o = np.zeros(3)
        for k in range(i):              # only joints before i move origin i
            v_o += np.cross(axes[k], origins[i] - origins[k]) * qd[k]
        o_dot.append(v_o)
        w_dot.append(np.cross(omega, axes[i]))
        omega = omega + axes[i] * qd[i]

    p_ee_dot = np.zeros(3)
    for k in range(N):
        p_ee_dot += np.cross(axes[k], p_ee - origins[k]) * qd[k]

    Jv_dot = np.zeros((3, N))
    Jw_dot = np.zeros((3, N))
    for i in range(N):
        Jv_dot[:, i] = (np.cross(w_dot[i], p_ee - origins[i])
                        + np.cross(axes[i], p_ee_dot - o_dot[i]))
        Jw_dot[:, i] = w_dot[i]
    return Jv_dot, Jw_dot


# ---------------------------------------------------------------- forward velocity / acceleration
def forward_velocity(q, qd):
    """EE linear and angular velocity: (v, w)."""
    Jv, Jw = jacobian(q)
    return Jv @ qd, Jw @ qd


def forward_acceleration(q, qd, qdd):
    """EE linear and angular acceleration: (a, alpha)."""
    Jv, Jw = jacobian(q)
    Jv_dot, Jw_dot = jacobian_dot(q, qd)
    return Jv @ qdd + Jv_dot @ qd, Jw @ qdd + Jw_dot @ qd


# ---------------------------------------------------------------- inverse kinematics
def damped_pinv(J, damping=1e-3):
    """Damped least-squares pseudo-inverse  J^T (J J^T + lambda^2 I)^-1."""
    return J.T @ np.linalg.inv(J @ J.T + damping**2 * np.eye(J.shape[0]))


def inverse_position(target, q0=None, tol=1e-8, max_iter=200, damping=1e-3):
    """Numerical IK (Newton / damped least squares) for the EE position. Returns (q, error)."""
    q = np.array([0.0, -0.3, 0.3, 0.0, 0.0] if q0 is None else q0, dtype=float)
    target = np.asarray(target, dtype=float)
    for _ in range(max_iter):
        err = target - ee_position(q)
        if np.linalg.norm(err) < tol:
            break
        Jv, _ = jacobian(q)
        q = q + damped_pinv(Jv, damping) @ err
        q = np.clip(q, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
    return q, float(np.linalg.norm(target - ee_position(q)))


def inverse_velocity(q, v, damping=1e-3, qd_null=None):
    """Joint velocities for a desired EE linear velocity v:  qd = J+ v  (+ null-space term)."""
    Jv, _ = jacobian(q)
    Jp = damped_pinv(Jv, damping)
    qd = Jp @ v
    if qd_null is not None:
        qd = qd + (np.eye(N) - Jp @ Jv) @ qd_null
    return qd


def inverse_acceleration(q, qd, a, damping=1e-3):
    """Joint accelerations for a desired EE linear acceleration a:  qdd = J+ (a - Jdot qd)."""
    Jv, _ = jacobian(q)
    Jv_dot, _ = jacobian_dot(q, qd)
    return damped_pinv(Jv, damping) @ (a - Jv_dot @ qd)


def manipulability(q):
    Jv, _ = jacobian(q)
    return float(np.sqrt(np.linalg.det(Jv @ Jv.T)))


if __name__ == "__main__":
    q = np.array([0.3, -0.4, 0.5, 0.2, 0.1])
    qd = np.array([0.2, -0.1, 0.3, 0.1, 0.0])
    qdd = np.array([0.1, 0.2, -0.1, 0.0, 0.1])
    print("EE position     :", ee_position(q).round(5))
    print("EE velocity     :", forward_velocity(q, qd)[0].round(5))
    print("EE acceleration :", forward_acceleration(q, qd, qdd)[0].round(5))
