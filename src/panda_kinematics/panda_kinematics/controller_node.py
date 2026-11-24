#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np

import logging

log = logging.getLogger("controller")

# ==========================================================
# PART A - copied unchanged from Challenge 1 (kinematics.py): the DH table of the Panda, the fixed
# link7 -> hand transform, dh_transform and the frame chain. Challenge 3 uses these frames to build
# the Jacobians of every link's centre of mass for the Lagrangian dynamics.
# ==========================================================

# Standard DH parameters for links 1 through 7.
#
# Format: one row per link, columns are [alpha, a, d].
# theta is the joint variable and is not stored here.
#
# Convention (standard / distal DH):
#   A_i = Rot_z(theta_i) * Trans_z(d_i) * Trans_x(a_i) * Rot_x(alpha_i)
#   z_{i-1} is the axis of joint i, so frame i sits on the axis of joint i+1.
#
# Derived from panda.xml (link0 -> link7):
#   link1 pos="0 0 0.333"                         -> d1 = 0.333
#   link2 quat="1 -1 0 0"  (Rx -90 deg)           -> alpha1 = -pi/2
#   link3 pos="0 -0.316 0" quat="1 1 0 0" (Rx+90) -> alpha2 = +pi/2, d3 = 0.316
#   link4 pos="0.0825 0 0" quat="1 1 0 0"         -> a3 = 0.0825,  alpha3 = +pi/2
#   link5 pos="-0.0825 0.384 0" quat="1 -1 0 0"   -> a4 = -0.0825, alpha4 = -pi/2, d5 = 0.384
#   link6 quat="1 1 0 0"                          -> alpha5 = +pi/2
#   link7 pos="0.088 0 0" quat="1 1 0 0"          -> a6 = 0.088,   alpha6 = +pi/2
#   joint 7 is the last joint, so row 7 is identity apart from theta7;
#   with d7 = 0, frame 7 coincides exactly with MuJoCo's 'link7' body frame.
DH_PARAMS = np.array([
    #  alpha        a        d
    [-np.pi / 2,  0.0,     0.333],   # link 1
    [ np.pi / 2,  0.0,     0.0  ],   # link 2
    [ np.pi / 2,  0.0825,  0.316],   # link 3
    [-np.pi / 2, -0.0825,  0.0  ],   # link 4
    [ np.pi / 2,  0.0,     0.384],   # link 5
    [ np.pi / 2,  0.088,   0.0  ],   # link 6
    [ 0.0,        0.0,     0.0  ],   # link 7
])

# Fixed transform from link7 to the 'hand' body.
# 4x4 numpy array.
#
# From <body name="hand" pos="0 0 0.107" quat="0.9238795 0 0 -0.3826834">:
#   translation 0.107 m along z7, and the quaternion (w, x, y, z) is a
#   rotation of -45 deg about z  (cos(-22.5 deg) = 0.9239, sin(-22.5 deg) = -0.3827).
_c45 = np.cos(-np.pi / 4)
_s45 = np.sin(-np.pi / 4)
T_L7_HAND = np.array([
    [_c45, -_s45, 0.0, 0.0  ],
    [_s45,  _c45, 0.0, 0.0  ],
    [0.0,   0.0,  1.0, 0.107],
    [0.0,   0.0,  0.0, 1.0  ],
])


def dh_transform(alpha, a, d, theta):
    """
    Single-link transformation matrix for standard DH
    (the same matrix as Tutorial 03, slide 13: i-1_T_i).

    alpha: twist angle (rad)
    a:     link length (m)
    d:     link offset (m)
    theta: joint angle (rad)

    Returns a 4x4 numpy array.
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,      sa,       ca,      d],
        [0.0,     0.0,      0.0,    1.0],
    ])


def _frames(q):
    """
    Cumulative transforms [T_0^0, T_0^1, ..., T_0^7, T_0^hand].
    Used by both FK and the Jacobian so they can never disagree.
    """
    T = np.eye(4)
    frames = [T]
    for (alpha, a, d), theta in zip(DH_PARAMS, q):
        T = T @ dh_transform(alpha, a, d, theta)
        frames.append(T)
    frames.append(T @ T_L7_HAND)
    return frames


def forward_kinematics(q):
    """
    Forward kinematics for the Panda arm.

    q: numpy array of 7 joint angles (rad).

    Returns a 4x4 homogeneous transform of the 'hand' frame
    relative to 'link0'.
    """
    q = np.asarray(q, dtype=float).reshape(7)
    T = np.eye(4)
    for (alpha, a, d), theta in zip(DH_PARAMS, q):
        T = T @ dh_transform(alpha, a, d, theta)
    return T @ T_L7_HAND


# ==========================================================
# PART B - NEW FOR CHALLENGE 3
# ==========================================================

# ----------------------------------------------------------
# 0. ROS 2 COMPATIBILITY SHIM
# ----------------------------------------------------------
# The template's start_trajectory_once() does  self.destroy_timer(self.timers[0]).  On ROS 2 Jazzy, Node.timers is a
# generator (not a list), so indexing it raises "TypeError: 'generator' object is not subscriptable" and the node never
# publishes. This makes Node.timers return a list on every distribution, so the template code below runs unchanged.
_timers_property = rclpy.node.Node.timers
if isinstance(_timers_property, property):
    rclpy.node.Node.timers = property(lambda self: list(_timers_property.fget(self)))


# ----------------------------------------------------------
# 1. LINK DATA (from panda.xml, MuJoCo Menagerie). Nothing in the model file is changed.
# ----------------------------------------------------------
GRAVITY = 9.81                 # m/s^2, along -z of the base frame
ARMATURE = 0.1                 # kg m^2, rotor inertia of every joint (<joint armature="0.1">); it adds to the diagonal of M
FINGER_OPEN = 0.04             # m, finger slide position when the gripper is open (GRIPPER_CTRL = 255)

# Franka limits used by the planner and for saturating the torque command
Q_LIM_LOW = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_LIM_HIGH = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
V_MAX = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100])     # rad/s
A_MAX = np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0])                    # rad/s^2
TAU_MAX = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])                 # Nm (forcerange of the actuators)

# Pose of every link body relative to its parent: (position [m], quaternion w x y z). Joint i turns about
# the z axis of its own body frame.
_BODY_OFFSETS = [
    ((0.0, 0.0, 0.333), (1, 0, 0, 0)),       # link1
    ((0.0, 0.0, 0.0), (1, -1, 0, 0)),        # link2
    ((0.0, -0.316, 0.0), (1, 1, 0, 0)),      # link3
    ((0.0825, 0.0, 0.0), (1, 1, 0, 0)),      # link4
    ((-0.0825, 0.384, 0.0), (1, -1, 0, 0)),  # link5
    ((0.0, 0.0, 0.0), (1, 1, 0, 0)),         # link6
    ((0.088, 0.0, 0.0), (1, 1, 0, 0)),       # link7
]
# Inertial data of every link: (mass [kg], centre of mass in the body frame [m],
# (Ixx, Iyy, Izz, Ixy, Ixz, Iyz) about the centre of mass in the body axes [kg m^2])
_LINK_INERTIAL = [
    (4.970684, (0.003875, 0.002081, -0.04762), (0.70337, 0.70661, 0.009117, -0.000139, 0.006772, 0.019169)),
    (0.646926, (-0.003141, -0.02872, 0.003495), (0.007962, 2.811e-2, 2.5995e-2, -3.925e-3, 1.0254e-2, 7.04e-4)),
    (3.228604, (2.7518e-2, 3.9252e-2, -6.6502e-2), (3.7242e-2, 3.6155e-2, 1.083e-2, -4.761e-3, -1.1396e-2, -1.2805e-2)),
    (3.587895, (-5.317e-2, 1.04419e-1, 2.7454e-2), (2.5853e-2, 1.9552e-2, 2.8323e-2, 7.796e-3, -1.332e-3, 8.641e-3)),
    (1.225946, (-1.1953e-2, 4.1065e-2, -3.8437e-2), (3.5549e-2, 2.9474e-2, 8.627e-3, -2.117e-3, -4.037e-3, 2.29e-4)),
    (1.666555, (6.0149e-2, -1.4117e-2, -1.0517e-2), (1.964e-3, 4.354e-3, 5.433e-3, 1.09e-4, -1.158e-3, 3.41e-4)),
    (0.735522, (1.0517e-2, -4.252e-3, 6.1597e-2), (1.2516e-2, 1.0027e-2, 4.815e-3, -4.28e-4, -1.196e-3, -7.41e-4)),
]
# Hand and the two fingers are rigidly attached to link 7 (the fingers do not move during the run), so they are
# lumped into link 7. Hand: mass, centre (hand frame), diagonal inertia. Finger: same, finger frame shifted by +-FINGER_OPEN.
_HAND = (0.73, (-0.01, 0.0, 0.03), (0.001, 0.0025, 0.0017))
_FINGER = (0.015, (0.0, 0.0, 0.0584), (2.375e-6, 2.375e-6, 7.5e-7))


def _quat_to_R(w, x, y, z):
    """Rotation matrix of a (not necessarily normalised) quaternion w x y z."""
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _inertia_matrix(i6):
    xx, yy, zz, xy, xz, yz = i6
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


def _build_link_parameters():
    """
    Mass, centre of mass and inertia of every link expressed in the DH frame of that link (frame i of the
    Challenge 1 table), which is the frame whose Jacobian we use.

    panda.xml gives the data in the MuJoCo body frames, the DH table uses its own frames; both are rigidly attached
    to the same link and both turn with joint i, so they differ by a CONSTANT transform C_i. C_i is computed once, at
    q = 0, from the two descriptions of the arm (tools/check_dynamics.py verifies it is constant for random q).
    """
    T_body = np.eye(4)
    bodies = []
    for pos, quat in _BODY_OFFSETS:
        step = np.eye(4)
        step[:3, :3] = _quat_to_R(*quat)
        step[:3, 3] = pos
        T_body = T_body @ step
        bodies.append(T_body.copy())
    frames0 = _frames(np.zeros(7))
    mass, com, inertia = np.zeros(7), np.zeros((7, 3)), np.zeros((7, 3, 3))
    for i in range(7):
        C = np.linalg.inv(frames0[i + 1]) @ bodies[i]            # body frame i expressed in DH frame i
        m, c, i6 = _LINK_INERTIAL[i]
        mass[i] = m
        com[i] = C[:3, :3] @ np.array(c) + C[:3, 3]
        inertia[i] = C[:3, :3] @ _inertia_matrix(i6) @ C[:3, :3].T

    # lump hand + fingers into link 7 (DH frame 7 = link7 body frame, so C_7 is the identity)
    Rh, ph = T_L7_HAND[:3, :3], T_L7_HAND[:3, 3]
    parts = [(mass[6], com[6], inertia[6])]
    m_h, c_h, i_h = _HAND
    parts.append((m_h, Rh @ np.array(c_h) + ph, Rh @ np.diag(i_h) @ Rh.T))
    m_f, c_f, i_f = _FINGER
    for sign in (+1.0, -1.0):
        centre = np.array(c_f) + np.array([0.0, sign * FINGER_OPEN, 0.0])
        parts.append((m_f, Rh @ centre + ph, Rh @ np.diag(i_f) @ Rh.T))
    m_tot = sum(p[0] for p in parts)
    c_tot = sum(p[0] * p[1] for p in parts) / m_tot
    i_tot = np.zeros((3, 3))
    for m_k, c_k, i_k in parts:                                   # parallel-axis theorem
        d = c_k - c_tot
        i_tot += i_k + m_k * (d @ d * np.eye(3) - np.outer(d, d))
    mass[6], com[6], inertia[6] = m_tot, c_tot, i_tot
    return mass, com, inertia


LINK_MASS, LINK_COM, LINK_INERTIA = _build_link_parameters()


def _skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


# ----------------------------------------------------------
# 2. EQUATIONS OF MOTION (Lagrangian)
# ----------------------------------------------------------
def compute_dynamics(q, dq, ddq):
    """
    q:   numpy array of 7 joint angles (rad).
    dq:  numpy array of 7 joint velocities (rad/s).
    ddq: numpy array of 7 joint accelerations (rad/s^2)  (not needed by the equations; accepted by the template).

    Returns:
        M_matrix: 7x7 numpy array (Mass/Inertia matrix).
        C_g_vector: 7x1 numpy array (Coriolis + Gravity forces).

    Lagrangian approach.  With L = T - V the equations of motion are  M(q) ddq + C(q,dq) dq + g(q) = tau.
      * Kinetic energy of link i:   T_i = 1/2 m_i |v_i|^2 + 1/2 w_i^T I_i w_i   with v_i = Jv_i dq, w_i = Jw_i dq,
        where Jv_i, Jw_i are the Jacobians of the link's centre of mass and I_i = R_i I_body R_i^T.
        Summing the links:  T = 1/2 dq^T M dq  with   M = sum_i ( m_i Jv_i^T Jv_i + Jw_i^T I_i Jw_i )  (+ rotor inertia).
      * Potential energy:           V = sum_i m_i g z_i(q)   ->   g(q) = dV/dq = sum_i m_i g Jv_i[z row]
      * Coriolis / centrifugal:     C(q,dq) dq = sum_jk c_ijk dq_j dq_k,  c_ijk = 1/2 (dM_ij/dq_k + dM_ik/dq_j - dM_jk/dq_i)
        (Christoffel symbols of the first kind, from the derivatives of the mass matrix).
    The Jacobian columns come from the DH frames: for joint j (axis z_j through o_j) a point p on a later link moves
    with z_j x (p - o_j) and the link turns with z_j. Their derivatives follow from "an axis only turns when an
    earlier joint turns":  dz_j/dq_k = z_k x z_j,  do_j/dq_k = z_k x (o_j - o_k)  (k < j),  dp/dq_k = z_k x (p - o_k).
    """
    q = np.asarray(q, dtype=float).reshape(7)
    dq = np.asarray(dq, dtype=float).reshape(7)
    frames = _frames(q)
    Z = np.array([frames[j][:3, 2] for j in range(7)])           # joint axes in the base frame
    O = np.array([frames[j][:3, 3] for j in range(7)])           # a point on each axis

    M = ARMATURE * np.eye(7)                                     # rotor inertia on the diagonal
    dM = np.zeros((7, 7, 7))                                     # dM[k, i, j] = dM_ij / dq_k
    g = np.zeros(7)

    for L in range(1, 8):                                        # link L depends on joints 1..L
        T = frames[L]
        R = T[:3, :3]
        p = R @ LINK_COM[L - 1] + T[:3, 3]                       # centre of mass in the base frame
        I_w = R @ LINK_INERTIA[L - 1] @ R.T                      # inertia about the centre of mass, base axes
        m = LINK_MASS[L - 1]
        Zs, Os = Z[:L], O[:L]
        r = p - Os                                               # lever arms of the joints to the centre of mass
        Jv = np.cross(Zs, r).T                                   # 3 x L   linear velocity Jacobian
        Jw = Zs.T                                                # 3 x L   angular velocity Jacobian

        M[:L, :L] += m * Jv.T @ Jv + Jw.T @ I_w @ Jw             # kinetic energy -> mass matrix
        g[:L] += m * GRAVITY * Jv[2, :]                          # potential energy -> gravity torque

        # derivatives with respect to every joint k < L at once (arrays are indexed [k, j, ...])
        mask = np.triu(np.ones((L, L)), k=1)[:, :, None]                             # 1 where j > k: only later axes turn with joint k
        dz = np.cross(Z[:L, None, :], Zs[None, :, :]) * mask                         # dz_j/dq_k = z_k x z_j
        do = np.cross(Z[:L, None, :], Os[None, :, :] - O[:L, None, :]) * mask        # do_j/dq_k = z_k x (o_j - o_k)
        dp = np.cross(Z[:L], p - O[:L])                                              # dp/dq_k  = z_k x (p - o_k)
        dJv = (np.cross(dz, r[None, :, :]) + np.cross(Zs[None, :, :], dp[:, None, :] - do)).transpose(0, 2, 1)   # [k, 3, j]
        dJw = dz.transpose(0, 2, 1)                                                  # [k, 3, j]
        S = np.zeros((L, 3, 3))                                                      # skew(z_k) for every k
        S[:, 0, 1], S[:, 0, 2], S[:, 1, 0] = -Z[:L, 2], Z[:L, 1], Z[:L, 2]
        S[:, 1, 2], S[:, 2, 0], S[:, 2, 1] = -Z[:L, 0], -Z[:L, 1], Z[:L, 0]
        dI_w = S @ I_w - I_w @ S                                                     # d(R I R^T)/dq_k
        A = np.einsum('kaj,ai->kji', dJv, Jv)                                        # dJv^T Jv
        B = np.einsum('kaj,ab,bi->kji', dJw, I_w, Jw)                                # dJw^T I Jw
        D = np.einsum('aj,kab,bi->kji', Jw, dI_w, Jw)                                # Jw^T dI Jw
        dM[:L, :L, :L] += m * (A + A.transpose(0, 2, 1)) + B + B.transpose(0, 2, 1) + D

    # Christoffel form:  (C dq)_i = sum_jk [ dM_ij/dq_k - 1/2 dM_jk/dq_i ] dq_j dq_k
    coriolis = np.einsum('kij,j,k->i', dM, dq, dq) - 0.5 * np.einsum('ijk,j,k->i', dM, dq, dq)
    return M, coriolis + g


# ----------------------------------------------------------
# 3. TRAJECTORY PLANNING (fast quintic; from Challenge 2, now planned as fast as the limits allow)
# ----------------------------------------------------------
SAFETY_FAST = 0.95             # use 95 % of Franka's speed / acceleration limits: Team 6 must be FAST
TORQUE_USE = 0.90              # the planned motion may use at most 90 % of the actuator torque limits
HOLD_TIME = 3.0                # s of "stand still at q_end" appended to the plan so the arm comes to a full stop under feedback


def _quintic(q_start, q_end, T, dt):
    """Quintic s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5: zero velocity and acceleration at both ends."""
    q_start = np.asarray(q_start, dtype=float)
    delta = np.asarray(q_end, dtype=float) - q_start
    n = max(1, int(round(T / dt)))
    T = n * dt
    t = np.arange(n + 1) * dt
    tau = t / T
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    ds = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / T
    dds = (60 * tau - 180 * tau**2 + 120 * tau**3) / T**2
    return t, q_start + np.outer(s, delta), np.outer(ds, delta), np.outer(dds, delta)


def minimum_time(q_start, q_end, factor=SAFETY_FAST):
    """Shortest T for which the quintic keeps every joint inside factor * (speed, acceleration) limits."""
    delta = np.abs(np.asarray(q_end, dtype=float) - np.asarray(q_start, dtype=float))
    t_vel = (15.0 / 8.0) * delta / (factor * V_MAX)              # peak speed of a quintic = 15/8 |dq| / T
    t_acc = np.sqrt((10.0 / np.sqrt(3.0)) * delta / (factor * A_MAX))   # peak acceleration = 10/sqrt(3) |dq| / T^2
    return float(max(t_vel.max(), t_acc.max()))


def _torque_peak_ratio(q, dq, ddq):
    """Largest |model torque| / torque limit over the trajectory (every 3rd sample)."""
    worst = 0.0
    for a, b, c in zip(q[::3], dq[::3], ddq[::3]):
        M, Cg = compute_dynamics(a, b, c)
        worst = max(worst, float((np.abs(M @ c + Cg) / TAU_MAX).max()))
    return worst


def plan_trajectory(q_start, q_end, T, dt):
    """
    Returns (t_array, q_array, dq_array, ddq_array), each with N rows (N, 7), from q_start to q_end.

    High-speed variant: T is treated as the LONGEST allowed duration. The planner uses the fastest time that keeps
    every joint inside 95 % of Franka's speed and acceleration limits and the required model torque inside 90 % of
    the actuator limits (the duration is stretched in 3 % steps if the torque limit would be exceeded).
    Joint limits: the quintic is monotonic, so checking the two end points covers the whole path.
    The plan ends with HOLD_TIME seconds of zero-velocity samples at q_end (a controlled stop, see the end of the function).
    """
    q_start = np.asarray(q_start, dtype=float).reshape(7)
    q_end = np.asarray(q_end, dtype=float).reshape(7)
    for name, qq in (("q_start", q_start), ("q_end", q_end)):
        if np.any(qq < Q_LIM_LOW - 1e-9) or np.any(qq > Q_LIM_HIGH + 1e-9):
            raise ValueError(f"{name} violates the joint limits: {np.round(qq, 3)}")

    T_use = min(float(T), minimum_time(q_start, q_end))          # as fast as the speed / acceleration limits allow
    for _ in range(60):
        t, q, dq, ddq = _quintic(q_start, q_end, T_use, dt)
        ratio = _torque_peak_ratio(q, dq, ddq)
        if ratio <= TORQUE_USE or T_use >= T:
            break
        T_use = min(float(T), T_use * 1.03)                       # too much torque: slow down a little
    log.info("planned %.2f s (asked for at most %.2f s), peak model torque %.0f %% of the limit", t[-1], T, 100 * ratio)

    # Come to a FULL stop: after the motion, keep "standing still at q_end" for HOLD_TIME seconds. The template keeps
    # running the model + FLC controller on every sample of the plan, so the arm is actively held (feedback) until its
    # velocity has died out; only afterwards the template switches to its gravity-only hold, which then holds a resting arm.
    n_hold = int(round(HOLD_TIME / dt))
    t = np.concatenate([t, t[-1] + dt * np.arange(1, n_hold + 1)])
    q = np.vstack([q, np.tile(q[-1], (n_hold, 1))])
    dq = np.vstack([dq, np.zeros((n_hold, 7))])
    ddq = np.vstack([ddq, np.zeros((n_hold, 7))])
    return t, q, dq, ddq


# ----------------------------------------------------------
# 4. FUZZY LOGIC CONTROLLER (MIMO = 7 decoupled single-joint FLCs working side by side)
# ----------------------------------------------------------
# Every joint has its own FLC with TWO inputs (error e, change of error de) and ONE output (a correction torque).
#   fuzzification   : each input is scaled to [-1, 1] and given a degree of membership in five triangular sets
#                     LN (large negative), MN, S (small), MP, LP (large positive), centres -1, -0.5, 0, 0.5, 1;
#                     neighbouring sets cross at 0.5, so the memberships of an input always add up to 1
#   rule base       : 25 IF-THEN rules "IF e is A AND de is B THEN u is C" with C = A + B (clipped to the 5 sets),
#                     e.g. IF e is LP and de is S THEN u is LP; IF e is LP and de is LN THEN u is S
#   inference       : Mamdani: AND = min, each rule clips its output set, all rules combined with max
#   defuzzification : centroid of area of the combined output set (sampled at 201 points)
# Then u (in [-1, 1]) is multiplied by the joint's maximum correction torque.
FLC_E_SCALE = np.full(7, 0.05)                                    # rad    error at which the input counts as "large"
FLC_DE_SCALE = np.full(7, 1.5)                                    # rad/s  change of error that counts as "large"
FLC_U_SCALE = np.array([20.0, 20.0, 20.0, 20.0, 8.0, 8.0, 8.0])   # Nm     correction torque for a "large" output
FLC_SMOOTH_TIME = 0.0                                             # s      optional low-pass on the output (0 = off; the
                                                                  #        output is already continuous, extra smoothing only adds delay)
# The scales were tuned in simulation WITH the real loop's latency (about 5-10 ms between reading the state and the
# torque reaching the motors): larger torque scales made the loop chatter at about 10 Hz. They are per-joint arrays,
# so each of the 7 decoupled FLCs can be tuned separately.

_SET_CENTRES = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])               # LN MN S MP LP
_U_GRID = np.linspace(-1.0, 1.0, 201)                              # output universe, sampled for the centroid
_RULES = np.clip(np.arange(-2, 3)[:, None] + np.arange(-2, 3)[None, :], -2, 2) + 2   # rule (e set, de set) -> output set
_flc_state = {"u": np.zeros(7)}                                    # previous (smoothed) output


def _memberships(x):
    """Degrees of membership of x (clipped to [-1, 1]) in the five triangular sets: shape x.shape + (5,)."""
    return np.clip(1.0 - np.abs(np.clip(x, -1.0, 1.0)[..., None] - _SET_CENTRES) / 0.5, 0.0, 1.0)


_OUTPUT_SETS = _memberships(_U_GRID)                               # (201, 5): the five output sets on the grid


def fuzzy_controller(error, delta_error, dt):
    """
    error:       7x1 numpy array (desired_q - current_q).
    delta_error: 7x1 numpy array (desired_dq - current_dq).
    dt:          Time step in seconds.

    Returns:
        tau_correction: 7x1 numpy array of correction torques.
    """
    mu_e = _memberships(np.asarray(error, dtype=float) / FLC_E_SCALE)             # (7, 5)
    mu_de = _memberships(np.asarray(delta_error, dtype=float) / FLC_DE_SCALE)     # (7, 5)

    # rule firing strengths: w[j, a, b] = min(mu_e[j, a], mu_de[j, b])
    w = np.minimum(mu_e[:, :, None], mu_de[:, None, :])
    # strength of each OUTPUT set = max over all rules that point to it
    strength = np.stack([np.where(_RULES == o, w, 0.0).reshape(7, -1).max(axis=1) for o in range(5)], axis=1)   # (7, 5)
    # clip every output set at its strength and take the union (max)
    combined = np.minimum(strength[:, None, :], _OUTPUT_SETS[None, :, :]).max(axis=2)                          # (7, 201)
    area = combined.sum(axis=1)
    u = np.where(area > 1e-9, (combined * _U_GRID).sum(axis=1) / np.maximum(area, 1e-9), 0.0)   # centroid of area

    # first-order smoothing of the output: avoids chattering when the active rules change
    a = np.exp(-dt / FLC_SMOOTH_TIME) if FLC_SMOOTH_TIME > 0 else 0.0
    _flc_state["u"] = a * _flc_state["u"] + (1.0 - a) * u
    return FLC_U_SCALE * _flc_state["u"]


# ----------------------------------------------------------
# 5. CONTROL LAW
# ----------------------------------------------------------
# Switches used only for the experiments in the report (the submitted values are the defaults):
USE_MODEL = True               # False: no Lagrangian torque (FLC alone)
USE_FLC = True                 # False: no fuzzy correction (model alone)
MODEL_TERMS = "full"           # "full" | "no_coriolis" | "decoupled" (diagonal M) | "gravity" (g only)


def compute_torques(q, dq, q_des, dq_des, ddq_des):
    """
    q:       current joint positions (7x1).
    dq:      current joint velocities (7x1).
    q_des:   desired joint positions (7x1).
    dq_des:  desired joint velocities (7x1).
    ddq_des: desired joint accelerations (7x1).

    Returns:
        final_torques: 7x1 numpy array to send to the motors.

    tau = [ M(q) ddq_des + C(q,dq) dq + g(q) ]  +  FLC(e, de)
          model feedforward (does the heavy lifting)   correction of what the model does not capture
    """
    tau_model = np.zeros(7)
    if USE_MODEL:
        M, Cg = compute_dynamics(q, dq, ddq_des)
        if MODEL_TERMS == "full":
            tau_model = np.dot(M, ddq_des) + Cg
        else:
            g = compute_dynamics(q, np.zeros(7), ddq_des)[1]               # gravity only
            if MODEL_TERMS == "no_coriolis":
                tau_model = np.dot(M, ddq_des) + g
            elif MODEL_TERMS == "decoupled":
                tau_model = np.diag(M) * ddq_des + Cg                       # joints treated as independent masses
            else:
                tau_model = g

    # 2. Calculate the error for the FLC
    error = q_des - q
    delta_error = dq_des - dq  # (Can also be (error - prev_error) / dt)

    # 3. Get the correction torque from the FLC
    tau_correction = fuzzy_controller(error, delta_error, 0.01) if USE_FLC else np.zeros(7)   # 0.01 is dt

    # 4. Combine them, and never ask for more than the actuators can deliver
    final_torques = np.clip(tau_model + tau_correction, -TAU_MAX, TAU_MAX)

    return final_torques



# ==========================================================
# DO NOT MODIFY BELOW
# ==========================================================

CMD_SIZE = 9
GRIPPER_CTRL = 255.0
TRAJ_DURATION = 4.0
TRAJ_DT = 0.01

# Team-specific start and end configurations.
# Team 6 - "The High-Speed Tracker": from the upright rest pose to a far, folded-back pose. plan_trajectory
# treats TRAJ_DURATION as a maximum and plays the motion at the fastest speed the limits allow (about 1.1 s),
# so inertial and Coriolis torques exceed gravity. Both poses are inside the joint limits and collision free.
Q_START = np.array([0.0, 0.0, 0.0, -0.1, 0.0, 0.3, 0.0])
Q_END   = np.array([0.6, -1.08, 0.81, -0.64, -1.47, 1.12, -1.37])

class ControllerNode(Node):
    def __init__(self):
        super().__init__('controller_node')
        self.q_current = np.zeros(7)
        self.dq_current = np.zeros(7)
        
        self.q_traj = None
        self.dq_traj = None
        self.ddq_traj = None
        self.step_idx = 0

        self.create_subscription(JointState, 'joint_state', self.on_joint_state, 10)
        self.cmd_pub = self.create_publisher(Float64MultiArray, 'joint_commands', 10)

        self.create_timer(0.5, self.start_trajectory_once)

    def on_joint_state(self, msg):
        self.q_current = np.array(msg.position[:7])
        if hasattr(msg, 'velocity') and len(msg.velocity) >= 7:
            self.dq_current = np.array(msg.velocity[:7])

    def start_trajectory_once(self):
        if self.q_traj is not None:
            return

        result = plan_trajectory(Q_START, Q_END, TRAJ_DURATION, TRAJ_DT)
        if result is None:
            return

        _, self.q_traj, self.dq_traj, self.ddq_traj = result
        self.get_logger().info("Trajectory generated. Starting Controller.")
        
        # Switch to the control loop timer
        self.destroy_timer(self.timers[0])
        self.create_timer(TRAJ_DT, self.control_loop)

    def control_loop(self):
        if self.step_idx >= len(self.q_traj):
            # Hold final position by sending zero velocity and holding torque
            cmd = Float64MultiArray()
            M, Cg = compute_dynamics(self.q_current, np.zeros(7), np.zeros(7))
            cmd.data = Cg.tolist() + [GRIPPER_CTRL, 0.0]
            self.cmd_pub.publish(cmd)
            return

        q_des = self.q_traj[self.step_idx]
        dq_des = self.dq_traj[self.step_idx]
        ddq_des = self.ddq_traj[self.step_idx]

        torques = compute_torques(
            self.q_current, self.dq_current, 
            q_des, dq_des, ddq_des
        )

        cmd = Float64MultiArray()
        cmd.data = torques.tolist() + [GRIPPER_CTRL, 0.0]
        self.cmd_pub.publish(cmd)

        self.step_idx += 1


def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
