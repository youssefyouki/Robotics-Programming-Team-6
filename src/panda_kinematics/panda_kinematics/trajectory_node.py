#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np


# ==========================================================
#  EDIT ONLY THIS SECTION
# ==========================================================

import logging
import time

log = logging.getLogger("trajectory_planner")

# ----------------------------------------------------------
# PART A - copied unchanged from Challenge 1 (kinematics.py):
#          DH_PARAMS, T_L7_HAND, dh_transform, forward_kinematics, inverse_kinematics
#          (plus the joint limits and the private helpers the inverse kinematics needs).
#          Challenge 2 only reuses the DH table and the frame chain; the inverse kinematics
#          is kept because the template asks for it.
# ----------------------------------------------------------

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

# Joint limits for joints 1 through 7, in radians.
# numpy arrays of shape (7,).
#
# Joints without their own range use the class default range="-2.8973 2.8973".
Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_MAX = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])


# ---------------- IK tuning ----------------
IK_POS_TOL = 1e-4        # m    (0.1 mm)  position convergence tolerance
IK_ROT_TOL = 1e-3        # rad  (0.06 deg) orientation convergence tolerance
IK_MAX_ITERS = 150       # iterations per attempt
IK_MAX_STEP = 0.2        # rad, largest change of any joint in one iteration
IK_MU_INIT = 1.0         # initial LM weight on the squared error
IK_LAMBDA_BIAS = 1e-5    # constant damping floor: keeps J J^T + lambda^2 I invertible
IK_RESTARTS = 10         # extra seeds tried if the solve from q0 fails
IK_TIME_BUDGET = 0.2     # s; hard cap so a hopeless target cannot stall the node

# Upper bound on the distance from the shoulder (joint 2 axis) to the hand
# origin: sum of the rigid segment lengths shoulder->elbow->wrist->hand.
SHOULDER = np.array([0.0, 0.0, 0.333])
MAX_REACH = (np.hypot(0.316, 0.0825)      # joint2 -> joint4
             + np.hypot(0.384, 0.0825)    # joint4 -> joint6
             + np.hypot(0.088, 0.107))    # joint6 -> hand origin


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


def _jacobian_from_frames(frames):
    """
    6x7 geometric Jacobian of the hand origin, expressed in link0.
    Standard DH: joint i rotates about z_{i-1} through o_{i-1}, so column i is
        J_v = z_{i-1} x (p_hand - o_{i-1}),   J_w = z_{i-1}
    """
    Z = np.array([F[:3, 2] for F in frames[:7]])
    O = np.array([F[:3, 3] for F in frames[:7]])
    p_e = frames[-1][:3, 3]
    J = np.empty((6, 7))
    J[:3] = np.cross(Z, p_e - O).T
    J[3:] = Z.T
    return J


def geometric_jacobian(q):
    """Returns (J, T_hand) at configuration q."""
    frames = _frames(q)
    return _jacobian_from_frames(frames), frames[-1]


def _rotation_log(R):
    """Axis-angle vector w (|w| = angle) such that exp([w]x) = R. Robust near 0 and pi."""
    cos_th = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    th = np.arccos(cos_th)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < 1e-6:
        return 0.5 * v                      # small angle: w ~ vee(R - R^T)/2
    if np.pi - th < 1e-4:
        # near 180 deg the skew part vanishes; read the axis off (R + I)/2 = k k^T
        B = (R + np.eye(3)) / 2.0
        k = int(np.argmax(np.diag(B)))
        axis = B[:, k] / np.sqrt(max(B[k, k], 1e-12))
        if np.dot(axis, v) < 0:
            axis = -axis
        return th * axis / np.linalg.norm(axis)
    return th / (2.0 * np.sin(th)) * v


def pose_error(T_target, T_current):
    """6-vector [position error ; orientation error], both in link0."""
    e_p = T_target[:3, 3] - T_current[:3, 3]
    e_o = _rotation_log(T_target[:3, :3] @ T_current[:3, :3].T)
    return np.concatenate([e_p, e_o])


def _converged(e):
    return np.linalg.norm(e[:3]) < IK_POS_TOL and np.linalg.norm(e[3:]) < IK_ROT_TOL


def _lm_solve(target_pose, q_start, deadline):
    """
    One Levenberg-Marquardt run from q_start (damped least squares).

    Link to the course (Tutorial 04, Newton-Raphson):
        q_{n+1} = q_n - (dF/dq)^-1 F(q_n),   with F(q) = FK(q) - target
    Here F(q_n) = -e and dF/dq = J, so Newton-Raphson is q_{n+1} = q_n + J^-1 e.
    For the Panda J is 6x7 (7 joints, 6 pose numbers), so J^-1 does not exist;
    it is replaced by the damped pseudo-inverse below. With lambda = 0 this is
    exactly Newton-Raphson with the pseudo-inverse J^T (J J^T)^-1.
    The two Newton-Raphson problems listed in the tutorial are handled too:
      "slope = 0"  (singular J)  -> damping keeps the step finite
      "diverges"                 -> a step that increases the error is rejected,
                                    and stuck runs are restarted from new guesses

        dq = J^T (J J^T + lambda^2 I)^-1 e          (minimum-norm damped step)
        lambda^2 = mu * |e|^2 + lambda_bias        (Sugihara-style damping)

    * Damping proportional to the squared error: far from the target, or when
      J is near-singular and the target is not reachable along the lost
      direction, lambda is large and the step is short and gradient-like; as
      the error goes to zero, lambda goes to zero and the step becomes the
      Gauss-Newton step, so convergence is fast at the end.
    * mu adapts like a trust region: halved after a step that reduces the
      error, x4 after a step that would increase it (step rejected). Every
      accepted step strictly lowers |e|, so the iteration cannot diverge.
    * Singularities: lambda_bias > 0 keeps (J J^T + lambda^2 I) invertible even
      when J loses rank, and IK_MAX_STEP bounds |dq|, so nothing blows up.
    * Joint limits: q is clamped after every step, and a joint that already
      sits on a limit and is being pushed further out is frozen for that step
      (its Jacobian column is zeroed) so the other joints take up the motion.
    Returns (q, converged, final_error_vector).
    """
    I6 = np.eye(6)
    q = np.clip(q_start, Q_MIN, Q_MAX)
    frames = _frames(q)
    J = _jacobian_from_frames(frames)
    e = pose_error(target_pose, frames[-1])
    cost = e @ e
    mu = IK_MU_INIT
    for _ in range(IK_MAX_ITERS):
        if _converged(e):
            return q, True, e
        if time.perf_counter() > deadline:
            break

        lam2 = mu * cost + IK_LAMBDA_BIAS
        free = np.ones(7, dtype=bool)
        for _ in range(7):                       # active set for the joint limits
            Jf = J * free
            dq = Jf.T @ np.linalg.solve(Jf @ Jf.T + lam2 * I6, e)
            pushing_out = ((q <= Q_MIN + 1e-9) & (dq < 0)) | ((q >= Q_MAX - 1e-9) & (dq > 0))
            if not np.any(pushing_out & free):
                break
            free &= ~pushing_out

        step = np.max(np.abs(dq))
        if step > IK_MAX_STEP:                   # step-size limit
            dq *= IK_MAX_STEP / step

        q_new = np.clip(q + dq, Q_MIN, Q_MAX)    # joint limits enforced inside the loop
        frames_new = _frames(q_new)
        e_new = pose_error(target_pose, frames_new[-1])
        cost_new = e_new @ e_new

        if cost_new < cost:                      # accept
            q, e, cost = q_new, e_new, cost_new
            J = _jacobian_from_frames(frames_new)
            mu = max(mu * 0.5, 1e-6)
        else:                                    # reject, damp harder
            mu *= 4.0
            if mu * cost > 1e3:                  # step ~0: local minimum / unreachable
                break

    return q, _converged(e), e


def _is_valid_pose(T):
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        return False
    R = T[:3, :3]
    return (np.allclose(R.T @ R, np.eye(3), atol=1e-6)
            and abs(np.linalg.det(R) - 1.0) < 1e-6)


def inverse_kinematics(target_pose, q0):
    """
    Inverse kinematics for the Panda arm.

    Method: numerical Levenberg-Marquardt (damped least squares) on the 6x7
    geometric Jacobian, with error-scaled damping, a per-step size limit,
    joint limits enforced inside the loop, and a few restarts if the seed
    gets stuck in a local minimum. See _lm_solve for the details.

    Which solution is returned: the solve starts from the seed q0 (the current
    joint state) and each damped step is the minimum-norm joint change that
    reduces the error, so it converges to a solution near the current
    configuration (redundancy is resolved by "move the joints as little as
    possible"). Only if that fails are other seeds tried, and among those the
    converged solution closest to q0 (in joint space) is returned.

    target_pose: 4x4 desired pose of the 'hand' frame in 'link0'.
    q0:          numpy array of 7 joint angles to seed the solve.

    Returns a tuple (q_solution, converged_flag), where
    q_solution is a numpy array of 7 joint angles and
    converged_flag is a bool. On failure q_solution is q0 clamped into the
    limits, i.e. "hold where you are", so a failed solve never moves the arm.
    """
    q0 = np.clip(np.asarray(q0, dtype=float).reshape(7), Q_MIN, Q_MAX)

    # 1) Reject malformed targets.
    if target_pose is None or not _is_valid_pose(target_pose):
        return q0, False
    target_pose = np.asarray(target_pose, dtype=float)

    # 2) Cheap reachability test: outside the reach sphere -> no solution exists.
    if np.linalg.norm(target_pose[:3, 3] - SHOULDER) > MAX_REACH:
        return q0, False

    deadline = time.perf_counter() + IK_TIME_BUDGET

    # 3) Solve from the seed.
    q, ok, _ = _lm_solve(target_pose, q0, deadline)
    if ok:
        return q, True

    # 4) Seed got stuck (joint limit / local minimum): deterministic restarts,
    #    keep the converged solution closest to q0.
    rng = np.random.default_rng(0)
    q_mid = 0.5 * (Q_MIN + Q_MAX)
    seeds = [q_mid] + [rng.uniform(Q_MIN, Q_MAX) for _ in range(IK_RESTARTS - 1)]
    solutions = []
    for s in seeds:
        if time.perf_counter() > deadline:
            break
        q, ok, _ = _lm_solve(target_pose, s, deadline)
        if ok:
            solutions.append(q)
    if solutions:
        return min(solutions, key=lambda qs: np.linalg.norm(qs - q0)), True

    # 5) Nothing converged: report unreachable and hold position.
    return q0, False


# ----------------------------------------------------------
# PART B - NEW FOR CHALLENGE 2
#          compute_jacobian, compute_jacobian_dot, plan_trajectory (+ helpers and limits)
# ----------------------------------------------------------

# Franka's published maximum joint speed (rad/s) and acceleration (rad/s^2) per joint.
V_MAX = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100])
A_MAX = np.array([15.0,   7.5,    10.0,   12.5,   15.0,   20.0,   20.0])
SAFETY = 0.6          # only use 60 % of those limits so the arm never runs at its maximum

# ----------------------------------------------------------
# 3. SINGULARITY GUARD SETTINGS
# ----------------------------------------------------------
# Two indicators of the 3x7 Jacobian are watched along the path:
#   sigma_min(J): smallest singular value = "how many m/s the hand can still make per rad/s of
#                 joint speed in its WORST direction". At 0.08, a hand speed of 0.1 m/s in the
#                 worst direction already needs 1.25 rad/s of joint speed.
#   w = sqrt(det(J J^T)): manipulability, the overall "volume" of hand velocities available.
#                 It collapses faster than sigma_min when the arm stretches out.
# The path is declared unsafe if EITHER falls below its threshold. (Typical values over random
# poses: sigma_min median 0.16, w median ~0.1. Fully stretched straight up: sigma_min 0.10, w 0.012.)
SIGMA_MIN_ALLOWED = 0.08
W_MIN_ALLOWED = 0.03
GUARD_SAMPLES = 80    # how many points along the path are checked by the guard

# ----------------------------------------------------------
# 5. JACOBIAN  (linear velocity only -> 3x7)
# ----------------------------------------------------------
def compute_jacobian(q):
    """
    Linear-velocity Jacobian of the hand origin, in the base frame (3 x 7).

        v_hand = J(q) @ dq

    Column i (one per joint) answers: "if ONLY joint i turns at 1 rad/s, how fast does the
    hand move?". For a revolute joint that spins about the unit axis z through the point o,
    a point p moves with velocity  z x (p - o)  (rotation axis cross lever arm), so

        column_i = z_{i-1} x ( p_hand - o_{i-1} )

    where z_{i-1} is the z axis and o_{i-1} the origin of DH frame i-1 (both read from the
    cumulative DH transforms). Rows are the hand speed along base x, y, z.
    """
    q = np.asarray(q, dtype=float).reshape(7)
    frames = _frames(q)
    p_hand = frames[-1][:3, 3]                       # where the hand is
    J = np.zeros((3, 7))
    for i in range(7):
        z = frames[i][:3, 2]                         # axis of joint i+1 (z of frame i)
        o = frames[i][:3, 3]                         # a point on that axis (origin of frame i)
        J[:, i] = np.cross(z, p_hand - o)            # axis x lever arm
    return J


def compute_jacobian_dot(q, dq):
    """
    Time derivative of the Jacobian, dJ/dt (3 x 7), computed analytically.

    J changes over time only because the joints move, so dJ/dt depends on q AND dq. It is what
    turns joint accelerations into hand acceleration:

        a_hand = J(q) @ ddq + dJ/dt(q, dq) @ dq

    Differentiating column_i = z_i x (p - o_i) with the product rule:

        dcol_i = (dz_i x (p - o_i))  +  (z_i x (dp - do_i))

    Every vector moves only because the joints BEFORE it rotate:
        angular velocity of frame i :  w_i  = sum_{k<i} z_k dq_k
        dz_i                        =  w_i x z_i                 (an axis just turns)
        do_i  (origin of frame i)   =  sum_{k<i} z_k x (o_i - o_k) dq_k
        dp    (hand velocity)       =  sum_{k<7} z_k x (p - o_k) dq_k  =  J dq
    """
    q = np.asarray(q, dtype=float).reshape(7)
    dq = np.asarray(dq, dtype=float).reshape(7)
    frames = _frames(q)
    z = [frames[i][:3, 2] for i in range(7)]         # joint axes
    o = [frames[i][:3, 3] for i in range(7)]         # a point on each axis
    p = frames[-1][:3, 3]                            # hand position

    # velocity of the hand: sum of every joint's contribution
    dp = np.zeros(3)
    for k in range(7):
        dp += np.cross(z[k], p - o[k]) * dq[k]

    Jdot = np.zeros((3, 7))
    omega = np.zeros(3)                              # angular velocity of frame i (grows with i)
    for i in range(7):
        dz = np.cross(omega, z[i])                   # how fast the axis z_i is turning
        do = np.zeros(3)                             # how fast the origin o_i is moving
        for k in range(i):
            do += np.cross(z[k], o[i] - o[k]) * dq[k]
        Jdot[:, i] = np.cross(dz, p - o[i]) + np.cross(z[i], dp - do)
        omega = omega + z[i] * dq[i]                 # joint i adds to the frames after it
    return Jdot


def compute_jacobian_dot_numeric(q, dq, h=1e-6):
    """Central finite-difference version of dJ/dt, used only to cross-check the analytic one."""
    q = np.asarray(q, dtype=float)
    dq = np.asarray(dq, dtype=float)
    return (compute_jacobian(q + h * dq) - compute_jacobian(q - h * dq)) / (2.0 * h)


def jacobian_metrics(q):
    """
    Singularity indicators of the 3x7 Jacobian:
      sigma_min : smallest singular value (-> 0 at a singularity)
      w         : Yoshikawa manipulability sqrt(det(J J^T)) (the "volume" of the speeds the
                  hand can reach with unit joint speed; -> 0 at a singularity)
    """
    J = compute_jacobian(q)
    sigma_min = np.linalg.svd(J, compute_uv=False)[-1]
    w = np.sqrt(max(np.linalg.det(J @ J.T), 0.0))
    return sigma_min, w


# ----------------------------------------------------------
# 6. TRAJECTORY PLANNING
# ----------------------------------------------------------
def _quintic(q_start, q_end, T, dt):
    """
    Raw quintic (5th-order polynomial) joint-space interpolation, no checks.

    Each joint follows   q(t) = q0 + s(tau) * (q1 - q0),   tau = t / T in [0, 1],
        s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5
    s has the six properties that pin down a 5th-order polynomial:
        s(0)=0, s(1)=1                (start / end position)
        s'(0)=0, s'(1)=0              (start / end velocity = 0)
        s''(0)=0, s''(1)=0            (start / end acceleration = 0)
    All joints share the same s, so they start and stop together and the path in joint space
    is a straight line. T is rounded to a whole number of time steps so the last sample is
    exactly tau = 1.
    """
    q_start = np.asarray(q_start, dtype=float)
    delta = np.asarray(q_end, dtype=float) - q_start
    n = max(1, int(round(T / dt)))                   # number of steps
    T = n * dt                                       # duration actually used
    times = np.arange(n + 1) * dt
    tau = times / T
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5                    # position profile
    ds = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / T            # ds/dt
    dds = (60 * tau - 180 * tau**2 + 120 * tau**3) / T**2         # d2s/dt2
    q = q_start + np.outer(s, delta)                 # (n+1, 7)
    dq = np.outer(ds, delta)
    ddq = np.outer(dds, delta)
    return times, q, dq, ddq


def minimum_time(q_start, q_end):
    """
    Shortest duration for which a quintic keeps every joint inside SAFETY * (speed, accel) limits.
    For the quintic, the peak speed is 15/8 * |dq| / T (at the middle of the motion) and the
    peak acceleration is 10/sqrt(3) * |dq| / T^2, so the limits turn into lower bounds on T.
    """
    delta = np.abs(np.asarray(q_end, dtype=float) - np.asarray(q_start, dtype=float))
    t_vel = (15.0 / 8.0) * delta / (SAFETY * V_MAX)
    t_acc = np.sqrt((10.0 / np.sqrt(3.0)) * delta / (SAFETY * A_MAX))
    return float(max(t_vel.max(), t_acc.max()))


def _path_is_safe(q_start, q_end, samples=GUARD_SAMPLES):
    """
    True if sigma_min(J) and w both stay above their thresholds at every sampled point of the
    straight joint-space segment q_start -> q_end (the quintic only changes the timing along
    this segment, not the segment itself).
    """
    for a in np.linspace(0.0, 1.0, samples):
        sigma_min, w = jacobian_metrics(q_start + a * (q_end - q_start))
        if sigma_min < SIGMA_MIN_ALLOWED or w < W_MIN_ALLOWED:
            return False
    return True


def plan_trajectory(q_start, q_end, T, dt):
    """
    Plan a smooth joint-space trajectory from q_start to q_end.

    Returns four arrays of equal length (T/dt + 1 rows):
        times (N,), q (N,7) positions, dq (N,7) velocities, ddq (N,7) accelerations.

    Method: quintic polynomial in joint space (zero velocity and zero acceleration at both
    ends, see _quintic), with three safeguards:

      1. Joint limits.  The quintic is monotonic, so every joint stays between its start and
         end value; checking the two end points is enough to guarantee the whole path is
         inside the limits. A goal outside the limits raises ValueError.
      2. Speed / acceleration limits.  If the requested T is too short, T is stretched to
         minimum_time() (limits x SAFETY).
      3. Singularity guard (the "High Reach" issue).  The path is sampled and sigma_min(J) and
         the manipulability w are monitored. If either drops below its threshold the goal is
         backed off along the same joint-space line (bisection) to the farthest point that is
         still safe, and a warning is logged. Because we plan in JOINT space, joint velocities never depend on J^-1,
         so they cannot blow up near a singularity; the guard protects the hand's ability to
         move, not the joint speeds.
    """
    q_start = np.asarray(q_start, dtype=float).reshape(7)
    q_end = np.asarray(q_end, dtype=float).reshape(7)

    # 1. joint limits (small tolerance for rounding)
    for name, qq in (("q_start", q_start), ("q_end", q_end)):
        if np.any(qq < Q_MIN - 1e-9) or np.any(qq > Q_MAX + 1e-9):
            raise ValueError(f"{name} violates the joint limits: {np.round(qq, 3)}")

    # 3. singularity guard (done before timing because it can shorten the move)
    if not _path_is_safe(q_start, q_end):
        lo, hi = 0.0, 1.0                            # fraction of the way to q_end that is safe
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if _path_is_safe(q_start, q_start + mid * (q_end - q_start)):
                lo = mid
            else:
                hi = mid
        log.warning("Path comes too close to a singularity (sigma_min < %.2f or w < %.2f): goal "
                    "backed off to %.0f %% of the requested move.",
                    SIGMA_MIN_ALLOWED, W_MIN_ALLOWED, 100 * lo)
        q_end = q_start + lo * (q_end - q_start)

    # 2. speed / acceleration limits
    t_min = minimum_time(q_start, q_end)
    if T < t_min:
        log.warning("T = %.2f s is too short for the joint limits, using %.2f s.", T, t_min)
        T = t_min

    return _quintic(q_start, q_end, T, dt)


def check_trajectory(q, dq, ddq):
    """Summary numbers used for logging and for the report (all from the planned arrays)."""
    sig = np.array([jacobian_metrics(qq)[0] for qq in q[::max(1, len(q) // 200)]])
    wman = np.array([jacobian_metrics(qq)[1] for qq in q[::max(1, len(q) // 200)]])
    margin = np.minimum(q - Q_MIN, Q_MAX - q).min(axis=0)       # distance to the nearest limit
    return {
        "peak_speed_ratio": (np.abs(dq).max(axis=0) / V_MAX),   # fraction of the speed limit
        "peak_accel_ratio": (np.abs(ddq).max(axis=0) / A_MAX),  # fraction of the accel limit
        "limit_margin_rad": margin,
        "sigma_min": float(sig.min()),
        "manipulability_min": float(wman.min()),
        "hand_start": forward_kinematics(q[0])[:3, 3],
        "hand_end": forward_kinematics(q[-1])[:3, 3],
    }


# ==========================================================
# INFRASTRUCTURE — DO NOT MODIFY BELOW
# ==========================================================

CMD_SIZE = 9
GRIPPER_CTRL = 255.0

# UPDATE THESE BASED ON YOUR ASSIGNED VARIANT
# You must choose reachable configurations within joint limits.
#
# Team 6 - "The High Reach": start in a LOW posture (hand ~0.56 m in front of the base, ~0.22 m
# above the floor, pointing down) and reach UP and OUT to near the top of the workspace (hand
# ~1.0 m high, ~90 % of full reach, elbow still bent so the arm is NOT fully stretched).
# Both poses are inside the joint limits (closest margin 41.8 deg) and collision free in MuJoCo.
Q_START = np.array([0.0, 0.35, 0.0, -2.2, 0.0, 2.55, 0.785])
Q_END   = np.array([0.3, -0.1, 0.0, -0.8, 0.0, 1.6, 0.785])

TRAJ_DURATION = 4.0
# The template value is 0.05 (20 Hz). It is 0.001 here on purpose: in VELOCITY mode the bridge does
# ctrl += command / simulation_frequency ONCE PER MESSAGE (simulation_frequency = 1000), so each
# message must stand for exactly 1 ms. With 0.05 a velocity run only moves 1/50 of the planned
# distance (measured, see the report). POSITION mode works with either value.
TRAJ_DT       = 0.001


class TrajectoryNode(Node):
    def __init__(self):
        super().__init__('trajectory_node')
        self.q_current = np.zeros(7)
        self.dq_current = np.zeros(7)

        self.create_subscription(
            JointState, 'joint_state', self.on_joint_state, 10)
        self.cmd_pub = self.create_publisher(
            Float64MultiArray, 'joint_commands', 10)

        self.t_array = None
        self.q_traj = None
        self.dq_traj = None
        self.ddq_traj = None
        self.step_idx = 0

        # Wait 0.5s before starting to ensure the sim is up
        self.create_timer(0.5, self.start_trajectory_once)

    def on_joint_state(self, msg):
        self.q_current = np.array(msg.position[:7])
        if hasattr(msg, 'velocity') and len(msg.velocity) >= 7:
            self.dq_current = np.array(msg.velocity[:7])

    def start_trajectory_once(self):
        if self.t_array is not None:
            return

        result = plan_trajectory(Q_START, Q_END, TRAJ_DURATION, TRAJ_DT)
        if result is None:
            return

        t_arr, q_arr, dq_arr, ddq_arr = result

        if q_arr.shape[1] != 7 or dq_arr.shape[1] != 7 or ddq_arr.shape[1] != 7:
            self.get_logger().error(
                "plan_trajectory returned wrong shape. Expected (N, 7).")
            return

        self.t_array = t_arr
        self.q_traj = q_arr
        self.dq_traj = dq_arr
        self.ddq_traj = ddq_arr

        self.get_logger().info(
            f"Trajectory generated: {len(t_arr)} samples over "
            f"{TRAJ_DURATION:.1f} s at {1.0/TRAJ_DT:.0f} Hz.")

        self.create_timer(TRAJ_DT, self.tick)

    def tick(self):
        if self.step_idx >= len(self.t_array):
            # Hold final target
            cmd = Float64MultiArray()
            
            # ==================================================
            # DEPENDING ON YOUR BRIDGE MODE, UNCOMMENT ONE BELOW:
            # ==================================================
            
            # -- POSITION MODE (Default) --
            cmd.data = Q_END.tolist() + [GRIPPER_CTRL, 0.0]
            
            # -- VELOCITY MODE --
            # cmd.data = [0.0] * 7 + [GRIPPER_CTRL, 0.0]
            
            self.cmd_pub.publish(cmd)
            return

        cmd = Float64MultiArray()
        
        # ==================================================
        # DEPENDING ON YOUR BRIDGE MODE, UNCOMMENT ONE BELOW:
        # ==================================================
        
        # -- POSITION MODE (Default) --
        q_target = self.q_traj[self.step_idx]
        
        # -- VELOCITY MODE --
        # q_target = self.dq_traj[self.step_idx]
        
        cmd.data = q_target.tolist() + [GRIPPER_CTRL, 0.0]
        self.cmd_pub.publish(cmd)

        self.step_idx += 1


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
