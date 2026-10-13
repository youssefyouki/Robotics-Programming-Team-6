#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np
import time

"""
In the setup.py, change the entry point to include this file's name without the brackets

    entry_points={
        'console_scripts': [
            'kinematics = panda_kinematics.kinematics:main',
        ],
    },
"""
# ==========================================================
# KINEMATICS IMPLEMENTATION — EDIT ONLY THIS SECTION
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

# Joint limits for joints 1 through 7, in radians.
# numpy arrays of shape (7,).
#
# Joints without their own range use the class default range="-2.8973 2.8973".
Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_MAX = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])

# The target pose the node will try to reach.
# 4x4 numpy array in the 'link0' frame.
#
# Generated from a joint configuration inside the limits that is
# collision-free and well above the floor (checked in MuJoCo).
# It is assigned at the end of this section, once forward_kinematics exists.
Q_TARGET = np.array([0.4, 0.2, -0.3, -2.0, 0.3, 2.2, 0.3])
TARGET_POSE = None


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


TARGET_POSE = forward_kinematics(Q_TARGET)


# =================================================================
# DO NOT MODIFY BELOW (OR MODIFY, IF YOU KNOW WHAT YOU ARE DOING)
# =================================================================

CMD_SIZE = 9
GRIPPER_CTRL = 255.0


class PandaKinematicsNode(Node):
    def __init__(self):
        super().__init__('panda_kinematics')
        self.q_current = np.zeros(7)

        self.create_subscription(
            JointState, 'joint_state', self.on_joint_state, 10)
        self.cmd_pub = self.create_publisher(
            Float64MultiArray, 'joint_commands', 10)

        self.create_timer(0.05, self.tick)

        if TARGET_POSE is not None:
            self.get_logger().info(
                f"Target pose:\n{np.round(TARGET_POSE, 4)}")
        else:
            self.get_logger().warn(
                "TARGET_POSE is None — set it before running.")

    def on_joint_state(self, msg):
        self.q_current = np.array(msg.position[:7])

    def tick(self):
        if TARGET_POSE is None:
            return

        result = inverse_kinematics(TARGET_POSE, self.q_current)
        q_sol, ok = result

        if not ok:
            self.get_logger().warn(
                "IK did not converge.",
                throttle_duration_sec=2.0)

        cmd = Float64MultiArray()
        cmd.data = q_sol.tolist() + [GRIPPER_CTRL, 0.0]
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = PandaKinematicsNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
