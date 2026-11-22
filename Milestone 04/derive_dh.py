#!/usr/bin/env python3
"""
Derivation and verification of the standard Denavit-Hartenberg model of the Koch v1.1 arm from the simulator's
kinematic chain and verify it against the simulator model.

Frames follow the standard (distal-free) DH convention  A_i = Rz(theta_i) Tz(d_i) Tx(a_i) Rx(alpha_i):
  z_{i-1}  = axis of joint i          (sign included, so q_i > 0 rotates exactly like the simulator joint)
  x_i      = common normal from z_{i-1} to z_i  (z_{i-1} x z_i when the axes intersect)
  O_i      = foot of that normal on z_i
The real arm has offset links (the upper arm and forearm are not straight, the shoulder is offset
sideways), so the table has a joint offset theta0_i for every joint and the chain is closed by a
constant base frame and a constant tool frame:

      T_ee(q) = T_base  A_1(q1 + th0_1) A_2(q2 + th0_2) ... A_5(q5 + th0_5)  T_tool

Run:  .venv/bin/python "Milestone 04/derive_dh.py"
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "milestone_03"))
from milestone_03 import kinematics as K


def closest_points(p1, u1, p2, u2):
    """Closest points of two skew lines p1 + t u1, p2 + s u2."""
    w0 = p1 - p2
    a, b, c, d, e = u1 @ u1, u1 @ u2, u2 @ u2, u1 @ w0, u2 @ w0
    den = a * c - b * b
    t = (b * e - c * d) / den
    s = (a * e - b * d) / den
    return p1 + t * u1, p2 + s * u2


def frame(x, z, origin):
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, origin
    return T


def dh_matrix(theta, d, a, alpha):
    ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
    return np.array([[ct, -st * ca,  st * sa, a * ct],
                     [st,  ct * ca, -ct * sa, a * st],
                     [0.0,     sa,       ca,      d],
                     [0.0,    0.0,      0.0,    1.0]])


def derive():
    origins, axes, p_ee, _ = K._geometry(np.zeros(5))
    T_ee0 = K.forward_kinematics(np.zeros(5))
    z = [np.asarray(a, float) for a in axes]                    # z_{i-1} = axis of joint i

    # frames F_0 ... F_4 (z_i = axis of joint i+1)
    O = [np.array([origins[0][0], origins[0][1], 0.0])]        # base frame origin on the first axis, at the ground
    X = [None]
    for i in range(1, 5):
        cross = np.cross(z[i - 1], z[i])
        if np.linalg.norm(cross) > 1e-9:                        # axes not parallel
            P, Q = closest_points(O[i - 1], z[i - 1], origins[i], z[i])
            if np.linalg.norm(Q - P) > 1e-12:
                x = (Q - P) / np.linalg.norm(Q - P)
            else:                                               # axes intersect: normal along z_{i-1} x z_i
                x = cross / np.linalg.norm(cross)
            O.append(Q)
        else:                                                   # parallel axes: shortest segment from z_{i-1}
            v = origins[i] - O[i - 1]
            vp = v - (v @ z[i - 1]) * z[i - 1]
            x = vp / np.linalg.norm(vp)
            O.append(O[i - 1] + vp)
        X.append(x)
    X[0] = X[1] - (X[1] @ z[0]) * z[0]
    X[0] /= np.linalg.norm(X[0])
    F = [frame(X[i], z[i], O[i]) for i in range(5)]
    # frame 5: on the roll axis at the wrist_roll origin, z_5 = z_4, x_5 = x_4
    F.append(frame(X[4], z[4], p_ee))

    rows = []
    for i in range(1, 6):
        T = np.linalg.inv(F[i - 1]) @ F[i]
        theta0 = np.arctan2(T[1, 0], T[0, 0])
        alpha = np.arctan2(T[2, 1], T[2, 2])
        d = T[2, 3]
        a = T[0, 3] * np.cos(theta0) + T[1, 3] * np.sin(theta0)
        assert np.allclose(dh_matrix(theta0, d, a, alpha), T, atol=1e-12), f"link {i} is not of DH form"
        rows.append((theta0, d, a, alpha))
    T_base = F[0]
    T_tool = np.linalg.inv(F[5]) @ T_ee0
    return rows, T_base, T_tool


def dh_forward(q, rows, T_base, T_tool):
    T = T_base.copy()
    for qi, (th0, d, a, al) in zip(q, rows):
        T = T @ dh_matrix(qi + th0, d, a, al)
    return T @ T_tool


if __name__ == "__main__":
    rows, T_base, T_tool = derive()
    np.set_printoptions(precision=6, suppress=True)
    print("Standard DH table of the Koch v1.1 arm  (theta_i = q_i + theta0_i)\n")
    print(" i | theta0_i [deg] |   d_i [m]  |   a_i [m]  | alpha_i [deg]")
    for i, (th0, d, a, al) in enumerate(rows, 1):
        print(f" {i} | {np.degrees(th0):14.4f} | {d:10.5f} | {a:10.5f} | {np.degrees(al):13.4f}")
    print("\nT_base (frame 0 in the world):\n", T_base)
    print("\nT_tool (end-effector frame in frame 5):\n", T_tool)

    rng = np.random.default_rng(0)
    worst_p = worst_R = 0.0
    for _ in range(2000):
        q = rng.uniform(K.JOINT_LIMITS[:, 0], K.JOINT_LIMITS[:, 1])
        A, B = dh_forward(q, rows, T_base, T_tool), K.forward_kinematics(q)
        worst_p = max(worst_p, np.linalg.norm(A[:3, 3] - B[:3, 3]))
        worst_R = max(worst_R, np.abs(A[:3, :3] - B[:3, :3]).max())
    print(f"\nverification against the simulator chain over 2000 random poses: "
          f"max position error {worst_p:.2e} m, max rotation error {worst_R:.2e}")
    print("DH model reproduces the simulator" if worst_p < 1e-12 and worst_R < 1e-12 else "MISMATCH")
