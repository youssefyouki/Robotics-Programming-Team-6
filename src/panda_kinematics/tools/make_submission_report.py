#!/usr/bin/env python3
"""
Builds RPC2_Report_Team6.pdf: the short formal report to hand in together with the code
(DH-based Jacobian, Jacobian derivative, trajectory planning, singularity handling, control-mode
experiments). All numbers come from the code and the recorded runs (see report_common.py).

  source /opt/ros/jazzy/setup.bash && source install/setup.bash
  python3 src/panda_kinematics/tools/make_submission_report.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report_common import *          # noqa: F401,F403

OUT_PDF = os.path.join(HERE, "..", "RPC2_Report_Team6.pdf")

# ---------------------------------------------------------------- title block
story.append(Paragraph("Robotics Programming Challenge 2", ParagraphStyle("t", parent=h1, fontSize=19, leading=24, spaceAfter=2)))
story.append(Paragraph("Linear Jacobian, Jacobian derivative and trajectory planning of the Franka Emika Panda &middot; MCTR911, Winter 2026 &middot; "
                       "<b>Team 6: The High Reach</b>", ParagraphStyle("t2", parent=body, fontSize=10.5, leading=14, textColor=GREY)))
story.append(Paragraph("Name / ID: ______________________________", body))
P("This report documents the Jacobian and its derivative, the trajectory planner, the singularity analysis for the assigned scenario and the POSITION / VELOCITY experiments. "
  "The code is in <i>panda_kinematics/trajectory_node.py</i> (the instructor's template; the EDIT section holds the Challenge 1 code and the three new functions). "
  "Verification is done by <i>tools/check_challenge2.py</i> (22 checks, all pass) and the recorded runs are in <i>challenge2_results/</i>. The robot model is panda.xml from MuJoCo Menagerie, "
  "the same as in Challenge 1; only the seven arm joints are used (the gripper command is fixed).")

# ---------------------------------------------------------------- 1
H1("1. Linear geometric Jacobian")
P("<b>Model.</b> The DH table, the fixed link7&rarr;hand transform and the joint limits are those of Challenge 1 (standard DH, A<sub>i</sub> = Rot<sub>z</sub>(q<sub>i</sub>) Trans<sub>z</sub>(d<sub>i</sub>) Trans<sub>x</sub>(a<sub>i</sub>) Rot<sub>x</sub>(&alpha;<sub>i</sub>)):")
rows = [["i", "&alpha;<sub>i</sub> [rad]", "a<sub>i</sub> [m]", "d<sub>i</sub> [m]", "q<sub>min</sub> .. q<sub>max</sub> [rad]"]]
for i, (al, a, d) in enumerate(T.DH_PARAMS):
    rows.append([str(i + 1), f"{al:+.4f}", f"{a:+.4f}", f"{d:.3f}", f"{T.Q_MIN[i]:+.4f} .. {T.Q_MAX[i]:+.4f}"])
table(rows, [1.0, 3.0, 3.0, 3.0, 6.6])
P("<b>Derivation.</b> A point p fixed to the hand, on a rigid body that rotates about an axis with unit direction z through the point o, moves with velocity z &times; (p &minus; o). "
  "In standard DH, joint i rotates about z<sub>i&minus;1</sub> through the origin o<sub>i&minus;1</sub> of frame i&minus;1, and the velocity of the hand origin is the sum over the joints, so "
  "v = J(q) dq with the column of joint i")
C("J_v,i = z_(i-1) x ( p_hand - o_(i-1) ),     i = 1..7\n"
  "z_(i-1) = 3rd column, o_(i-1) = 4th column of T_0^(i-1) = A1 A2 ... A(i-1)\n"
  "p_hand  = 4th column of T_0^hand")
P("The result is a <b>3 x 7</b> matrix: rows are the hand velocity along the base x, y, z; column i is the hand velocity produced by 1 rad/s of joint i with the other joints frozen. "
  "Seven columns and three rows give a redundancy of four degrees of freedom (null space).")
P(f"<b>Worked columns</b> at the end pose q = {f3(q1)}: p_hand = {f3(p_h)}.")
table([["Joint", "z<sub>i&minus;1</sub>", "o<sub>i&minus;1</sub>", "Column J<sub>v,i</sub> = z &times; (p &minus; o)"],
       ["1", f3(z0), f3(o0), f"(&minus;p<sub>y</sub>, p<sub>x</sub>, 0) = {f3(J1[:, 0])}"],
       ["2", f3(z1), f3(o1), f3(J1[:, 1])],
       ["4", f3(z3), f3(o3), f3(col4)],
       ["7", f3(frames1[6][:3, 2]), f3(frames1[6][:3, 3]), f"{f3(J1[:, 6])} (axis passes through the hand origin)"]], [1.3, 4.3, 4.3, 6.7])
P("<b>Implementation.</b> compute_jacobian(q) chains the DH matrices once (the same frames as the forward kinematics) and evaluates the cross product for each joint.")
C("frames = _frames(q);  p_hand = frames[-1][:3, 3]\n"
  "for i in range(7):  J[:, i] = np.cross(frames[i][:3, 2], p_hand - frames[i][:3, 3])")

# ---------------------------------------------------------------- 2
H1("2. Jacobian derivative")
P("The Jacobian depends on q, so it changes while the arm moves. Its time derivative is what turns joint accelerations into hand acceleration:")
C("v_hand = J dq          a_hand = J ddq + (dJ/dt) dq")
P("The second term is the acceleration the hand has even when the joint accelerations are zero (for one joint at constant speed it is the centripetal term w<super>2</super> r). "
  "dJ/dt is computed analytically by differentiating each column z<sub>i</sub> &times; (p &minus; o<sub>i</sub>) with the product rule:")
C("d col_i/dt = ( dz_i x (p - o_i) ) + ( z_i x (dp - do_i) )\n"
  "w_i = sum_{k<i} z_k dq_k;   dz_i = w_i x z_i;   do_i = sum_{k<i} z_k x (o_i - o_k) dq_k;   dp = J dq")
P("Each axis and origin moves only because the joints <i>before</i> it rotate, which gives the sums above. A central finite-difference version, (J(q + h dq) &minus; J(q &minus; h dq)) / 2h, is kept only as a check.")
table([["Check (tools/check_challenge2.py, 200 random poses in the joint limits)", "Result"],
       ["J vs MuJoCo's own Jacobian (mj_jac at the hand origin)", "max difference 1.3e-15"],
       ["J vs finite differences of the forward kinematics", "max difference 1.8e-10"],
       ["dJ/dt analytic vs finite differences", "max difference 1.9e-10"]], [11.6, 5.0])

# ---------------------------------------------------------------- 3
H1("3. Trajectory planning")
H2("3.1 Scenario: Team 6, \"The High Reach\"")
table([["", "q (rad)", "Hand position (m)", "Distance from shoulder / max reach"],
       ["Start (low)", f3(q0), f3(hand0), f"{100 * reach0:.0f} %"],
       ["End (high reach)", f3(q1), f3(hand1), f"{100 * reach1:.0f} %"]], [2.8, 6.5, 4.0, 3.3])
P(f"The arm starts low (hand {hand0[2]:.2f} m above the floor, pointing down) and reaches up and out to {hand1[2]:.2f} m, close to the top of the workspace (the hand origin reaches at most about 1.19 m). "
  f"The elbow stays bent ({np.degrees(q1[3]):.0f}&deg;), so the arm is not fully stretched. Both poses are inside the joint limits (closest margin "
  f"{np.degrees(info['limit_margin_rad'].min()):.1f}&deg;) and MuJoCo reports no contacts along the path.")
H2("3.2 Method: quintic polynomial in joint space")
C("q(t)   = q_start + s(tau) (q_end - q_start),         tau = t / T\n"
  "s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5\n"
  "dq(t)  = (30 tau^2 - 60 tau^3 + 30 tau^4)/T * (q_end - q_start)\n"
  "ddq(t) = (60 tau - 180 tau^2 + 120 tau^3)/T^2 * (q_end - q_start)")
P("The six coefficients of a fifth-order polynomial are fixed by six conditions: position, velocity and acceleration at both ends (s = 0, 1; s' = 0, 0; s'' = 0, 0). "
  "All joints share the same s(tau), so they start and stop together and the path in joint space is a straight line.")
table([["Moment", "Position", "Velocity", "Acceleration"],
       ["t = 0", "q_start", "0", "0"],
       ["t = T", "q_end", "0", "0"],
       ["t = T/2", "midpoint", "maximum, 15/8 &middot; |dq|/T", "0"],
       ["t = 0.21 T, 0.79 T", "", "", "maximum, 10/&radic;3 &middot; |dq|/T<super>2</super>"]], [3.6, 3.4, 5.0, 4.6])
P(f"For joint 4, which moves most (|dq| = {abs(delta[3]):.2f} rad) with T = {T.TRAJ_DURATION:.0f} s: peak speed {1.875 * abs(delta[3]) / T.TRAJ_DURATION:.3f} rad/s "
  f"({100 * info['peak_speed_ratio'][3]:.0f} % of 2.175 rad/s) and peak acceleration {5.7735 * abs(delta[3]) / T.TRAJ_DURATION ** 2:.3f} rad/s<super>2</super> ({100 * info['peak_accel_ratio'][3]:.0f} % of 12.5). "
  "A cubic would leave a non-zero acceleration at both ends and linear interpolation has an infinite one, so the quintic was chosen.")
fig(fig_traj(), 16.0, "Figure 1: planned motion (positions, speeds, accelerations) of the joints that move. Every curve starts and ends at zero speed and acceleration.")
P("<b>plan_trajectory(q_start, q_end, T, dt)</b> returns four arrays of equal length N = T/dt + 1 (times, positions, velocities, accelerations) and contains three safeguards:")
B([f"<b>Joint limits.</b> The quintic is monotonic, so every joint stays between its start and end value; checking the two end points guarantees the whole path (a goal outside the limits raises an error).",
   f"<b>Speed and acceleration limits.</b> Peaks are known in closed form, so T must be at least (15/8)|dq|/(0.6 v<sub>max</sub>) and sqrt((10/&radic;3)|dq|/(0.6 a<sub>max</sub>)) for every joint "
   f"(60 % of Franka's published limits). A shorter T is stretched automatically. Here the minimum is {t_min:.2f} s and T = {T.TRAJ_DURATION:.0f} s is used.",
   "<b>Singularity guard</b> (section 4)."])

# ---------------------------------------------------------------- 4
H1("4. Singularities near the top of the workspace")
P("At a singularity J loses rank: the hand cannot move in some direction, and the joint speeds needed for a hand velocity v, dq = J<super>+</super>v, grow like 1/&sigma; with the small singular value &sigma;. "
  "Close to the upper boundary of the workspace the arm is almost stretched, so this matters for Team 6. Two indicators of the 3 x 7 Jacobian are monitored along the path:")
table([["Indicator", "Meaning", "Threshold", "Our path (min)", "Arm straight up"],
       ["&sigma;<sub>min</sub>(J)", "hand speed per rad/s of joint speed in the worst direction", f"&ge; {T.SIGMA_MIN_ALLOWED}", f"{info['sigma_min']:.3f}", f"{sig_up:.3f}"],
       ["w = sqrt(det(J J<super>T</super>))", "manipulability: volume of the available hand velocities", f"&ge; {T.W_MIN_ALLOWED}", f"{info['manipulability_min']:.3f}", f"{w_up:.4f}"]], [3.4, 5.2, 2.6, 2.7, 2.7])
fig(fig_sing(), 15.5, "Figure 2: &sigma;<sub>min</sub> and w along our path (solid) and along a path ending with the arm stretched straight up (dashed red).")
P("<b>How the motion is kept safe.</b> (1) The goal is at 90 % of full reach with a bent elbow, away from the stretched configuration (w = "
  f"{w1:.3f}, &sigma;<sub>min</sub> = {sig1:.3f}). (2) Planning is done in joint space, so J is never inverted and the joint speeds are bounded by 15/8 &middot; |dq|/T whatever the pose "
  f"({100 * info['peak_speed_ratio'].max():.0f} % of the speed limit here). (3) T is stretched if a limit would be exceeded. (4) The guard samples the path and, if &sigma;<sub>min</sub> or w falls below its threshold, backs the goal off along the same joint line by bisection and logs a warning; "
  "for a goal with the arm straight up it shortened the move to 88 %. <b>Limits:</b> only the position Jacobian is monitored, and a joint-space plan protects the joint speeds but does not keep the hand path straight; a Cartesian planner would need a damped pseudo-inverse "
  "dq = J<super>T</super>(J J<super>T</super> + &lambda;<super>2</super>I)<super>&minus;1</super>v and slower motion near singularities.")

# ---------------------------------------------------------------- 5
H1("5. POSITION and VELOCITY control modes")
P("<b>Bridge behaviour</b> (mujoco_ros.cpp): in POSITION mode the received numbers are copied to the actuator targets (ctrl = command). In VELOCITY mode the bridge adds command / simulation_frequency to the target "
  "<b>once for every message</b> (simulation_frequency = 1000). Each joint is a position servo, force = k<sub>p</sub>(ctrl &minus; q) &minus; k<sub>v</sub> dq with k<sub>v</sub>/k<sub>p</sub> = 0.1 s for all joints. "
  "The node publishes q_traj[k] (POSITION) or dq_traj[k] (VELOCITY) by switching the two commented places in tick(); the bridge mode was changed to match. All runs used the real C++ bridge drawing at 20 Hz, with a logger recording planned and actual joint angles.")
table([["Set", "Node", "Msgs / s", "Arm starts at", "Purpose"],
       ["A", "trajectory_node.py as submitted", "1000", "all zeros", "following the challenge text"],
       ["B", "same with the template's TRAJ_DT = 0.05", "20", "all zeros", "the template's own rate"],
       ["C", "extended runner (set-up move first)", "1000", "Q_START", "clean tracking comparison"]], [1.0, 6.2, 2.0, 3.0, 4.4])
H2("5.1 Template node (sets A and B)")
fig(fig_template(), 16.2, "Figure 3: left: final joint angles against the goal; right: hand error against the plan (template node).")
table([["Run", "Hand error RMS / max", "Settled hand error", "Commanded motion delivered"],
       ["POSITION, 1 kHz", f"{tP1k['ee_rms']:.0f} / {tP1k['ee_max']:.0f} mm", f"{tP1k['settled_ee']:.1f} mm", "n/a (absolute target)"],
       ["POSITION, 20 Hz", f"{tP20['ee_rms']:.0f} / {tP20['ee_max']:.0f} mm", f"{tP20['settled_ee']:.1f} mm", "n/a (absolute target)"],
       ["VELOCITY, 1 kHz", f"{tV1k['ee_rms']:.0f} / {tV1k['ee_max']:.0f} mm", f"{tV1k['settled_ee']:.0f} mm", f"joints 1, 2: {100 * tV1k['delivered'][0]:.0f} %, {100 * tV1k['delivered'][1]:.0f} %; joints 4, 6: about 0"],
       ["VELOCITY, 20 Hz", f"{tV20['ee_rms']:.0f} / {tV20['ee_max']:.0f} mm", f"{tV20['settled_ee']:.0f} mm", f"{100 * tV20['delivered'][0]:.1f} % (= 1/50)"]], [3.4, 4.2, 3.4, 5.6])
B(["The simulator starts with all joints at 0, but the template does not move the arm to Q_START first. In <b>POSITION</b> mode the arm snaps to Q_START in about 0.7 s (the large first-second error) and then tracks; it settles within "
   f"{np.abs(tP1k['settled_joint']).max():.2f}&deg;.",
   "In <b>VELOCITY</b> mode the bridge's target starts at 0 and only accumulates, so the arm performs the relative motion Q_END &minus; Q_START from zeros and never reaches Q_END. Joints 4 and 6 do not move at all: "
   "their integrated targets (+1.4 rad, &minus;0.95 rad) lie outside the actuator range (joint 4 &lt; &minus;0.07, joint 6 &gt; &minus;0.0175) and are clamped.",
   f"At the template's TRAJ_DT = 0.05 only {100 * tV20['delivered'][0]:.1f} % of the commanded motion arrives (20 messages per second, each adding command/1000). "
   "<b>TRAJ_DT was therefore set to 0.001 (1 kHz)</b>, the only change below the template's marked line besides Q_START / Q_END; setting 0.05 again reproduces the template behaviour."])
H2("5.2 Clean comparison from the same start pose (set C)")
fig(fig_modes(), 16.2, "Figure 4: top: joints 4 and 6, plan (solid) vs actual (dashed); bottom: joint errors over the whole run.")
table([["Measure (main motion unless noted)", "POSITION", "VELOCITY"],
       ["Hand position error, RMS / max", f"{mP['ee_rms']:.1f} / {mP['ee_max']:.1f} mm", f"{mV['ee_rms']:.1f} / {mV['ee_max']:.1f} mm"],
       ["Largest joint error during the motion", f"{mP['max_joint'].max():.1f}&deg;", f"{mV['max_joint'].max():.1f}&deg;"],
       ["Time lag behind the plan (joint 4)", f"{mP['lag']:.0f} ms", "not meaningful"],
       ["Settled joint error vs the goal", f"&lt; {np.abs(mP['settled_joint']).max():.2f}&deg;", f"up to {np.abs(mV['settled_joint']).max():.1f}&deg;"],
       ["Settled hand error vs the goal", f"{mP['settled_ee']:.1f} mm", f"{mV['settled_ee']:.1f} mm"],
       ["Commanded motion delivered", f"{100 * mP['delivered'].mean():.1f} %", f"{100 * mV['delivered'].mean():.1f} %"]], [7.2, 4.6, 4.8])
H2("5.3 Observations")
B([f"<b>Which mode tracks more closely?</b> POSITION. Its error is the servo lag only: about {mP['lag']:.0f} ms (predicted k<sub>v</sub>/k<sub>p</sub> = 100 ms), proportional to joint speed, peaking mid-motion and returning to "
   f"zero when the arm stops; the remaining {mP['settled_ee']:.1f} mm is gravity sag. In VELOCITY mode the hand RMS error looks smaller ({mV['ee_rms']:.1f} vs {mP['ee_rms']:.1f} mm) only because the missing motion happens to cancel part of the lag; "
   f"the joints stop up to {np.abs(mV['settled_joint']).max():.1f}&deg; away from the goal.",
   f"<b>Does VELOCITY mode drift or lag toward the end?</b> It drifts. The bridge integrates the messages it receives with no feedback, and only {100 * mV['delivered'].mean():.1f} % of the commanded motion arrived; every lost message is a permanent offset, "
   "so the error builds up during the motion and is the same after the arm has stopped. A lost message in POSITION mode is harmless because the next one carries the absolute target.",
   f"<b>Why is motion lost?</b> The bridge keeps only the newest message (queue depth 1) and draws frames on the same thread as the 1 kHz physics. Repeating the VELOCITY run with other render rates: "
   + ", ".join(f"{hz} Hz &rarr; {100 * sens[hz]['delivered'].mean():.0f} %" for hz in (5, 20, 30, 60))
   + " of the motion delivered (one run each). The node itself published at "
   f"{min(s['rate'] for s in sens.values()):.0f}&ndash;{max(s['rate'] for s in sens.values()):.0f} Hz, so the loss is inside the bridge."])

# ---------------------------------------------------------------- 6
H1("6. Conclusion and limitations")
B(["The linear Jacobian (3 x 7) and its derivative follow directly from the DH table, agree with MuJoCo and with finite differences to machine precision / 1e-10, and are used to evaluate the singularity indicators along the path.",
   "The quintic joint-space plan starts and stops with zero speed and acceleration, respects position, speed (30 %) and acceleration (4 %) limits, and is protected by a singularity guard; the Team 6 path needs no back-off.",
   "POSITION mode follows the plan with a 0.1 s servo lag and no accumulated error. VELOCITY mode is open loop: with the template's 20 Hz it delivers 2 % of the motion, at 1 kHz about 97 % (less when the bridge draws more often), and from the simulator's start pose it cannot "
   "move joints 4 and 6 because the integrated targets leave the actuator range.",
   "Limitations: only the position Jacobian is monitored; the singularity thresholds are design choices; the results of each configuration come from one run; the start-up transient comes from the simulator's non-physical initial pose (joint 4 at 0, hand overlapping link 5)."])

H2("How to run")
C("colcon build --packages-select panda_kinematics && source install/setup.bash\n"
  "ros2 launch mujoco_ros2 example.py            # bridge, control_mode POSITION or VELOCITY\n"
  "ros2 run panda_kinematics trajectory_node     # the two mode lines in tick() must match the bridge\n"
  "python3 src/panda_kinematics/tools/check_challenge2.py     # 22 self-checks (needs mujoco python)")

# ---------------------------------------------------------------- build
def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("DV", 8)
    canvas.setFillColor(GREY)
    canvas.drawString(2 * cm, 1.1 * cm, "MCTR911 - Robotics Programming Challenge 2 - Panda Jacobian and trajectory planning - Team 6")
    canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"{doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(OUT_PDF, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.7 * cm, bottomMargin=1.8 * cm,
                        title="RPC 2 - Panda Jacobian and trajectory planning (Team 6)", author="Team 6")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", os.path.abspath(OUT_PDF))
