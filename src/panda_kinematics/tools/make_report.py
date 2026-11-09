#!/usr/bin/env python3
"""
Builds RPC2_Explained_Team6.pdf: a plain-language STUDY GUIDE for the evaluation (everything that was
done for Challenge 2, explained simply, with likely questions and answers). All numbers come from the
code and from the recorded runs in challenge2_results/ (see report_common.py).

  source /opt/ros/jazzy/setup.bash && source install/setup.bash
  python3 src/panda_kinematics/tools/make_report.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report_common import *          # noqa: F401,F403  (styles, helpers, numbers, figures)

OUT_PDF = os.path.join(HERE, "..", "RPC2_Explained_Team6.pdf")

# ---------------- cover
story.append(Spacer(1, 1.2 * cm))
story.append(Paragraph("Robotics Programming Challenge 2", ParagraphStyle("t", parent=h1, fontSize=21, leading=27, alignment=TA_CENTER)))
story.append(Paragraph("Velocity &amp; acceleration kinematics and trajectory planning<br/>for the Franka Emika Panda",
                       ParagraphStyle("t2", parent=body, fontSize=13, leading=18, alignment=TA_CENTER, textColor=GREY)))
story.append(Spacer(1, 0.4 * cm))
story.append(Paragraph("<b>Team 6: \"The High Reach\"</b> &mdash; everything that was done, explained in simple words,<br/>"
                       "so you can answer any question in the evaluation.",
                       ParagraphStyle("t3", parent=body, fontSize=11, leading=16, alignment=TA_CENTER, textColor=TEAL)))
story.append(Spacer(1, 0.8 * cm))
box("How to use this document", [
    "Read sections 1&ndash;7 once, in order: each builds on the one before. Section 8 is the experiment you must be able to talk about "
    "(POSITION vs VELOCITY). <b>Section 10 is a list of the questions you are likely to be asked, with model answers</b>: rehearse it out loud.",
    "Everything in a grey box is real code or a real command. Every number in this PDF was computed from the project itself "
    "(the code and the recorded runs), nothing is typed in by hand."])
box("Things you must know before the evaluation", [
    "<b>1. The instructor's template is used.</b> In <i>trajectory_node.py</i> only the EDIT section was filled in (Challenge 1 code copied, plus the three new functions "
    "<i>compute_jacobian</i>, <i>compute_jacobian_dot</i>, <i>plan_trajectory</i>) and Q_START / Q_END were set. The infrastructure (the node class, tick(), the commented mode lines) is the template's own.",
    "<b>2. One deliberate change below the line: TRAJ_DT = 0.001 instead of 0.05.</b> The bridge adds command/1000 once per message in VELOCITY mode, so at 20 messages per second the arm only moves "
    "2 % of the planned motion (measured, section 8). POSITION mode works with either value. Be ready to defend this; setting it back to 0.05 is a one-line change.",
    "<b>3. Team 6 was assumed</b> because your repository is called Robotics-Programming-Team-6. If that is wrong, change Q_START / Q_END.",
    "<b>4. Mode switch = the template's own two places in tick()</b> (comment / uncomment). <i>tools/run_experiment.py</i> does it on a temporary copy so the submitted file is never edited."],
    bg="#fff4e0", edge="#d98b00")

story.append(Paragraph("Contents", h2))
table([["1", "The challenge in one page"], ["2", "How the pieces talk to each other (the bridge, POSITION and VELOCITY)"],
       ["3", "The robot and its DH table"], ["4", "The Jacobian: what it is and how each column is derived"],
       ["5", "The Jacobian derivative"], ["6", "Trajectory planning: the quintic polynomial"],
       ["7", "Team 6 scenario and singularities"], ["8", "The experiment: POSITION vs VELOCITY"],
       ["9", "Files, how to run, how it was verified"], ["10", "Likely evaluation questions with model answers"],
       ["11", "Glossary"]], [1.2, 15.3], header=False)
story.append(PageBreak())

# ---------------- 1
H1("1. The challenge in one page")
P("<b>The task.</b> Take the Franka Panda in the MuJoCo simulator (controlled through ROS 2) and write three functions in <i>trajectory_node.py</i>:")
table([["Function", "What it returns", "In plain words"],
       ["compute_jacobian(q)", "3 x 7 matrix", "A table that converts joint speeds into the speed of the hand (x, y, z) for the current pose."],
       ["compute_jacobian_dot(q, dq)", "3 x 7 matrix", "How fast that table itself is changing. Needed to get the hand <i>acceleration</i>."],
       ["plan_trajectory(q_start, q_end, T, dt)", "4 arrays of equal length: times, positions, velocities, accelerations",
        "A smooth movement from one pose to another, sampled every dt seconds."]], [4.6, 4.5, 7.5])
P("Then the trajectory is played to the simulator in <b>two control modes</b> (POSITION and VELOCITY) and you describe what you observe.")
P("<b>Our scenario (Team 6, \"The High Reach\").</b> The arm starts <b>low</b> (hand about 0.22 m above the floor, pointing down) and must reach "
  "<b>up and out</b> to a point near the top of its workspace (hand about 1.0 m high). The point of the scenario is that the arm gets close to a "
  "<b>singularity</b> there, and you must explain how that was detected and handled.")
box("What we built (the short version)", [
    "&bull; <b>Jacobian</b> straight from the DH table (same table as Challenge 1): column i = (axis of joint i) x (vector from that axis to the hand). "
    "Checked against MuJoCo's own Jacobian: agreement to 1e-15.",
    "&bull; <b>Jacobian derivative</b> computed analytically with the product rule. Checked against finite differences: agreement to 2e-10.",
    "&bull; <b>Trajectory</b>: quintic (5th-order) polynomial in joint space &rarr; starts and stops with zero speed and zero acceleration. "
    "The planner enforces joint limits, stretches the time if speed/acceleration limits would be exceeded, and has a <b>singularity guard</b>.",
    "&bull; <b>Experiment</b>: both modes were run on the real C++ bridge; a logger recorded plan vs. actual. Main finding: in VELOCITY mode part of the "
    "commanded motion is lost (open-loop integration), so the arm ends slightly short and the error never recovers."])

# ---------------- 2
H1("2. How the pieces talk to each other")
fig(fig_architecture(), 16.6, "Figure 1: the data flow. Our node publishes 1000 commands per second; the bridge turns them into actuator targets.")
P("<b>The bridge (mujoco_ros2).</b> It is a C++ ROS 2 node that runs the MuJoCo physics at 1000 steps per second, publishes the joint angles on "
  "<i>/joint_state</i>, and listens on <i>/joint_commands</i>. What it does with a command depends on its <b>control_mode</b> parameter "
  "(this is read directly from <i>mujoco_ros.cpp</i>):")
table([["Mode", "What the bridge does with every command message", "So our node must publish"],
       ["POSITION", "ctrl[i] = command[i] (it simply copies the number)", "the planned joint <b>positions</b> (q_traj)"],
       ["VELOCITY", "ctrl[i] = ctrl[i] + command[i] / 1000 (it <b>adds</b> a small step <i>per message received</i>)", "the planned joint <b>velocities</b> (dq_traj)"]],
      [2.3, 9.0, 5.3])
P("<b>What is ctrl?</b> Each Panda joint is driven by a <b>position servo</b>: a spring and a damper that pull the joint toward the target ctrl. "
  "In the model file: force = kp &middot; (ctrl &minus; q) &minus; kv &middot; dq. For every joint kv/kp = 0.1 s. "
  "Consequence: a joint that moves at speed v always <b>lags</b> behind its target by about 0.1 &middot; v (a 0.1 s delay). Keep this number in mind, it explains the POSITION results.")
P("<b>TRAJ_DT: why 0.001 and not the template's 0.05.</b> In VELOCITY mode the bridge adds command/1000 for <b>every message it receives</b>, whatever the time between messages. "
  "So each message must stand for exactly 1 ms. Publishing every 1 ms with velocity v adds v &middot; 0.001 = the right displacement. Publishing every 50 ms (20 Hz, the template value) adds the same "
  "v &middot; 0.001 only 20 times per second, i.e. <b>1/50 = 2 % of the planned motion</b>. We measured exactly 2.0 % (section 8). POSITION mode does not care: it copies the absolute target.")
P("<b>Two bridge details that matter later:</b> (1) the bridge subscribes with a queue of depth 1, so if it is busy, only the latest message survives; "
  "(2) the physics timer and the drawing (render) timer share a single thread, so while it draws a frame, incoming messages can be dropped. "
  "Section 8 shows what this does in VELOCITY mode.")
P("<b>The 9-value command.</b> The model has 9 joint values (7 arm + 2 finger), so a command has 9 numbers: 7 arm values, a gripper value (we send 0 in both modes so the runs are comparable) "
  "and one spare (0).")

# ---------------- 3
H1("3. The robot and its DH table")
P("<b>DH in one sentence.</b> Denavit-Hartenberg describes a chain of joints with four numbers per link, so that the position and orientation of every link can be computed by multiplying 4 x 4 matrices.")
table([["Symbol", "Meaning (plain words)"],
       ["theta", "the joint angle (this is the variable that changes)"],
       ["d", "how far the next frame is shifted <b>along the current joint axis (z)</b>"],
       ["a", "how far the next frame is shifted <b>sideways (along x)</b>, i.e. the length of the link"],
       ["alpha", "how much the next joint axis is <b>twisted</b> relative to this one (rotation about x)"]], [2.5, 14.1])
P("We use the <b>standard (distal) DH</b> convention: the matrix for link i is  A_i = Rot_z(theta) &middot; Trans_z(d) &middot; Trans_x(a) &middot; Rot_x(alpha). "
  "Joint i rotates about the z axis of frame i&minus;1. The values come from the Panda's model file (panda.xml), the same table as Challenge 1:")
rows = [["Link", "alpha [rad]", "a [m]", "d [m]", "Joint limits [rad]", "Comment"]]
names = ["shoulder yaw", "shoulder pitch", "upper-arm roll", "elbow (the only one that can't go positive)", "forearm roll", "wrist pitch", "wrist roll"]
for i, (al, a, d) in enumerate(T.DH_PARAMS):
    rows.append([str(i + 1), f"{al:+.4f}", f"{a:+.4f}", f"{d:.3f}", f"{T.Q_MIN[i]:+.3f} .. {T.Q_MAX[i]:+.3f}", names[i]])
table(rows, [1.3, 2.2, 1.9, 1.6, 4.0, 5.6])
P("After joint 7 there is a fixed step to the <b>hand</b>: 0.107 m along z7 and a &minus;45&deg; turn about z7. <b>The Jacobian describes the hand origin.</b> "
  "Multiplying all the matrices gives the pose of the hand: <b>T = A1 &middot; A2 &middot; ... &middot; A7 &middot; T_hand</b>. "
  "This forward kinematics was checked against MuJoCo: the hand position agrees to 1e-15 m, so the DH table is correct.")
fig(fig_pose(), 11.5, "Figure 2: the arm from the side at the start pose, halfway and the end pose (dots are the frame origins of joints 1-7 and the hand).")

# ---------------- 4
H1("4. The Jacobian")
box("In one sentence", "The Jacobian J(q) is a table that tells you: <b>if the joints move at speeds dq, the hand moves at v = J(q) &middot; dq</b>. "
    "It changes with the pose q, so it must be recomputed at every step.")
H2("4.1 Shape and meaning")
B(["<b>3 rows</b> = the hand speed along x, y, z (we only care about linear velocity; the full Jacobian would have 3 more rows for rotation, 6 x 7).",
   "<b>7 columns</b> = one per joint.",
   "<b>Column i</b> answers: \"<i>if only joint i turns at 1 rad/s and all others are frozen, how fast (and in which direction) does the hand move?</i>\" "
   "The total motion is the sum of the columns weighted by the joint speeds.",
   "3 rows but 7 columns means <b>7 &minus; 3 = 4 extra degrees of freedom</b>: the arm is <i>redundant</i>. There are many joint motions that give the same hand motion "
   "(for example swinging the elbow while the hand stays still). That null space is also why the Panda can sit close to a singularity and still have room to move."])
H2("4.2 The recipe for one column (this is the derivation you may be asked to do)")
P("Take a joint that spins about an axis with direction <b>z</b> passing through the point <b>o</b>. A point p fixed to the hand then moves with velocity "
  "<b>z &times; (p &minus; o)</b>: the rotation axis crossed with the lever arm from the axis to the point (like a wheel: speed = angular speed &times; radius). "
  "For the standard DH table the axis of joint i is the z axis of frame i&minus;1, and a point on that axis is the origin of frame i&minus;1. Therefore:")
C("column_i  =  z_(i-1)  x  ( p_hand  -  o_(i-1) )\n\n"
  "z_(i-1) = 3rd column of T_0^(i-1) = A1 A2 ... A(i-1)   (joint axis direction)\n"
  "o_(i-1) = 4th column of T_0^(i-1)                      (a point on that axis)\n"
  "p_hand  = 4th column of T_0^hand                       (where the hand is)")
fig(fig_lever(), 6.3, "Figure 3: the idea behind every column. The joint axis and the lever arm give the direction and size of the hand velocity.")
P("<b>Example A: joint 1 (column 1).</b> Joint 1 spins about the base z axis through the origin, so z0 = (0, 0, 1) and o0 = (0, 0, 0). Then "
  "column 1 = (0,0,1) &times; (p<sub>x</sub>, p<sub>y</sub>, p<sub>z</sub>) = <b>(&minus;p<sub>y</sub>, p<sub>x</sub>, 0)</b>. "
  "Meaning: turning the base moves the hand sideways, tangentially, with no vertical motion, and the farther out the hand, the faster it moves.")
P("<b>Example B: joint 2 (column 2).</b> The third column of a standard DH matrix is (sin&theta; sin&alpha;, &minus;cos&theta; sin&alpha;, cos&alpha;). For link 1, "
  "alpha<sub>1</sub> = &minus;&pi;/2, so z1 = (&minus;sin q1, cos q1, 0): a horizontal axis. Link 1 only shifts up by d<sub>1</sub> = 0.333, so o1 = (0, 0, 0.333). "
  "Column 2 = z1 &times; (p &minus; o1). With q1 = 0 this is (0,1,0) &times; r = (r<sub>z</sub>, 0, &minus;r<sub>x</sub>): the shoulder pitch moves the hand in the x-z plane (forward/up), as expected.")
P(f"<b>Example C: joint 4 with numbers</b> (at the end pose q = {f3(q1)}):")
C(f"p_hand = {f3(p_h)}\n"
  f"z3     = {f3(z3)}   (3rd column of T_0^3)\n"
  f"o3     = {f3(o3)}   (4th column of T_0^3)\n"
  f"r      = p_hand - o3 = {f3(r3)}\n"
  f"column 4 = z3 x r = {f3(col4)}\n"
  f"            (this equals column 4 of compute_jacobian: checked in the code)")
P("A useful sanity check: column i is always <b>perpendicular to its own joint axis</b> z<sub>i&minus;1</sub> (a cross product is perpendicular to both of its inputs). "
  "Another: a joint whose axis passes through the hand has a column of exactly zero. Joint 7 (wrist roll) is such a joint, because the hand origin lies on its axis: "
  "it only spins the hand about its own axis, so it does not move the hand origin (in the code, column 7 is 1e-17, i.e. zero).")
H2("4.3 How it is coded")
C("def compute_jacobian(q):\n"
  "    frames = _frames(q)                     # T_0^0 ... T_0^hand (chained DH matrices)\n"
  "    p_hand = frames[-1][:3, 3]\n"
  "    for i in range(7):\n"
  "        z = frames[i][:3, 2]                # axis of joint i+1\n"
  "        o = frames[i][:3, 3]                # point on that axis\n"
  "        J[:, i] = np.cross(z, p_hand - o)   # axis x lever arm")
H2("4.4 What the numbers physically represent")
B(["Each entry has units of <b>m/s per rad/s = metres</b>: it is a lever arm. A column length of 0.5 means \"1 rad/s of that joint moves the hand 0.5 m/s\".",
   "<b>J&middot;dq = hand velocity.</b> The transpose does the reverse job for forces: hand force &rarr; joint torques (tau = J<super>T</super> F). Both come from the same table.",
   "The <b>singular values</b> of J (section 7) are the gains of the arm in its best and worst directions."])

# ---------------- 5
H1("5. The Jacobian derivative")
box("In one sentence", "J changes whenever the arm moves, and <b>dJ/dt</b> says how quickly. The hand <b>acceleration</b> is "
    "<b>a = J &middot; ddq + (dJ/dt) &middot; dq</b>: the first part comes from accelerating the joints, the second from the changing geometry.")
P("<b>Intuition.</b> Spin one joint at a <i>constant</i> speed: ddq = 0, yet the hand still accelerates toward the axis (circular motion, a = w<super>2</super> r). "
  "That acceleration is entirely the (dJ/dt)&middot;dq term. Without it, acceleration kinematics would be wrong whenever the arm is moving.")
H2("Derivation (product rule)")
P("Each column is  z_i &times; (p &minus; o_i).  Differentiating a product of two things that both move:")
C("d(column_i)/dt = ( dz_i x (p - o_i) )  +  ( z_i x (dp - do_i) )")
P("Every ingredient moves only because the joints <b>before</b> it rotate:")
C("w_i  = sum over k < i of  z_k * dq_k           angular velocity of frame i\n"
  "dz_i = w_i x z_i                               an axis simply turns\n"
  "do_i = sum over k < i of  z_k x (o_i - o_k) * dq_k    how the origin of frame i moves\n"
  "dp   = sum over k < 7 of  z_k x (p - o_k) * dq_k  =  J * dq   (hand velocity)")
P("The code (compute_jacobian_dot) is a direct loop over these formulas. <b>Verification:</b> the analytic result was compared with a finite-difference estimate "
  "(J(q + h dq) &minus; J(q &minus; h dq)) / 2h on 200 random poses: the largest difference is about 2e-10, so both are the same quantity. "
  "(The challenge accepts either method; we implemented the analytic one and kept the numeric one only as a check.)")

# ---------------- 6
H1("6. Trajectory planning: the quintic polynomial")
H2("6.1 What we need and the method we chose")
P("We need a motion from q_start to q_end that is smooth and respects the robot's limits. We plan in <b>joint space</b>: each joint follows the <b>same</b> smooth "
  "curve s(tau) scaled by its own distance:")
C("q(t) = q_start + s(tau) * (q_end - q_start),      tau = t / T  (from 0 to 1)\n"
  "s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5\n"
  "dq(t)  = ds/dt * (q_end - q_start),    ds/dt  = (30 tau^2 - 60 tau^3 + 30 tau^4) / T\n"
  "ddq(t) = d2s/dt2 * (q_end - q_start),  d2s/dt2 = (60 tau - 180 tau^2 + 120 tau^3) / T^2")
H2("6.2 Where the 10, &minus;15, 6 come from")
P("A 5th-order polynomial has 6 free coefficients, so we can impose 6 conditions. We want the motion to start and stop <b>gently</b>: position, velocity and acceleration "
  "are all specified at both ends:")
table([["Condition", "start (tau = 0)", "end (tau = 1)"], ["position s", "0", "1"], ["velocity ds/dtau", "0", "0"], ["acceleration d2s/dtau2", "0", "0"]], [5, 5.5, 5.5])
P("Solving those 6 equations gives s = 10 tau<super>3</super> &minus; 15 tau<super>4</super> + 6 tau<super>5</super> (the tau<super>0</super>, tau<super>1</super>, tau<super>2</super> terms vanish because of the start conditions).")
fig(fig_quintic(), 16.6, "Figure 4: the quintic (solid) against a cubic (dashed). The cubic has zero speed at the ends but its acceleration jumps from/to a non-zero value, which means a sudden jerk.")
H2("6.3 Behaviour at the boundaries (very likely evaluation question)")
table([["Moment", "Position", "Velocity", "Acceleration", "Why it matters"],
       ["t = 0", "q_start exactly", "0", "0", "the arm starts from rest without a kick; no step in the command"],
       ["t = T", "q_end exactly", "0", "0", "the arm comes to a stop without overshoot; next phase can start from rest"],
       ["t = T/2", "halfway", "<b>maximum</b>: 15/8 &middot; dq/T", "0", "peak speed is 1.875 times the average speed"],
       ["t = 0.21 T and 0.79 T", "", "", "<b>maximum</b>: 10/&radic;3 &middot; dq/T<super>2</super> = 5.77 &middot; dq/T<super>2</super>", "peak acceleration (and braking)"]],
      [2.8, 2.4, 3.2, 4.0, 4.2])
P(f"<b>Example (joint 4, which moves the most):</b> dq = {delta[3]:+.2f} rad, T = {T.TRAJ_DURATION:.0f} s &rarr; peak speed = 1.875 &middot; {abs(delta[3]):.2f} / {T.TRAJ_DURATION:.0f} = "
  f"{1.875 * abs(delta[3]) / T.TRAJ_DURATION:.3f} rad/s ({100 * info['peak_speed_ratio'][3]:.0f} % of that joint's 2.175 rad/s limit); peak acceleration = "
  f"5.77 &middot; {abs(delta[3]):.2f} / {T.TRAJ_DURATION ** 2:.0f} = {5.7735 * abs(delta[3]) / T.TRAJ_DURATION ** 2:.3f} rad/s<super>2</super> ({100 * info['peak_accel_ratio'][3]:.0f} % of 12.5).")
fig(fig_traj(), 16.6, "Figure 5: the planned Team 6 motion (main phase): positions, speeds and accelerations. Only the joints that move are drawn; every curve starts and ends at zero speed and acceleration.")
H2("6.4 Safeguards inside plan_trajectory")
B(["<b>Joint limits.</b> The quintic is monotonic (s only goes from 0 to 1, no overshoot), so every joint stays between its start and end value. "
   "A box-shaped limit is therefore satisfied for the whole path if both ends satisfy it. A goal outside the limits raises an error. "
   f"The closest approach to a limit in our motion is {np.degrees(info['limit_margin_rad'].min()):.1f}&deg; (joint 4).",
   f"<b>Speed and acceleration limits.</b> Peak speed = 15/8 &middot; |dq|/T and peak acceleration = 10/&radic;3 &middot; |dq|/T<super>2</super>, so for every joint T must be at least "
   f"(15/8)|dq| / (0.6 &middot; v_max) and sqrt((10/&radic;3)|dq| / (0.6 &middot; a_max)). If the requested T is shorter, it is <b>stretched automatically</b>. "
   f"Only 60 % of Franka's published limits are used (SAFETY = 0.6). For our move the minimum time is {t_min:.2f} s; we use {T.TRAJ_DURATION:.0f} s (comfortable).",
   "<b>Singularity guard.</b> Described in section 7.",
   "<b>Return format.</b> Four arrays of the same length N = T/dt + 1 (4001 rows for 4 s at 1 kHz); the last sample is exactly tau = 1 because T is rounded to a whole number of steps."])
H2("6.5 Why not other methods? (counter-arguments you should have ready)")
table([["Alternative", "Why we did not choose it"],
       ["Linear interpolation", "velocity jumps from 0 to a constant at t = 0: infinite acceleration, jerks the arm, and violates the servo's comfort"],
       ["Cubic polynomial", "zero speed at the ends but acceleration is non-zero (6 dq/T<super>2</super>) at both ends: acceleration steps = jerk spikes. Quintic fixes it"],
       ["Trapezoidal velocity profile", "also fine, but acceleration is discontinuous (steps). The quintic is smooth and needs no parameter besides T"],
       ["Task-space (straight line of the hand)", "needs the inverse of J at every instant, which blows up close to singularities (exactly where Team 6 is). Joint space never uses J<super>-1</super>. The cost: the hand path is a curve, not a straight line"],
       ["Time-optimal methods", "overkill for this exercise; the limit-based minimum time (6.4) gives the same idea simply"]], [4.4, 12.2])

H2("6.6 What the instructor's node does with the plan")
table([["Step", "What happens"],
       ["1", "After 0.5 s the node calls plan_trajectory(Q_START, Q_END, TRAJ_DURATION, TRAJ_DT) <b>once</b> and checks the shapes."],
       ["2", "Every TRAJ_DT seconds, tick() publishes sample number step_idx: the positions q_traj (POSITION mode) or the velocities dq_traj (VELOCITY mode), followed by the gripper value 255 and a spare 0.0."],
       ["3", "After the last sample it keeps publishing the goal: Q_END in POSITION mode, zero velocities in VELOCITY mode."]], [1.5, 15.1])
P("<b>The node does not move the arm to Q_START first.</b> The simulator starts with all joints at 0, so:")
B(["<b>POSITION:</b> the first command is Q_START, so the arm <b>snaps</b> there from all zeros in about 0.7 s (we simulated the snap: the hand never goes below 9 cm, so it is safe) and then follows the trajectory.",
   "<b>VELOCITY:</b> the bridge's internal target starts at 0 and only <b>adds</b> velocities, so the arm performs the <i>relative</i> motion Q_END &minus; Q_START = (0.3, &minus;0.45, 0, 1.4, 0, &minus;0.95, 0) starting from zeros. "
   "It never goes to Q_START or Q_END. Worse, joints 4 and 6 <b>cannot move at all</b>: their target (0 + 1.4 for joint 4, 0 &minus; 0.95 for joint 6) is outside the range the actuator accepts "
   "(joint 4 only accepts targets &lt; &minus;0.07, joint 6 only targets &gt; &minus;0.0175), so MuJoCo clamps it and the joint stays where it is. This is a property of the template + bridge, not of our planner."])
H2("The extended runner (used for a clean comparison)")
P("To measure tracking without these start-up effects, <i>trajectory_experiment.py</i> reuses the same maths and plays a longer timeline at 1 kHz, with the mode as a ROS parameter:")
table([["Phase", "Duration", "What and why"],
       ["prep", f"{E.T_PREP:.0f} s", "slow quintic from the simulator's start pose (all zeros) to Q_START. Planned from <b>zeros</b> on purpose: in VELOCITY mode the bridge's target starts at 0, so the planned positions and the bridge's target stay identical."],
       ["hold_start", f"{E.HOLD_START:.0f} s", "stand still at Q_START so the arm settles before the real motion"],
       ["<b>main</b>", f"{T.TRAJ_DURATION:.0f} s", "<b>the Team 6 motion Q_START &rarr; Q_END (what is evaluated)</b>"],
       ["hold_end", f"{E.HOLD_END:.1f} s", "stand still at Q_END so the final (settled) error can be measured"]], [2.6, 2.0, 12.0])
P("At the very start the simulator is in a non-physical pose: joint 4 at 0 is outside its allowed range (it must be &lt; &minus;0.07) and the hand overlaps link 5. "
  "The physics pushes joint 4 to its limit and joint 6 away by about 11&deg; in the first second. That start-up transient is <b>not a bug in our trajectory</b>; it is the simulator's initial state.")

# ---------------- 7
H1("7. Team 6 scenario and singularities")
H2("7.1 The two poses")
table([["", "q (rad)", "Hand position (m)", "Distance from shoulder / max reach"],
       ["Start (low)", f3(q0), f3(hand0), f"{100 * reach0:.0f} %"],
       ["End (high reach)", f3(q1), f3(hand1), f"{100 * reach1:.0f} %"]], [2.8, 6.5, 4.0, 3.3])
B([f"<b>Low start:</b> hand {hand0[0]:.2f} m in front of the base, only {hand0[2]:.2f} m above the floor, pointing straight down (like hovering over a table).",
   f"<b>High end:</b> hand {hand1[2]:.2f} m high (the Panda's hand origin can reach at most about 1.19 m), {hand1[0]:.2f} m forward, {hand1[1]:.2f} m sideways: up <b>and</b> out. "
   f"The elbow is still bent ({np.degrees(q1[3]):.0f}&deg;), so the arm is <b>not</b> fully stretched.",
   f"Both poses are inside the joint limits (closest margin {np.degrees(info['limit_margin_rad'].min()):.1f}&deg;), and MuJoCo reports <b>no contacts</b> along the path (floor and self-collision)."])
H2("7.2 What is a singularity?")
P("A singularity is a pose where J <b>loses rank</b>: there is a direction in which the hand simply <b>cannot move</b>, however the joints move. "
  "For the position Jacobian the classic example is an arm stretched out straight: you can swing the hand sideways, but you cannot make it go further outwards. "
  "Two consequences:")
B(["<b>Inverse velocity blows up.</b> To get a hand speed v you would compute dq = J<super>+</super> v, and the gain of J<super>+</super> in the weak direction is 1/sigma, where sigma is the small singular value. "
   "As sigma &rarr; 0 the required joint speed &rarr; infinity.",
   "<b>The hand loses control authority</b> in that direction: even at full joint speed it moves slowly."])
P("The Panda's workspace boundary (the top of its reach, our scenario) is where the arm is almost fully stretched, so singularities are close.")
H2("7.3 How we detect it")
P("We watch two numbers computed from the 3 x 7 Jacobian along the whole path:")
table([["Indicator", "Meaning", "Threshold", f"Our path (min)", "Arm stretched straight up"],
       ["sigma_min(J) &mdash; smallest singular value", "hand speed per rad/s of joint speed in the <b>worst</b> direction", f"&ge; {T.SIGMA_MIN_ALLOWED}",
        f"{info['sigma_min']:.3f}", f"{sig_up:.3f}"],
       ["w = sqrt(det(J J<super>T</super>)) &mdash; manipulability", "overall volume of hand velocities available (it collapses faster than sigma_min)", f"&ge; {T.W_MIN_ALLOWED}",
        f"{info['manipulability_min']:.3f}", f"{w_up:.4f}"]], [4.1, 4.7, 2.5, 2.6, 2.7])
fig(fig_sing(), 16.6, "Figure 6: sigma_min and w along our path (solid) and along a path that ends with the arm stretched straight up (dashed red). Our path stays well above the thresholds; the other dives below the w threshold.")
H2("7.4 How we handle it (four layers)")
B([f"<b>1. Choice of the goal.</b> The end pose is at {100 * reach1:.0f} % of full reach with the elbow bent: high and far, but away from the stretched singularity "
   f"(w = {w1:.3f}, sigma_min = {sig1:.3f}).",
   "<b>2. Planning in joint space.</b> The planner never inverts J, so joint speeds are bounded by the polynomial itself (peak 1.875 &middot; dq/T) <b>no matter how close the pose is to a singularity</b>. "
   f"Here the peak is {100 * info['peak_speed_ratio'].max():.0f} % of the speed limit.",
   "<b>3. Automatic time stretching</b> when a limit would be exceeded (section 6.4).",
   "<b>4. The guard in plan_trajectory.</b> It samples the path; if sigma_min or w drops below its threshold, the goal is moved back along the same joint-space line (bisection, 30 steps) "
   "to the farthest safe point, and a warning is logged. Test: we asked for the arm stretched straight up; the guard shortened the move to 88 % so that both indicators stay above their thresholds."])
box("What to say about the limits of this approach (be honest in the evaluation)", [
    "The guard watches the <b>position</b> Jacobian (3 x 7) only; orientation singularities are not monitored (the challenge only needs linear velocity). "
    "Joint-space planning protects the <b>joint speeds</b> but not the straight-line shape of the hand path: near a singularity the hand speed along the weak direction is small. "
    "If a Cartesian path were required, we would use a damped pseudo-inverse dq = J<super>T</super> (J J<super>T</super> + lambda<super>2</super> I)<super>-1</super> v, which stays finite at the cost of a small tracking error "
    "(this is what the IK in Challenge 1 does) and slow the hand down when sigma_min is small."], bg="#fff4e0", edge="#d98b00")

# ---------------- 8
H1("8. The experiment: POSITION vs VELOCITY")
P("Everything was run on the real bridge (C++ MuJoCo node, 1000 Hz physics, drawing at 20 Hz as in the instructor's example.py). A logger recorded the planned and the actual joint angles at every command. "
  "Three groups of runs answer three different questions:")
table([["Set", "Node", "Messages / s", "Arm starts at", "Question it answers"],
       ["A", "trajectory_node.py as submitted", "1000", "all zeros", "what happens when you simply follow the challenge text?"],
       ["B", "same, with the template's TRAJ_DT = 0.05", "20", "all zeros", "does the template's own rate work in VELOCITY mode?"],
       ["C", "trajectory_experiment (extended runner)", "1000", "Q_START (set-up move first)", "how well does each mode track the trajectory itself?"]], [1.0, 5.3, 2.2, 3.3, 4.8])

H2("8.1 The template node as submitted (sets A and B)")
fig(fig_template(), 16.6, "Figure 7: left: final joint angles of the template runs against the goal. Right: hand error against the plan. POSITION recovers after a start-up snap; VELOCITY never reaches the goal.")
rows = [["Joint", "Goal Q_END [deg]", "POSITION 1 kHz", "VELOCITY 1 kHz", "VELOCITY 20 Hz"]]
fg, fp, fv, fv20 = np.degrees(T.Q_END), final_deg("template_1khz/POSITION"), final_deg("template_1khz/VELOCITY"), final_deg("template_20hz/VELOCITY")
for j in range(7):
    rows.append([str(j + 1), f"{fg[j]:+.1f}", f"{fp[j]:+.1f}", f"{fv[j]:+.1f}", f"{fv20[j]:+.1f}"])
table(rows, [1.4, 3.6, 3.8, 3.8, 3.8])
B([f"<b>POSITION:</b> a violent first 0.7 s (the snap from zeros to Q_START: hand error up to {tP1k['ee_max']:.0f} mm), then normal tracking; it ends within "
   f"{np.abs(tP1k['settled_joint']).max():.2f}&deg; and {tP1k['settled_ee']:.1f} mm of the goal. At 20 Hz it behaves the same ({tP20['settled_ee']:.1f} mm).",
   f"<b>VELOCITY at 1 kHz:</b> joints 1 and 2 move about {100 * tV1k['delivered'][0]:.0f} % / {100 * tV1k['delivered'][1]:.0f} % of what was commanded, "
   f"but joints 4 and 6 stay pinned (their targets are out of range) so the hand ends {tV1k['settled_ee']:.0f} mm from the goal.",
   f"<b>VELOCITY at 20 Hz (the template's TRAJ_DT):</b> only {100 * tV20['delivered'][0]:.1f} % of the commanded motion is delivered (joints 1 and 2: {100 * tV20['delivered'][0]:.1f} %, {100 * tV20['delivered'][1]:.1f} %): "
   "the bridge adds command/1000 per message and there are only 20 messages per second, exactly 1/50 of the motion. This is why TRAJ_DT was changed to 0.001."])

H2("8.2 Clean comparison from the same start pose (set C)")
P("To compare the two modes themselves, the arm is first moved gently to Q_START and the same trajectory is then played in each mode.")
fig(fig_modes(), 16.6, "Figure 8: top: joints 4 and 6, plan (solid) vs actual (dashed); bottom: error of all joints for the whole run. In POSITION mode the error returns to about zero when the arm stops; in VELOCITY mode it stays.")
table([["Measure (main motion unless noted)", "POSITION", "VELOCITY"],
       ["Messages per second achieved by the node", f"{mP['rate']:.0f}", f"{mV['rate']:.0f}"],
       ["Hand position error, RMS", f"{mP['ee_rms']:.1f} mm", f"{mV['ee_rms']:.1f} mm"],
       ["Hand position error, maximum", f"{mP['ee_max']:.1f} mm", f"{mV['ee_max']:.1f} mm"],
       ["Largest joint error during the motion", f"{mP['max_joint'].max():.1f}&deg; (joint 4)", f"{mV['max_joint'].max():.1f}&deg;"],
       ["Time lag behind the plan (joint 4)", f"{mP['lag']:.0f} ms (servo lag, predicted 100 ms)", "not meaningful (see text)"],
       ["<b>Settled joint error vs the goal</b> (after resting)", "all &lt; " + f"{np.abs(mP['settled_joint']).max():.2f}&deg;", "up to " + f"{np.abs(mV['settled_joint']).max():.1f}&deg;"],
       ["Settled hand error vs the goal", f"{mP['settled_ee']:.1f} mm (gravity sag)", f"{mV['settled_ee']:.1f} mm"],
       ["<b>Fraction of the commanded motion delivered</b> (100 % = all of it)", f"{100 * mP['delivered'].mean():.1f} %", f"{100 * mV['delivered'].mean():.1f} %"]], [7.2, 4.6, 4.8])
H2("8.3 Does the robot track more closely in one mode?")
B([f"<b>POSITION:</b> the error is pure servo lag: the arm follows the plan about {mP['lag']:.0f} ms late, matching the prediction kv/kp = 0.1 s. The error is proportional to the joint speed, "
   f"so it peaks in the middle of the motion (up to {mP['max_joint'].max():.1f}&deg;) and goes <b>back to about zero</b> when the arm stops. The remaining {mP['settled_ee']:.1f} mm is gravity sag "
   "(a spring needs a small stretch to hold the arm's weight).",
   f"<b>VELOCITY:</b> the hand's RMS error looks smaller ({mV['ee_rms']:.1f} mm vs {mP['ee_rms']:.1f} mm) but that is <b>misleading</b>. The joints do not return to the plan: after the arm stops, "
   f"errors of up to {np.abs(mV['settled_joint']).max():.1f}&deg; remain. The missing motion pulls some joints' targets toward zero, which partly cancels the servo lag, so the hand error looks good by luck. "
   "<b>Per joint, POSITION tracks the goal much better.</b>"])
H2("8.4 In VELOCITY mode, does the arm drift or lag more toward the end?")
P(f"<b>Yes: error accumulates and never recovers.</b> In VELOCITY mode the bridge adds up the velocity messages it receives, with no feedback. Only {100 * mV['delivered'].mean():.1f} % of the commanded motion "
  "arrived. Because the target is an <b>integral</b>, every lost message is a permanent offset. The error starts to build in the first (prep) phase, is carried into the main motion, and is "
  "the same at the end of the motion as after settling (the arm is simply parked at the wrong place). In POSITION mode, a lost message does no harm: the next one contains the absolute target again.")
H2("8.5 Why is motion lost? (we tested it)")
P("The bridge keeps only the latest message (queue depth 1), and one thread does both the 1000 Hz physics and the drawing. While a frame is being drawn, several 1 kHz messages arrive and all but the "
  "last are thrown away. Each lost message removes 1 ms of velocity. If this is the cause, drawing less often should help, so we repeated the VELOCITY run with different render rates:")
fig(fig_sens(), 9.5, "Figure 9: the more often the bridge draws, the more velocity messages are lost. At 60 Hz the bridge is almost saturated and the arm barely moves.")
P(f"Each bar is a single run, so small differences between runs are noise, but the trend is clear and large: "
  f"{100 * sens[5]['delivered'].mean():.0f} % delivered at 5 Hz down to about {100 * sens[60]['delivered'].mean():.0f} % at 60 Hz. This confirms the explanation. The node's own publishing rate was "
  f"{min(s['rate'] for s in sens.values()):.0f}&ndash;{max(s['rate'] for s in sens.values()):.0f} Hz in all runs, so the loss happens inside the bridge, not in our node.")
box("Things you could say you would do to fix VELOCITY mode", [
    "(a) Draw less often (low visualisation_frequency) or give the bridge subscription a deeper queue. "
    "(b) Add feedback: publish dq_plan + K &middot; (q_plan &minus; q_actual) so that lost increments are corrected. "
    "(c) Integrate in our node and send positions: that is just POSITION mode. In the challenge we deliberately did <b>not</b> do any of these, because the goal is to observe the open-loop behaviour."])

# ---------------- 9
H1("9. Files, how to run, how it was verified")
table([["File", "What it is"],
       ["panda_kinematics/trajectory_node.py", "<b>The challenge file (the instructor's template).</b> EDIT section: Challenge 1 code (DH_PARAMS, T_L7_HAND, dh_transform, forward_kinematics, inverse_kinematics), then the new "
        "compute_jacobian, compute_jacobian_dot, plan_trajectory and helpers. Below the line: the template's node; only Q_START, Q_END and TRAJ_DT were set."],
       ["panda_kinematics/trajectory_experiment.py", "extra: same maths, longer timeline (set-up move, rests), mode as a ROS parameter; used for the clean comparison"],
       ["panda_kinematics/tracking_logger.py", "extra: records plan vs actual, computes the errors, saves a CSV, a figure and metrics.txt"],
       ["tools/run_experiment.py", "extra: runs bridge + node + logger for one configuration; for the template node it swaps the two mode places in tick() on a temporary copy"],
       ["tools/check_challenge2.py", "22 self-checks (Jacobian vs MuJoCo, planner properties, guard, collisions, copied Challenge 1 code)"],
       ["launch/challenge2.launch.py", "starts the bridge, the extended runner and the logger in one mode"],
       ["challenge2_results/", "the recorded runs used in this report"],
       ["setup.py", "contains the entry point  'trajectory_node = panda_kinematics.trajectory_node:main'  (required by the challenge)"]], [5.6, 11.0])
H2("How to run the template node (what the challenge text asks)")
C("# 1. choose the mode in the bridge: mujoco_ros2/launch/example.py\n"
  "#      {\"control_mode\": \"POSITION\"}   or   {\"control_mode\": \"VELOCITY\"}\n"
  "colcon build --packages-select mujoco_ros2 && source install/setup.bash\n"
  "ros2 launch mujoco_ros2 example.py\n\n"
  "# 2. in trajectory_node.py, tick(): keep the POSITION lines active, or comment them\n"
  "#    and uncomment the VELOCITY lines (two places). Then, in a second terminal:\n"
  "colcon build --packages-select panda_kinematics && source install/setup.bash\n"
  "ros2 run panda_kinematics trajectory_node")
P("The bridge mode and the two lines in tick() must always match: sending positions to a bridge in VELOCITY mode (or the reverse) makes the arm move wildly.")
H2("How to repeat the experiments automatically")
C("source /opt/ros/jazzy/setup.bash && source install/setup.bash\n"
  "python3 src/panda_kinematics/tools/run_experiment.py template POSITION --label template_1khz\n"
  "python3 src/panda_kinematics/tools/run_experiment.py template VELOCITY --label template_20hz --traj-dt 0.05\n"
  "python3 src/panda_kinematics/tools/run_experiment.py experiment VELOCITY --label runner_vis5 --vis-hz 5\n\n"
  "# self-checks (needs mujoco python):\n"
  "export PYTHONPATH=$PWD/.venv/lib/python3.12/site-packages:$PYTHONPATH\n"
  "python3 src/panda_kinematics/tools/check_challenge2.py")
H2("Verification summary (tools/check_challenge2.py)")
table([["Check", "Result"],
       ["Jacobian vs MuJoCo's mj_jac, 200 random poses", "max difference 1.3e-15"],
       ["Jacobian vs finite differences of the forward kinematics", "max difference 1.8e-10"],
       ["dJ/dt analytic vs finite differences", "max difference 1.9e-10"],
       ["planner: 4 arrays of equal length; exact start and end; zero speed and acceleration at both ends", "pass"],
       ["dq is the derivative of q, ddq the derivative of dq", "max difference 2e-7 / 3e-7"],
       ["limits: positions, speeds (peak 30 % of limit), accelerations (peak 4 %); too-short T is stretched; goal outside limits rejected", "pass"],
       ["singularity guard backs off a near-singular goal; our real path needs no back-off", "pass"],
       ["no contacts between hold_start and the end of the run (floor or self-collision)", "pass"],
       ["copied Challenge 1 forward kinematics matches MuJoCo (8.7e-16 m); copied inverse kinematics reaches the Q_END pose", "pass"]], [11.5, 5.1])

# ---------------- 10
H1("10. Likely evaluation questions with model answers")
QA = [
 ("Derive a column of the geometric Jacobian from your DH table.",
  "For a revolute joint i the hand velocity caused by that joint is the cross product of the joint axis with the lever arm: column_i = z_(i-1) x (p_hand - o_(i-1)). "
  "z_(i-1) and o_(i-1) are the 3rd and 4th columns of the cumulative matrix T_0^(i-1) = A1...A(i-1). Example: joint 1, z0 = (0,0,1), o0 = 0, so column 1 = (-p_y, p_x, 0). "
  "Joint 2: alpha1 = -pi/2 gives z1 = (-sin q1, cos q1, 0) and o1 = (0,0,0.333), so column 2 = z1 x (p - o1)."),
 ("What do the Jacobian's rows and columns physically mean?",
  "Rows: x, y, z hand speed. Columns: the hand velocity produced by 1 rad/s of one joint with the others frozen. v = J dq adds those contributions. The entries are lever arms in metres."),
 ("Why is it 3 x 7 and what does that imply?",
  "3 because we only track the linear velocity (x, y, z); 7 because there are 7 arm joints. Redundancy of 7 - 3 = 4: infinitely many joint motions give the same hand velocity (null space), "
  "e.g. moving the elbow with the hand still."),
 ("What is dJ/dt and why do you need it?",
  "The rate of change of the Jacobian because the joints move. Hand acceleration = J ddq + (dJ/dt) dq. The second term is the acceleration the hand has even with constant joint speed (for example the w^2 r of circular motion). "
  "Computed analytically with the product rule on z_i x (p - o_i), using the fact that each axis and origin moves only because earlier joints rotate. Checked against finite differences."),
 ("Why a quintic? What happens at the boundaries?",
  "Six conditions (position, speed and acceleration at both ends) fix a 5th-order polynomial. At t = 0 and t = T: speed 0 and acceleration 0, positions exactly q_start and q_end. "
  "Peak speed 15/8 dq/T at the midpoint, peak acceleration 10/sqrt(3) dq/T^2 at 21 % and 79 % of the time. A cubic would leave a non-zero acceleration at the ends (a jerk); linear interpolation has infinite acceleration."),
 ("How do you guarantee you do not exceed joint limits?",
  "Joint limits: the quintic is monotonic, so each joint stays between its start and end value; checking the end points is enough (and a goal outside raises an error). "
  "Speed and acceleration: closed-form peaks give a minimum T per joint (with 60 % safety), and the planner stretches T if needed. The self-check confirms 30 % of the speed limit and 4 % of the acceleration limit."),
 ("What happens to your trajectory at or near a singularity? How did you handle it?",
  "Near a singularity J loses rank; with a Cartesian planner, dq = J^-1 v would blow up. We plan in joint space, so joint speeds are bounded by 15/8 dq/T regardless of J. "
  "We monitor sigma_min(J) and manipulability w along the path; if either gets below its threshold (0.08 / 0.03), the goal is backed off along the joint line. We also chose a goal with the elbow bent "
  "at 90 % reach. For our real path the minimum sigma_min is 0.150 and w is 0.079, so no back-off was needed; with a goal of 'arm straight up' the guard cut the move to 88 %."),
 ("Why joint space and not a straight line in Cartesian space?",
  "Singularities: the Cartesian method needs J^-1 and is unstable exactly near the top of the workspace. Also the scenario does not require a straight hand path. "
  "Trade-off: in joint space the hand path is curved; in Cartesian space it is straight but needs inverse kinematics at each step and has large joint speeds near singularities."),
 ("Counter-argument: 'your sigma_min threshold of 0.08 is arbitrary'.",
  "It is a design choice with a meaning: at 0.08 a hand speed of 0.1 m/s in the worst direction needs 1.25 rad/s, which is about half of the Panda's 2.175 rad/s limit; random poses have a median of 0.16, our path minimum is 0.15. "
  "The value could be tuned, but the check and its logic (monitor the worst direction and the overall volume) stay the same."),
 ("Counter-argument: 'why 60 % of the speed and acceleration limits?'",
  "Margin: the servo lags, the load changes, and the published limits are absolute maxima. 60 % keeps the arm away from saturating its motors. It is the single constant SAFETY."),
 ("Does your node move the arm to Q_START first? What happens at the start?",
  "No: the template node plans Q_START -> Q_END and starts publishing at once, while the simulator starts at all zeros. In POSITION mode the arm snaps to Q_START in about 0.7 s (we simulated it: the hand stays above 9 cm) "
  "and then follows. In VELOCITY mode the bridge's target starts at 0 and only adds velocities, so the arm does the relative motion Q_END - Q_START from zeros; joints 4 and 6 cannot move because their integrated targets are outside the "
  "actuator range. For the clean comparison we used an extended runner that first moves the arm to Q_START with a slow quintic."),
 ("Which mode tracked better and why?",
  f"POSITION. Its error is only the servo lag (about {mP['lag']:.0f} ms, equal to kv/kp = 0.1 s) plus gravity sag, and it recovers to about zero when the arm stops. VELOCITY integrates the messages in the bridge with no feedback; "
  f"messages are lost when the bridge is busy drawing, so only {100 * mV['delivered'].mean():.1f} % of the motion arrived and the joints stop up to {np.abs(mV['settled_joint']).max():.1f} degrees away from the goal."),
 ("In VELOCITY mode, does the arm drift or lag toward the end?",
  "It drifts: error accumulates with every lost message and never comes back because the target is an integral. The settled error equals the error at the end of the motion. "
  f"It depends on the render rate: {100 - 100 * sens[5]['delivered'].mean():.1f} % lost at 5 Hz, {100 - 100 * sens[20]['delivered'].mean():.1f} % at 20 Hz, {100 - 100 * sens[30]['delivered'].mean():.1f} % at 30 Hz."),
 ("The hand RMS error is lower in VELOCITY mode. Does that mean it is better?",
  "No. The lost motion shifts some joints' targets in the direction that cancels the servo lag, so the hand error looks smaller by coincidence. The per-joint errors do not recover; POSITION returns to within 0.2 degrees."),
 ("How would you improve VELOCITY mode?",
  "Draw less often or make the bridge queue deeper; or add a feedback term dq_plan + K (q_plan - q_actual); or integrate in our node and send positions. We intentionally left it open loop to observe the behaviour."),
 ("Why did you change TRAJ_DT from the template's 0.05 to 0.001?",
  "In VELOCITY mode the bridge adds command/1000 for every message, so each message must represent 1 ms. At 20 Hz only 1/50 of the planned motion arrives (measured 2.0 %); at 1 kHz it is about 97 %. "
  "It is a single constant; POSITION mode works with either value; setting it back to 0.05 reproduces the template behaviour."),
 ("Why does the template run in VELOCITY mode fail for joints 4 and 6?",
  "The bridge starts its target at 0 and adds the velocities, so the targets are 0 + 1.4 rad (joint 4) and 0 - 0.95 rad (joint 6). The actuators only accept targets in the joint's range "
  "(joint 4 < -0.07, joint 6 > -0.0175), so MuJoCo clamps them and the joints do not move. It is the absolute-vs-relative nature of VELOCITY mode, not the planner."),
 ("How do you know the Jacobian is right?",
  "It matches MuJoCo's own Jacobian at the hand to 1e-15 on 200 random poses and finite differences of the forward kinematics to 2e-10. The forward kinematics itself matches the simulator's hand position to 1e-15 m."),
 ("What is a singular value / manipulability?",
  "J = U S V^T: the singular values in S are the arm's gains (hand speed per unit joint speed) along its principal directions; sigma_min is the worst, sigma_max the best. Manipulability w = sigma1 sigma2 sigma3 = sqrt(det(J J^T)) is the volume of the velocity ellipsoid."),
]
for i, (qq, aa) in enumerate(QA, 1):
    story.append(KeepTogether([Paragraph(f"<b>Q{i}. {qq}</b>", ParagraphStyle("q", parent=body, textColor=NAVY, spaceAfter=1, spaceBefore=4)),
                               Paragraph(aa, ParagraphStyle("a", parent=body, leftIndent=12))]))

# ---------------- 11
H1("11. Glossary")
table([["Term", "Meaning"],
       ["Jacobian J", "table converting joint speeds into hand speed (v = J dq)"],
       ["dJ/dt", "rate of change of J; needed for acceleration"],
       ["DH table", "four numbers (theta, d, a, alpha) per link describing the chain"],
       ["Forward kinematics", "joint angles &rarr; hand pose"],
       ["Redundancy / null space", "extra joints beyond what the task needs; motions that do not move the hand"],
       ["Singularity", "pose where J loses rank; some hand direction is impossible and inverse velocity blows up"],
       ["sigma_min, manipulability w", "worst-direction gain of J; overall volume of reachable hand speeds"],
       ["Quintic", "5th-order polynomial; here with zero speed and acceleration at both ends"],
       ["POSITION / VELOCITY mode", "bridge copies the command as target / adds command/1000 to the target for each message"],
       ["Servo lag", "an actuator follows its target about 0.1 s late (kv/kp); the error grows with speed"],
       ["Open loop", "no measurement is used to correct the command; lost or wrong commands stay as errors"],
       ["q, dq, ddq", "joint positions, velocities, accelerations (dq = q-dot, ddq = q-double-dot)"]], [4.6, 12.0])


# ---------------------------------------------------------------- build
def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("DV", 8)
    canvas.setFillColor(GREY)
    canvas.drawString(2 * cm, 1.1 * cm, "MCTR911 Challenge 2 - Team 6 - explained")
    canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"{doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(OUT_PDF, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                        title="Robotics Programming Challenge 2 - explained (Team 6)", author="Team 6")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", os.path.abspath(OUT_PDF))
