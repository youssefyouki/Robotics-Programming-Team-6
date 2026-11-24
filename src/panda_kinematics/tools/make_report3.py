#!/usr/bin/env python3
"""
Builds RPC3_Explained_Team6.pdf: a plain-language STUDY GUIDE for the Challenge 3 evaluation (everything that was
done, explained simply, with likely questions and answers). Numbers come from the code and the recorded runs.

  source /opt/ros/jazzy/setup.bash && source install/setup.bash
  export PYTHONPATH=$PWD/.venv/lib/python3.12/site-packages:$PYTHONPATH
  python3 src/panda_kinematics/tools/make_report3.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report3_common import *          # noqa: F401,F403

OUT_PDF = os.path.join(HERE, "..", "RPC3_Explained_Team6.pdf")
m, mo, fo, dc, nc, gr = (RM[k] for k in ("main", "model_only", "flc_only", "decoupled", "no_coriolis", "gravity_only"))
frac_dom = 100 * float(np.mean(RATIO > 1.0))
mu_e, mu_d, fired, strength, combined, u_w = worked_example()

# ================================================================ cover
story.append(Spacer(1, 1.0 * cm))
story.append(Paragraph("Robotics Programming Challenge 3", ParagraphStyle("t", parent=h1, fontSize=21, leading=27, alignment=TA_CENTER)))
story.append(Paragraph("Lagrangian dynamics and a Fuzzy Logic Controller<br/>for the Franka Emika Panda", ParagraphStyle("t2", parent=body, fontSize=13, leading=18, alignment=TA_CENTER, textColor=GREY)))
story.append(Spacer(1, 0.4 * cm))
story.append(Paragraph("<b>Team 6: \"The High-Speed Tracker\"</b> &mdash; everything that was done, explained in simple words,<br/>so you can answer any question in the evaluation.",
                       ParagraphStyle("t3", parent=body, fontSize=11, leading=16, alignment=TA_CENTER, textColor=TEAL)))
story.append(Spacer(1, 0.6 * cm))
box("How to use this document", [
    "Read sections 1&ndash;8 once, in order. Section 3 (dynamics) and section 5 (fuzzy logic) are what the instructors will ask about most. <b>Section 9 lists the likely questions with model answers: rehearse it out loud.</b>",
    "Every number in this PDF was computed from the project itself (the code and the recorded runs)."])
box("Things you must know before the evaluation", [
    "<b>1. The template is followed.</b> In <i>controller_node.py</i> everything above the marked line is ours (Challenge 1 code, the dynamics, the planner, the FLC, compute_torques); below it is the instructor's code with only Q_START and Q_END set.",
    "<b>2. The robot model (panda.xml) was NOT edited.</b> Only the C++ bridge and a launch file were changed, as the challenge text says. The Panda's motors in the model are position servos, so the bridge converts each torque into the servo target that produces exactly that torque (section 2).",
    "<b>3. One template bug had to be worked around:</b> on ROS 2 Jazzy the template's <i>self.timers[0]</i> crashes (Node.timers is a generator). A 4-line shim above the marked line fixes it without touching the template code.",
    "<b>4. Team 6 was assumed</b> because your repository is called Robotics-Programming-Team-6. If that is wrong, change Q_START / Q_END.",
    "<b>5. Honest limits:</b> the first 0.3 s are a start-up convergence (the arm rests 6&deg; away from the first desired pose). The template's own final hold (gravity only, no feedback) would let leftover velocity coast, so plan_trajectory appends a 3 s controlled stop; the arm then ends at rest."],
    bg="#fff4e0", edge="#d98b00")
story.append(Paragraph("Contents", h2))
table([["1", "The challenge in one page"], ["2", "How the torque-controlled system works (bridge, servos, latency)"], ["3", "Lagrangian dynamics: what it is and how the code computes it"],
       ["4", "The fast Team 6 trajectory"], ["5", "The fuzzy logic controller from scratch"], ["6", "The control law: compute_torques"],
       ["7", "Experiments and results (what happens with / without the FLC and the model)"], ["8", "Files, how to run, how it was verified"],
       ["9", "Likely evaluation questions with model answers"], ["10", "Glossary"]], [1.2, 15.4], header=False)
story.append(PageBreak())

# ================================================================ 1
H1("1. The challenge in one page")
P("<b>Part (a), dynamics.</b> Derive the equations of motion of the Panda with the Lagrangian method, M(q) ddq + C(q,dq) dq + g(q) = tau, implement them in <i>compute_dynamics</i>, and write a script that checks the result against MuJoCo's own dynamics for random q, dq, ddq.")
P("<b>Part (b), controller.</b> Build a fuzzy logic controller (FLC) that tracks the trajectory in TORQUE mode: membership functions, a rule base and centroid defuzzification, one FLC per joint (decoupled MIMO). "
  "<i>compute_torques</i> adds the model torque and the FLC correction. Plot desired vs actual and say whether the FLC reduces the error.")
P("<b>Team 6 variant, \"The High-Speed Tracker\".</b> The trajectory must be very fast, so that inertial and Coriolis forces dominate over gravity. Debate: how did the FLC handle the coupling between joints (moving joint 1 throws off joint 4), and was a decoupled FLC enough?")
box("What was built (the short version)", [
    "&bull; <b>Dynamics</b>: M, C dq + g from the energies of every link, with Jacobians from the Challenge 1 DH frames. Matches MuJoCo to about 1e-8 Nm.",
    f"&bull; <b>Trajectory</b>: the Challenge 2 quintic, planned as fast as Franka's limits allow: <b>{T_MOVE:.2f} s</b>, fastest joint at {100 * SPEED_RATIO.max():.0f} % of its speed limit.",
    "&bull; <b>FLC</b>: seven independent fuzzy controllers (5 triangular sets, 25 rules, Mamdani, centroid) on top of the model torque.",
    "&bull; <b>Bridge</b>: new TORQUE behaviour so that the numbers we publish are real joint torques.",
    f"&bull; <b>Result on the real bridge</b>: {m['rms_late']:.2f}&deg; RMS tracking error with model + FLC, against {mo['rms_late']:.0f}&deg; for the model alone and {fo['rms_late']:.1f}&deg; for the FLC alone."])

# ================================================================ 2
H1("2. How the torque-controlled system works")
fig(fig_architecture(), 16.4, "Figure 1: the control loop. The controller node runs at 100 Hz; the bridge runs the physics at 1 kHz.")
H2("2.1 The problem: the Panda's motors are position servos")
P("In the MuJoCo model each arm joint is driven by a <b>position servo</b>: a spring and a damper pulling the joint toward a target ctrl, force = k<sub>p</sub>(ctrl &minus; q) &minus; k<sub>v</sub> dq. "
  "So whatever we send is treated as a <i>target angle</i>, not a torque. The challenge forbids touching the model, so the fix is in the bridge.")
H2("2.2 What the bridge does now (new option torque_passthrough)")
B(["<b>Torque &rarr; servo target.</b> The bridge solves the servo law for the target that gives exactly the torque we want: <b>ctrl = (tau &minus; bias) / gain</b>, using the model's own gain and bias numbers (bias = &minus;k<sub>p</sub> q &minus; k<sub>v</sub> dq). It is recomputed at every 1 ms step with the current q and dq, so the actuator force equals tau.",
   "<b>Held, not pulsed.</b> The old TORQUE mode cleared the command after one 1 ms step. With our 100 Hz commands that would mean 1 ms of torque every 10 ms. Now the torque is held until the next message.",
   "<b>No extra compensation.</b> The old mode also added gravity/Coriolis compensation and damping by itself, which would double-count our model. The new option applies only what we send.",
   "<b>Euler integrator.</b> MuJoCo's default (implicitfast) treats the servo's velocity gain k<sub>v</sub> as extra stiffness inside the integration and adds k<sub>v</sub> &middot; dt = 0.2&ndash;0.45 kg m<super>2</super> of artificial inertia to every joint, which would ruin an accurate dynamic model (the wrist joints have an inertia of only about 0.1&ndash;0.15 kg m<super>2</super>). The bridge switches to plain Euler when the option is on.",
   "<b>Hold until the first command.</b> The arm would otherwise collapse before the controller node starts. Until the first torque message the bridge holds the initial pose with the servo law.",
   "<b>Safety.</b> Writes stay inside the 8 actuators (the Panda has 9 joint values but 8 actuators). The default behaviour of the bridge for other robots is unchanged."])
P("<b>Limits.</b> The actuators still saturate at the Franka torque limits (87 Nm for joints 1&ndash;4, 12 Nm for joints 5&ndash;7); compute_torques clips to the same values.")
H2("2.3 Timing: why the controller must not be too aggressive")
P("The controller reads the state, computes (about 1.7 ms for the dynamics, in Python) and publishes; the bridge applies the torque. Together with the 10 ms hold this is a delay of roughly 5&ndash;10 ms between measuring and acting. "
  "A feedback loop with delay becomes unstable if its gains are too high. This is exactly what happened with the first tuning (section 5.8), so the final FLC gains are modest and the model feedforward does most of the work.")
H2("2.4 The template on ROS 2 Jazzy")
P("The template's start_trajectory_once calls <i>self.destroy_timer(self.timers[0])</i>. On Jazzy, <i>Node.timers</i> is a generator, so indexing it raises <i>TypeError: 'generator' object is not subscriptable</i> and the node never publishes. "
  "Because the infrastructure must not be modified, a small shim above the marked line makes <i>Node.timers</i> return a list (it does nothing on older ROS versions where it already is one).")
P("After the last plan sample, the template sends only the gravity torque g(q) (no feedback). That would keep a resting arm where it is, but any leftover velocity would coast. To avoid it, <b>plan_trajectory appends 3 s of \"stand still at q_end\" samples</b> (zero velocity and acceleration): the template keeps running the model + FLC on them, so the arm is brought to a full stop <i>under feedback</i>, and only then does the gravity-only hold take over a resting arm.")

# ================================================================ 3
H1("3. Lagrangian dynamics")
H2("3.1 The idea in simple words")
P("Instead of balancing forces on every link (Newton), the Lagrangian method uses <b>energy</b>. Write down how much <b>kinetic energy</b> T the moving arm has and how much <b>potential energy</b> V it has because of its height; the equations of motion follow mechanically from L = T &minus; V. "
  "For a robot arm the result always has the same shape:")
C("M(q) ddq  +  C(q,dq) dq  +  g(q)  =  tau")
table([["Term", "Name", "What it physically represents"],
       ["M(q) ddq", "inertia term", "the torque needed to <b>accelerate</b> the arm. M (7x7) is the \"effective mass\" the joints feel; it depends on the pose (an outstretched arm is much harder to swing than a folded one). "
        "The off-diagonal entries are <b>coupling</b>: accelerating joint 1 also asks torque from joint 4."],
       ["C(q,dq) dq", "Coriolis and centrifugal", "forces that exist only while the arm <b>moves</b>, because turning one joint changes the effective inertia seen by the others. They grow with speed squared, so they matter at high speed."],
       ["g(q)", "gravity", "the torque needed just to <b>hold the arm up</b> against its own weight; it depends on the pose only, not on motion"],
       ["tau", "joint torques", "what the motors must deliver"]], [2.6, 3.0, 11.0])
H2("3.2 A complete small example: two links")
fig(fig_two_link(), 6.0, "Figure 2: a two-link arm with a point mass at the end of each link (angles from the horizontal, gravity downwards).")
P("<b>Kinetic energy.</b> Mass 1 moves only with joint 1: v<sub>1</sub> = l<sub>1</sub> dq<sub>1</sub>. Mass 2 moves with both: its velocity squared is "
  "l<sub>1</sub><super>2</super> dq<sub>1</sub><super>2</super> + l<sub>2</sub><super>2</super>(dq<sub>1</sub>+dq<sub>2</sub>)<super>2</super> + 2 l<sub>1</sub> l<sub>2</sub> cos(q<sub>2</sub>) dq<sub>1</sub>(dq<sub>1</sub>+dq<sub>2</sub>). Adding &frac12; m v<super>2</super> for both masses and writing T = &frac12; dq<super>T</super> M dq gives")
C("M11 = (m1+m2) l1^2 + m2 l2^2 + 2 m2 l1 l2 cos q2\nM12 = M21 = m2 l2^2 + m2 l1 l2 cos q2\nM22 = m2 l2^2")
P("<b>Potential energy</b> V = m<sub>1</sub> g l<sub>1</sub> sin q<sub>1</sub> + m<sub>2</sub> g (l<sub>1</sub> sin q<sub>1</sub> + l<sub>2</sub> sin(q<sub>1</sub>+q<sub>2</sub>)); differentiating, "
  "<b>g<sub>1</sub> = (m1+m2) g l1 cos q1 + m2 g l2 cos(q1+q2),  g<sub>2</sub> = m2 g l2 cos(q1+q2)</b>.")
P("<b>Coriolis.</b> M depends on q<sub>2</sub> only (through cos q<sub>2</sub>). With h = m2 l1 l2 sin q2 the Christoffel formula gives <b>C dq = ( &minus;h (2 dq1 dq2 + dq2<super>2</super>) ,  h dq1<super>2</super> )</b>: "
  "joint 2 spinning pushes on joint 1 (centrifugal/Coriolis) and joint 1 spinning pushes on joint 2.")
P("<b>Numbers</b> (m1 = m2 = 1 kg, l1 = l2 = 1 m, q1 = 0, q2 = 90&deg;): M = [[3, 1], [1, 1]]; g = (19.62, 0) Nm: joint 1 carries both masses, joint 2 carries none because the second link is vertical. "
  "The off-diagonal 1 is the coupling: an acceleration of joint 2 of 1 rad/s<super>2</super> needs 1 Nm at joint 1.")
H2("3.3 The same steps for the Panda (what compute_dynamics does)")
fig(fig_lagrange(), 16.0, "Figure 3: the data flow inside compute_dynamics.")
B(["<b>Link data.</b> panda.xml gives, for every link, its mass, the position of its centre of mass (CoM) and its inertia matrix. They are stored in the code as constants (nothing in the model file changes).",
   "<b>Frames.</b> The data are given in MuJoCo's body frames; the DH table of Challenge 1 uses its own frames. Both are attached to the same link and turn with the same joint, so they differ by a <b>constant transform</b> (computed once at q = 0; check 4 shows it is constant for random q). The CoM and inertia are expressed in the DH frame of the link.",
   "<b>Hand and fingers.</b> They are rigid with link 7 (the gripper is held open), so their mass, CoM and inertia are lumped into link 7 with the parallel-axis theorem. The fingers contribute very little (15 g each).",
   "<b>Jacobians of the CoM.</b> For link i and joint j &le; i with axis z<sub>j</sub> through o<sub>j</sub>: column j of Jv is z<sub>j</sub> &times; (p<sub>CoM</sub> &minus; o<sub>j</sub>); column j of Jw is z<sub>j</sub>. These come from the DH frames, exactly as in Challenge 2.",
   "<b>M from the kinetic energy.</b> T<sub>i</sub> = &frac12; m |Jv dq|<super>2</super> + &frac12; (Jw dq)<super>T</super> I (Jw dq) &rArr; M = &Sigma;<sub>i</sub> ( m Jv<super>T</super>Jv + Jw<super>T</super> I Jw ), with I = R I<sub>body</sub> R<super>T</super> in the base frame. A rotor inertia (armature) of 0.1 kg m<super>2</super> is added to each diagonal entry, as in panda.xml.",
   "<b>g from the potential energy.</b> V = &Sigma; m g z<sub>CoM</sub>, so g = &Sigma; m g Jv[z row]: only the vertical part of the CoM Jacobian matters.",
   "<b>C from the derivatives of M.</b> Differentiate M with respect to each joint angle (dM[k] = &part;M/&part;q<sub>k</sub>) using \"an axis only turns when an earlier joint turns\": dz<sub>j</sub>/dq<sub>k</sub> = z<sub>k</sub> &times; z<sub>j</sub>, do<sub>j</sub>/dq<sub>k</sub> = z<sub>k</sub> &times; (o<sub>j</sub> &minus; o<sub>k</sub>), dp/dq<sub>k</sub> = z<sub>k</sub> &times; (p &minus; o<sub>k</sub>) for k &lt; j, and d(R I R<super>T</super>)/dq<sub>k</sub> = S I<sub>w</sub> &minus; I<sub>w</sub> S with S = skew(z<sub>k</sub>). "
   "Then <b>(C dq)<sub>i</sub> = &Sigma;<sub>jk</sub> [ dM[k]<sub>ij</sub> &minus; &frac12; dM[i]<sub>jk</sub> ] dq<sub>j</sub> dq<sub>k</sub></b> (Christoffel symbols).",
   "<b>Output.</b> compute_dynamics returns M (7x7) and the vector C dq + g (7). The template's compute_torques multiplies M by the desired acceleration and adds the vector."])
H2("3.4 Validation against MuJoCo")
P("The script <i>tools/check_dynamics.py</i> draws random q, dq, ddq inside the joint limits (skipping poses with contacts, because the model has no contact forces) and compares:")
table([["Check", "Result"],
       ["M(q) vs MuJoCo's inertia matrix (mj_fullM), 300 states", f"max difference {check_value(1)} kg m<super>2</super>"],
       ["C dq + g vs MuJoCo's bias force (qfrc_bias)", f"max difference {check_value(2)} Nm"],
       ["M ddq + C dq + g vs MuJoCo inverse dynamics (mj_inverse)", f"max difference {check_value(3)} Nm"],
       ["XML body frame = DH frame x constant transform", f"max difference {check_value(4)}"],
       ["power balance: d(kinetic energy)/dt = dq . (tau &minus; g)  (the Coriolis terms do no work)", f"max difference {check_value(5)} W"],
       ["compute_dynamics speed (the loop needs &lt; 10 ms)", "1.7 ms per call"]], [10.6, 6.0])
fig(fig_dyn_validation(), 12.5, "Figure 4: analytic torque against MuJoCo's inverse-dynamics torque for 150 random states and 7 joints: identity line.")
P("<b>Joint damping.</b> panda.xml gives every joint a viscous damping of 1 Nms/rad. It is friction, not energy storage, so it is <b>not part of a Lagrangian model</b>; MuJoCo's inverse dynamics contains it, so the check adds b &middot; dq to our torque. "
  "In the controller it is left out on purpose: it is one of the \"model inaccuracies\" the FLC has to absorb (section 7).")

# ================================================================ 4
H1("4. The fast Team 6 trajectory")
P(f"<b>Start:</b> the upright rest pose {f3(q0)} rad (hand at {f3(hand0)} m). <b>End:</b> {f3(q1)} rad (hand at {f3(hand1)} m): a folded-back pose. Both are inside the joint limits and the path has no contacts in MuJoCo. "
  "The start is chosen close to where the simulator's arm rests after start-up, so the first desired pose is only about 6&deg; away from the real pose.")
P(f"<b>How fast?</b> The template gives TRAJ_DURATION = 4 s and the planner is ours, so plan_trajectory treats it as the <i>longest</i> allowed time and plays the quintic (zero speed and acceleration at both ends) as fast as the limits allow: 95 % of Franka's speed and acceleration limits "
  f"and at most 90 % of the actuator torque. Result: <b>{T_MOVE:.2f} s</b>. The fastest joint (joint {int(np.argmax(SPEED_RATIO)) + 1}) peaks at {100 * SPEED_RATIO.max():.0f} % of its speed limit; the model torque peaks at {100 * TAU_USE.max():.0f} % of the limit (joint {int(np.argmax(TAU_USE)) + 1}).")
fig(fig_fast(), 16.4, "Figure 5: planned joints, joint speeds as a percentage of the limit, and the torque norms (inertial + Coriolis against gravity).")
P(f"<b>Is it really dominated by inertial and Coriolis forces?</b> The ratio (norm of inertial + Coriolis torque) / (norm of gravity torque) is {RATIO.mean():.2f} on average, {RATIO.max():.1f} at its peak, and above 1 during {frac_dom:.0f} % of the move. "
  f"The first half is clearly dynamics-dominated; later gravity grows because the arm folds back. Played in the template's 4 s, the same move would need only {100 * (T_MOVE / cn.TRAJ_DURATION) ** 2:.0f} % of the accelerations (they scale with 1/T<super>2</super>) and gravity would dominate everywhere.")

# ================================================================ 5
H1("5. The fuzzy logic controller from scratch")
H2("5.1 The idea in simple words")
P("A fuzzy controller works like a person following simple rules of thumb: \"if the error is <i>large and positive</i> and it is <i>not shrinking</i>, push <i>hard</i> in the positive direction\". Words like <i>large</i> or <i>small</i> are not sharp: an error of 0.03 rad is "
  "partly \"medium\" and partly \"large\". A <b>membership function</b> says <i>how much</i> (a number from 0 to 1) a value belongs to each word. Several rules fire at once, each with a strength, and their conclusions are blended into one number.")
H2("5.2 The four steps")
table([["Step", "What happens", "In our code"],
       ["1. Fuzzification", "turn each crisp input into degrees of membership in the sets LN, MN, S, MP, LP", "_memberships(): e / E_scale and de / DE_scale, clipped to [-1, 1], then triangles"],
       ["2. Rule base", "IF e is A AND de is B THEN u is C (25 rules)", "_RULES: C = A + B clipped to the five sets"],
       ["3. Inference", "strength of a rule = min(membership of A, membership of B) (AND = min); an output set takes the max strength of the rules that point to it; each output set is clipped at its strength; all are united (max)", "w = min(...), strength = max(...), combined = max(min(strength, set))"],
       ["4. Defuzzification", "centroid of area: the centre of mass of the united output shape gives one number u in [-1, 1]", "(combined * u_grid).sum() / combined.sum(), 201 points"]], [3.0, 7.6, 6.0])
P("Finally the torque correction is  <b>tau_FLC = U_scale &middot; u</b>.")
H2("5.3 The design: shapes, boundaries, scales")
fig(fig_flc_design(), 16.6, "Figure 6: the membership functions, the 25-rule table (output set for every pair) and the control surface of joint 1 (output torque against error and change of error).")
table([["Choice", "Value", "Why"],
       ["Number of sets", "5 (LN, MN, S, MP, LP)", "the standard minimum for a signed quantity: negative/positive, small/large, and zero; 5 &times; 5 = 25 rules stays readable"],
       ["Shape", "triangles with centres &minus;1, &minus;0.5, 0, 0.5, 1; the outer sets saturate", "each input always belongs to at most two sets with memberships adding to 1, so the output changes smoothly when a rule switches (no chattering); simple and cheap to compute"],
       ["Error scale E", "0.05 rad (about 3&deg;)", "an error of this size counts as \"large\"; larger errors saturate. Tuned together with U so the loop is stable with the real latency"],
       ["Change-of-error scale DE", "1.5 rad/s", "of the order of the planned peak joint speeds (0.9&ndash;2.5 rad/s); a larger value means less weight on the derivative and was found more robust against the delay"],
       ["Output scale U", "20 Nm for joints 1&ndash;4, 8 Nm for joints 5&ndash;7", "a fraction of the actuator limits (87 and 12 Nm): the FLC only corrects, the model supplies the main torque; lower than the first tuning (60 Nm) which chattered"],
       ["Smoothing time", "0 (off)", "the output is already continuous; extra low-pass only adds delay (it was slightly worse in tests)"]], [3.3, 4.7, 8.6])
P("<b>Reading the control surface:</b> along the anti-diagonal (error and change of error cancelling) the output is zero; when both are positive it saturates at about +0.83 U (the centroid of the outer set cannot reach U); near the centre the surface is nearly linear, like a PD controller, "
  "with the error acting as the \"P\" part and the change of error as the \"D\" part.")
H2("5.4 One rule walked through")
P(f"Joint 1 with e = +0.03 rad and de = &minus;0.45 rad/s. Normalised: e/E = {WORK_E_N:+.1f}, de/DE = {WORK_DE_N:+.1f}. <b>Fuzzification:</b> e is MP 0.8 and LP 0.2; de is MN 0.6 and S 0.4. "
  "<b>Rules that fire</b> (strength = min of the two memberships): " + "; ".join(f"IF e is {a} AND de is {b} THEN u is {c_} ({w:.1f})" for a, b, c_, w, _ in fired)
  + f". <b>Inference:</b> output set S gets 0.6, MP gets max(0.4, 0.2) = 0.4, LP gets 0.2. <b>Defuzzification:</b> the three clipped triangles are united and their centroid is u = {u_w:+.3f}. "
  f"<b>Torque:</b> {u_w:+.3f} &times; 20 Nm = <b>{u_w * cn.FLC_U_SCALE[0]:+.2f} Nm</b> added to the model torque of joint 1. Reading it in words: the error is mostly \"medium positive\" and the joint is already approaching (de negative), so the controller pushes gently, not hard.")
fig(fig_worked(), 16.6, "Figure 7: the same step as a picture: the memberships of e and de, and the united output set with its centroid.")
H2("5.5 MIMO and decoupled")
P("The Panda has 7 joints, so the controller has 14 inputs (7 errors, 7 changes of error) and 7 outputs. A single rule base over 14 inputs would be impossible, so it is split into <b>seven independent FLCs</b>, one per joint, each with the 25 rules above and its own scales (they are arrays of 7 numbers). "
  "The code evaluates all seven at once with numpy. The FLCs do not know about each other: the coupling between joints is handled by the model torque (section 7.3).")
H2("5.6 Tuning story")
B(["First tuning (zero-latency simulation): U = 60 Nm, E = 0.05 rad, DE = 2 rad/s. In the simulator without delay it tracked to 0.3&deg;.",
   "On the real bridge the same numbers made the arm chatter at about 10 Hz (2.6&deg; RMS) because of the 5&ndash;10 ms delay (section 2.3); the leftover velocity then made the arm coast during the template's feedback-free hold.",
   "<b>offline_sim.py</b> reproduces the loop with a latency parameter. A sweep at 8 ms (E, DE, U, smoothing) found stable gains: U = 20/8 Nm, DE = 1.5 rad/s, no smoothing; they also work for 4 ms and 12 ms. On the real bridge the error then dropped to about 0.4&deg; RMS."])

# ================================================================ 6
H1("6. The control law: compute_torques")
C("tau_model      = M(q) * ddq_des + C(q, dq) dq + g(q)   # model, at the CURRENT q, dq\n"
  "error          = q_des  - q\n"
  "delta_error    = dq_des - dq\n"
  "tau_correction = fuzzy_controller(error, delta_error, 0.01)\n"
  "tau            = clip(tau_model + tau_correction, -TAU_MAX, TAU_MAX)")
B(["<b>Feedforward.</b> If the model were perfect and nothing disturbed the arm, tau_model alone would produce exactly the desired acceleration: it is the torque of the motion.",
   "<b>Feedback.</b> The FLC measures how far the arm is from the plan and corrects. It is what keeps the loop stable (the upright arm is unstable) and absorbs everything the model leaves out: joint damping, the 6&deg; start offset, the 10 ms hold and delay, the discretisation.",
   "<b>Why ddq_des and not the measured acceleration?</b> It is noise-free and known in advance; the measured acceleration would need a derivative of the (noisy) velocity.",
   "<b>Clip.</b> The actuators cannot give more than 87 / 12 Nm; clipping makes the logged command equal to what is applied."])

# ================================================================ 7
H1("7. Experiments and results")
P("All runs use the real C++ bridge in TORQUE mode, with the same trajectory and the same start. The switches USE_MODEL, USE_FLC and MODEL_TERMS in controller_node.py (defaults: model and FLC on, \"full\" model) were changed on a temporary copy by <i>tools/run_control.py</i>, so the submitted file is untouched. \"After start-up\" ignores the first 0.3 s.")
H2("7.1 The submitted controller")
fig(fig_tracking(), 16.4, "Figure 8: desired (solid) and actual (dashed) joints, joint errors, commanded torques and hand error. After the dotted line (end of the motion) the plan holds q_end for 3 s (controlled stop).")
P(f"Tracking error over the motion: <b>{m['rms']:.2f}&deg; RMS</b> ({m['mx']:.1f}&deg; maximum, from the start-up); after the first 0.3 s <b>{m['rms_late']:.2f}&deg; RMS and {m['mx_late']:.1f}&deg; at most</b>. Hand position error {m['ee_rms']:.1f} mm RMS, {m['ee_max']:.1f} mm maximum. "
  f"Peak commanded torque {m['tau_use']:.0f} % of the limit; the largest torque step between two commands is {m['dtau']:.1f} Nm (smooth). "
  f"<b>Full stop:</b> after the motion the arm is held for 3 s under feedback and then by the template's gravity hold; the final error is {np.abs(R['main']['final']).max():.2f}&deg; and the joint speed is 0.000 rad/s (all four runs of the submitted controller end at rest).")
H2("7.2 With and without the FLC and the model")
rows = [["Controller", "RMS all [deg]", "RMS after start-up [deg]", "max after start-up [deg]", "hand RMS [mm]"]]
for k, name in RUNS:
    rng = f" ({RM[k]['lo']:.2f}-{RM[k]['hi']:.2f}, {RM[k]['n']} runs)" if RM[k]['n'] > 1 else " (1 run)"
    rows.append([name, f"{RM[k]['rms']:.2f}", f"{RM[k]['rms_late']:.2f}{rng}", f"{RM[k]['mx_late']:.1f}", f"{RM[k]['ee_rms']:.1f}"])
table(rows, [5.4, 2.2, 4.6, 2.6, 1.8])
fig(fig_ablation(), 16.4, "Figure 9: left, RMS error of the six controllers (log scale); right, how the error norm evolves.")
B([f"<b>Does the FLC reduce the tracking error? Yes:</b> {mo['rms_late']:.1f}&deg; (model alone) &rarr; {m['rms_late']:.2f}&deg; (model + FLC).",
   "<b>Why does the model alone fail?</b> Two reasons. (1) The model contains no joint damping (1 Nms/rad in panda.xml, a friction effect that a Lagrangian model leaves out); the wrist joints have an inertia of only about 0.1&ndash;0.15 kg m<super>2</super>, so a damping torque of up to 2&ndash;2.5 Nm is huge for them. "
   "(2) The arm is open-loop <i>unstable</i>: any small error grows because gravity pushes it further. <b>We tested it:</b> adding only the term b &middot; dq to the open-loop torque lowers the model-alone error from 28.6&deg; to 1.1&deg; RMS from a perfect start (tools/offline_sim.py). "
   "So the \"small model inaccuracy\" the FLC fills in is essentially viscous friction, and the FLC also supplies the feedback stability.",
   f"<b>FLC only</b> (no model, Team 2's variant is a separate question): {fo['rms_late']:.1f}&deg; RMS. Joint 2, which carries most of the weight, is {fo['pj_late'][1]:.1f}&deg; off: a bounded fuzzy correction cannot replace gravity compensation."])
H2("7.3 Debate focus: coupling at high speed")
fig(fig_perjoint(), 12.0, "Figure 10: per-joint RMS error after start-up for each controller with a model or with the FLC alone (mean of the runs).")
P(f"<b>How big is the coupling?</b> Along the trajectory the off-diagonal terms of M make up about <b>{100 * OFF_SHARE:.0f} % of the inertial torque</b> (accelerating one joint really does ask torque from the others), and the Coriolis/centrifugal torque peaks at {CORI_PEAK:.1f} Nm "
  f"(joint 2: {CORI_PEAK_J[1]:.1f} Nm, joint 4: {CORI_PEAK_J[3]:.1f} Nm, joint 1: {CORI_PEAK_J[0]:.1f} Nm), about a third of the peak inertial torque ({INERT_PEAK:.0f} Nm). So the coupling is not small.")
P(f"<b>How did the FLC handle it?</b> It does not, directly: the seven FLCs are independent. To see what matters, the model was degraded while keeping the same FLC. Mean RMS error after start-up over repeated runs: "
  f"full model <b>{RM['main']['rms_late']:.2f}&deg;</b> ({RM['main']['lo']:.2f}&ndash;{RM['main']['hi']:.2f}, {RM['main']['n']} runs); <b>Coriolis/centrifugal term removed</b> {RM['no_coriolis']['rms_late']:.2f}&deg; ({RM['no_coriolis']['lo']:.2f}&ndash;{RM['no_coriolis']['hi']:.2f}): no measurable difference; "
  f"<b>inertia coupling removed</b> (diagonal M, joints treated as independent masses) {RM['decoupled']['rms_late']:.2f}&deg; ({RM['decoupled']['lo']:.2f}&ndash;{RM['decoupled']['hi']:.2f}): consistently about {100 * (RM['decoupled']['rms_late'] / RM['main']['rms_late'] - 1):.0f} % worse; "
  f"gravity only {RM['gravity_only']['rms_late']:.1f}&deg;; no model {RM['flc_only']['rms_late']:.1f}&deg;.")
box("Conclusion for the debate", [
    "&bull; A decoupled FLC <b>alone</b> is not enough (6&deg;): it has no gravity and no inertia knowledge.",
    "&bull; The <b>inertial feedforward M ddq_des</b> is what matters most: even with only the diagonal of M the error drops from 3.0&deg; (gravity only) to 0.5&deg;.",
    "&bull; The coupling terms are large but <b>smooth over the 1.1 s motion</b>, slowly varying compared with the bandwidth of the FLC, so the independent FLCs absorb most of the part the model leaves out: the off-diagonal inertia buys a further ~0.15&deg;, the Coriolis term nothing measurable.",
    "&bull; So a decoupled FLC performed well enough <b>on top of a model that contains the inertia</b>; full coupling compensation gives a small extra gain. At higher speeds or with a faster loop the Coriolis term would matter more."])
H2("7.4 Offline simulation against the real bridge, and run-to-run variation")
P(f"The offline loop (8 ms latency, tools/offline_sim.py) gives, after start-up: model + FLC 0.40&deg; (real: {RM['main']['rms_late']:.2f}&deg;), without Coriolis 0.46&deg; (real {RM['no_coriolis']['rms_late']:.2f}&deg;), diagonal inertia 0.53&deg; (real {RM['decoupled']['rms_late']:.2f}&deg;), "
  f"gravity only 3.2&deg; (real {RM['gravity_only']['rms_late']:.1f}&deg;), FLC only 6.2&deg; (real {RM['flc_only']['rms_late']:.1f}&deg;) and model only 33&deg; (real {RM['model_only']['rms_late']:.0f}&deg;): a faithful test bench. "
  f"On the real bridge the submitted controller varies between {RM['main']['lo']:.2f}&deg; and {RM['main']['hi']:.2f}&deg; from run to run (the 100 Hz Python loop has timing jitter, and an earlier set of runs that was disturbed by a leftover process showed how sensitive the arm is to it), "
  "so differences below about 0.2&deg; are not interpreted.")

# ================================================================ 8
H1("8. Files, how to run, how it was verified")
table([["File", "What it is"],
       ["panda_kinematics/controller_node.py", "<b>The challenge file.</b> Above the marked line: Challenge 1 code (DH_PARAMS, T_L7_HAND, dh_transform, forward kinematics), the Jazzy shim, link data, compute_dynamics, the planner, the FLC, compute_torques. Below it: the template's node with Q_START and Q_END set."],
       ["panda_kinematics/control_logger.py", "extra: records desired / actual / torque for every command, writes metrics.txt, tracking.csv and tracking.png"],
       ["tools/check_dynamics.py", "the required comparison with MuJoCo (6 checks)"],
       ["tools/run_control.py", "runs bridge + node + logger for one controller variant on a temporary copy of the node"],
       ["tools/offline_sim.py", "the loop without ROS (latency parameter) used for tuning and for the damping test"],
       ["mujoco_ros2/launch/panda_torque.py", "launches the bridge for the unmodified Panda in TORQUE mode with torque_passthrough"],
       ["mujoco_ros2/src/mujoco_ros.cpp (+ .hpp)", "bridge changes: torque_passthrough option; stays inside ctrl; default behaviour unchanged"],
       ["setup.py", "contains the entry point  'controller_node = panda_kinematics.controller_node:main'  (required)"],
       ["challenge3_results/", "recorded runs used here (6 controllers) and the check and offline outputs"]], [5.6, 11.0])
H2("How to run")
C("cd ~/Robo_arm && source /opt/ros/jazzy/setup.bash\n"
  "colcon build --packages-select mujoco_ros2 panda_kinematics && source install/setup.bash\n\n"
  "ros2 launch mujoco_ros2 panda_torque.py                 # terminal 1: bridge in TORQUE mode\n"
  "ros2 run panda_kinematics controller_node               # terminal 2 (after sourcing again)\n\n"
  "# repeat the experiments / checks:\n"
  "python3 src/panda_kinematics/tools/run_control.py main --out challenge3_results\n"
  "python3 src/panda_kinematics/tools/run_control.py model_only --no-flc\n"
  "export PYTHONPATH=$PWD/.venv/lib/python3.12/site-packages:$PYTHONPATH\n"
  "python3 src/panda_kinematics/tools/check_dynamics.py")

# ================================================================ 9
H1("9. Likely evaluation questions with model answers")
QA = [
 ("Explain how you used the Lagrangian approach to compute the kinetic and potential energies, and how they lead to the mass matrix and the gravity vector.",
  "For every link I take the velocity of its centre of mass and its angular velocity, both linear in the joint speeds: v = Jv dq, w = Jw dq, with Jacobians from the DH frames. The kinetic energy of the link is 1/2 m |v|^2 + 1/2 w' I w, so the total is "
  "1/2 dq' M dq with M = sum(m Jv'Jv + Jw' I Jw) (plus the rotor inertia). The potential energy is sum(m g z of the centre of mass); its derivative with respect to q is the gravity vector g = sum(m g Jv[z row]). "
  "Coriolis terms come from differentiating M with respect to q (Christoffel symbols)."),
 ("What do M(q) ddq, C(q,dq) dq and g(q) physically represent?",
  "M ddq is the torque to accelerate the arm (M is the pose-dependent effective inertia; off-diagonal terms are coupling between joints). C dq are the velocity-dependent forces: centrifugal (from a joint's own speed) and Coriolis (from two joints moving together); they do no work and are zero at rest. "
  "g is the torque that holds the arm up against gravity; it depends only on the pose."),
 ("Why does C come from M?",
  "Because the Coriolis and centrifugal forces appear when the inertia the joints feel changes as the arm moves. If M did not depend on q there would be no such forces. The Christoffel formula (C dq)_i = sum[dM_ij/dq_k - 1/2 dM_jk/dq_i] dq_j dq_k expresses that."),
 ("How do you connect the DH table to the dynamics, and why a constant transform?",
  "The Jacobian of each link's centre of mass is built from the axes z_j and points o_j of the DH frames. But panda.xml gives mass, centre and inertia in MuJoCo's body frames. Both frames are fixed to the same link and turn with the same joint, so they differ by one constant transform per link, "
  "computed once at q = 0; check 4 verifies it is constant for random q (error 1e-15)."),
 ("How did you validate the model?",
  f"With random q, dq, ddq inside the limits: M against mj_fullM ({check_value(1)}), C dq + g against qfrc_bias ({check_value(2)}), and M ddq + C dq + g (+ joint damping) against mj_inverse ({check_value(3)} Nm). Poses in contact were skipped. A power-balance check confirms that C does no work."),
 ("Why is joint damping not in your model? Why is the armature?",
  "The Lagrangian equation M ddq + C dq + g = tau comes from energies; damping is dissipative friction and is not part of it. The armature is a physical rotor inertia (0.1 kg m^2) that really adds to the mass matrix in MuJoCo, so including it makes M exact. Damping is left to the FLC; adding b*dq to the model would improve the model-only result to about 1 degree."),
 ("Why triangular membership functions and five sets? How did you choose the boundaries?",
  "Triangles overlapping at 0.5 give memberships that add to 1, so the output changes smoothly when the active rules change; five sets (LN, MN, S, MP, LP) are the usual minimum to separate sign and size and give 25 readable rules. The boundaries are the scales: E = 0.05 rad (an error of about 3 degrees counts as large), "
  "DE = 1.5 rad/s (of the order of the planned joint speeds), U = 20 Nm (joints 1-4) and 8 Nm (wrist), a fraction of the torque limits because the model supplies the main torque. They were tuned in simulation including the real loop's delay."),
 ("Walk through one rule: IF error is Large Positive and change of error is Small THEN torque is ...",
  "The rule says Large Positive: the joint is far behind (positive error) and the error is not changing, so push strongly positive: output set LP. Its strength is min(membership of e in LP, membership of de in S). Other rules fire at the same time (for example e MP with de S gives MP), each clips its output set at its strength, "
  "the clipped sets are united with max, and the centroid of the united area gives one number u in [-1, 1], which is multiplied by the joint's torque scale. In the worked example (e = +0.6, de = -0.3 normalised) four rules fire and u = +0.22, that is +4.5 Nm for joint 1."),
 ("Why Mamdani min/max inference and centroid defuzzification?",
  "They are the standard choices, give a continuous output (the centroid blends all active rules) and match the assignment (centroid of area). Min for AND is the usual conservative intersection; max for combining rules keeps the strongest conclusion of each output set."),
 ("Why 25 rules with C = A + B?",
  "That table is the fuzzy version of a PD controller: positive error pushes positive, positive change of error (error growing) pushes harder, a positive error with a strongly negative change of error (already catching up) gives about zero. It is symmetric, complete (every input pair has a rule) and easy to explain."),
 ("What happens to the tracking error when the FLC is removed?",
  f"With only the model torque the error grows to {mo['rms_late']:.0f} degrees RMS and the arm ends 30-90 degrees off. Reasons: the model has no joint damping (which is huge for the light wrist joints), and the upright arm is unstable, so open-loop errors grow. I tested it: adding only b*dq brings the model-only error from 28.6 to 1.1 degrees from a perfect start, "
  "so the FLC is mainly absorbing viscous friction and supplying feedback stability."),
 ("And when the model is removed and only the FLC is used?",
  f"The error is {fo['rms_late']:.1f} degrees RMS; joint 2 is {fo['pj_late'][1]:.0f} degrees off, because a bounded correction torque cannot hold the arm up, and the coupling forces at high speed are not compensated. The model does the heavy lifting; the FLC fills the gaps."),
 ("Team 6: how did the FLC handle the coupling between joints (joint 1 throws off joint 4)? Was a decoupled FLC enough?",
  f"The seven FLCs are independent; the coupling is cancelled by the model torque. The coupling is large (off-diagonal inertia is about {100 * OFF_SHARE:.0f} % of the inertial torque; Coriolis peaks at {CORI_PEAK:.0f} Nm). Degrading the model while keeping the FLC (mean RMS after start-up): full {RM['main']['rms_late']:.2f} degrees, no Coriolis term {RM['no_coriolis']['rms_late']:.2f} (no measurable change), "
  f"diagonal inertia only {RM['decoupled']['rms_late']:.2f}, gravity only {RM['gravity_only']['rms_late']:.1f}, no model {RM['flc_only']['rms_late']:.1f}. So the decoupled FLC alone is not enough, but on top of an inertial feedforward it is: the coupling terms are smooth over the motion and the FLCs absorb most of what the model leaves out; "
  "the off-diagonal inertia gives a small extra gain and the Coriolis term none at this speed."),
 ("How did you make sure the trajectory is fast?",
  f"The planner plays the quintic at the fastest time that keeps every joint within 95 % of Franka's speed and acceleration limits and the model torque within 90 % of the actuator limits: {T_MOVE:.2f} s instead of the template's 4 s. Inertial + Coriolis torque is {RATIO.mean():.1f} times the gravity torque on average, {RATIO.max():.1f} at the peak, and above gravity during {frac_dom:.0f} % of the move."),
 ("Why did you modify the bridge, and why not the model?",
  "The model's actuators are position servos, so a number sent as 'torque' would be a target angle. The challenge says to change the bridge and launch file, not the model. The bridge now computes the servo target that gives exactly the commanded torque (ctrl = (tau - bias)/gain), holds the torque between messages, uses an explicit integrator and does not add its own compensation."),
 ("Why the Euler integrator?",
  "The default implicit integrator treats the servo's velocity gain as extra inertia (k_v dt, 0.2 to 0.45 kg m^2) which is large compared with the wrist inertias (about 0.1-0.15). Our feedforward would then be wrong. Plain Euler uses the force at the start of the step, as the model assumes."),
 ("Why are the FLC gains so modest? Could you use higher gains?",
  "The loop has a delay of 5-10 ms (compute time plus the held command) and the template fixes the rate at 100 Hz. With the first, higher gains the real arm chattered at about 10 Hz. A model-based feedforward needs only a modest correction, so a lower gain is both stable and accurate."),
 ("What does your shim do, and why is it needed?",
  "The template calls self.timers[0]. On ROS 2 Jazzy Node.timers is a generator, so this raises a TypeError and the node never publishes. The shim above the marked line makes Node.timers return a list; the template code is untouched and it does nothing on older versions."),
 ("Why is the error large in the first 0.3 s?",
  "The simulator's arm rests about 0.1 rad (6 degrees) away from Q_START (the hand overlaps link 5 at q = 0, which pushes the wrist). The controller needs a few ticks to pull it onto the trajectory. After 0.3 s the RMS error is 0.3 degrees."),
 ("What happens after the trajectory ends? How does the arm stop?",
  "The template publishes only the gravity torque after the last plan sample, with no feedback, so leftover velocity would coast (joint damping slows it). That is why plan_trajectory appends a 3 s controlled stop (zero-velocity samples at q_end, still under feedback): the arm is brought to rest first, and the gravity hold then only has to hold a resting arm (final error 0.03 degrees, speed 0). Without the stop, with the first chattering tuning, the arm coasted up to 61 degrees."),
 ("Counterargument: a plain PD controller would do the same. Why a FLC?",
  "The challenge asks for a FLC. The FLC is a nonlinear PD: near the centre it behaves like a PD and it saturates for large errors, which keeps the torque bounded. I do not claim it is better than a tuned PD for this smooth trajectory; what the data show is that a bounded decoupled correction on top of the Lagrangian model is enough."),
 ("Counterargument: your model has no friction, so it is incomplete.",
  "True, and deliberately: the assignment asks for the Lagrangian M, C, g and expects the FLC to fill model inaccuracies. The damping test shows exactly how much of the model-alone error is friction (28.6 to 1.1 degrees). Adding b*dq would be a one-line change if desired."),
 ("What would change with a payload?",
  "The mass matrix and gravity vector would be wrong by the payload's contribution (m_payload Jv'Jv and m_payload g Jv). The FLC would have to absorb it: with bounded output it can fix small payloads but the arm would sag for a large one; the better fix is to add the payload to link 7 in the model or scale the output sets."),
]
for i, (qq, aa) in enumerate(QA, 1):
    story.append(KeepTogether([Paragraph(f"<b>Q{i}. {qq}</b>", ParagraphStyle("q", parent=body, textColor=NAVY, spaceAfter=1, spaceBefore=4)),
                               Paragraph(aa, ParagraphStyle("a", parent=body, leftIndent=12))]))

# ================================================================ 10
H1("10. Glossary")
table([["Term", "Meaning"],
       ["Lagrangian", "L = T &minus; V (kinetic minus potential energy); the equations of motion follow from it"],
       ["M(q)", "mass / inertia matrix (7x7): effective inertia at the joints, depends on the pose"],
       ["C(q,dq) dq", "Coriolis and centrifugal forces: velocity-dependent coupling between joints"],
       ["g(q)", "gravity torque vector"],
       ["Armature", "rotor inertia of a joint motor (0.1 kg m^2 in panda.xml), added to the diagonal of M"],
       ["Viscous damping", "friction torque proportional to joint speed (1 Nms/rad in panda.xml); not in the Lagrangian model"],
       ["Feedforward / feedback", "feedforward: the torque the model says the desired motion needs; feedback: the correction computed from the measured error (here: the FLC)"],
       ["Membership function", "tells how much a value belongs to a fuzzy set (0 to 1)"],
       ["Rule base / Mamdani", "the IF-THEN rules; AND = min, each conclusion = output set clipped at the rule strength, combination = max"],
       ["Centroid of area", "defuzzification: the centre of mass of the combined output set"],
       ["Decoupled MIMO", "several single-joint controllers working side by side without sharing information"],
       ["Latency", "delay between measuring the state and the torque reaching the motor (about 5-10 ms here)"],
       ["TORQUE mode", "the bridge applies the numbers as joint torques (with torque_passthrough: exactly, held until the next message)"]], [4.6, 12.0])


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("DV", 8)
    canvas.setFillColor(GREY)
    canvas.drawString(2 * cm, 1.1 * cm, "MCTR911 Challenge 3 - Team 6 - explained")
    canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"{doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(OUT_PDF, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                        title="Robotics Programming Challenge 3 - explained (Team 6)", author="Team 6")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", os.path.abspath(OUT_PDF))
