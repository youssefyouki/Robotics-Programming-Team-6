#!/usr/bin/env python3
"""
Builds RPC3_Report_Team6.pdf: the formal report to hand in with controller_node.py.

  source /opt/ros/jazzy/setup.bash && source install/setup.bash
  export PYTHONPATH=$PWD/.venv/lib/python3.12/site-packages:$PYTHONPATH
  python3 src/panda_kinematics/tools/make_submission_report3.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report3_common import *          # noqa: F401,F403

OUT_PDF = os.path.join(HERE, "..", "RPC3_Report_Team6.pdf")
m, mo, fo = RM["main"], RM["model_only"], RM["flc_only"]
frac_dom = 100 * float(np.mean(RATIO > 1.0))

# ---------------------------------------------------------------- title
story.append(Paragraph("Robotics Programming Challenge 3", ParagraphStyle("t", parent=h1, fontSize=19, leading=24, spaceAfter=2)))
story.append(Paragraph("Lagrangian dynamics and a fuzzy logic controller for the Franka Emika Panda &middot; MCTR911, Winter 2026 &middot; "
                       "<b>Team 6: The High-Speed Tracker</b>", ParagraphStyle("t2", parent=body, fontSize=10.5, leading=14, textColor=GREY)))
story.append(Paragraph("Name / ID: ______________________________", body))
P("This report documents the Lagrangian model of the Panda, its validation against MuJoCo, the fast Team 6 trajectory, the fuzzy logic controller (FLC) and the results on the simulated robot in TORQUE mode. "
  "The code is <i>panda_kinematics/controller_node.py</i> (the instructor's template; the part above the marked line holds the Challenge 1 code, the dynamics, the planner, the FLC and compute_torques). "
  "Checks are in <i>tools/check_dynamics.py</i> (6 checks, all pass) and the recorded runs in <i>challenge3_results/</i>. The robot model (panda.xml, MuJoCo Menagerie) was <b>not modified</b>; "
  "only the bridge (C++) and a launch file were changed, as the challenge asks.")

# ---------------------------------------------------------------- 1
H1("1. Lagrangian dynamics")
P("<b>Equations of motion.</b> With the Lagrangian L = T &minus; V the dynamics of the seven arm joints are")
C("M(q) ddq + C(q,dq) dq + g(q) = tau")
B(["<b>Kinetic energy.</b> Link i has mass m<sub>i</sub> and inertia I<sub>i</sub> about its centre of mass (CoM). Its CoM moves with v<sub>i</sub> = Jv<sub>i</sub> dq and the link turns with w<sub>i</sub> = Jw<sub>i</sub> dq, so "
   "T<sub>i</sub> = &frac12; m<sub>i</sub> |v<sub>i</sub>|<super>2</super> + &frac12; w<sub>i</sub><super>T</super> I<sub>i</sub> w<sub>i</sub>, and T = &frac12; dq<super>T</super> M dq with "
   "<b>M = &Sigma;<sub>i</sub> ( m<sub>i</sub> Jv<sub>i</sub><super>T</super> Jv<sub>i</sub> + Jw<sub>i</sub><super>T</super> I<sub>i</sub> Jw<sub>i</sub> )</b> (plus the rotor inertia 0.1 kg m<super>2</super> of every joint, the armature of panda.xml). "
   "M(q) is the inertia the joints feel; M ddq is the torque needed to accelerate the arm.",
   "<b>Potential energy.</b> V = &Sigma;<sub>i</sub> m<sub>i</sub> g z<sub>i</sub>(q), so <b>g(q) = &part;V/&part;q = &Sigma;<sub>i</sub> m<sub>i</sub> g Jv<sub>i</sub>[z row]</b>: the torque that holds the arm up against gravity.",
   "<b>Coriolis and centrifugal terms</b> come from how M changes with the configuration (Christoffel symbols): "
   "<b>(C dq)<sub>i</sub> = &Sigma;<sub>j,k</sub> [ &part;M<sub>ij</sub>/&part;q<sub>k</sub> &minus; &frac12; &part;M<sub>jk</sub>/&part;q<sub>i</sub> ] dq<sub>j</sub> dq<sub>k</sub></b>. They are the coupling forces between joints that appear when the arm moves fast "
   "(joint 1 turning throws joint 4)."])
fig(fig_lagrange(), 15.2, "Figure 1: from the link data and the DH frames to M, g and C.")
P("<b>Implementation.</b> compute_dynamics(q, dq, ddq) uses the DH frames of Challenge 1: joint j has the axis z<sub>j</sub> through o<sub>j</sub>, a point p on a later link moves with z<sub>j</sub> &times; (p &minus; o<sub>j</sub>) and the link turns with z<sub>j</sub>. "
  "The derivatives needed for C follow because an axis only turns when an earlier joint turns (dz<sub>j</sub>/dq<sub>k</sub> = z<sub>k</sub> &times; z<sub>j</sub>, dp/dq<sub>k</sub> = z<sub>k</sub> &times; (p &minus; o<sub>k</sub>) for k &lt; j). "
  "The link data (mass, CoM, inertia) come from panda.xml and are converted once from the MuJoCo body frames to the DH frames (a constant transform per link, verified for random q). The hand and the two fingers (open gripper) are lumped into link 7 with the parallel-axis theorem.")
P("<b>Validation against MuJoCo</b> (tools/check_dynamics.py, random q, dq, ddq inside the joint limits, poses in contact skipped):")
table([["Check", "Result"],
       ["M(q) vs MuJoCo inertia matrix (mj_fullM), 300 states", f"max difference {check_value(1)} kg m<super>2</super>"],
       ["C dq + g vs MuJoCo bias force (qfrc_bias)", f"max difference {check_value(2)} Nm"],
       ["M ddq + C dq + g vs MuJoCo inverse dynamics (mj_inverse)", f"max difference {check_value(3)} Nm"],
       ["XML body frame = DH frame x constant transform, every link", f"max difference {check_value(4)}"],
       ["power balance d(kinetic energy)/dt = dq.(tau &minus; g)", f"max difference {check_value(5)} W"],
       ["time per call of compute_dynamics (100 Hz loop needs &lt; 10 ms)", "1.7 ms"]], [10.6, 6.0])
fig(fig_dyn_validation(), 11.0, "Figure 2: the analytic torques (vertical) against MuJoCo's inverse dynamics (horizontal): all points lie on the identity line. MuJoCo's torque also contains the joint damping b dq of panda.xml (b = 1 Nms/rad), which a Lagrangian model leaves out and which was added for this comparison.")

# ---------------------------------------------------------------- 2
H1("2. Team 6 trajectory: fast")
P(f"The arm starts in the upright rest pose ({f3(q0)} rad, hand at {f3(hand0)} m) and sweeps to {f3(q1)} rad (hand at {f3(hand1)} m), a folded-back pose. "
  f"Both poses are inside the joint limits and the path has no contacts in MuJoCo. plan_trajectory is the quintic of Challenge 2 (zero speed and acceleration at both ends), but T = TRAJ_DURATION ({cn.TRAJ_DURATION:.0f} s) is treated as the <b>longest</b> allowed time: "
  f"the planner plays the move as fast as 95 % of Franka's speed and acceleration limits allow and stretches it only if the model torque would exceed 90 % of the actuator limit. The plan is followed by a 3 s controlled stop (the arm keeps being held with feedback at q_end, so it comes to a full stop). Result: <b>{T_MOVE:.2f} s</b> of motion, "
  f"fastest joint at {100 * SPEED_RATIO.max():.0f} % of its speed limit (joint {int(np.argmax(SPEED_RATIO)) + 1}), peak model torque {100 * TAU_USE.max():.0f} % of the limit (joint {int(np.argmax(TAU_USE)) + 1}).")
P(f"<b>Is it dominated by inertial and Coriolis forces?</b> The norm of the inertial + Coriolis torques divided by the norm of the gravity torque is {RATIO.mean():.2f} on average and reaches {RATIO.max():.1f}; "
  f"it is above 1 during {frac_dom:.0f} % of the move (Figure 3, right). The first half of the motion is clearly dynamics-dominated; gravity grows again as the arm folds back. Played in the template's 4 s, the same move would need only {100 * (T_MOVE / cn.TRAJ_DURATION) ** 2:.0f} % of these accelerations (they scale with 1/T<super>2</super>).")
fig(fig_fast(), 16.2, "Figure 3: the planned motion, the speed of every joint as a percentage of its limit, and the torque norms (inertial + Coriolis against gravity).")

# ---------------------------------------------------------------- 3
H1("3. Fuzzy logic controller (decoupled MIMO)")
P("The MIMO controller is seven independent FLCs, one per joint, evaluated together. Each has two inputs, the tracking error e = q<sub>des</sub> &minus; q and its rate of change de = dq<sub>des</sub> &minus; dq, and one output, a correction torque.")
B([f"<b>Membership functions.</b> Five triangular sets LN, MN, S, MP, LP on the normalised range [&minus;1, 1] with centres &minus;1, &minus;0.5, 0, 0.5, 1 for e, de and the output. Neighbouring triangles cross at 0.5, so the memberships of any input add up to 1 and the output changes smoothly "
   f"(no chattering); the outer sets saturate. <b>Boundaries:</b> e is divided by 0.05 rad (about 3&deg;: an error of this size already counts as \"large\"), de by 1.5 rad/s (the planned peak joint speeds are 1&ndash;2.5 rad/s), and the output is multiplied by 20 Nm "
   "(joints 1&ndash;4, about a quarter of the 87 Nm limit) or 8 Nm (wrist joints, two thirds of the 12 Nm limit). The values were tuned in simulation with the real loop's latency (section 4).",
   "<b>Rule base.</b> 25 rules \"IF e is A AND de is B THEN u is C\" with C = A + B limited to the five sets (Figure 4): a PD-like surface. Examples: e LP and de S &rarr; LP; e LP and de LN &rarr; S (the joint is already catching up); e S and de S &rarr; S.",
   "<b>Inference</b> (Mamdani): AND = min; each output set is clipped at the strength of its strongest rule; the clipped sets are united with max. <b>Defuzzification:</b> centroid of area of the united set, sampled at 201 points.",
   "<b>Control law</b> (compute_torques): tau = M(q) ddq<sub>des</sub> + C(q,dq) dq + g(q) + FLC(e, de), saturated at the actuator limits. The Lagrangian model does the heavy lifting; the FLC corrects what it does not contain."])
fig(fig_flc_design(), 16.4, "Figure 4: the membership functions, the 25-rule table and the resulting control surface of joint 1.")
mu_e, mu_d, fired, strength, combined, u = worked_example()
P(f"<b>One inference step.</b> For joint 1 with e = +0.03 rad (normalised {WORK_E_N:+.1f}) and de = &minus;0.45 rad/s (normalised {WORK_DE_N:+.1f}): e belongs to MP (0.8) and LP (0.2), de to MN (0.6) and S (0.4). Four rules fire: "
  + "; ".join(f"({a}, {b}) &rarr; {c_} with strength {w:.1f}" for a, b, c_, w, _ in fired)
  + f". The output set MP gets max(0.4, 0.2) = 0.4. The clipped sets S (0.6), MP (0.4) and LP (0.2) are united and their centroid is u = {u:+.3f}, i.e. a correction of {u * cn.FLC_U_SCALE[0]:+.2f} Nm (Figure 5).")
fig(fig_worked(), 16.4, "Figure 5: the inference step above: memberships of the two inputs and the clipped, united output set with its centroid.")

# ---------------------------------------------------------------- 4
H1("4. Torque control in the simulator")
B(["<b>Bridge.</b> The Panda's actuators in panda.xml are position servos (force = k<sub>p</sub> ctrl &minus; k<sub>p</sub> q &minus; k<sub>v</sub> dq). New bridge option <i>torque_passthrough</i> (default off, other robots unaffected): the received numbers are joint torques and the bridge sets "
   "ctrl = (tau &minus; bias)/gain with the model's own gain and bias, so the actuator produces <b>exactly tau</b>; the torque is held until the next message (the old TORQUE mode cleared it after one 1 ms step and added gravity compensation of its own); the integrator is switched to Euler "
   "(an implicit integrator would add k<sub>v</sub> dt = 0.2&ndash;0.45 kg m<super>2</super> of artificial inertia to every joint); until the first command the initial pose is held; writes stay inside the 8 actuators. New launch file <i>panda_torque.py</i>. The model is untouched.",
   "<b>Template on ROS 2 Jazzy.</b> The template's <i>self.timers[0]</i> fails because Node.timers is a generator in Jazzy; a 4-line shim above the marked line makes it a list, so the template code below the line is unchanged "
   "(only Q_START and Q_END were set).",
   "<b>Loop latency.</b> Reading the state, computing the torque (1.7 ms for the dynamics) and applying it takes about 5&ndash;10 ms. A first tuning with zero latency (torque scale 60 Nm, error scale 0.05 rad) chattered at about 10 Hz on the real bridge (2.6&deg; RMS, and the arm then coasted up to 61&deg; during the feedback-free hold). "
   "An offline simulation of the loop (tools/offline_sim.py) with 8 ms of latency was used to retune: lower torque scales and a larger de scale gave a stable controller that works for latencies of 4&ndash;12 ms."])
fig(fig_architecture(), 15.5, "Figure 6: signal flow of the torque-controlled setup.")

# ---------------------------------------------------------------- 5
H1("5. Results on the simulated robot")
P("All runs use the real C++ bridge in TORQUE mode, the same trajectory and the same start. Errors are joint errors over the planned motion; \"after start-up\" ignores the first 0.3 s, in which the arm (resting about 0.1 rad away from Q_START) converges onto the trajectory.")
fig(fig_tracking(), 15.8, "Figure 7: desired and actual trajectory of the submitted controller (model + FLC), the commanded torques and the hand error. The dotted line marks the end of the motion; the plan then holds q_end for 3 s (controlled stop) before the template's gravity-only hold takes over.")
rows = [["Controller", "RMS error all [deg]", "RMS after start-up [deg]", "max after start-up [deg]", "hand error RMS [mm]"]]
for k, name in RUNS:
    rng = f" ({RM[k]['lo']:.2f}-{RM[k]['hi']:.2f}, {RM[k]['n']} runs)" if RM[k]['n'] > 1 else " (1 run)"
    rows.append([name, f"{RM[k]['rms']:.2f}", f"{RM[k]['rms_late']:.2f}{rng}", f"{RM[k]['mx_late']:.1f}", f"{RM[k]['ee_rms']:.1f}"])
table(rows, [5.4, 2.2, 4.6, 2.6, 1.8])
fig(fig_ablation(), 16.0, "Figure 8: the six controllers on the same trajectory (left: RMS error, log scale; right: error norm over time).")
B([f"<b>Does the FLC reduce the tracking error? Yes.</b> The Lagrangian model alone leaves {mo['rms_late']:.1f}&deg; RMS (the arm is open-loop unstable and the model has no joint damping); adding the FLC gives {m['rms_late']:.2f}&deg; RMS and {m['mx_late']:.1f}&deg; at most after start-up, "
   f"hand error {m['ee_rms']:.1f} mm RMS ({m['ee_max']:.1f} mm max).",
   "<b>What happens without the FLC?</b> The error grows steadily and the arm ends 30&ndash;90&deg; off per joint. The cause was isolated offline (tools/offline_sim.py): adding only the joint damping b dq of panda.xml (1 Nms/rad, not part of a Lagrangian model) to the open-loop torque brings the model-alone error from "
   "28.6&deg; to 1.1&deg; RMS from a perfect start. So the \"small model inaccuracy\" the FLC fills in is mainly this viscous friction, whose effect is amplified because the upright arm is an unstable equilibrium.",
   f"<b>FLC alone</b> (no model): {fo['rms_late']:.1f}&deg; RMS; joint 2 is {fo['pj_late'][1]:.1f}&deg; off because nothing compensates gravity."])

# ---------------------------------------------------------------- 6
H1("6. Debate focus: coupling at high speed")
P(f"<b>How did the FLC handle the coupling between joints, and was a decoupled FLC enough?</b> The seven FLCs are independent; the coupling is handled by the model feedforward. The coupling is large here: the off-diagonal inertia terms make up about {100 * OFF_SHARE:.0f} % of the inertial torque "
  f"and the Coriolis/centrifugal torque peaks at {CORI_PEAK:.1f} Nm (joint 2: {CORI_PEAK_J[1]:.1f} Nm, joint 4: {CORI_PEAK_J[3]:.1f} Nm, joint 1: {CORI_PEAK_J[0]:.1f} Nm). To see what matters, the model was degraded while keeping the same decoupled FLC (mean RMS after start-up over repeated runs): "
  f"full model {RM['main']['rms_late']:.2f}&deg; ({RM['main']['lo']:.2f}&ndash;{RM['main']['hi']:.2f}); <b>Coriolis/centrifugal term removed</b> {RM['no_coriolis']['rms_late']:.2f}&deg; ({RM['no_coriolis']['lo']:.2f}&ndash;{RM['no_coriolis']['hi']:.2f}), not measurably different; "
  f"<b>inertia coupling removed</b> (diagonal M) {RM['decoupled']['rms_late']:.2f}&deg; (about {100 * (RM['decoupled']['rms_late'] / RM['main']['rms_late'] - 1):.0f} % worse); gravity only {RM['gravity_only']['rms_late']:.1f}&deg;; no model {RM['flc_only']['rms_late']:.1f}&deg;. "
  "<b>Conclusion:</b> a decoupled FLC <i>alone</i> is not enough (6&deg;), but a decoupled FLC on top of the inertial feedforward M ddq<sub>des</sub> is. The coupling torques are large but smooth over the 1.1 s motion, so the independent FLCs absorb most of what the model leaves out; "
  "the diagonal inertia terms already remove most of the error (3.0&deg; &rarr; 0.5&deg;), the off-diagonal coupling brings a further small gain, and the Coriolis term makes no measurable difference at this speed.")
fig(fig_perjoint(), 10.0, "Figure 9: per-joint RMS error after start-up for the controllers with a model or with the FLC alone.")

# ---------------------------------------------------------------- 7
H1("7. Conclusion and limitations")
B(["The Lagrangian model of the Panda (M, C dq + g) agrees with MuJoCo to 1e-8 Nm; the controller is model feedforward plus a decoupled 25-rule FLC (triangular sets, Mamdani, centroid).",
   f"On the fast {T_MOVE:.2f} s Team 6 trajectory it tracks to {m['rms_late']:.2f}&deg; RMS ({m['ee_rms']:.1f} mm at the hand) on the real bridge in TORQUE mode, against {mo['rms_late']:.0f}&deg; for the model alone and {fo['rms_late']:.1f}&deg; for the FLC alone.",
   "Limitations: the start-up error (the arm rests about 6&deg; from the first desired pose, so the first 0.3 s are an initial convergence); the template's own hold (gravity compensation only, no feedback) would let leftover velocity coast, so the planner appends a 3 s controlled stop (zero-velocity samples at q_end, still under feedback): the arm ends at rest, "
   f"{np.abs(R['main']['final']).max():.2f}&deg; from the goal; the controllers were run 1&ndash;4 times on the bridge and the submitted one varies by about &plusmn;0.15&deg; between runs (timing jitter of the 100 Hz loop), which is why differences below about 0.2&deg; are not interpreted; the model has no friction term by design.",
   "The FLC scales are tuned for this trajectory and the nominal robot; a payload or a 20 % mass change would shift the model error and call for larger output sets or an adaptive term."])
H2("How to run")
C("colcon build --packages-select mujoco_ros2 panda_kinematics && source install/setup.bash\n"
  "ros2 launch mujoco_ros2 panda_torque.py              # bridge, TORQUE mode, unmodified Panda\n"
  "ros2 run panda_kinematics controller_node            # second terminal\n"
  "python3 src/panda_kinematics/tools/check_dynamics.py # dynamics vs MuJoCo (needs mujoco python)")


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("DV", 8)
    canvas.setFillColor(GREY)
    canvas.drawString(2 * cm, 1.1 * cm, "MCTR911 - Robotics Programming Challenge 3 - Panda dynamics and fuzzy control - Team 6")
    canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"{doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(OUT_PDF, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.7 * cm, bottomMargin=1.8 * cm,
                        title="RPC 3 - Panda Lagrangian dynamics and fuzzy control (Team 6)", author="Team 6")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", os.path.abspath(OUT_PDF))
