#!/usr/bin/env python3
"""
Shared helpers for the two Challenge 2 PDFs (make_report.py = study guide, make_submission_report.py =
report to hand in): fonts, styles, layout helpers, all numbers (computed from trajectory_node.py and
from the recorded runs in challenge2_results/) and all figures.
"""
import gzip
import os
import re
import sys
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from panda_kinematics import trajectory_node as T            # noqa: E402
from panda_kinematics import trajectory_experiment as E      # noqa: E402

RES = os.path.join(HERE, "..", "challenge2_results")
TMP = tempfile.mkdtemp()

# ---------------------------------------------------------------- fonts / styles
FD = "/usr/share/fonts/truetype/dejavu/"
for name, f in [("DV", "DejaVuSans.ttf"), ("DV-B", "DejaVuSans-Bold.ttf"),
                ("DV-I", "DejaVuSans-Oblique.ttf"), ("DV-BI", "DejaVuSans-BoldOblique.ttf"),
                ("DVM", "DejaVuSansMono.ttf")]:
    pdfmetrics.registerFont(TTFont(name, FD + f))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

NAVY, TEAL, GREY = colors.HexColor("#1f3a5f"), colors.HexColor("#1b7f79"), colors.HexColor("#555555")
body = ParagraphStyle("body", fontName="DV", fontSize=9.6, leading=14, spaceAfter=5)
small = ParagraphStyle("small", parent=body, fontSize=8.4, leading=11.5, textColor=GREY)
h1 = ParagraphStyle("h1", fontName="DV-B", fontSize=16, leading=20, textColor=NAVY, spaceBefore=6, spaceAfter=8)
h2 = ParagraphStyle("h2", fontName="DV-B", fontSize=11.5, leading=15, textColor=TEAL, spaceBefore=8, spaceAfter=4)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=14, bulletIndent=3, spaceAfter=2)
code = ParagraphStyle("code", fontName="DVM", fontSize=8.4, leading=11, backColor=colors.HexColor("#f2f4f7"),
                      borderPadding=5, spaceBefore=3, spaceAfter=7, leftIndent=4)
cell = ParagraphStyle("cell", parent=body, fontSize=8.6, leading=11.2, spaceAfter=0)
cellb = ParagraphStyle("cellb", parent=cell, fontName="DV-B")
caption = ParagraphStyle("caption", parent=small, alignment=TA_CENTER, spaceAfter=8)

story = []


def H1(t):
    story.append(Paragraph(t, h1))


def H2(t):
    story.append(Paragraph(t, h2))


def P(t):
    story.append(Paragraph(t, body))


def B(items):
    for it in items:
        story.append(Paragraph(it, bullet, bulletText="•"))
    story.append(Spacer(1, 3))


def C(t):
    story.append(Preformatted(t, code))


def box(title, text, bg="#e8f4f3", edge="#1b7f79"):
    inner = [Paragraph(f"<b>{title}</b>", cell), Spacer(1, 2)] + \
            [Paragraph(x, cell) for x in (text if isinstance(text, list) else [text])]
    t = Table([[inner]], colWidths=[16.6 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg)),
                           ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor(edge)),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story.append(t)
    story.append(Spacer(1, 7))


def table(rows, widths, header=True, size=8.6):
    data = [[Paragraph(str(c), cellb if (header and i == 0) else cell) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b8c0cc")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe7f1")))
    t.setStyle(TableStyle(st))
    story.append(t)
    story.append(Spacer(1, 7))


def fig(path, width_cm, cap):
    from PIL import Image as PI
    w, h = PI.open(path).size
    img = Image(path, width=width_cm * cm, height=width_cm * cm * h / w)
    story.append(KeepTogether([img, Paragraph(cap, caption)]))


def f3(v):
    return "[" + ", ".join(f"{x:+.3f}" for x in v) + "]"


# ---------------------------------------------------------------- numbers from the code
q0, q1 = T.Q_START, T.Q_END
hand0, hand1 = T.forward_kinematics(q0)[:3, 3], T.forward_kinematics(q1)[:3, 3]
SHOULDER = np.array([0, 0, 0.333])
MAXREACH = np.hypot(0.316, 0.0825) + np.hypot(0.384, 0.0825) + np.hypot(0.088, 0.107)
reach0, reach1 = np.linalg.norm(hand0 - SHOULDER) / MAXREACH, np.linalg.norm(hand1 - SHOULDER) / MAXREACH
dt = T.TRAJ_DT
t_main, qm, dqm, ddqm = T.plan_trajectory(q0, q1, T.TRAJ_DURATION, dt)
info = T.check_trajectory(qm, dqm, ddqm)
t_min = T.minimum_time(q0, q1)
sig0, w0 = T.jacobian_metrics(q0)
sig1, w1 = T.jacobian_metrics(q1)
q_up = np.array([0.0, 0.0, 0.0, -0.0698, 0.0, 0.0, 0.785])
sig_up, w_up = T.jacobian_metrics(q_up)
delta = q1 - q0


def parse(rel):
    """Numbers from challenge2_results/<rel>/metrics.txt (written by tracking_logger)."""
    txt = open(os.path.join(RES, rel, "metrics.txt")).read()
    num = lambda pat: float(re.search(pat, txt).group(1))
    arr = lambda pat: np.array([float(x) for x in re.search(pat, txt, re.S).group(1).split()])
    return dict(
        rate=num(r"publish rate achieved: (\d+)"),
        ee_rms=num(r"rms ([\d.]+) mm"), ee_max=num(r"max ([\d.]+) mm"),
        settled_ee=num(r"settled hand error vs goal: ([\d.]+) mm"),
        settled_joint=arr(r"settled joint error vs goal \[deg\]: \[(.*?)\]"),
        start_offset=arr(r"main motion starts \[deg\]: \[(.*?)\]"),
        delivered=arr(r"delivered fraction.*?: \[(.*?)\]"),
        lag=num(r"lag behind the plan \(joint \d\): (\d+) ms"),
        max_joint=arr(r"per-joint max error \[deg\]: \[(.*?)\]"))


# extended runner (set C): clean comparison, started from Q_START
mP, mV = parse("runner_vis20/POSITION"), parse("runner_vis20/VELOCITY")
sens = {hz: parse(f"runner_vis{hz}/VELOCITY") for hz in (5, 20, 30, 60)}
# instructor's node as submitted (set A, TRAJ_DT = 0.001) and with the template's own TRAJ_DT = 0.05 (set B)
tP1k, tV1k = parse("template_1khz/POSITION"), parse("template_1khz/VELOCITY")
tP20, tV20 = parse("template_20hz/POSITION"), parse("template_20hz/VELOCITY")


def load(rel):
    path = os.path.join(RES, rel, "tracking.csv")
    d = np.loadtxt(path if os.path.exists(path) else path + ".gz", delimiter=",")   # csv or csv.gz
    return dict(wall=d[:, 0], t=d[:, 1], phase=d[:, 2].astype(int), plan=d[:, 3:10], act=d[:, 10:17])


def final_deg(rel):
    """Settled joint angles (deg) of a recorded run: mean of the last 0.5 s."""
    d = load(rel)
    n = max(2, int(0.5 / (d["t"][1] - d["t"][0])))
    return np.degrees(d["act"][-n:].mean(axis=0))


# ---------------------------------------------------------------- figures
def fig_architecture():
    fig_, ax = plt.subplots(figsize=(10, 3.8))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.8); ax.axis("off")

    def b(x, y, w, h, text, c):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=c, ec="#333", lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)

    b(0.2, 1.4, 2.6, 1.3, "trajectory_node\n(Python, our code)\nplans, then plays\n1000 messages / s", "#cfe8e6")
    b(3.9, 1.4, 2.6, 1.3, "mujoco_node (the bridge)\nC++ ROS 2 node\nfills the actuator\ncontrol values", "#dfe7f1")
    b(7.3, 1.4, 2.5, 1.3, "MuJoCo physics\nPanda + 7 position\nservos (spring-damper)", "#f6e7c8")
    b(3.9, 0.05, 2.6, 0.8, "tracking_logger\n(records + compares)", "#e9e3f3")
    ax.annotate("", xy=(3.9, 2.25), xytext=(2.8, 2.25), arrowprops=dict(arrowstyle="->", lw=1.6))
    ax.text(3.35, 2.45, "/joint_commands", ha="center", fontsize=8)
    ax.annotate("", xy=(7.3, 2.25), xytext=(6.5, 2.25), arrowprops=dict(arrowstyle="->", lw=1.6))
    ax.annotate("", xy=(2.8, 1.75), xytext=(3.9, 1.75), arrowprops=dict(arrowstyle="->", lw=1.6, color="#1b7f79"))
    ax.text(3.35, 1.45, "/joint_state", ha="center", fontsize=8, color="#1b7f79")
    ax.annotate("", xy=(5.2, 1.4), xytext=(5.2, 0.85), arrowprops=dict(arrowstyle="<-", lw=1.2, color="#7a5fb3"))
    ax.text(5.35, 1.05, "listens to both topics", fontsize=7.5, color="#7a5fb3")
    ax.text(5, 3.45, "POSITION mode: ctrl = command          VELOCITY mode: ctrl = ctrl + command / 1000",
            ha="center", fontsize=9, color="#333")
    p = os.path.join(TMP, "arch.png"); fig_.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig_)
    return p


def fig_pose():
    fig_, ax = plt.subplots(figsize=(7.4, 5))
    for qq, c, lab in [(q0, "#1b7f79", "start (low)"), (q0 + 0.5 * delta, "#aaaaaa", "halfway"),
                       (q1, "#c0392b", "end (high reach)")]:
        fr = T._frames(qq)
        pts = np.array([f[:3, 3] for f in fr])
        ax.plot(pts[:, 0], pts[:, 2], "-o", color=c, lw=3, ms=5, label=lab)
    ax.plot([0, 0.858 * np.sin(np.linspace(0, np.pi, 100))[50]], [0.333, 0.333], alpha=0)
    th = np.linspace(-np.pi / 2, np.pi / 2, 100)
    ax.plot(MAXREACH * np.cos(th), 0.333 + MAXREACH * np.sin(th), "k:", lw=1, label="max reach sphere (side cut)")
    ax.axhline(0, color="k", lw=2)
    ax.set_aspect("equal"); ax.set_xlabel("x forward [m]"); ax.set_ylabel("z up [m]")
    ax.set_title("Panda from the side: joint positions at start, halfway and end"); ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    p = os.path.join(TMP, "pose.png"); fig_.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig_)
    return p


def fig_lever():
    fig_, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.plot(0, 0, "ko", ms=8); ax.annotate("", xy=(2.2, 1.4), xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=2, color="#1f3a5f"))
    ax.text(0.9, 0.35, "lever arm  r = p - o", color="#1f3a5f", fontsize=9, rotation=32)
    ax.plot(2.2, 1.4, "o", color="#c0392b", ms=9)
    ax.annotate("", xy=(1.2, 2.5), xytext=(2.2, 1.4), arrowprops=dict(arrowstyle="->", lw=2.4, color="#c0392b"))
    ax.text(0.2, 2.45, "hand velocity\n= z x r", color="#c0392b", fontsize=9)
    th = np.linspace(-0.2, 1.3, 50); ax.plot(0.7 * np.cos(th), 0.7 * np.sin(th) - 0.1, color="#1b7f79", lw=1.6)
    ax.text(0.75, -0.5, "joint spins at 1 rad/s\n(axis z points at you)", fontsize=8.5, color="#1b7f79")
    ax.set_xlim(-0.5, 3); ax.set_ylim(-1, 3); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Top view: one spinning joint moves the hand sideways", fontsize=10)
    p = os.path.join(TMP, "lever.png"); fig_.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig_)
    return p


def fig_quintic():
    tau = np.linspace(0, 1, 400)
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    ds = 30 * tau**2 - 60 * tau**3 + 30 * tau**4
    dds = 60 * tau - 180 * tau**2 + 120 * tau**3
    cub = 3 * tau**2 - 2 * tau**3
    dds_cub = 6 - 12 * tau
    fig_, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    ax[0].plot(tau, s, label="quintic"); ax[0].plot(tau, cub, "--", label="cubic"); ax[0].set(title="position s(tau)", xlabel="tau = t/T")
    ax[1].plot(tau, ds); ax[1].plot(tau, 6 * tau - 6 * tau**2, "--"); ax[1].set(title="speed ds/dtau (peak 1.875)", xlabel="tau")
    ax[2].plot(tau, dds); ax[2].plot(tau, dds_cub, "--"); ax[2].set(title="acceleration (quintic starts/ends at 0)", xlabel="tau")
    ax[0].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    p = os.path.join(TMP, "quintic.png"); fig_.tight_layout(); fig_.savefig(p, dpi=150); plt.close(fig_)
    return p


def fig_traj():
    fig_, ax = plt.subplots(1, 3, figsize=(11, 3.3))
    for j in range(7):
        if abs(delta[j]) > 1e-9:
            ax[0].plot(t_main, np.degrees(qm[:, j]), label=f"q{j+1}")
            ax[1].plot(t_main, dqm[:, j], label=f"q{j+1}")
            ax[2].plot(t_main, ddqm[:, j], label=f"q{j+1}")
    ax[0].set(title="joint positions [deg]", xlabel="t [s]"); ax[1].set(title="joint speeds [rad/s]", xlabel="t [s]")
    ax[2].set(title="joint accelerations [rad/s^2]", xlabel="t [s]")
    ax[0].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    p = os.path.join(TMP, "traj.png"); fig_.tight_layout(); fig_.savefig(p, dpi=150); plt.close(fig_)
    return p


def fig_sing():
    s_vals = np.linspace(0, 1, 120)
    path = np.array([T.jacobian_metrics(q0 + a * delta) for a in s_vals])
    path_up = np.array([T.jacobian_metrics(q0 + a * (q_up - q0)) for a in s_vals])
    fig_, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    for a_, k, thr, name in [(ax[0], 0, T.SIGMA_MIN_ALLOWED, "sigma_min(J)  [m per rad/s]"),
                             (ax[1], 1, T.W_MIN_ALLOWED, "manipulability w")]:
        a_.plot(s_vals, path[:, k], lw=2, label="our Team 6 path")
        a_.plot(s_vals, path_up[:, k], lw=2, ls="--", color="#c0392b", label="path to arm stretched straight up")
        a_.axhline(thr, color="k", ls=":", label=f"guard threshold {thr}")
        a_.set(xlabel="fraction of the move completed", title=name); a_.grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    p = os.path.join(TMP, "sing.png"); fig_.tight_layout(); fig_.savefig(p, dpi=150); plt.close(fig_)
    return p


def fig_modes():
    P_, V_ = load("runner_vis20/POSITION"), load("runner_vis20/VELOCITY")
    fig_, ax = plt.subplots(2, 2, figsize=(11, 6.6))
    for k, (d, name) in enumerate([(P_, "POSITION"), (V_, "VELOCITY")]):
        main = d["phase"] == 2
        tm = d["t"][main] - d["t"][main][0]
        for j, c in [(3, "C3"), (5, "C5")]:
            ax[0, k].plot(tm, np.degrees(d["plan"][main, j]), color=c, lw=2, label=f"plan q{j+1}")
            ax[0, k].plot(tm, np.degrees(d["act"][main, j]), color=c, ls="--", lw=1.6, label=f"actual q{j+1}")
        ax[0, k].set(title=f"{name}: planned vs actual (main motion)", xlabel="t [s]", ylabel="deg")
        ax[0, k].legend(fontsize=8)
        err = np.degrees(d["act"] - d["plan"])
        for j in range(7):
            ax[1, k].plot(d["t"], err[:, j], label=f"q{j+1}")
        ax[1, k].set(title=f"{name}: joint error (actual - plan), whole run", xlabel="plan time [s]", ylabel="deg",
                     ylim=(-8, 12))
        ax[1, k].axvspan(7, 11, color="green", alpha=0.07)
        ax[1, k].text(7.1, 10, "main motion", fontsize=8)
        ax[1, k].legend(fontsize=7, ncol=4, loc="lower right")
        for a in (ax[0, k], ax[1, k]):
            a.grid(alpha=0.3)
    p = os.path.join(TMP, "modes.png"); fig_.tight_layout(); fig_.savefig(p, dpi=150); plt.close(fig_)
    return p


def fig_sens():
    fig_, ax = plt.subplots(figsize=(6.4, 3.3))
    hz = sorted(sens)
    vals = [100 * sens[h]["delivered"].mean() for h in hz]
    bars = ax.bar([str(h) for h in hz], vals, color=["#1b7f79", "#2a9d8f", "#e9c46a", "#c0392b"])
    for b_, v in zip(bars, vals):
        ax.text(b_.get_x() + b_.get_width() / 2, v + 1.5, f"{v:.1f} %", ha="center", fontsize=9)
    ax.axhline(100, color="k", ls=":")
    ax.set(xlabel="bridge render rate [Hz]", ylabel="% of commanded motion delivered", ylim=(0, 112),
           title="VELOCITY mode: motion that actually reaches the arm")
    p = os.path.join(TMP, "sens.png"); fig_.tight_layout(); fig_.savefig(p, dpi=150); plt.close(fig_)
    return p


def fig_template():
    runs = [("goal Q_END", np.degrees(T.Q_END), "#333333"),
            ("POSITION, 1 kHz", final_deg("template_1khz/POSITION"), "#1b7f79"),
            ("VELOCITY, 1 kHz", final_deg("template_1khz/VELOCITY"), "#e9c46a"),
            ("VELOCITY, 20 Hz", final_deg("template_20hz/VELOCITY"), "#c0392b")]
    fig_, ax = plt.subplots(1, 2, figsize=(11.5, 3.9))
    w = 0.2
    for i, (name, vals, c) in enumerate(runs):
        ax[0].bar(np.arange(1, 8) + (i - 1.5) * w, vals, w, label=name, color=c)
    ax[0].set(title="Where the arm ends up (template node)", xlabel="joint", ylabel="final angle [deg]")
    ax[0].legend(fontsize=8)
    for rel, name, c in [("template_1khz/POSITION", "POSITION, 1 kHz", "#1b7f79"),
                         ("template_1khz/VELOCITY", "VELOCITY, 1 kHz", "#e9c46a"),
                         ("template_20hz/VELOCITY", "VELOCITY, 20 Hz", "#c0392b")]:
        d = load(rel)
        step = max(1, len(d["t"]) // 600)
        ee_a = np.array([T.forward_kinematics(q)[:3, 3] for q in d["act"][::step]])
        ee_p = np.array([T.forward_kinematics(q)[:3, 3] for q in d["plan"][::step]])
        ax[1].plot(d["t"][::step], 1000 * np.linalg.norm(ee_a - ee_p, axis=1), label=name, color=c, lw=2)
    ax[1].set(title="Hand position error vs the plan", xlabel="plan time [s]", ylabel="mm")
    ax[1].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    p = os.path.join(TMP, "template.png"); fig_.tight_layout(); fig_.savefig(p, dpi=150); plt.close(fig_)
    return p


# ---------------------------------------------------------------- worked Jacobian example
frames1 = T._frames(q1)
p_h = frames1[-1][:3, 3]
z0, o0 = frames1[0][:3, 2], frames1[0][:3, 3]
z1, o1 = frames1[1][:3, 2], frames1[1][:3, 3]
z3, o3 = frames1[3][:3, 2], frames1[3][:3, 3]
J1 = T.compute_jacobian(q1)
r3 = p_h - o3
col4 = np.cross(z3, r3)
assert np.allclose(col4, J1[:, 3])

