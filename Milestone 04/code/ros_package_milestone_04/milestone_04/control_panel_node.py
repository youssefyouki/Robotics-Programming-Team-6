#!/usr/bin/env python3
"""
Single-window control panel for the sorting cell (Milestone 04) - tkinter GUI + ROS 2 node.

Left   live view of the simulation (streamed from the simulator node as JPEG frames).
       Mouse: left-drag orbit, right-drag or wheel zoom, middle-drag or Shift+left-drag pan,
       double-click resets the view.
       The planned tool-point path is drawn in the view (blue = joint-space moves, orange =
       task-space straight lines, purple = manual moves) with a green trail of where the tool
       point actually went (toggles below the view).
       Below the view: Simulation (Run / Pause / Reset / speed), Program (Start sorting / Stop /
       Home, sorted status per colour), the robot state readout, and Controller & validation:
       choose the motor-level controller (also while the robot moves), see its control law, the
       live tracking errors and the validation result of every finished trajectory segment.
Right  Joint angles (sliders in degrees + gripper) and Cartesian control (tool point x, y, z,
       pitch, roll; joint-space or straight-line moves; jog; presets above the belt and bins).

World frame: the arm points along -X when all joints are 0, so positions in front of the robot
(over the conveyor) have negative x.

  Publishes  : m4/command, m4/sim_control, m4/camera, m4/controller
  Subscribes : m4/state, m4/status, m4/boxes, m4/image, m4/controller_info, m4/monitor, m4/validation
"""
import io
import json
import os
import threading
import time
import tkinter as tk
from tkinter import ttk

import numpy as np
import rclpy
from PIL import Image, ImageTk
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64MultiArray, String

from milestone_03 import kinematics as K
from milestone_04 import cell as C, planning as P, tcp_kinematics as TK, topics as T

BG, PANEL, FG, ACCENT, GOOD, WARN = "#1e1e2e", "#313244", "#cdd6f4", "#89b4fa", "#a6e3a1", "#f38ba8"
JOINT_LABELS = ["base_rotation", "pitch", "elbow", "wrist_pitch", "wrist_roll"]
CONTROLLER_LABELS = {"Position servo": "servo", "PD + gravity": "pd_gravity", "Computed torque": "computed_torque"}
CONTROLLER_SHORT = {"servo": "servo", "pd_gravity": "PD+g", "computed_torque": "CTC"}


class PanelNode(Node):
    def __init__(self):
        super().__init__("control_panel")
        self.pub_cmd = self.create_publisher(String, T.COMMAND, 20)
        self.pub_sim = self.create_publisher(String, T.SIM_CONTROL, 20)
        self.pub_cam = self.create_publisher(String, T.CAMERA, 50)
        self.pub_ctrl = self.create_publisher(String, T.CONTROLLER, 10)
        self.info = None                               # description of the active controller
        self.monitor = None                            # (t, phase, max joint error [rad], TCP error [mm])
        self.valid = []                                # validation results of finished segments
        self.q = np.zeros(5)
        self.grip = 0.0
        self.sim_t = 0.0
        self.last_state = 0.0
        self.status = "waiting for the simulator..."
        self.boxes = {c: False for c in C.COLORS}
        self.frame = None                              # latest JPEG bytes from the view node
        self.frame_event = threading.Event()
        self.create_subscription(Float64MultiArray, T.STATE, self._on_state, 5)
        self.create_subscription(String, T.STATUS, self._on_status, 10)
        self.create_subscription(Float64MultiArray, T.BOXES, self._on_boxes, 5)
        self.create_subscription(CompressedImage, T.IMAGE, self._on_image, qos_profile_sensor_data)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(String, T.CONTROLLER_INFO, self._on_info, latched)
        self.create_subscription(Float64MultiArray, T.MONITOR, self._on_monitor, 10)
        self.create_subscription(String, T.VALIDATION, self._on_valid, 50)

    def _on_state(self, msg):
        self.q = np.array(msg.data[1:6])
        self.grip = msg.data[11]
        self.sim_t = msg.data[0]
        self.last_state = time.time()

    def _on_status(self, msg):
        self.status = msg.data

    def _on_boxes(self, msg):
        d = msg.data
        self.boxes = {c: bool(d[5 + 5 * i]) for i, c in enumerate(C.COLORS)}

    def _on_info(self, msg):
        self.info = json.loads(msg.data)

    def _on_monitor(self, msg):
        self.monitor = tuple(msg.data)

    def _on_valid(self, msg):
        self.valid.append(json.loads(msg.data))

    def select_controller(self, name):
        self.pub_ctrl.publish(String(data=name))

    def _on_image(self, msg):
        self.frame = bytes(msg.data)
        self.frame_event.set()

    def command(self, **kw):
        self.pub_cmd.publish(String(data=json.dumps(kw)))

    def sim(self, **kw):
        self.pub_sim.publish(String(data=json.dumps(kw)))

    def camera(self, **kw):
        self.pub_cam.publish(String(data=json.dumps(kw)))


class PanelApp:
    # view quality: largest frame requested from the renderer (None = the whole window).
    # Fewer pixels = smoother; the Tk thread has to draw every pixel of every frame.
    VIEW_MODES = {"fit window": None, "high (2.0 MP)": 2.0e6, "medium (1.3 MP)": 1.3e6, "low (0.6 MP)": 0.6e6}

    def __init__(self, node):
        self.node = node
        self.paused = False
        self.synced = False
        self._photo = None
        self._photo_size = None
        self._img_pos = None
        self._drag = None
        self.canvas_size = (640, 480)
        self.ready = None                              # decoded frame waiting to be shown
        self.ready_lock = threading.Lock()
        self.disp_cost = 0.03                          # measured seconds to draw one frame
        self.next_show = 0.0
        self._fps_sent = None
        self.running = True
        self._valid_shown = 0
        self._info_shown = None
        self.worst = {}
        self._cam_acc = {"rotate": [0.0, 0.0], "pan": [0.0, 0.0], "zoom": [0.0, 0.0]}
        self._resize_job = None
        self._sent_size = None

        self.root = tk.Tk()
        self.root.withdraw()                           # stay hidden until the final size is set (avoids a lost first paint)
        self.root.title("Koch v1.1 sorting cell")
        self.root.configure(bg=BG)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TCombobox", fieldbackground=PANEL, background=PANEL, foreground=FG,
                        arrowcolor=FG, selectbackground=PANEL, selectforeground=FG)
        style.map("TCombobox", fieldbackground=[("readonly", PANEL)], foreground=[("readonly", FG)],
                  selectbackground=[("readonly", PANEL)], selectforeground=[("readonly", FG)],
                  background=[("readonly", PANEL)])
        self.root.option_add("*TCombobox*Listbox.background", PANEL)
        self.root.option_add("*TCombobox*Listbox.foreground", FG)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.left = tk.Frame(self.root, bg=BG, width=1, height=1)
        self.left.grid(row=0, column=0, sticky="nsew")
        self.left.grid_propagate(False)                # the view takes whatever space the controls leave
        self.left.columnconfigure(0, weight=1)
        self.left.rowconfigure(0, weight=1)
        self.right = tk.Frame(self.root, bg=BG)
        self.right.grid(row=0, column=1, sticky="ns")
        # under the video: controller + robot state on the left, the validation table takes the rest
        self.bottom = tk.Frame(self.left, bg=BG)
        self.bottom.grid(row=2, column=0, sticky="ew")
        self.bottom.columnconfigure(1, weight=1)
        self.colA = tk.Frame(self.bottom, bg=BG)
        self.colA.grid(row=0, column=0, sticky="n")
        self.colB = tk.Frame(self.bottom, bg=BG)
        self.colB.grid(row=0, column=1, sticky="nsew")
        self.colB.columnconfigure(0, weight=1)
        self.colB.rowconfigure(0, weight=1)

        self._build_view()
        self._build_sim_row()
        self._build_readouts()
        self._build_controller_frame()
        self._build_signal_frame()
        self._build_joint_frame()
        self._build_cartesian_frame()

        self._update_labels()
        self.live_label.config(text="transfer joint 000.00 mrad  TCP 00.000 mm")   # worst-case width for the measurement
        self.root.update_idletasks()
        rw, rh = self.right.winfo_reqwidth() + 12, self.right.winfo_reqheight() + 12
        self.root.columnconfigure(1, weight=0, minsize=rw)       # controls are never squeezed
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        # narrowest left column that shows every control (the video box itself may shrink)
        left_min = max(w.winfo_reqwidth() for w in self.left.winfo_children() if w is not self.canvas.master) + 24
        width = min(rw + max(left_min, int(sw * 0.45)), int(sw * 0.95))
        height = min(max(rh, 760), int(sh * 0.92))
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(rw + left_min, min(rh, int(sh * 0.9)))
        self._repaint_job = None
        threading.Thread(target=self._decode_loop, daemon=True).start()
        self.root.after(8, self._refresh_video)
        self.root.after(100, self._refresh)
        self.root.after(40, self._flush_camera)
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.update_idletasks()
        self.root.deiconify()                          # first map happens at the final size
        for ms in (300, 1200, 3000):
            self.root.after(ms, self._repaint_all)
        self.root.bind("<Configure>", lambda e: self._repaint_soon() if e.widget is self.root else None)
        self.root.bind("<FocusIn>", lambda e: self._repaint_soon())
        self.right.bind("<Enter>", lambda e: self._repaint_soon())
        if os.environ.get("PANEL_DUMP_LAYOUT"):
            self.root.after(5000, self._dump_layout)

    # ------------------------------------------------------------------ widgets
    def _frame(self, parent, title, row, col, colspan=1, sticky="nsew"):
        f = tk.LabelFrame(parent, text=title, bg=BG, fg=ACCENT, font=("Helvetica", 11, "bold"),
                          bd=1, relief="groove", padx=8, pady=6)
        f.grid(row=row, column=col, columnspan=colspan, padx=6, pady=5, sticky=sticky)
        return f

    def _button(self, parent, text, cmd, color=PANEL, width=14):
        return tk.Button(parent, text=text, command=cmd, bg=color, fg="white" if color != PANEL else FG,
                         relief="flat", width=width, activebackground=ACCENT, padx=4, pady=3)

    def _label(self, parent, text, **kw):
        return tk.Label(parent, text=text, bg=BG, fg=kw.pop("fg", FG), **kw)

    # ----- the simulator view
    def _build_view(self):
        box = tk.Frame(self.left, bg=BG, bd=1, relief="sunken")
        box.grid(row=0, column=0, sticky="nsew", padx=6, pady=(6, 2))
        box.columnconfigure(0, weight=1)
        box.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(box, bg="#11111b", width=640, height=480, highlightthickness=0, cursor="fleur")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas_text = self.canvas.create_text(320, 240, fill="#9399b2", font=("Helvetica", 13),
                                                   text="waiting for the simulator view...")
        self.canvas_img = self.canvas.create_image(0, 0, anchor="center")
        c = self.canvas
        c.bind("<Configure>", self._on_canvas_resize)
        c.bind("<ButtonPress-1>", lambda e: self._drag_start(e, "pan" if e.state & 0x1 else "rotate"))
        c.bind("<ButtonPress-2>", lambda e: self._drag_start(e, "pan"))
        c.bind("<ButtonPress-3>", lambda e: self._drag_start(e, "zoom"))
        for b in ("<B1-Motion>", "<B2-Motion>", "<B3-Motion>"):
            c.bind(b, self._drag_move)
        c.bind("<Double-Button-1>", lambda e: self.node.camera(action="reset"))
        c.bind("<Button-4>", lambda e: self._wheel(-1))       # Linux wheel
        c.bind("<Button-5>", lambda e: self._wheel(1))
        c.bind("<MouseWheel>", lambda e: self._wheel(-1 if e.delta > 0 else 1))
        self._label(self.left, "drag: orbit | wheel: zoom | shift-drag: pan | double-click: reset", fg="#9399b2",
                    font=("Helvetica", 9)).grid(row=1, column=0, sticky="w", padx=8)
        legend = tk.Frame(self.left, bg=BG)
        legend.grid(row=1, column=0)
        for text, colour in (("joint move", "#59bfff"), ("straight line", "#ff8c1a"),
                             ("manual", "#cc73ff"), ("actual", "#59ff73")):
            tk.Label(legend, text="\u2501 " + text, bg=BG, fg=colour, font=("Helvetica", 10, "bold")).pack(side=tk.LEFT, padx=8)
        q = tk.Frame(self.left, bg=BG)
        q.grid(row=1, column=0, sticky="e", padx=8)
        self._label(q, "view quality", fg="#9399b2", font=("Helvetica", 9)).pack(side=tk.LEFT, padx=4)
        self.quality = ttk.Combobox(q, values=list(self.VIEW_MODES), width=15, state="readonly")
        self.quality.set("high (2.0 MP)")                # sharp and smooth; "fit window" is sharpest but slower
        self.quality.pack(side=tk.LEFT)
        self.quality.bind("<<ComboboxSelected>>", lambda e: (setattr(self, "_sent_size", None), self._send_size()))

    def _on_canvas_resize(self, event):
        """Tell the simulator the pixel size of the view (debounced) so it renders at full resolution."""
        self.canvas_size = (event.width, event.height)
        if self._resize_job is not None:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(250, self._send_size)

    def _send_size(self):
        self._resize_job = None
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        limit = self.VIEW_MODES[self.quality.get()]
        scale = 1.0 if limit is None else min(1.0, float(np.sqrt(limit / max(cw * ch, 1))))
        size = (int(cw * scale), int(ch * scale))
        if size[0] > 64 and size[1] > 64 and size != self._sent_size:
            self._sent_size = size
            self.node.camera(action="resize", w=size[0], h=size[1])

    def _drag_start(self, e, mode):
        self._drag = (mode, e.x, e.y)

    def _drag_move(self, e):
        if self._drag is None:
            return
        mode, x0, y0 = self._drag
        h = max(self.canvas.winfo_height(), 1)
        acc = self._cam_acc[mode]
        acc[0] += (e.x - x0) / h
        acc[1] += (e.y - y0) / h
        self._drag = (mode, e.x, e.y)

    def _wheel(self, direction):
        self._cam_acc["zoom"][1] += 0.05 * direction

    def _flush_camera(self):
        for action, acc in self._cam_acc.items():
            if acc[0] or acc[1]:
                self.node.camera(action=action, dx=acc[0], dy=acc[1])
                acc[0] = acc[1] = 0.0
        self.root.after(40, self._flush_camera)

    def _decode_loop(self):
        """Worker thread: JPEG decode (releases the GIL) so the UI thread only has to draw."""
        seen = None
        while self.running:
            self.node.frame_event.wait(0.1)
            self.node.frame_event.clear()
            data = self.node.frame
            if data is None or data is seen:
                continue
            seen = data
            try:
                img = Image.open(io.BytesIO(data))
                img.load()
                cw, ch = self.canvas_size
                if img.width > cw or img.height > ch:        # window got smaller since the frame was rendered
                    s = min(cw / img.width, ch / img.height)
                    img = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.BILINEAR)
                with self.ready_lock:
                    self.ready = img
            except Exception:
                pass

    def _refresh_video(self):
        """Show the newest decoded frame, but only as often as the UI thread can afford:
        drawing takes `disp_cost`; waiting 2.2x that keeps more than half of the thread free for clicks."""
        now = time.time()
        if self.ready is not None and now >= self.next_show:
            with self.ready_lock:
                img, self.ready = self.ready, None
            t0 = time.time()
            if self._photo is None or self._photo_size != img.size:
                self._photo, self._photo_size = ImageTk.PhotoImage(img), img.size
            else:
                self._photo.paste(img)
            cw, ch = self.canvas_size
            self.canvas.itemconfig(self.canvas_img, image=self._photo)
            if self._img_pos != (cw // 2, ch // 2):          # only touch the canvas when something moved
                self._img_pos = (cw // 2, ch // 2)
                self.canvas.coords(self.canvas_img, *self._img_pos)
                self.canvas.itemconfig(self.canvas_text, state="hidden")
            self.root.update_idletasks()                      # draw now, so the cost includes the drawing
            self.disp_cost = 0.7 * self.disp_cost + 0.3 * (time.time() - t0)
            interval = max(1 / 30, 2.2 * self.disp_cost)
            self.next_show = time.time() + interval
            fps = float(np.clip(1.15 / interval, 4, 30))     # tell the renderer what it can use
            if self._fps_sent is None or abs(fps / self._fps_sent - 1) > 0.2:
                self._fps_sent = fps
                self.node.camera(action="fps", fps=fps)
        self.root.after(8, self._refresh_video)

    # ----- simulation / program (right column, under the Cartesian controls)
    def _build_sim_row(self):
        f = self._frame(self.right, "Simulation & program", 2, 0)
        run = tk.Frame(f, bg=BG)
        run.grid(row=0, column=0, sticky="w")
        self._button(run, "Run", self._run, "#40a02b", 9).pack(side=tk.LEFT, padx=3, pady=2)
        self._button(run, "Pause", self._pause, "#df8e1d", 9).pack(side=tk.LEFT, padx=3, pady=2)
        self._button(run, "Reset", self._reset, "#d20f39", 9).pack(side=tk.LEFT, padx=3, pady=2)
        self._label(run, "speed x").pack(side=tk.LEFT, padx=(14, 4))
        self.speed = ttk.Combobox(run, values=["0.1", "0.25", "0.5", "1", "2", "4"], width=5, state="readonly")
        self.speed.set("1")
        self.speed.pack(side=tk.LEFT)
        self.speed.bind("<<ComboboxSelected>>", lambda e: self.node.sim(cmd="speed", value=float(self.speed.get())))
        self.sim_label = self._label(f, "", font=("Courier", 12, "bold"))
        self.sim_label.grid(row=1, column=0, sticky="w", pady=(4, 0))
        tk.Frame(f, bg="#45475a", height=1).grid(row=2, column=0, sticky="ew", pady=8)
        prog = tk.Frame(f, bg=BG)
        prog.grid(row=3, column=0, sticky="w")
        self._button(prog, "Start sorting", lambda: self.node.command(cmd="start_sorting"), "#1e66f5", 13).pack(side=tk.LEFT, padx=3, pady=2)
        self._button(prog, "Stop (hold)", lambda: self.node.command(cmd="hold"), "#df8e1d", 12).pack(side=tk.LEFT, padx=3, pady=2)
        self._button(prog, "Home", lambda: self.node.command(cmd="home"), width=8).pack(side=tk.LEFT, padx=3, pady=2)
        self.box_labels = {}
        row = tk.Frame(f, bg=BG)
        row.grid(row=4, column=0, sticky="w", pady=(6, 0))
        for c in C.COLORS:
            lbl = tk.Label(row, text=f"{c}: -", bg=PANEL, fg=FG, width=14, font=("Helvetica", 10))
            lbl.pack(side=tk.LEFT, padx=3)
            self.box_labels[c] = lbl

    def _build_readouts(self):
        f = self._frame(self.colA, "Robot state", 1, 0)
        self.state_label = self._label(f, "", font=("Courier", 10, "bold"), justify="left", anchor="w")
        self.state_label.pack(fill="x")
        self.status_label = self._label(f, "", font=("Helvetica", 11, "italic"), fg=ACCENT, anchor="w",
                                        justify="left", wraplength=500)
        self.status_label.pack(fill="x", pady=(6, 0))

    # ----- controller (left) and validation (fills the rest of the bottom area)
    def _build_controller_frame(self):
        f = self._frame(self.colA, "Controller", 0, 0)
        top = tk.Frame(f, bg=BG)
        top.grid(row=0, column=0, sticky="w")
        self._label(top, "controller").pack(side=tk.LEFT, padx=(0, 8))
        self.ctrl_box = ttk.Combobox(top, values=list(CONTROLLER_LABELS), width=17, state="readonly")
        self.ctrl_box.set("Computed torque")
        self.ctrl_box.pack(side=tk.LEFT)
        self.ctrl_box.bind("<<ComboboxSelected>>", lambda e: self.node.select_controller(CONTROLLER_LABELS[self.ctrl_box.get()]))
        opts = tk.Frame(f, bg=BG)
        opts.grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.show_path = tk.BooleanVar(value=True)
        self.show_trail = tk.BooleanVar(value=True)
        for text, var in (("planned path", self.show_path), ("trail", self.show_trail)):
            tk.Checkbutton(opts, text=text, variable=var, bg=BG, fg=FG, selectcolor=PANEL, activebackground=BG,
                           command=self._overlay).pack(side=tk.LEFT, padx=(0, 10))
        self._button(opts, "clear", lambda: self.node.camera(action="overlay", clear_trail=True), width=7).pack(side=tk.LEFT)
        self.info_label = self._label(f, "", font=("Courier", 10), justify="left", anchor="w", fg="#bac2de", wraplength=500)
        self.info_label.grid(row=2, column=0, sticky="ew", pady=(8, 4))
        self.live_label = self._label(f, "", font=("Courier", 11, "bold"), justify="left", anchor="w", wraplength=500)
        self.live_label.grid(row=3, column=0, sticky="ew")

        v = self._frame(self.colB, "Validation of every finished trajectory segment", 0, 0)
        style = ttk.Style()
        style.configure("V.Treeview", background=PANEL, fieldbackground=PANEL, foreground=FG, rowheight=28,
                        borderwidth=0, font=("Helvetica", 11))
        style.configure("V.Treeview.Heading", background="#45475a", foreground=FG, relief="flat",
                        font=("Helvetica", 10, "bold"), padding=(2, 4))
        style.map("V.Treeview", background=[("selected", "#585b70")])
        box = tk.Frame(v, bg=BG)
        box.pack(fill="both", expand=True)
        cols = ("t", "phase", "space", "ctrl", "max", "end", "result")
        self.table = ttk.Treeview(box, columns=cols, show="headings", height=7, style="V.Treeview")
        for c, w, head in (("t", 62, "t [s]"), ("phase", 100, "segment"), ("space", 74, "space"), ("ctrl", 90, "controller"),
                           ("max", 130, "max error"), ("end", 130, "end error"), ("result", 80, "result")):
            self.table.heading(c, text=head)
            self.table.column(c, width=w, minwidth=w, anchor="center", stretch=True)
        self.table.tag_configure("pass", foreground=GOOD)
        self.table.tag_configure("fail", foreground=WARN)
        sb = ttk.Scrollbar(box, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill="y")
        self.table.pack(side=tk.LEFT, fill="both", expand=True)
        self.summary_label = self._label(v, "no segments validated yet", font=("Courier", 11), justify="left",
                                         anchor="w", wraplength=760)
        self.summary_label.pack(fill="x", pady=(8, 0))

    # ----- signal builder: constants, steps and waveforms as joint inputs
    def _build_signal_frame(self):
        f = self._frame(self.colB, "Signal builder", 1, 0)
        row = tk.Frame(f, bg=BG)
        row.grid(row=0, column=0, sticky="w")
        self._label(row, "joint").pack(side=tk.LEFT, padx=(0, 4))
        self.sig_joint = ttk.Combobox(row, values=[f"{i + 1} {n}" for i, n in enumerate(JOINT_LABELS)], width=14, state="readonly")
        self.sig_joint.set("2 pitch")
        self.sig_joint.pack(side=tk.LEFT)
        self._label(row, "signal").pack(side=tk.LEFT, padx=(10, 4))
        self.sig_kind = ttk.Combobox(row, values=list(P.SIGNAL_KINDS), width=9, state="readonly")
        self.sig_kind.set("sine")
        self.sig_kind.pack(side=tk.LEFT)
        row2 = tk.Frame(f, bg=BG)
        row2.grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.sig_entries = {}
        for label, key, default, unit in (("amp", "amp", "15", "deg"), ("freq", "freq", "0.5", "Hz"), ("dur", "dur", "10", "s")):
            self._label(row2, label).pack(side=tk.LEFT, padx=(0 if key == "amp" else 8, 3))
            e = tk.Entry(row2, width=5, bg=PANEL, fg=FG, insertbackground=FG, justify="right")
            e.insert(0, default)
            e.pack(side=tk.LEFT)
            self._label(row2, unit, fg="#9399b2").pack(side=tk.LEFT, padx=(2, 0))
            self.sig_entries[key] = e
        row3 = tk.Frame(f, bg=BG)
        row3.grid(row=2, column=0, sticky="w", pady=(6, 0))
        self._button(row3, "Run signal", self._run_signal, "#1e66f5", 12).pack(side=tk.LEFT, padx=(0, 6))
        self._button(row3, "Stop", lambda: self.node.command(cmd="hold"), "#df8e1d", 8).pack(side=tk.LEFT)

    def _run_signal(self):
        try:
            amp = np.radians(float(self.sig_entries["amp"].get()))
            freq = float(self.sig_entries["freq"].get())
            dur = float(self.sig_entries["dur"].get())
        except ValueError:
            self.node.status = "signal builder: amplitude, frequency and duration must be numbers"
            return
        self.node.command(cmd="signal", joint=int(self.sig_joint.get().split()[0]), kind=self.sig_kind.get(),
                          amp=amp, freq=freq, duration=dur)

    def _overlay(self):
        self.node.camera(action="overlay", path=bool(self.show_path.get()), trail=bool(self.show_trail.get()))

    def _clear_validation(self):
        self.table.delete(*self.table.get_children())
        self.node.valid.clear()
        self._valid_shown = 0
        self.worst = {}
        self.summary_label.config(text="no segments validated yet")

    def _update_controller_panel(self):
        n = self.node
        if n.info is not None and n.info != self._info_shown:        # shown when the controller changes
            self._info_shown = n.info
            self.ctrl_box.set(n.info["label"])
            self.info_label.config(text=f"{n.info['law']}\n{n.info['gains']}")
        if n.monitor is not None:
            _, phase, jerr, terr = n.monitor
            name = T.PHASES[int(phase)] if 0 <= int(phase) < len(T.PHASES) else "-"
            self.live_label.config(text=f"{name:<8s} joint {jerr * 1000:6.2f} mrad  TCP {terr:6.3f} mm",
                                   fg=GOOD if terr < 3.0 else WARN)
        while self._valid_shown < len(n.valid):
            ev = n.valid[self._valid_shown]
            self._valid_shown += 1
            unit = ev["unit"]
            ok = ev["ok"]
            self.table.insert("", tk.END, tags=("pass" if ok else "fail",), values=(
                f"{ev['t']:.1f}", ev["phase"], "task" if ev["space"] == "task" else "joint",
                CONTROLLER_SHORT.get(ev.get("controller", ""), "-"), f"{ev['value']:.2f} {unit}",
                f"{ev['end']:.2f} {unit}", "PASS" if ok else "FAIL"))
            self.table.yview_moveto(1.0)
            w = self.worst.setdefault(ev["space"], {"n": 0, "fail": 0, "worst": 0.0, "unit": unit, "tol": ev["tol"]})
            w["n"] += 1
            w["fail"] += 0 if ok else 1
            w["worst"] = max(w["worst"], ev["value"])
        if self.worst:
            parts = []
            for space, label in (("task", "task-space "), ("joint", "joint-space")):
                if space in self.worst:
                    w = self.worst[space]
                    verdict = "all pass" if w["fail"] == 0 else f"{w['fail']} FAIL"
                    parts.append(f"{label}: {w['n']} segments, worst {w['worst']:.2f} {w['unit']} (tol {w['tol']:g}) {verdict}")
            self.summary_label.config(text="\n".join(parts))

    # ----- joints
    def _build_joint_frame(self):
        f = self._frame(self.right, "Joint angles", 0, 0)
        self.j_scale, self.j_meas = [], []
        self._label(f, "target [deg]").grid(row=0, column=1)
        self._label(f, "measured").grid(row=0, column=2)
        for i, name in enumerate(JOINT_LABELS):
            lo, hi = np.degrees(K.JOINT_LIMITS[i])
            self._label(f, name, width=13, anchor="w").grid(row=i + 1, column=0, sticky="w")
            s = tk.Scale(f, from_=lo, to=hi, resolution=0.5, orient=tk.HORIZONTAL, length=230,
                         bg=PANEL, fg=FG, troughcolor="#45475a", highlightthickness=0)
            s.grid(row=i + 1, column=1, padx=4)
            s.bind("<ButtonRelease-1>", lambda e: self._live_joint())
            self.j_scale.append(s)
            m = self._label(f, "0.0", width=7, font=("Courier", 10), fg=GOOD)
            m.grid(row=i + 1, column=2)
            self.j_meas.append(m)
        self._label(f, "gripper [rad]", anchor="w").grid(row=6, column=0, sticky="w")
        self.g_scale = tk.Scale(f, from_=P.GRIPPER_LIMITS[0], to=P.GRIPPER_LIMITS[1], resolution=0.01,
                                orient=tk.HORIZONTAL, length=230, bg=PANEL, fg=FG,
                                troughcolor="#45475a", highlightthickness=0)
        self.g_scale.set(0.0)
        self.g_scale.grid(row=6, column=1, padx=4)
        self.g_scale.bind("<ButtonRelease-1>", lambda e: self._live_joint())
        self.g_meas = self._label(f, "0.00", width=7, font=("Courier", 10), fg=GOOD)
        self.g_meas.grid(row=6, column=2)

        row = tk.Frame(f, bg=BG)
        row.grid(row=7, column=0, columnspan=3, pady=(8, 0))
        self._button(row, "Go to angles", self._go_joints, "#1e66f5", 12).pack(side=tk.LEFT, padx=2)
        self._button(row, "Read robot", self._read_joints, width=10).pack(side=tk.LEFT, padx=2)
        self._button(row, "Open grip", lambda: self._grip(-0.6), width=9).pack(side=tk.LEFT, padx=2)
        self._button(row, "Close grip", lambda: self._grip(0.0), width=9).pack(side=tk.LEFT, padx=2)
        self.live = tk.BooleanVar(value=False)
        tk.Checkbutton(f, text="send on slider release", variable=self.live, bg=BG, fg=FG,
                       selectcolor=PANEL, activebackground=BG).grid(row=8, column=0, columnspan=3, sticky="w")

    # ----- cartesian
    def _build_cartesian_frame(self):
        f = self._frame(self.right, "Cartesian (tool point, world frame)", 1, 0)
        self.xyz = []
        for i, (n, v) in enumerate(zip("xyz", (-0.103, 0.0, 0.085))):
            self._label(f, f"{n} [m]").grid(row=i, column=0, sticky="w")
            e = tk.Entry(f, width=9, bg=PANEL, fg=FG, insertbackground=FG, justify="right")
            e.insert(0, f"{v:.3f}")
            e.grid(row=i, column=1, padx=4, pady=1)
            self.xyz.append(e)
        self._label(f, "x < 0 = in front", fg="#9399b2").grid(row=0, column=2, sticky="w", padx=6)
        self._label(f, "tool pitch [deg]").grid(row=3, column=0, sticky="w")
        self.pitch = tk.Entry(f, width=9, bg=PANEL, fg=FG, insertbackground=FG, justify="right")
        self.pitch.insert(0, "-90")
        self.pitch.grid(row=3, column=1)
        self._label(f, "-90 = pointing down", fg="#9399b2").grid(row=3, column=2, sticky="w", padx=6)
        self._label(f, "roll [deg]").grid(row=4, column=0, sticky="w")
        self.roll = tk.Entry(f, width=9, bg=PANEL, fg=FG, insertbackground=FG, justify="right")
        self.roll.grid(row=4, column=1)
        self._label(f, "blank = keep", fg="#9399b2").grid(row=4, column=2, sticky="w", padx=6)

        row = tk.Frame(f, bg=BG)
        row.grid(row=5, column=0, columnspan=3, pady=(8, 2))
        self._button(row, "Go (joint motion)", lambda: self._go_tcp(False), "#1e66f5", 15).pack(side=tk.LEFT, padx=2)
        self._button(row, "Go (straight)", lambda: self._go_tcp(True), "#1e66f5", 12).pack(side=tk.LEFT, padx=2)
        self._button(row, "Read TCP", self._read_tcp, width=9).pack(side=tk.LEFT, padx=2)

        self._label(f, "jog step [mm]").grid(row=6, column=0, sticky="w", pady=(8, 0))
        self.step = tk.Entry(f, width=9, bg=PANEL, fg=FG, insertbackground=FG, justify="right")
        self.step.insert(0, "10")
        self.step.grid(row=6, column=1, pady=(8, 0))
        pad = tk.Frame(f, bg=BG)
        pad.grid(row=7, column=0, columnspan=3, pady=4)
        for c, (lab, d) in enumerate([("X -", (-1, 0, 0)), ("X +", (1, 0, 0)), ("Y -", (0, -1, 0)),
                                      ("Y +", (0, 1, 0)), ("Z -", (0, 0, -1)), ("Z +", (0, 0, 1))]):
            self._button(pad, lab, lambda d=d: self._jog(d), width=5).grid(row=0, column=c, padx=2)

        self._label(f, "presets").grid(row=8, column=0, sticky="w")
        pre = tk.Frame(f, bg=BG)
        pre.grid(row=9, column=0, columnspan=3)
        self._button(pre, "belt", lambda: self._preset(C.BELT_CENTER[0], 0.0), width=7).pack(side=tk.LEFT, padx=2)
        for c in C.COLORS:
            bc = C.bin_center(c)
            self._button(pre, f"{c} bin", lambda bc=bc: self._preset(bc[0], bc[1]), width=9).pack(side=tk.LEFT, padx=2)

    # ------------------------------------------------------------------ actions
    def _run(self):
        self.paused = False
        self.node.sim(cmd="resume")

    def _pause(self):
        self.paused = True
        self.node.sim(cmd="pause")

    def _reset(self):
        self.node.sim(cmd="reset")
        self._clear_validation()
        self.root.after(500, self._read_joints)

    def _go_joints(self):
        q = np.radians([s.get() for s in self.j_scale])
        self.node.command(cmd="goto_joint", q=q.tolist(), grip=self.g_scale.get())

    def _live_joint(self):
        if self.live.get():
            self._go_joints()

    def _grip(self, value):
        self.g_scale.set(value)
        self.node.command(cmd="gripper", value=value)

    def _read_joints(self):
        for s, v in zip(self.j_scale, np.degrees(self.node.q)):
            s.set(round(v * 2) / 2)
        self.g_scale.set(round(self.node.grip, 2))

    def _read_tcp(self):
        p = TK.tcp_position(self.node.q)
        for e, v in zip(self.xyz, p):
            e.delete(0, tk.END)
            e.insert(0, f"{v:.3f}")
        self.pitch.delete(0, tk.END)
        self.pitch.insert(0, f"{np.degrees(TK.tool_pitch(self.node.q)):.0f}")

    def _preset(self, x, y):
        for e, v in zip(self.xyz, (x, y, 0.085)):
            e.delete(0, tk.END)
            e.insert(0, f"{v:.3f}")
        self.pitch.delete(0, tk.END)
        self.pitch.insert(0, "-90")
        self._go_tcp(False)

    def _go_tcp(self, linear):
        try:
            p = [float(e.get()) for e in self.xyz]
            pitch = np.radians(float(self.pitch.get()))
            roll = np.radians(float(self.roll.get())) if self.roll.get().strip() else None
        except ValueError:
            self.node.status = "check the Cartesian entries (numbers only)"
            return
        self.node.command(cmd="goto_tcp", p=p, pitch=pitch, roll=roll, linear=linear, grip=self.g_scale.get())

    def _jog(self, d):
        try:
            step = float(self.step.get()) / 1000.0
        except ValueError:
            return
        self.node.command(cmd="jog", d=[step * x for x in d])

    # ------------------------------------------------------------------ periodic refresh
    def _refresh(self):
        self._update_labels()
        self.root.after(100, self._refresh)

    def _update_labels(self):
        n = self.node
        alive = time.time() - n.last_state < 1.5
        if alive and not self.synced:                  # copy the robot's real pose into the controls once
            self.synced = True
            self._read_joints()
            self._read_tcp()
        for m, v in zip(self.j_meas, np.degrees(n.q)):
            m.config(text=f"{v:6.1f}")
        self.g_meas.config(text=f"{n.grip:5.2f}")
        p = TK.tcp_position(n.q)
        self.state_label.config(
            text=f"TCP  x={p[0]:+.3f}  y={p[1]:+.3f}  z={p[2]:+.3f} m\n"
                 f"tool pitch {np.degrees(TK.tool_pitch(n.q)):+6.1f} deg\n"
                 f"q  " + " ".join(f"{v:+6.1f}" for v in np.degrees(n.q)) + " deg")
        # a paused simulation sends no updates, so "paused" has to be checked before "no simulator"
        state = "PAUSED" if self.paused else ("RUNNING" if alive else "NO SIMULATOR")
        self.sim_label.config(text=f"t = {n.sim_t:7.2f} s   {state}", fg=GOOD if state == "RUNNING" else WARN)
        for c, lbl in self.box_labels.items():
            ok = n.boxes[c]
            lbl.config(text=f"{c}: {'sorted' if ok else 'waiting'}", fg=GOOD if ok else FG)
        self.status_label.config(text=n.status)
        self._update_controller_panel()

    def _repaint_soon(self):
        """Debounced full repaint. Some X11/Wayland setups lose the paint of widgets that never change
        (sliders, buttons), so every widget is asked to redraw after the window is shown or resized."""
        if self._repaint_job is not None:
            self.root.after_cancel(self._repaint_job)
        self._repaint_job = self.root.after(350, self._repaint_all)

    def _repaint_all(self):
        self._repaint_job = None

        def walk(w):
            yield w
            for c in w.winfo_children():
                yield from walk(c)
        for w in walk(self.root):
            if w is self.canvas:
                continue
            try:
                w.configure(background=w.cget("background"))     # re-configuring schedules a redraw
            except tk.TclError:
                pass                                             # ttk widgets have no such option

    def _dump_layout(self):
        """Debug aid: PANEL_DUMP_LAYOUT=/path writes the size of every widget."""
        def walk(w, depth=0):
            yield depth, w
            for c in w.winfo_children():
                yield from walk(c, depth + 1)
        with open(os.environ["PANEL_DUMP_LAYOUT"], "w") as f:
            f.write(f"root {self.root.winfo_width()}x{self.root.winfo_height()} scaling {self.root.tk.call('tk', 'scaling')}\n")
            for d, w in walk(self.root):
                f.write(f"{'  ' * d}{w.winfo_class():12s} {w.winfo_width():5d}x{w.winfo_height():<5d} req {w.winfo_reqwidth():5d}x{w.winfo_reqheight():<5d} "
                        f"at ({w.winfo_x()},{w.winfo_y()}) mapped={w.winfo_ismapped()} {str(w.cget('text'))[:18] if 'text' in w.keys() else ''}\n")

    def _close(self):
        self.running = False
        self.root.destroy()
        rclpy.try_shutdown()

    def run(self):
        self.root.mainloop()


def main(args=None):
    rclpy.init(args=args)
    node = PanelNode()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    app = PanelApp(node)
    try:
        app.run()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
