import glob
import os
import sys

# offscreen rendering of the embedded simulator view uses EGL (no window needed)
os.environ.setdefault("MUJOCO_GL", "egl")


def _ensure_mujoco():
    """ROS nodes run under the system Python; if `mujoco` is only in the project's .venv, add it."""
    try:
        import mujoco  # noqa: F401
        return
    except ImportError:
        pass
    d = os.path.dirname(os.path.abspath(__file__))
    roots = []
    for _ in range(8):
        roots.append(d)
        d = os.path.dirname(d)
    roots.append(os.getcwd())
    for root in roots:
        for sp in glob.glob(os.path.join(root, ".venv", "lib", "python3*", "site-packages")):
            if os.path.isdir(os.path.join(sp, "mujoco")):
                sys.path.append(sp)
                return


_ensure_mujoco()
