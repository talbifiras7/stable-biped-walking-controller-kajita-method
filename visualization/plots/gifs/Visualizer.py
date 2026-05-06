"""
visualiser.py
=============
Plots and animated GIFs comparing the ZMP reference (from zmp.py)
against the CoM trajectory (from lipm.py).

What this file produces
-----------------------
  Static plots (PNG):
    1. zmp_com_timeseries.png  — ZMP ref vs CoM position over time (X and Y)
    2. zmp_com_topview.png     — top-down view: ZMP path, CoM path, foot placements
    3. foot_trajectories.png   — foot Z height over time (swing arcs)
    4. com_state.png           — full CoM state: pos / vel / acc per axis

  Animated GIFs:
    5. walk_topview.gif        — top-down animation: CoM dot + ZMP dot + feet
    6. walk_sideview.gif       — side-view animation: CoM height + ZMP point

"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from dynamics.zmp          import ZMPPlanner, ZMPData
from dynamics.lipm         import LIPM, CoMTrajectory


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
FOOT_COLOR = {"right": "#2196F3", "left": "#FF9800"}   # blue / orange
ZMP_COLOR  = "#E53935"    # red
COM_COLOR  = "#43A047"    # green


def _save(fig: plt.Figure, path: str,
          tight: bool = True) -> None:
    kwargs = {"dpi": 120}
    if tight:
        try:
            fig.tight_layout()
        except Exception:
            pass
        kwargs["bbox_inches"] = "tight"
    fig.savefig(path, **kwargs)
    plt.close(fig)
    print(f"  saved → {path}")


def _foot_rect(pos: np.ndarray, side: str,
               fw: float = 0.06, fh: float = 0.03) -> mpatches.FancyBboxPatch:
    """Rounded rectangle representing a foot in the XY plane."""
    return mpatches.FancyBboxPatch(
        (pos[0] - fw/2, pos[1] - fh/2), fw, fh,
        boxstyle="round,pad=0.005",
        linewidth=1.2,
        edgecolor=FOOT_COLOR[side],
        facecolor=FOOT_COLOR[side] + "55",   # 33% alpha
    )


# ─────────────────────────────────────────────────────────────
# ZMP  →  CoM  (simple proportional jerk controller for demo)
# ─────────────────────────────────────────────────────────────
def _com_from_zmp(zmp_data: ZMPData, lipm: LIPM) -> CoMTrajectory:
    """
    Compute a CoM trajectory from the ZMP reference using a simple
    proportional jerk controller:

        u_k = Kp · (p_zmp_ref[k] − C·x_k)

    This is NOT Kajita's preview control — it is intentionally kept
    simple so the visualiser only depends on lipm.py and zmp.py.
    The CoM will lag slightly behind the ZMP reference, which is
    physically correct for the LIPM (the CoM leads the ZMP in steady
    gait but lags during rapid changes).

    Parameters
    ----------
    zmp_data : ZMPData
    lipm     : LIPM instance

    Returns
    -------
    CoMTrajectory
    """
    T    = zmp_data.T
    Kp   = 600.0    # proportional gain on ZMP error → jerk

    u_x = np.zeros(T)
    u_y = np.zeros(T)

    x_x = np.zeros(3)
    x_y = np.zeros(3)

    for k in range(T):
        # ZMP error drives the jerk
        u_x[k] = Kp * (zmp_data.zmp_x[k] - lipm.compute_zmp(x_x))
        u_y[k] = Kp * (zmp_data.zmp_y[k] - lipm.compute_zmp(x_y))
        # Advance state
        x_x, _ = lipm.step(x_x, u_x[k])
        x_y, _ = lipm.step(x_y, u_y[k])

    return lipm.integrate(u_x, u_y)


# ─────────────────────────────────────────────────────────────
# PLOT 1 — ZMP vs CoM time series
# ─────────────────────────────────────────────────────────────
def plot_timeseries(zmp: ZMPData, com: CoMTrajectory,
                    outdir: str) -> None:
    """
    Four-panel plot:
      top-left   — ZMP ref x  vs  CoM x
      top-right  — ZMP ref y  vs  CoM y
      bottom-left  — CoM velocity x and y
      bottom-right — CoM acceleration x and y
    """
    t   = np.arange(zmp.T) * zmp.dt
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    fig.suptitle("ZMP Reference vs CoM Trajectory — Time Series", fontsize=13)

    # X axis
    ax = axes[0, 0]
    ax.plot(t, zmp.zmp_x, color=ZMP_COLOR, lw=1.5, ls="--", label="ZMP ref x")
    ax.plot(t, com.pos_x, color=COM_COLOR,  lw=1.5,           label="CoM x")
    ax.set_ylabel("position [m]"); ax.set_title("X axis — position")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

    # Y axis
    ax = axes[0, 1]
    ax.plot(t, zmp.zmp_y, color=ZMP_COLOR, lw=1.5, ls="--", label="ZMP ref y")
    ax.plot(t, com.pos_y, color=COM_COLOR,  lw=1.5,           label="CoM y")
    ax.set_ylabel("position [m]"); ax.set_title("Y axis — position")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

    # Velocities
    ax = axes[1, 0]
    ax.plot(t, com.vel_x, color="#1565C0", lw=1.2, label="vel x")
    ax.plot(t, com.vel_y, color="#E65100", lw=1.2, label="vel y")
    ax.set_ylabel("velocity [m/s]"); ax.set_xlabel("time [s]")
    ax.set_title("CoM velocity")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

    # Accelerations
    ax = axes[1, 1]
    ax.plot(t, com.acc_x, color="#1565C0", lw=1.2, label="acc x")
    ax.plot(t, com.acc_y, color="#E65100", lw=1.2, label="acc y")
    ax.set_ylabel("acceleration [m/s²]"); ax.set_xlabel("time [s]")
    ax.set_title("CoM acceleration")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────
# PLOT 2 — Top-view (XY plane)
# ─────────────────────────────────────────────────────────────
def plot_topview(zmp: ZMPData, com: CoMTrajectory,
                 outdir: str) -> None:
    """
    Top-down view showing:
      - ZMP reference path (dashed red)
      - CoM path           (solid green)
      - Foot placements    (rectangles, blue = right, orange = left)
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title("Top View — ZMP Reference Path vs CoM Path", fontsize=12)

    x_all = np.concatenate([zmp.zmp_x, com.pos_x,
                             zmp.foot_pos["right"][:, 0],
                             zmp.foot_pos["left"][:, 0]])
    y_all = np.concatenate([zmp.zmp_y, com.pos_y,
                             zmp.foot_pos["right"][:, 1],
                             zmp.foot_pos["left"][:, 1]])

    # Paths
    ax.plot(zmp.zmp_x, zmp.zmp_y,
            color=ZMP_COLOR, lw=1.5, ls="--", label="ZMP reference", zorder=3)
    ax.plot(com.pos_x, com.pos_y,
            color=COM_COLOR, lw=2.0,           label="CoM",           zorder=4)

    # Start / end markers
    ax.plot(com.pos_x[0],  com.pos_y[0],
            "o", color=COM_COLOR, ms=8, zorder=5, label="CoM start")
    ax.plot(com.pos_x[-1], com.pos_y[-1],
            "s", color=COM_COLOR, ms=8, zorder=5, label="CoM end")

    # Foot rectangles at each step transition
    for side in ("right", "left"):
        fp  = zmp.foot_pos[side]
        fc  = zmp.foot_contact[side]
        # Detect landing events (False→True transitions of contact)
        landings = np.where(np.diff(fc.astype(int)) == 1)[0] + 1
        # Also draw initial position
        positions = [fp[0]] + [fp[k] for k in landings]
        for pos in positions:
            ax.add_patch(_foot_rect(pos, side))

    # Legend handles
    from matplotlib.lines import Line2D
    extra_handles = [
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor=FOOT_COLOR["right"] + "99",
               markeredgecolor=FOOT_COLOR["right"],
               markersize=10, label="Right foot"),
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor=FOOT_COLOR["left"] + "99",
               markeredgecolor=FOOT_COLOR["left"],
               markersize=10, label="Left foot"),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + extra_handles,
              labels + ["Right foot", "Left foot"],
              fontsize=8)

    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_xlim(x_all.min() - 0.1, x_all.max() + 0.1)
    ax.set_ylim(y_all.min() - 0.15, y_all.max() + 0.15)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────
# PLOT 3 — Foot trajectories (Z height)
# ─────────────────────────────────────────────────────────────
def plot_foot_trajectories(zmp: ZMPData, outdir: str) -> None:
    """
    Shows the Z height of each foot over time.
    Swing arcs are clearly visible; support phases are at z = 0.
    Contact bands are shaded.
    """
    t   = np.arange(zmp.T) * zmp.dt
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    fig.suptitle("Foot Trajectories — Z Height (Swing Arcs)", fontsize=12)

    for idx, side in enumerate(("right", "left")):
        ax  = axes[idx]
        z   = zmp.foot_pos[side][:, 2]
        fc  = zmp.foot_contact[side]

        ax.plot(t, z, color=FOOT_COLOR[side], lw=1.8, label=f"{side} foot z")

        # Shade contact phases
        in_contact = False
        t_start    = 0.
        for k in range(zmp.T):
            if fc[k] and not in_contact:
                t_start    = t[k]
                in_contact = True
            elif not fc[k] and in_contact:
                ax.axvspan(t_start, t[k], alpha=0.12,
                           color=FOOT_COLOR[side], label="_nolegend_")
                in_contact = False
        if in_contact:
            ax.axvspan(t_start, t[-1], alpha=0.12, color=FOOT_COLOR[side])

        ax.set_ylabel("z [m]"); ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

    axes[-1].set_xlabel("time [s]")
    plt.tight_layout()
    _save(fig, os.path.join(outdir, "foot_trajectories.png"))


# ─────────────────────────────────────────────────────────────
# PLOT 4 — Full CoM state (pos / vel / acc)
# ─────────────────────────────────────────────────────────────
def plot_com_state(com: CoMTrajectory, outdir: str) -> None:
    """Six-panel plot: position, velocity, acceleration for X and Y."""
    t   = np.arange(com.T) * com.dt
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    fig.suptitle("Full CoM State  (position / velocity / acceleration)", fontsize=12)

    pairs = [
        ("Position [m]",      com.pos_x, com.pos_y),
        ("Velocity [m/s]",    com.vel_x, com.vel_y),
        ("Acceleration [m/s²]", com.acc_x, com.acc_y),
    ]
    for row, (ylabel, dx, dy) in enumerate(pairs):
        for col, (data, label, color) in enumerate([
                (dx, "X", "#1565C0"), (dy, "Y", "#E65100")]):
            ax = axes[row, col]
            ax.plot(t, data, color=color, lw=1.5, label=f"CoM {label}")
            ax.set_ylabel(ylabel if col == 0 else "")
            ax.set_title(f"{label} axis")
            ax.legend(fontsize=8); ax.grid(True, alpha=0.4)
    axes[-1, 0].set_xlabel("time [s]")
    axes[-1, 1].set_xlabel("time [s]")
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────
# GIF 1 — Top-view animation
# ─────────────────────────────────────────────────────────────
def animate_topview(zmp: ZMPData, com: CoMTrajectory,
                    outdir: str, stride: int = 8) -> None:
    """
    Top-down animated GIF showing:
      - ZMP dot   (red)     moving along its reference path
      - CoM dot   (green)   following its trajectory
      - Foot rectangles (blue / orange) updating at each landing
    Every `stride` timesteps becomes one animation frame.
    """
    frames    = range(0, zmp.T, stride)
    n_frames  = len(list(frames))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Top View — ZMP vs CoM  (animated)", fontsize=11)

    # Pre-compute axis limits
    x_anim = np.concatenate([zmp.zmp_x, com.pos_x])
    y_anim = np.concatenate([zmp.zmp_y, com.pos_y,
                             zmp.foot_pos["right"][:, 1],
                             zmp.foot_pos["left"][:,  1]])
    margin = 0.08
    ax.set_xlim(x_anim.min() - margin, x_anim.max() + margin)
    ax.set_ylim(y_anim.min() - margin, y_anim.max() + margin)

    # Static background: full paths
    ax.plot(zmp.zmp_x, zmp.zmp_y,
            color=ZMP_COLOR, lw=0.8, ls="--", alpha=0.35, zorder=1)
    ax.plot(com.pos_x, com.pos_y,
            color=COM_COLOR,  lw=0.8, alpha=0.35, zorder=1)

    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.grid(True, alpha=0.3)

    # Dynamic elements — topview
    zmp_dot, = ax.plot([], [], "o", color=ZMP_COLOR, ms=9,
                       zorder=5, label="ZMP")
    com_dot, = ax.plot([], [], "o", color=COM_COLOR,  ms=9,
                       zorder=5, label="CoM")
    time_txt = ax.text(0.02, 0.95, "", transform=ax.transAxes,
                       fontsize=9, va="top")
    ax.legend(fontsize=8, loc="upper right")

    # Pre-compute axis limits
    x_anim = np.concatenate([zmp.zmp_x, com.pos_x])
    y_anim = np.concatenate([zmp.zmp_y, com.pos_y,
                             zmp.foot_pos["right"][:, 1],
                             zmp.foot_pos["left"][:,  1]])
    margin = 0.08
    ax.set_xlim(x_anim.min() - margin, x_anim.max() + margin)
    ax.set_ylim(y_anim.min() - margin, y_anim.max() + margin)

    # Foot patch containers
    foot_patches = {"right": [], "left": []}

    def _update_feet(k):
        for side in ("right", "left"):
            for p in foot_patches[side]:
                p.remove()
            foot_patches[side].clear()
            pos = zmp.foot_pos[side][k]
            p   = _foot_rect(pos, side)
            ax.add_patch(p)
            foot_patches[side].append(p)

    def init():
        zmp_dot.set_data([], [])
        com_dot.set_data([], [])
        return zmp_dot, com_dot, time_txt

    def update(frame_idx):
        k = list(frames)[frame_idx]
        zmp_dot.set_data([zmp.zmp_x[k]], [zmp.zmp_y[k]])
        com_dot.set_data([com.pos_x[k]], [com.pos_y[k]])
        time_txt.set_text(f"t = {k * zmp.dt:.2f} s")
        _update_feet(k)
        return zmp_dot, com_dot, time_txt

    anim = FuncAnimation(fig, update, frames=n_frames,
                         init_func=init, blit=False, interval=40)

    plt.close(fig)
    print(f"  saved → {path}")


# ─────────────────────────────────────────────────────────────
# GIF 2 — Side-view animation
# ─────────────────────────────────────────────────────────────
def animate_sideview(zmp: ZMPData, com: CoMTrajectory,
                     outdir: str, stride: int = 8) -> None:
    """
    Side-view (XZ plane) animated GIF showing:
      - CoM dot at height zc        (green)
      - ZMP dot at ground level     (red)
      - Right and left foot Z arcs  as they swing
    """
    frames   = range(0, zmp.T, stride)
    n_frames = len(list(frames))
    zc       = com.zc

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.set_title("Side View — CoM height vs ZMP  (animated)", fontsize=11)

    # Static ground line
    x_all  = np.concatenate([zmp.zmp_x, com.pos_x])
    x_min  = x_all.min() - 0.1
    x_max  = x_all.max() + 0.1
    ax.axhline(0., color="saddlebrown", lw=1.5, alpha=0.5, label="ground")
    ax.axhline(zc, color=COM_COLOR, lw=0.6, ls=":", alpha=0.4,
               label=f"CoM height ({zc:.2f} m)")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.05, zc + 0.15)
    ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]")
    ax.grid(True, alpha=0.3)

    # CoM trajectory trace
    ax.plot(com.pos_x, np.full(com.T, zc),
            color=COM_COLOR, lw=0.8, alpha=0.3)

    # Dynamic elements
    com_dot, = ax.plot([], [], "o", color=COM_COLOR, ms=10,
                       zorder=5, label="CoM")
    zmp_dot, = ax.plot([], [], "^", color=ZMP_COLOR, ms=10,
                       zorder=5, label="ZMP")
    r_foot,  = ax.plot([], [], "s", color=FOOT_COLOR["right"],
                       ms=8, zorder=4, label="right foot")
    l_foot,  = ax.plot([], [], "s", color=FOOT_COLOR["left"],
                       ms=8, zorder=4, label="left foot")
    time_txt  = ax.text(0.02, 0.93, "", transform=ax.transAxes,
                        fontsize=9, va="top")
    ax.legend(fontsize=8, loc="upper right")

    # Pendulum rod
    rod, = ax.plot([], [], "-", color="gray", lw=1.2, alpha=0.6, zorder=3)

    def init():
        for artist in (com_dot, zmp_dot, r_foot, l_foot):
            artist.set_data([], [])
        rod.set_data([], [])
        return com_dot, zmp_dot, r_foot, l_foot, rod, time_txt

    def update(frame_idx):
        k = list(frames)[frame_idx]
        cx = com.pos_x[k]
        zx = zmp.zmp_x[k]
        com_dot.set_data([cx],  [zc])
        zmp_dot.set_data([zx],  [0.])
        rod.set_data([zx, cx],  [0., zc])

        rfz = zmp.foot_pos["right"][k, 2]
        lfz = zmp.foot_pos["left"][k,  2]
        r_foot.set_data([zmp.foot_pos["right"][k, 0]], [rfz])
        l_foot.set_data([zmp.foot_pos["left"][k,  0]], [lfz])
        time_txt.set_text(f"t = {k * zmp.dt:.2f} s")
        return com_dot, zmp_dot, r_foot, l_foot, rod, time_txt

    anim = FuncAnimation(fig, update, frames=n_frames,
                         init_func=init, blit=False, interval=40)

    plt.close(fig)
    print(f"  saved → {path}")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="ZMP vs CoM plots and GIFs")
    parser.add_argument("--steps",    type=int,   default=6,
                        help="Number of forward footsteps  [default 6]")
    parser.add_argument("--length",   type=float, default=0.10,
                        help="Step length [m]  [default 0.10]")
    parser.add_argument("--dt",       type=float, default=0.005,
                        help="Control timestep [s]  [default 0.005]")
    parser.add_argument("--no-gif",   action="store_true",
                        help="Skip GIF generation (faster)")
    parser.add_argument("--outdir",   type=str,   default="/tmp/biped_plots",
                        help="Output folder  [default /tmp/biped_plots]")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 60)
    print("  VISUALISER — ZMP vs CoM")
    print("=" * 60)

    # ── 1. Build ZMP reference ────────────────────────────────────────
    print("\n[1/4] Building ZMP reference ...")
    planner  = ZMPPlanner(dt=args.dt, step_duration=0.8,
                          ds_ratio=0.2, swing_height=0.04)
    steps    = planner.make_forward_steps(n=args.steps,
                                          step_length=args.length)
    zmp_data = planner.plan(steps, verbose=False)
    print(f"  {zmp_data}")

    # ── 2. Integrate LIPM ─────────────────────────────────────────────
    print("\n[2/4] Integrating LIPM ...")
    lipm = LIPM(dt=args.dt, zc=GEO.pelvis_height)
    com  = _com_from_zmp(zmp_data, lipm)
    print(f"  {com}")

    # ── 3. Static plots ───────────────────────────────────────────────
    print("\n[3/4] Generating static plots ...")
    plot_timeseries(zmp_data, com, args.outdir)
    plot_topview(zmp_data, com, args.outdir)
    plot_foot_trajectories(zmp_data, args.outdir)
    plot_com_state(com, args.outdir)

    # ── 4. Animated GIFs ──────────────────────────────────────────────
    if not args.no_gif:
        print("\n[4/4] Generating GIFs ...")
        stride = max(1, zmp_data.T // 200)   # ~200 frames per GIF
        animate_topview(zmp_data, com, args.outdir, stride=stride)
        animate_sideview(zmp_data, com, args.outdir, stride=stride)
    else:
        print("\n[4/4] GIFs skipped (--no-gif).")

    print(f"\nAll outputs saved to: {args.outdir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()