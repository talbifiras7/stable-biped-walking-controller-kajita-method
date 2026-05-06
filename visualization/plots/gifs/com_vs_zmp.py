"""
visualization.py
================
Main visualization script:
- Generates ZMP trajectory
- Runs LIPM to get CoM
- Plots CoM vs ZMP
- Creates GIF animations
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dynamics.lipm import LIPM
from dynamics.zmp import ZMPPlanner  


# ─────────────────────────────────────────────
# VISUALIZER CLASS
# ─────────────────────────────────────────────
class BipedVisualizer:

    @staticmethod
    def compare(com_traj, zmp_ref):
        T = com_traj.T
        t = np.arange(T) * com_traj.dt

        # X axis
        plt.figure()
        plt.title("X axis: CoM vs ZMP")
        plt.plot(t, com_traj.pos_x, label="CoM x")
        plt.plot(t, com_traj.zmp_x, "--", label="ZMP actual x")
        plt.plot(t, zmp_ref.x, ":", label="ZMP ref x")
        plt.xlabel("Time [s]")
        plt.ylabel("Position [m]")
        plt.legend()
        plt.grid(True)

        # Y axis
        plt.figure()
        plt.title("Y axis: CoM vs ZMP")
        plt.plot(t, com_traj.pos_y, label="CoM y")
        plt.plot(t, com_traj.zmp_y, "--", label="ZMP actual y")
        plt.plot(t, zmp_ref.y, ":", label="ZMP ref y")
        plt.xlabel("Time [s]")
        plt.ylabel("Position [m]")
        plt.legend()
        plt.grid(True)

        plt.show()

    @staticmethod
    def animate(com_traj, zmp_ref, save_path="biped.gif"):
        T = com_traj.T

        fig, ax = plt.subplots()
        ax.set_title("CoM vs ZMP (Top View)")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.grid(True)
        ax.axis("equal")

        margin = 0.1
        ax.set_xlim(com_traj.pos_x.min() - margin,
                    com_traj.pos_x.max() + margin)
        ax.set_ylim(com_traj.pos_y.min() - margin,
                    com_traj.pos_y.max() + margin)

        # Points
        com_point, = ax.plot([], [], "bo", label="CoM")
        zmp_point, = ax.plot([], [], "ro", label="ZMP")

        # Trails
        com_line, = ax.plot([], [], "b--", lw=1)
        zmp_line, = ax.plot([], [], "r--", lw=1)

        # ZMP reference
        zmp_ref_line, = ax.plot([], [], "g:", lw=1, label="ZMP ref")

        ax.legend()

        def init():
            com_point.set_data([], [])
            zmp_point.set_data([], [])
            com_line.set_data([], [])
            zmp_line.set_data([], [])
            zmp_ref_line.set_data([], [])
            return com_point, zmp_point, com_line, zmp_line, zmp_ref_line

        def update(k):
            com_point.set_data(com_traj.pos_x[k], com_traj.pos_y[k])
            zmp_point.set_data(com_traj.zmp_x[k], com_traj.zmp_y[k])

            com_line.set_data(com_traj.pos_x[:k], com_traj.pos_y[:k])
            zmp_line.set_data(com_traj.zmp_x[:k], com_traj.zmp_y[:k])

            zmp_ref_line.set_data(zmp_ref.x[:k], zmp_ref.y[:k])

            return com_point, zmp_point, com_line, zmp_line, zmp_ref_line

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=T,
            init_func=init,
            interval=20,
            blit=True
        )

        writer = animation.PillowWriter(fps=30)
        ani.save(save_path, writer=writer)

        print(f"GIF saved → {save_path}")


# ─────────────────────────────────────────────
# MAIN PROGRAM
# ─────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 50)
    print("   BIPED WALKING PIPELINE TEST")
    print("=" * 50)

    # ── 1. Create ZMP reference ─────────────────────
    planner = ZMPPlanner(dt=0.005)

    steps = ZMPPlanner.make_forward_steps(
        n=6,
        step_length=0.2,
        step_width=0.1
    )

    zmp_data = planner.generate(steps)

    # ── 2. Create dummy jerk (for now) ──────────────
    T = zmp_data.T
    u_x = np.zeros(T)
    u_y = np.zeros(T)

    # (later: replace with preview control)

    # ── 3. Run LIPM ────────────────────────────────
    lipm = LIPM(dt=zmp_data.dt)

    com_traj = lipm.integrate(u_x, u_y)

    # ── 4. Visualization ───────────────────────────
    BipedVisualizer.compare(com_traj, zmp_data)
    BipedVisualizer.animate(com_traj, zmp_data)