"""
visualizer.py
==============

Creates animated GIFs for:
1. Top-view CoM vs ZMP trajectory
2. ZMP reference vs CoM position (X axis)
3. ZMP reference vs CoM position (Y axis)

Outputs
-------
visualization/gifs/
    ├── top_view.gif
    ├── zmp_ref_vs_com_x.gif
    └── zmp_ref_vs_com_y.gif
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


class Visualizer:

    @staticmethod
    def _make_dirs():
        os.makedirs("visualization/gifs", exist_ok=True)

    @staticmethod
    def animate_top_view(zmp_data, com_traj, fps=30):

        Visualizer._make_dirs()

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.set_title("Top View — CoM vs ZMP")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.grid(True)
        ax.axis("equal")

        # limits
        margin = 0.05

        x_all = np.concatenate([
            zmp_data.zmp_x,
            com_traj.pos_x
        ])

        y_all = np.concatenate([
            zmp_data.zmp_y,
            com_traj.pos_y
        ])

        ax.set_xlim(x_all.min() - margin, x_all.max() + margin)
        ax.set_ylim(y_all.min() - margin, y_all.max() + margin)

        # reference trajectories
        ax.plot(
            zmp_data.zmp_x,
            zmp_data.zmp_y,
            "--",
            alpha=0.5,
            label="ZMP Reference"
        )

        # animated lines
        com_line, = ax.plot([], [], linewidth=2, label="CoM")
        zmp_line, = ax.plot([], [], linewidth=2, label="ZMP")

        com_dot, = ax.plot([], [], "o")
        zmp_dot, = ax.plot([], [], "o")

        ax.legend()

        def init():
            com_line.set_data([], [])
            zmp_line.set_data([], [])
            return com_line, zmp_line

        def update(frame):

            com_line.set_data(
                com_traj.pos_x[:frame],
                com_traj.pos_y[:frame]
            )

            zmp_line.set_data(
                zmp_data.zmp_x[:frame],
                zmp_data.zmp_y[:frame]
            )

            com_dot.set_data(
                [com_traj.pos_x[frame]],
                [com_traj.pos_y[frame]]
            )

            zmp_dot.set_data(
                [zmp_data.zmp_x[frame]],
                [zmp_data.zmp_y[frame]]
            )

            return (
                com_line,
                zmp_line,
                com_dot,
                zmp_dot
            )

        anim = FuncAnimation(
            fig,
            update,
            frames=com_traj.T,
            init_func=init,
            interval=1000 / fps,
            blit=True
        )

        path = "visualization/gifs/top_view.gif"

        anim.save(
            path,
            writer=PillowWriter(fps=fps)
        )

        plt.close()

        print(f"[GIF SAVED] {path}")

    @staticmethod
    def animate_x_axis(zmp_data, com_traj, fps=30):

        Visualizer._make_dirs()

        t = np.arange(com_traj.T) * com_traj.dt

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.set_title("ZMP Reference vs CoM Position — X Axis")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("X Position [m]")
        ax.grid(True)

        ax.set_xlim(t[0], t[-1])

        y_all = np.concatenate([
            zmp_data.zmp_x,
            com_traj.pos_x
        ])

        margin = 0.05

        ax.set_ylim(
            y_all.min() - margin,
            y_all.max() + margin
        )

        ax.plot(
            t,
            zmp_data.zmp_x,
            "--",
            label="ZMP Reference X"
        )

        com_line, = ax.plot([], [], linewidth=2, label="CoM X")

        com_dot, = ax.plot([], [], "o")

        ax.legend()

        def init():
            com_line.set_data([], [])
            return com_line,

        def update(frame):

            com_line.set_data(
                t[:frame],
                com_traj.pos_x[:frame]
            )

            com_dot.set_data(
                [t[frame]],
                [com_traj.pos_x[frame]]
            )

            return com_line, com_dot

        anim = FuncAnimation(
            fig,
            update,
            frames=com_traj.T,
            init_func=init,
            interval=1000 / fps,
            blit=True
        )

        path = "visualization/gifs/zmp_ref_vs_com_x.gif"

        anim.save(
            path,
            writer=PillowWriter(fps=fps)
        )

        plt.close()

        print(f"[GIF SAVED] {path}")

    @staticmethod
    def animate_y_axis(zmp_data, com_traj, fps=30):

        Visualizer._make_dirs()

        t = np.arange(com_traj.T) * com_traj.dt

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.set_title("ZMP Reference vs CoM Position — Y Axis")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Y Position [m]")
        ax.grid(True)

        ax.set_xlim(t[0], t[-1])

        y_all = np.concatenate([
            zmp_data.zmp_y,
            com_traj.pos_y
        ])

        margin = 0.05

        ax.set_ylim(
            y_all.min() - margin,
            y_all.max() + margin
        )

        ax.plot(
            t,
            zmp_data.zmp_y,
            "--",
            label="ZMP Reference Y"
        )

        com_line, = ax.plot([], [], linewidth=2, label="CoM Y")

        com_dot, = ax.plot([], [], "o")

        ax.legend()

        def init():
            com_line.set_data([], [])
            return com_line,

        def update(frame):

            com_line.set_data(
                t[:frame],
                com_traj.pos_y[:frame]
            )

            com_dot.set_data(
                [t[frame]],
                [com_traj.pos_y[frame]]
            )

            return com_line, com_dot

        anim = FuncAnimation(
            fig,
            update,
            frames=com_traj.T,
            init_func=init,
            interval=1000 / fps,
            blit=True
        )

        path = "visualization/gifs/zmp_ref_vs_com_y.gif"

        anim.save(
            path,
            writer=PillowWriter(fps=fps)
        )

        plt.close()

        print(f"[GIF SAVED] {path}")


# =========================================================
# TEST
# =========================================================
if __name__ == "__main__":
    import os 
    import sys
    sys.path.append( os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    from dynamics.zmp import ZMPPlanner
    from dynamics.lipm import LIPM

    planner = ZMPPlanner(
        dt=0.005,
        step_duration=0.6,
        ds_ratio=0.2,
        swing_height=0.05
    )

    steps = planner.make_forward_steps(
        n=6,
        step_length=0.1
    )

    zmp_data = planner.plan(steps)

    T = zmp_data.T

    zmp_x = zmp_data.zmp_x
    zmp_y = zmp_data.zmp_y

    lipm = LIPM(dt=0.005)

    com_traj = lipm.integrate(zmp_x, zmp_y)

    Visualizer.animate_top_view(zmp_data, com_traj)

    Visualizer.animate_x_axis(zmp_data, com_traj)

    Visualizer.animate_y_axis(zmp_data, com_traj)