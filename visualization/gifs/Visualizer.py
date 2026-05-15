"""
visualizer.py
==============

GIF visualization for:
1. Top-view CoM / ZMP / Footsteps
2. X-axis CoM vs ZMP
3. Y-axis CoM vs ZMP

Compatible with:
- zmp.py
- preview_controler.py
- lipm.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


# =========================================================
# IMPORT FIX
# =========================================================
ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

sys.path.append(ROOT_DIR)

from dynamics.zmp import ZMPPlanner
from controler.preview_controler import PreviewController
from dynamics.lipm import LIPM


# =========================================================
# VISUALIZER
# =========================================================
class Visualizer:

    # =====================================================
    # TOP VIEW
    # =====================================================
    @staticmethod
    def animate_top_view(zmp_data, com_traj, fps=30):

        fig, ax = plt.subplots(figsize=(10, 7))

        ax.set_title("Top View — CoM / ZMP / Footsteps")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.grid(True)
        ax.axis("equal")

        margin = 0.08

        x_all = np.concatenate([
            zmp_data.zmp_x,
            com_traj.pos_x,
            zmp_data.foot_pos["left"][:, 0],
            zmp_data.foot_pos["right"][:, 0]
        ])

        y_all = np.concatenate([
            zmp_data.zmp_y,
            com_traj.pos_y,
            zmp_data.foot_pos["left"][:, 1],
            zmp_data.foot_pos["right"][:, 1]
        ])

        ax.set_xlim(x_all.min() - margin, x_all.max() + margin)
        ax.set_ylim(y_all.min() - margin, y_all.max() + margin)

        # reference ZMP
        ax.plot(
            zmp_data.zmp_x,
            zmp_data.zmp_y,
            "k--",
            linewidth=1.5,
            label="ZMP Reference"
        )

        # animated lines
        com_line, = ax.plot([], [], linewidth=2.5, label="CoM")
        zmp_line, = ax.plot([], [], linewidth=2.0, label="Generated ZMP")

        # animated points
        com_dot, = ax.plot([], [], "o", markersize=8)
        zmp_dot, = ax.plot([], [], "o", markersize=6)

        # feet
        left_foot, = ax.plot([], [], "s", markersize=10, label="Left Foot")
        right_foot, = ax.plot([], [], "s", markersize=10, label="Right Foot")

        ax.legend()

        def init():

            com_line.set_data([], [])
            zmp_line.set_data([], [])

            return com_line, zmp_line

        def update(frame):

            # CoM path
            com_line.set_data(
                com_traj.pos_x[:frame],
                com_traj.pos_y[:frame]
            )

            # ZMP path
            zmp_line.set_data(
                zmp_data.zmp_x[:frame],
                zmp_data.zmp_y[:frame]
            )

            # current CoM
            com_dot.set_data(
                [com_traj.pos_x[frame]],
                [com_traj.pos_y[frame]]
            )

            # current ZMP
            zmp_dot.set_data(
                [zmp_data.zmp_x[frame]],
                [zmp_data.zmp_y[frame]]
            )

            # left foot
            left_foot.set_data(
                [zmp_data.foot_pos["left"][frame, 0]],
                [zmp_data.foot_pos["left"][frame, 1]]
            )

            # right foot
            right_foot.set_data(
                [zmp_data.foot_pos["right"][frame, 0]],
                [zmp_data.foot_pos["right"][frame, 1]]
            )

            return (
                com_line,
                zmp_line,
                com_dot,
                zmp_dot,
                left_foot,
                right_foot
            )

        anim = FuncAnimation(
            fig,
            update,
            frames=com_traj.T,
            init_func=init,
            interval=1000 / fps,
            blit=True
        )

        plt.show()
        path=f'visualization/gifs/top_view.gif'
        anim.save(path, writer=PillowWriter(fps=fps))

    # =====================================================
    # X AXIS
    # =====================================================
    @staticmethod
    def animate_x_axis(zmp_data, com_traj, fps=30):

        t = np.arange(com_traj.T) * com_traj.dt

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.set_title("X Axis — CoM vs ZMP")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("X Position [m]")
        ax.grid(True)

        ax.plot(
            t,
            zmp_data.zmp_x,
            "k--",
            linewidth=2,
            label="ZMP Reference"
        )

        com_line, = ax.plot([], [], linewidth=2.5, label="CoM")
        zmp_line, = ax.plot([], [], linewidth=2.0, label="Generated ZMP")

        ax.legend()

        def init():

            com_line.set_data([], [])
            zmp_line.set_data([], [])

            return com_line, zmp_line

        def update(frame):

            com_line.set_data(
                t[:frame],
                com_traj.pos_x[:frame]
            )

            zmp_line.set_data(
                t[:frame],
                zmp_data.zmp_x[:frame]
            )

            return com_line, zmp_line

        anim = FuncAnimation(
            fig,
            update,
            frames=com_traj.T,
            init_func=init,
            interval=1000 / fps,
            blit=True
        )

        plt.show()
        path=f'visualization/gifs/x_axis.gif'
        anim.save(path, writer=PillowWriter(fps=fps))

    # =====================================================
    # Y AXIS
    # =====================================================
    @staticmethod
    def animate_y_axis(zmp_data, com_traj, fps=30):

        t = np.arange(com_traj.T) * com_traj.dt

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.set_title("Y Axis — CoM vs ZMP")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Y Position [m]")
        ax.grid(True)

        ax.plot(
            t,
            zmp_data.zmp_y,
            "k--",
            linewidth=2,
            label="ZMP Reference"
        )

        com_line, = ax.plot([], [], linewidth=2.5, label="CoM")
        zmp_line, = ax.plot([], [], linewidth=2.0, label="Generated ZMP")

        ax.legend()

        def init():

            com_line.set_data([], [])
            zmp_line.set_data([], [])

            return com_line, zmp_line

        def update(frame):

            com_line.set_data(
                t[:frame],
                com_traj.pos_y[:frame]
            )

            zmp_line.set_data(
                t[:frame],
                zmp_data.zmp_y[:frame]
            )

            return com_line, zmp_line

        anim = FuncAnimation(
            fig,
            update,
            frames=com_traj.T,
            init_func=init,
            interval=1000 / fps,
            blit=True
        )

        plt.show()
        path=f'visualization/gifs/y_axis.gif'
        anim.save(path, writer=PillowWriter(fps=fps))


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    # -----------------------------------------------------
    # ZMP PLANNER
    # -----------------------------------------------------
    planner = ZMPPlanner(
        dt=0.01,
        step_duration=0.8,
        ds_ratio=0.2,
        swing_height=0.05
    )

    footsteps = planner.make_forward_steps(
        n=8,
        step_length=0.15
    )

    zmp_data = planner.plan(footsteps)

    # -----------------------------------------------------
    # PREVIEW CONTROLLER
    # -----------------------------------------------------
    controller = PreviewController(
        dt=0.01,
        N=200
    )

    u_x, u_y = controller.compute(zmp_data)

    # -----------------------------------------------------
    # LIPM
    # -----------------------------------------------------
    lipm = LIPM(dt=0.01)

    com_traj = lipm.integrate(u_x, u_y)

    # -----------------------------------------------------
    # VISUALIZATION
    # -----------------------------------------------------
    Visualizer.animate_top_view(zmp_data, com_traj)

    Visualizer.animate_x_axis(zmp_data, com_traj)

    Visualizer.animate_y_axis(zmp_data, com_traj)

    print("\nGIF generation complete.")