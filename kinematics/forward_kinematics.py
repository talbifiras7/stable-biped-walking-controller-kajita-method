"""
Forward Kinematics for one leg of a 12-DOF biped.
Clean symmetric implementation (left/right mirrored correctly).
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.robot_config import GEO, Rx, Ry, Rz, homogeneous

class ForwardKinematics:

    def __init__(self, side: str = "right"):
        assert side in ("right", "left")
        self.side = side

        # pelvis offset
        self.lat = -1.0 if side == "right" else 1.0
        self._t_pelvis_hip = np.array([0.0, self.lat * GEO.hip_width, 0.0])

        # fixed links
        self._t_hip_knee   = np.array([0.0, 0.0, -GEO.thigh_length])
        self._t_knee_ankle = np.array([0.0, 0.0, -GEO.shank_length])
        self._t_ankle_foot = np.array([0.0, 0.0, -GEO.foot_height])

    def compute(self, q: np.ndarray) -> np.ndarray:

        if q.shape != (6,):
            raise ValueError("q must be shape (6,)")

        # base
        T = homogeneous(np.eye(3), self._t_pelvis_hip)

        # ---------------- HIP ----------------
        if self.side == "left":
            R_hip = Rx(-q[0]) @ Ry(q[1]) @ Rz(-q[2])
        else:
            R_hip = Rx(q[0]) @ Ry(q[1]) @ Rz(q[2])

        T = T @ homogeneous(R_hip, np.zeros(3))

        # thigh
        T = T @ homogeneous(np.eye(3), self._t_hip_knee)

        # knee (same both sides)
        T = T @ homogeneous(Ry(q[3]), np.zeros(3))

        # shank
        T = T @ homogeneous(np.eye(3), self._t_knee_ankle)

        # ---------------- ANKLE ----------------
        if self.side == "left":
            R_ankle = Ry(q[4]) @ Rx(-q[5])
        else:
            R_ankle = Ry(q[4]) @ Rx(q[5])

        T = T @ homogeneous(R_ankle, np.zeros(3))

        # foot
        T = T @ homogeneous(np.eye(3), self._t_ankle_foot)

        return T

    def position(self, q: np.ndarray) -> np.ndarray:
        return self.compute(q)[:3, 3]

    def rotation(self, q: np.ndarray) -> np.ndarray:
        return self.compute(q)[:3, :3]

    def jacobian(self, q: np.ndarray, delta: float = 1e-7) -> np.ndarray:

        J = np.zeros((6, 6))
        T0 = self.compute(q)
        R0 = T0[:3, :3]

        for i in range(6):
            dq = np.zeros(6)
            dq[i] = delta

            T_p = self.compute(q + dq)
            T_m = self.compute(q - dq)

            # linear velocity
            J[:3, i] = (T_p[:3, 3] - T_m[:3, 3]) / (2 * delta)

            # angular velocity
            dR = (T_p[:3, :3] - T_m[:3, :3]) / (2 * delta)
            S = dR @ R0.T

            J[3, i] = S[2, 1]
            J[4, i] = S[0, 2]
            J[5, i] = S[1, 0]

        return J

    def print_state(self, q: np.ndarray):

        T = self.compute(q)
        p = T[:3, 3]
        R = T[:3, :3]
        J = self.jacobian(q)

        print(f"\n[FK | {self.side} leg]")
        print("q (deg):", np.round(np.degrees(q), 3))
        print("position:", np.round(p, 6))
        print("rotation:\n", np.round(R, 3))
        print("cond(J):", np.linalg.cond(J))


    # ---------------- PLOT ----------------

    def plot_feet(self,q):

        fk_r = ForwardKinematics("right")
        p_r = fk_r.position(q)

        fk_l= ForwardKinematics("left")
        p_l = fk_l.position(q)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ax.scatter(0, 0, 0)

        ax.scatter(*p_r)
        ax.scatter(*p_l)

        ax.plot([0, p_r[0]], [0, p_r[1]], [0, p_r[2]])
        ax.plot([0, p_l[0]], [0, p_l[1]], [0, p_l[2]])

        ax.text(*p_r, "Right")
        ax.text(*p_l, "Left")

        ax.set_title("Feet positions")

        plt.show()
        
        image_path = "visualization/plots/fk/feet_positions.png"
        fig.savefig(image_path, dpi=300, bbox_inches="tight")


# ---------------- TEST ----------------

if __name__ == "__main__":

    fk = ForwardKinematics("right")

    q0 = np.zeros(6)
    print("ZERO POSE RIGHT")
    fk.print_state(q0)

    fk = ForwardKinematics("left")
    print("ZERO POSE LEFT")
    fk.print_state(q0)

    q_bent = np.array([0.05, -0.35, 0.0, 0.70, -0.35, -0.05])

    fk = ForwardKinematics("right")
    fk.print_state(q_bent)

    fk = ForwardKinematics("left")
    fk.print_state(q_bent)
    
    fk=ForwardKinematics()
    fk.plot_feet(q_bent)