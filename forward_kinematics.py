"""
Forward Kinematics for one leg of the 12-DOF biped.
All transforms are built from elementary rotation matrices (Rx, Ry, Rz)
and fixed translation vectors.
"""

import numpy as np
from robot_config import GEO, Rx, Ry, Rz, homogeneous


class ForwardKinematics:

    def __init__(self, side: str = "right"):
        assert side in ("right", "left"), "side must be 'right' or 'left'"
        self.side = side
        # Right: hip is at -Y; Left: hip is at +Y
        self.lat = -1.0 if side == "right" else 1.0

        # Fixed translation vectors (constant, precomputed once)
        self._t_pelvis_hip = np.array([0.0, self.lat * GEO.hip_width, 0.0])
        self._t_hip_knee   = np.array([0.0, 0.0, -GEO.thigh_length])
        self._t_knee_ankle = np.array([0.0, 0.0, -GEO.shank_length])
        self._t_ankle_foot = np.array([0.0, 0.0, -GEO.foot_height])

    def compute(self, q: np.ndarray) -> np.ndarray:
        
        
        if q.shape != (6,):
            raise ValueError(f"Expected q of shape (6,), got {q.shape}")

        # pelvis → hip (fixed translation)
        T = homogeneous(np.eye(3), self._t_pelvis_hip)

        # hip rotation: roll → pitch → yaw
        T = T @ homogeneous(Rx(q[0]) @ Ry(q[1]) @ Rz(q[2]), np.zeros(3))

        # thigh segment
        T = T @ homogeneous(np.eye(3), self._t_hip_knee)

        # knee rotation: pitch only
        T = T @ homogeneous(Ry(q[3]), np.zeros(3))

        # shank segment
        T = T @ homogeneous(np.eye(3), self._t_knee_ankle)

        # ankle rotation: pitch → roll
        T = T @ homogeneous(Ry(q[4]) @ Rx(q[5]), np.zeros(3))

        # foot sole offset
        T = T @ homogeneous(np.eye(3), self._t_ankle_foot)

        return T

    def position(self, q: np.ndarray) -> np.ndarray:
        return self.compute(q)[:3, 3]

    def rotation(self, q: np.ndarray) -> np.ndarray:
        return self.compute(q)[:3, :3]

    def jacobian(self, q: np.ndarray, delta: float = 1e-7) -> np.ndarray:
        
        J  = np.zeros((6, 6))
        T0 = self.compute(q)
        R0 = T0[:3, :3]

        for i in range(6):
            dq    = np.zeros(6)
            dq[i] = delta

            T_p = self.compute(q + dq)
            T_m = self.compute(q - dq)

            # linear rows  (central difference on position)
            J[:3, i] = (T_p[:3, 3] - T_m[:3, 3]) / (2.0 * delta)

            # angular rows  (skew-symmetric part of dR · R0ᵀ)
            dR      = (T_p[:3, :3] - T_m[:3, :3]) / (2.0 * delta)
            S       = dR @ R0.T     # skew-symmetric approximation
            J[3, i] = S[2, 1]       # ω_x
            J[4, i] = S[0, 2]       # ω_y
            J[5, i] = S[1, 0]       # ω_z

        return J

    def print_state(self, q: np.ndarray) -> None:
    
        T = self.compute(q)
        p = T[:3, 3]
        R = T[:3, :3]
        J = self.jacobian(q)
        print(f"\n[FK debug | {self.side} leg]")
        print(f"  q (deg)  : {np.round(np.degrees(q), 3)}")
        print(f"  position : {np.round(p, 6)} m")
        print(f"  rotation :\n{np.round(R, 4)}")
        print(f"  Jacobian cond. number : {np.linalg.cond(J):.4f}")


if __name__ == "__main__":
    print("=== Forward Kinematics self-test ===\n")

    for side in ("right", "left"):
        fk = ForwardKinematics(side)

        # zero configuration — leg straight down
        q0 = np.zeros(6)
        print(f"── {side.upper()} leg | q = 0 ──")
        fk.print_state(q0)

        # bent configuration
        q_bent = np.array([0.05, -0.35, 0.0, 0.70, -0.35, -0.05])
        print(f"\n── {side.upper()} leg | q = bent ──")
        fk.print_state(q_bent)

        print()