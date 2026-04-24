"""
Inverse Kinematics for one leg of the 12-DOF biped.

Jacobian pseudo-inverse refinement
"""

import numpy as np
from typing import Optional, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from kinematics.forward_kinematics import ForwardKinematics


class InverseKinematics:
    def __init__(self, side: str = "right",
                 fk: Optional[ForwardKinematics] = None):
        assert side in ("right", "left")
        self.side = side
        self.lat = -1.0 if side == "right" else 1.0
        self.fk = fk if fk is not None else ForwardKinematics(side)


    def newton_raphson(self,
                       q: np.ndarray,
                       p_target: np.ndarray,
                       R_target: Optional[np.ndarray] = None,
                       max_iter: int = 60,
                       tol: float = 1e-4) -> Tuple[np.ndarray, float]:

        for _ in range(max_iter):

            T_cur = self.fk.compute(q)
            p_cur = T_cur[:3, 3]
            R_cur = T_cur[:3, :3]

            e_p = p_target - p_cur

            if R_target is not None:
                R_err = R_target @ R_cur.T
                e_r = 0.5 * np.array([
                    R_err[2, 1] - R_err[1, 2],
                    R_err[0, 2] - R_err[2, 0],
                    R_err[1, 0] - R_err[0, 1],
                ])
            else:
                e_r = np.zeros(3)

            e = np.concatenate([e_p, e_r])

            if np.linalg.norm(e) < tol:
                break

            J = self.fk.jacobian(q)

            J_pinv = np.linalg.pinv(J)

            dq = J_pinv @ e
            dq *= 0.5 
            
            q = q + dq
            q = np.clip(q, -np.pi, np.pi)

        final_error = np.linalg.norm(self.fk.position(q) - p_target)
        return q, final_error

    # ---------------- USER API ----------------

    def solve(self,
              p_foot: np.ndarray,
              R_foot: Optional[np.ndarray] = None,
              q_init: Optional[np.ndarray] = None,
              verbose: bool = False) -> Tuple[np.ndarray, bool]:

        if R_foot is None:
            R_foot = np.eye(3)

        if q_init is None:
            q = np.zeros(6)
        else:
            q = q_init.copy()

        q, final_error = self.newton_raphson(q, p_foot, R_foot)

        success = final_error < 1e-3

        if verbose:
            self.print_result(q, p_foot, final_error, success)

        return q, success

    # ---------------- DEBUG ----------------

    def print_result(self,
                     q: np.ndarray,
                     p_target: np.ndarray,
                     error: float,
                     success: bool) -> None:

        p_achieved = self.fk.position(q)

        print(f"\n[IK debug | {self.side} leg]")
        print(f"target pos   : {np.round(p_target, 6)}")
        print(f"achieved pos : {np.round(p_achieved, 6)}")
        print(f"error        : {error * 1000:.4f} mm")
        print(f"q (deg)      : {np.round(np.degrees(q), 3)}")
        print(f"success      : {success}")


# ---------------- TEST ----------------

if __name__ == "__main__":

    print("=== IK self-test (FK → IK round-trip) ===\n")

    for side in ("right", "left"):

        fk = ForwardKinematics(side)
        ik = InverseKinematics(side, fk)

        print(f"── {side.upper()} leg ──")

        q_ref = np.array([0.05, -0.35, 0.0, 0.70, -0.35, -0.05])

        T_ref = fk.compute(q_ref)
        p_ref = T_ref[:3, 3]
        R_ref = T_ref[:3, :3]


        q_sol, ok = ik.solve(p_ref, R_ref, q_init=q_ref, verbose=True)

        p_sol = fk.position(q_sol)
        err = np.linalg.norm(p_sol - p_ref) * 1000

       