"""
Inverse Kinematics for one leg of a 12-DOF biped
Jacobian pseudo-inverse + debugging + convergence plot
"""

import numpy as np
from typing import Optional, Tuple
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from kinematics.forward_kinematics import ForwardKinematics


class InverseKinematics:
    def __init__(self, side: str = "right",
                 fk: Optional[ForwardKinematics] = None):

        assert side in ("right", "left")
        self.side = side
        self.fk = fk if fk is not None else ForwardKinematics(side)

    # ---------------- CORE IK ----------------

    def newton_raphson(self,
                       q: np.ndarray,
                       p_target: np.ndarray,
                       R_target: Optional[np.ndarray] = None,
                       max_iter: int = 60,
                       tol: float = 1e-4):

        errors = []

        for i in range(max_iter):

            T = self.fk.compute(q)
            p = T[:3, 3]
            R = T[:3, :3]

            # position error
            e_p = p_target - p

            # orientation error
            if R_target is not None:
                R_err = R_target @ R.T
                e_r = 0.5 * np.array([
                    R_err[2, 1] - R_err[1, 2],
                    R_err[0, 2] - R_err[2, 0],
                    R_err[1, 0] - R_err[0, 1],
                ])
            else:
                e_r = np.zeros(3)

            e = np.concatenate([e_p, e_r])
            errors.append(np.linalg.norm(e_p))

            # convergence check
            if np.linalg.norm(e) < tol:
                break

            # Jacobian
            J = self.fk.jacobian(q)
            J_pinv = np.linalg.pinv(J)

            # DEBUG (clean + reliable)
            if i == 0 or i == max_iter - 1:
                print("\n" + "=" * 60)
                print(f"[{self.side.upper()} LEG] ITERATION {i}")

                print("\nJacobian J shape:", J.shape)
                print(J)

                print("\nPseudo-inverse J+ shape:", J_pinv.shape)
                print(J_pinv)
                print("=" * 60)

            # update rule
            dq = J_pinv @ e
            dq *= 0.5  # damping

            q = q + dq
            q = np.clip(q, -np.pi, np.pi)

        final_error = np.linalg.norm(self.fk.position(q) - p_target)

        return q, final_error, errors

    # ---------------- PUBLIC API ----------------

    def solve(self,
              p_foot: np.ndarray,
              R_foot: Optional[np.ndarray] = None,
              q_init: Optional[np.ndarray] = None,
              verbose: bool = False):

        if R_foot is None:
            R_foot = np.eye(3)

        q = np.zeros(6) if q_init is None else q_init.copy()

        q, final_error, errors = self.newton_raphson(q, p_foot, R_foot)

        success = final_error < 1e-3

        if verbose:
            self.print_result(q, p_foot, final_error, success)
            self.plot_convergence(errors)

        return q, success

    # ---------------- DEBUG PRINT ----------------

    def print_result(self,
                     q: np.ndarray,
                     p_target: np.ndarray,
                     error: float,
                     success: bool):

        p_final = self.fk.position(q)

        print(f"\n[IK RESULT - {self.side.upper()} LEG]")
        print("Target   :", np.round(p_target, 5))
        print("Achieved :", np.round(p_final, 5))
        print("Error mm :", error * 1000)
        print("q (deg)  :", np.round(np.degrees(q), 2))
        print("Success  :", success)

    # ---------------- PLOT ----------------

    def plot_convergence(self, errors):

        if len(errors) == 0:
            print("[WARN] No convergence data")
            return
 
        fig,ax = plt.subplots()
        plt.plot(errors, marker='o')
        plt.title(f"IK Convergence - {self.side} leg")
        plt.xlabel("Iteration")
        plt.ylabel("Position Error")
        plt.grid(True)
        
        image_path = f"visualization/plots/ik/convergence_{self.side}leg.png"        
        fig.savefig(image_path, dpi=300, bbox_inches="tight")
        
        plt.show(block=True)
        plt.close(fig)
        
        


# ---------------- TEST ----------------

if __name__ == "__main__":

    print("=== IK SELF TEST ===\n")

    for side in ("right", "left"):

        fk = ForwardKinematics(side)
        ik = InverseKinematics(side, fk)

        print(f"\n--- {side.upper()} LEG ---")

        q_ref = np.array([0.05, -0.35, 0.0, 0.70, -0.35, -0.05])

        T = fk.compute(q_ref)
        p = T[:3, 3]
        R = T[:3, :3]

        q_sol, ok = ik.solve(p, R, q_init=q_ref, verbose=True)