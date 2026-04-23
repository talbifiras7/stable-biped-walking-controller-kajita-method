"""
Inverse Kinematics for one leg of the 12-DOF biped.

Jacobian pseudo-inverse refinement
Iterates  Δq = J⁺ · [e_pos ; e_rot]  until position error < 1 mm.
Corrects the small error left by the analytical approximations.

"""

import numpy as np
from typing import Optional, Tuple
from kinematics.forward_kinematics import ForwardKinematics
from configs.robot_config import GEO, Rx, Ry, Rz


class InverseKinematics:
    def __init__(self, side: str = "right",
                 fk: Optional[ForwardKinematics] = None):
        assert side in ("right", "left"), "side must be 'right' or 'left'"
        self.side = side
        self.lat  = -1.0 if side == "right" else 1.0
        self.fk   =ForwardKinematics(side)

    def newton_raphson(self, q: np.ndarray,
                p_target: np.ndarray,
                R_target: Optional[np.ndarray] = None,
                max_iter: int = 60,
                tol: float = 1e-4) -> Tuple[np.ndarray, float]:
    
    
        for iteration in range(max_iter):
            T_cur = self.fk.compute(q)
            p_cur = T_cur[:3, 3]
            R_cur = T_cur[:3, :3]

            # position error
            e_p = p_target - p_cur

            # orientation error (axis-angle from skew part of R_err)
            if R_target is not None:
                R_err = R_target @ R_cur.T
                e_r   = 0.5 * np.array([
                    R_err[2, 1] - R_err[1, 2],
                    R_err[0, 2] - R_err[2, 0],
                    R_err[1, 0] - R_err[0, 1],
                ])
            else:
                e_r = np.zeros(3)

            err_norm = np.linalg.norm(e_p)
            if err_norm < tol:
                break

            # 6×6 Jacobian and its pseudo-inverse
            J  = self.fk.jacobian(q)
            Jp = np.linalg.pinv(J)

            # joint update — clamped step
            dq = np.clip(Jp @ np.concatenate([e_p, e_r]), -0.05, 0.05)
            q  = np.clip(q + dq, -np.pi, np.pi)

        final_error = np.linalg.norm(self.fk.position(q) - p_target)
        return q, final_error

    def solve(self, p_foot: np.ndarray,
              R_foot: Optional[np.ndarray] = None,
              q_init: Optional[np.ndarray] = None,
              verbose: bool = False) -> Tuple[np.ndarray, bool]:
        
        
        if R_foot is None:
            R_foot = np.eye(3)

        q, final_error = self.newton_raphson(q, p_foot, R_foot)

        success = final_error < 1e-3

        if verbose:
            self.print_result(q, p_foot, final_error, success)

        return q, success

    def print_result(self, q: np.ndarray, p_target: np.ndarray,
                     error: float, success: bool) -> None:
        """Debug helper — print IK solve summary."""
        p_achieved = self.fk.position(q)
        print(f"\n[IK debug | {self.side} leg]")
        print(f"  target pos   : {np.round(p_target,   6)} m")
        print(f"  achieved pos : {np.round(p_achieved, 6)} m")
        print(f"  position err : {error * 1000:.4f} mm")
        print(f"  q (deg)      : {np.round(np.degrees(q), 3)}")
        print(f"  success      : {success}")


if __name__ == "__main__":
    print("=== Inverse Kinematics self-test (FK → IK round-trip) ===\n")

    for side in ("right", "left"):
        fk = ForwardKinematics(side)
        ik = InverseKinematics(side, fk)

        print(f"── {side.upper()} leg ──")

        # Generate a target from a known joint config via FK
        q_ref = np.array([0.05, -0.35, 0.0, 0.70, -0.35, -0.05])
        T_ref  = fk.compute(q_ref)
        p_ref  = T_ref[:3, 3]
        R_ref  = T_ref[:3, :3]

        print(f"  Reference q (deg) : {np.round(np.degrees(q_ref), 2)}")

        # Solve IK from the FK-generated target
        q_sol, ok = ik.solve(q_ref,p_ref, R_ref, verbose=True)

        # Verify the solution with FK
        p_sol = fk.position(q_sol)
        err   = np.linalg.norm(p_sol - p_ref) * 1000
        print(f"  Final pos error   : {err:.4f} mm\n")