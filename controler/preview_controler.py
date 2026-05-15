"""
preview_controller.py
=====================

Computes optimal jerk inputs  u_x, u_y  that drive the LIPM CoM
to track a ZMP reference sequence.  The jerk inputs are then fed
into  LIPM.integrate()  to produce the full CoM trajectory.

----------------------
  LIPM state:  x = [p, ṗ, p̈]   (position, velocity, acceleration)
  ZMP output:  p_zmp = C·x = p − (zc/g)·p̈
  Jerk input:  u

  Augmented state to track integral of ZMP error:
      X = [eᵢ, p, ṗ, p̈]   (4-dimensional)

  where  eᵢ(k+1) = eᵢ(k) + C·x(k) − p_ref(k)

  Augmented system:
      Ã = [[1,  C·A],     B̃ = [[C·B],
           [0,   A ]]          [ B  ]]

  LQR cost:  Σ  Q_e · eᵢ²  +  R_u · u²

  Optimal gains from DARE:
      K   = (R_u + B̃ᵀ P B̃)⁻¹ B̃ᵀ P Ã    →  split into Gi, Gx, Gp

  Control law  (Kajita 2003, eq. 26):
      u_k = −Gi · eᵢ  −  Gx · x_k  −  Σⱼ₌₀ᴺ⁻¹  Gp[j] · p_ref[k+j]

  Note on Gp computation
  ----------------------
  Gp is computed using FORWARD powers of the closed-loop matrix A_cl:
      Gp[j] = −K · A_cl^j · b₀     b₀ = [1, 0, 0, 0]
  This ensures Gp → 0 as j → N (gains decay, bounded preview sum).
  Using A_cl.T instead produces diverging gains — forward powers are correct.

Stand-alone test
----------------
  python preview_controller.py
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

from scipy.linalg import solve_discrete_are
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__))) 

from configs.robot_config import GEO


# PREVIEW CONTROLLER
class PreviewController:
    """""
    Takes a ZMPData object, outputs jerk arrays (u_x, u_y) for LIPM.

    Parameters
    ----------
    dt  : control period         [s]   must match ZMPData.dt and LIPM.dt
    zc  : constant CoM height    [m]   must match LIPM.zc
    N   : preview horizon        [steps]   default 320  (~1.6 s at dt=0.005)
    Q_e : LQR weight on ZMP integral error   default 1.0
    R_u : LQR weight on jerk input           default 1e-6
    """

    def __init__(self,
                 dt:  float = 0.005,
                 zc:  float = None,
                 N:   int   = 320,
                 Q_e: float = 1.0,
                 R_u: float = 1e-6):

        self.dt  = dt
        self.zc  = zc if zc is not None else GEO.pelvis_height
        self.N   = N
        self.Q_e = Q_e
        self.R_u = R_u
        self.g   = 9.81

        # ── LIPM matrices  (1-D / 2-D, no column vectors) ─────────────
        self.A = np.array([[1.,  dt,  dt**2 / 2.],
                           [0.,  1.,  dt         ],
                           [0.,  0.,  1.         ]])       # (3,3)

        self.B = np.array([dt**3 / 6., dt**2 / 2., dt])   # (3,) flat

        self.C = np.array([1., 0., -self.zc / self.g])    # (3,) flat

        # ── Augmented system  (4-D) ───────────────────────────────────
        CA = self.C @ self.A    # (3,) flat
        CB = float(self.C @ self.B)

        self._A_aug = np.array(
            [[1.,    CA[0],        CA[1],        CA[2]       ],
             [0.,    self.A[0,0],  self.A[0,1],  self.A[0,2] ],
             [0.,    self.A[1,0],  self.A[1,1],  self.A[1,2] ],
             [0.,    self.A[2,0],  self.A[2,1],  self.A[2,2] ]])  # (4,4)

        self._B_aug = np.array(
            [CB, self.B[0], self.B[1], self.B[2]])  # (4,) flat

        # ── Solve DARE and store gains ────────────────────────────────
        self.Gi, self.Gx, self.Gp = self._solve_dare()

    # GAIN COMPUTATION

    def _solve_dare(self) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Solve the discrete-time algebraic Riccati equation (DARE) for
        the augmented system and compute gains Gi, Gx, Gp.

        DARE:
            P = Ãᵀ P Ã − Ãᵀ P B̃ (R_u + B̃ᵀ P B̃)⁻¹ B̃ᵀ P Ã + Q_aug

        Optimal gain row:
            K = (R_u + B̃ᵀ P B̃)⁻¹ B̃ᵀ P Ã    shape (4,)

        Split:
            Gi = K[0]   integral-error gain   (scalar)
            Gx = K[1:]  state feedback gain   (3,)

        Preview gains  (FORWARD powers of A_cl):
            Gp[j] = −K · A_cl^j · b₀
            A_cl  = Ã − B̃ ⊗ K    (closed-loop matrix)
            b₀    = [1, 0, 0, 0]

        Returns
        -------
        Gi : float
        Gx : (3,) ndarray
        Gp : (N,) ndarray   values are initially negative, decay to ~0
        """
        Q_aug         = np.zeros((4, 4))
        Q_aug[0, 0]   = self.Q_e

        P = solve_discrete_are(
            self._A_aug,
            self._B_aug.reshape(-1, 1),
            Q_aug,
            np.array([[self.R_u]]))

        denom = self.R_u + float(self._B_aug @ P @ self._B_aug)
        K     = (self._B_aug @ P @ self._A_aug) / denom    # (4,) flat

        Gi = float(K[0])
        Gx = K[1:].copy()    # (3,)

        # Closed-loop matrix
        A_cl = self._A_aug - np.outer(self._B_aug, K)      # (4,4)

        # Preview gains via forward powers  A_cl^j
        b0   = np.array([1., 0., 0., 0.])
        Gp   = np.zeros(self.N)
        pw   = np.eye(4)
        for j in range(self.N):
            Gp[j] = -float(K @ pw @ b0)
            pw     = pw @ A_cl    # A_cl^j  (forward, not transpose)

        return Gi, Gx, Gp

    # SINGLE AXIS: ONE STEP
    def _control_step(self,
                      x:      np.ndarray,
                      ei:     float,
                      ref_k:  float,
                      refs:   np.ndarray,
                      ) -> Tuple[float, float, np.ndarray, float]:
        """
        Apply the preview control law for one timestep on one axis.

        Control law (Kajita 2003, eq. 26):
            u_k = −Gi·eᵢ  −  Gx·x_k  −  Gp · p_ref[k:k+N]

        Parameters
        ----------
        x     : (3,) current LIPM state  [p, ṗ, p̈]
        ei    : current integral error
        ref_k : ZMP reference at current timestep  (scalar)
        refs  : (≤N,) padded ZMP reference window starting at k

        Returns
        -------
        u     : jerk input  [m/s³]
        ei_new: updated integral error
        x_new : (3,) updated state
        pzmp  : actual ZMP before update  [m]
        """
        # Actual ZMP from current state
        pzmp = float(self.C @ x)

        # Integral error
        ei_new = ei + pzmp - ref_k

        # Preview control law
        N_av = min(self.N, len(refs))
        u    = (-self.Gi * ei_new
                - float(self.Gx @ x)
                - float(self.Gp[:N_av] @ refs[:N_av]))

        # State update
        x_new = self.A @ x + self.B * u

        return u, ei_new, x_new, pzmp

    # ══════════════════════════════════════════════════════════════════
    # PUBLIC: COMPUTE JERK INPUTS FROM ZMPData
    # ══════════════════════════════════════════════════════════════════
    def compute(self,
                zmp_data,
                x0_x: Optional[np.ndarray] = None,
                x0_y: Optional[np.ndarray] = None,
                ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the preview controller over the full ZMP reference sequence
        and return the jerk input arrays for both axes.

        The ZMP reference is padded by N steps at the end (holding the
        last value) so the preview window is always fully populated.

        Parameters
        ----------
        zmp_data : ZMPData  (from zmp.py)
        x0_x     : (3,) initial X-axis LIPM state   default = zeros
        x0_y     : (3,) initial Y-axis LIPM state   default = zeros

        Returns
        -------
        u_x : (T,) jerk inputs for X axis  [m/s³]
        u_y : (T,) jerk inputs for Y axis  [m/s³]

        Raises
        ------
        ValueError  if zmp_data.dt does not match self.dt
        """
        if abs(zmp_data.dt - self.dt) > 1e-9:
            raise ValueError(
                f"ZMPData.dt={zmp_data.dt} ≠ PreviewController.dt={self.dt}. "
                f"Both must use the same timestep.")

        T = zmp_data.T

        # Pad references so the preview window never runs off the edge
        pad_x = np.concatenate([
            zmp_data.zmp_x, np.full(self.N, zmp_data.zmp_x[-1])])
        pad_y = np.concatenate([
            zmp_data.zmp_y, np.full(self.N, zmp_data.zmp_y[-1])])

        # Initial states
        x_x = np.zeros(3) if x0_x is None else np.asarray(x0_x, float)
        x_y = np.zeros(3) if x0_y is None else np.asarray(x0_y, float)

        u_x  = np.zeros(T)
        u_y  = np.zeros(T)
        ei_x = 0.0
        ei_y = 0.0

        for k in range(T):
            refs_x = pad_x[k: k + self.N]
            refs_y = pad_y[k: k + self.N]

            u_x[k], ei_x, x_x, _ = self._control_step(
                x_x, ei_x, pad_x[k], refs_x)

            u_y[k], ei_y, x_y, _ = self._control_step(
                x_y, ei_y, pad_y[k], refs_y)

        return u_x, u_y

    # ══════════════════════════════════════════════════════════════════
    # UTILITIES / DEBUG
    # ══════════════════════════════════════════════════════════════════
    def gain_summary(self) -> dict:
        """
        Return a summary dict of all computed gains.

        Returns
        -------
        dict with keys:
            Gi, Gx, Gp_first5, Gp_last, Gp_abssum
            cl_eigenvalues, cl_stable
        """
        K_full  = np.concatenate([[self.Gi], self.Gx])
        A_cl    = self._A_aug - np.outer(self._B_aug, K_full)
        ev      = np.abs(np.linalg.eigvals(A_cl))

        return {
            "Gi":            self.Gi,
            "Gx":            self.Gx,
            "Gp_first5":     self.Gp[:5],
            "Gp_last":       self.Gp[-1],
            "Gp_abssum":     float(np.abs(self.Gp).sum()),
            "cl_eigenvalues": ev,
            "cl_stable":     bool(np.all(ev < 1.0)),
        }

    def print_gains(self) -> None:
        """Print all LQR gains and stability info."""
        s = self.gain_summary()
        print("\n[PreviewController] Gain summary:")
        print(f"  dt  = {self.dt} s   zc = {self.zc} m   N = {self.N}")
        print(f"  Q_e = {self.Q_e}   R_u = {self.R_u}")
        print(f"  Gi          : {s['Gi']:.6f}")
        print(f"  Gx          : {np.round(s['Gx'], 4)}")
        print(f"  Gp[0..4]    : {np.round(s['Gp_first5'], 4)}")
        print(f"  Gp[N-1]     : {s['Gp_last']:.8f}")
        print(f"  |Gp| sum    : {s['Gp_abssum']:.4f}")
        print(f"  CL |eigs|   : {np.round(s['cl_eigenvalues'], 6)}")
        print(f"  CL stable   : {s['cl_stable']}")


# ─────────────────────────────────────────────────────────────
# STAND-ALONE TEST
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from dynamics.lipm import LIPM, CoMTrajectory

    print("=" * 64)
    print("  PREVIEW CONTROLLER — stand-alone test")
    print("=" * 64)

    dt = 0.005
    zc = GEO.pelvis_height
    g  = 9.81
    N  = 320

    # ── Instantiate ───────────────────────────────────────────────────
    ctrl = PreviewController(dt=dt, zc=zc, N=N, Q_e=1.0, R_u=1e-6)
    lipm = LIPM(dt=dt, zc=zc)
    ctrl.print_gains()

    # ── Minimal ZMPData stub ──────────────────────────────────────────
    class _FakeZMP:
        def __init__(self, dt, zx, zy):
            self.dt    = dt
            self.zmp_x = zx
            self.zmp_y = zy
            self.T     = len(zx)

    # ── Test 1: constant ZMP reference ───────────────────────────────
    print("\n[ Test 1 — constant ZMP ref = 0.18 m ]")
    T_test = 3000
    ref    = np.full(T_test, 0.18)
    zd     = _FakeZMP(dt, ref, ref)

    u_x, u_y = ctrl.compute(zd)
    com       = lipm.integrate(u_x, u_y)

    # ZMP from state
    zmp_from_state = com.pos_x - (zc / g) * com.acc_x
    err = np.abs(zmp_from_state[500:] - ref[500:])

    print(f"  CoM final x   : {com.pos_x[-1]:.6f} m  (target 0.18)")
    print(f"  CoM max x     : {com.pos_x.max():.6f} m")
    print(f"  ZMP ss error  : mean={np.mean(err)*1000:.3f} mm  "
          f"max={np.max(err)*1000:.3f} mm")
    print(f"  PASS          : {abs(com.pos_x[-1] - 0.18) < 1e-3}")

    # ── Test 2: step change ───────────────────────────────────────────
    print("\n[ Test 2 — step change 0 → 0.18 at t = 0.5 s ]")
    ref2   = np.zeros(T_test)
    ref2[int(0.5 / dt):] = 0.18
    zd2    = _FakeZMP(dt, ref2, ref2)
    u_x2, u_y2 = ctrl.compute(zd2)
    com2        = lipm.integrate(u_x2, u_y2)
    zmp2        = com2.pos_x - (zc / g) * com2.acc_x

    ss_start = int(2.0 / dt)
    err2     = np.abs(zmp2[ss_start:] - ref2[ss_start:])
    print(f"  CoM final x   : {com2.pos_x[-1]:.6f} m  (target 0.18)")
    print(f"  ZMP ss error  : mean={np.mean(err2)*1000:.3f} mm  "
          f"max={np.max(err2)*1000:.3f} mm")
    print(f"  PASS          : {abs(com2.pos_x[-1] - 0.18) < 1e-3}")

    # ── Test 3: realistic gait ZMP ref ───────────────────────────────
    print("\n[ Test 3 — realistic gait ZMP reference (6 steps) ]")
    from dynamics.zmp import ZMPPlanner

    planner  = ZMPPlanner(dt=dt, step_duration=0.8, ds_ratio=0.2)
    steps    = planner.make_forward_steps(n=6, step_length=0.10)
    zmp_data = planner.plan(steps, verbose=False)

    u_x3, u_y3 = ctrl.compute(zmp_data)
    com3        = lipm.integrate(u_x3, u_y3)

    zmp3_x = com3.pos_x - (zc / g) * com3.acc_x
    zmp3_y = com3.pos_y - (zc / g) * com3.acc_y
    err3_x = np.abs(zmp3_x - zmp_data.zmp_x)
    err3_y = np.abs(zmp3_y - zmp_data.zmp_y)

    print(f"  CoM x range   : [{com3.pos_x.min():.4f}, {com3.pos_x.max():.4f}] m")
    print(f"  CoM y range   : [{com3.pos_y.min():.4f}, {com3.pos_y.max():.4f}] m")
    print(f"  ZMP err x     : mean={np.mean(err3_x)*1000:.3f} mm  "
          f"max={np.max(err3_x)*1000:.3f} mm")
    print(f"  ZMP err y     : mean={np.mean(err3_y)*1000:.3f} mm  "
          f"max={np.max(err3_y)*1000:.3f} mm")
    lat_ok = com3.pos_y.max() < GEO.hip_width + 0.05
    print(f"  Laterally OK  : {lat_ok}")

    # ── Test 4: dt mismatch raises ValueError ─────────────────────────
    print("\n[ Test 4 — dt mismatch raises ValueError ]")
    class _BadZMP:
        dt = 0.010; T = 100
        zmp_x = np.zeros(100); zmp_y = np.zeros(100)
    try:
        ctrl.compute(_BadZMP())
        print("  FAIL — no exception raised")
    except ValueError as e:
        print(f"  ValueError correctly raised  PASS")

    print("\n" + "=" * 64)
    print("  All tests complete.")
    print("=" * 64)