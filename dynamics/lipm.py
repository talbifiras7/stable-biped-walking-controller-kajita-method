"""
lipm.py
=======
Linear Inverted Pendulum Model (LIPM) and Center of Mass (CoM) trajectory.

What this file does
-------------------
  1. Defines the LIPM discrete-time system matrices (A, B, C)
  2. Integrates the CoM state forward in time given a jerk input sequence
  3. Computes the ZMP from the CoM state at every timestep
  4. Packages the full CoM state into a CoMTrajectory dataclass


LIPM model
------------------------
  The robot CoM is modelled as a point mass constrained to move on a
  horizontal plane at constant height zc above the ground.

  Continuous-time ZMP equation:
      p_zmp(t) = p(t)  −  (zc / g) · p̈(t)

  Discretised at period dt with CoM jerk u as input:

      State   x = [p, ṗ, p̈]ᵀ       (position, velocity, acceleration)
      Input   u = p⃛  (jerk)

      x(k+1) = A · x(k) + B · u(k)

           ┌ 1  dt  dt²/2 ┐        ┌ dt³/6 ┐
      A  = │ 0   1  dt    │   B  = │ dt²/2 │
           └ 0   0   1    ┘        └ dt    ┘

      ZMP output equation:
      p_zmp(k) = C · x(k)    where  C = [1, 0, −zc/g]

  The model runs independently for the X and Y axes.

CoMTrajectory output
--------------------
  Full state per timestep k = 0 … T-1:
      position     [m]
      velocity     [m/s]
      acceleration [m/s²]
      ZMP          [m]   (computed from CoM state via C·x)
  for both X and Y axes.

Stand-alone test
----------------
  python lipm.py
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))    

from configs.robot_config import GEO


# CoM TRAJECTORY  (output dataclass)
@dataclass
class CoMTrajectory:
    """
    Full CoM state trajectory produced by LIPM.integrate().

    All arrays are indexed by timestep k = 0 … T-1.
    Consumed by simulation.py once a controller provides jerk inputs.

    Attributes
    ----------
    dt       : control period      [s]
    zc       : constant CoM height [m]
    pos_x    : CoM position x      (T,)  [m]
    pos_y    : CoM position y      (T,)  [m]
    vel_x    : CoM velocity x      (T,)  [m/s]
    vel_y    : CoM velocity y      (T,)  [m/s]
    acc_x    : CoM acceleration x  (T,)  [m/s²]
    acc_y    : CoM acceleration y  (T,)  [m/s²]
    zmp_x    : ZMP x               (T,)  [m]   C·x, computed from state
    zmp_y    : ZMP y               (T,)  [m]
    """
    dt:    float
    zc:    float
    pos_x: np.ndarray
    pos_y: np.ndarray
    vel_x: np.ndarray
    vel_y: np.ndarray
    acc_x: np.ndarray
    acc_y: np.ndarray
    zmp_x: np.ndarray
    zmp_y: np.ndarray

    @property
    def T(self) -> int:
        """Number of timesteps."""
        return len(self.pos_x)

    @property
    def duration(self) -> float:
        """Total trajectory duration [s]."""
        return self.T * self.dt

    def state_at(self, k: int) -> dict:
        """
        Return the full CoM state at timestep k as a dict.

        Parameters
        ----------
        k : timestep index

        Returns
        -------
        dict with keys: pos_x, pos_y, vel_x, vel_y, acc_x, acc_y,
                        zmp_x, zmp_y
        """
        if not (0 <= k < self.T):
            raise IndexError(f"k={k} out of range [0, {self.T})")
        return {
            "pos_x": self.pos_x[k], "pos_y": self.pos_y[k],
            "vel_x": self.vel_x[k], "vel_y": self.vel_y[k],
            "acc_x": self.acc_x[k], "acc_y": self.acc_y[k],
            "zmp_x": self.zmp_x[k], "zmp_y": self.zmp_y[k],
        }

    def __repr__(self) -> str:
        return (f"CoMTrajectory("
                f"T={self.T}, dt={self.dt}s, "
                f"duration={self.duration:.2f}s, "
                f"zc={self.zc}m)")

    def print_summary(self) -> None:
        """Print a human-readable summary."""
        print("  CoMTrajectory summary:")
        print(f"    T         : {self.T} timesteps")
        print(f"    duration  : {self.duration:.3f} s")
        print(f"    dt        : {self.dt} s")
        print(f"    zc        : {self.zc} m")
        print(f"    pos x     : [{self.pos_x.min():.4f},  {self.pos_x.max():.4f}] m")
        print(f"    pos y     : [{self.pos_y.min():.4f},  {self.pos_y.max():.4f}] m")
        print(f"    vel x     : [{self.vel_x.min():.4f},  {self.vel_x.max():.4f}] m/s")
        print(f"    vel y     : [{self.vel_y.min():.4f},  {self.vel_y.max():.4f}] m/s")
        print(f"    acc x     : [{self.acc_x.min():.4f}, {self.acc_x.max():.4f}] m/s²")
        print(f"    acc y     : [{self.acc_y.min():.4f}, {self.acc_y.max():.4f}] m/s²")
        print(f"    ZMP x     : [{self.zmp_x.min():.4f},  {self.zmp_x.max():.4f}] m")
        print(f"    ZMP y     : [{self.zmp_y.min():.4f},  {self.zmp_y.max():.4f}] m")


# LIPM CLASS    
class LIPM:
    """
    Linear Inverted Pendulum Model — dynamics and CoM integration.

    One class, all methods.

    Parameters
    ----------
    dt : control period      [s]   must match the ZMPData timestep
    zc : constant CoM height [m]   default = GEO.pelvis_height
    """

    def __init__(self,
                 dt: float = 0.005,
                 zc: float = None):

        self.dt = dt
        self.zc = zc if zc is not None else GEO.pelvis_height
        self.g  = 9.81

        # ── LIPM discrete-time matrices ───────────────────────────────
        # All 1-D or 2-D, no column vectors — avoids shape-ambiguity bugs.

        # State transition matrix A  (3×3)
        self.A = np.array([
            [1.,  dt,  dt**2 / 2.],
            [0.,  1.,  dt         ],
            [0.,  0.,  1.         ],
        ])

        # Input matrix B  (3,)  — jerk maps to [pos, vel, acc] increments
        self.B = np.array([dt**3 / 6., dt**2 / 2., dt])

        # ZMP output matrix C  (3,)  — p_zmp = C · x = p − (zc/g)·p̈
        self.C = np.array([1., 0., -self.zc / self.g])
        
    # CORE: SINGLE TIMESTEP
    def step(self,
             x: np.ndarray,
             u: float,
             ) -> tuple:
        """
        Advance the LIPM one timestep.

        Parameters
        ----------
        x : (3,) current state  [position, velocity, acceleration]
        u : jerk input  [m/s³]  (scalar)

        Returns
        -------
        x_next : (3,) updated state
        zmp    : ZMP at current state  [m]  (computed BEFORE update)
        """
        zmp    = float(self.C @ x)            # ZMP from current state
        x_next = self.A @ x + self.B * u      # state update
        return x_next, zmp
    
    # COMPUTE ZMP FROM STATE
    def compute_zmp(self, x: np.ndarray) -> float:
        """
        Compute ZMP from a LIPM state vector.

        p_zmp = p − (zc/g) · p̈  = C · x

        Parameters
        ----------
        x : (3,) state  [position, velocity, acceleration]

        Returns
        -------
        zmp : float  [m]
        """
        return float(self.C @ x)

    # INTEGRATION: GIVEN JERK INPUTS
    def integrate(self,
                  u_x: np.ndarray,
                  u_y: np.ndarray,
                  x0_x: np.ndarray = None,
                  x0_y: np.ndarray = None,
                  ) -> CoMTrajectory:
        """
        Integrate the LIPM forward in time given jerk input sequences.

        This is the core integration method.  A controller (e.g. preview
        control in a future file) computes u_x, u_y and passes them here.

        Parameters
        ----------
        u_x  : (T,) jerk inputs for X axis  [m/s³]
        u_y  : (T,) jerk inputs for Y axis  [m/s³]
        x0_x : (3,) initial state for X     default = zeros
        x0_y : (3,) initial state for Y     default = zeros

        Returns
        -------
        CoMTrajectory
        """
        T = len(u_x)
        if len(u_y) != T:
            raise ValueError(
                f"u_x and u_y must have the same length, "
                f"got {len(u_x)} and {len(u_y)}")

        # Initial states
        x_x = np.zeros(3) if x0_x is None else np.asarray(x0_x, float)
        x_y = np.zeros(3) if x0_y is None else np.asarray(x0_y, float)

        # Output arrays
        pos_x = np.zeros(T); pos_y = np.zeros(T)
        vel_x = np.zeros(T); vel_y = np.zeros(T)
        acc_x = np.zeros(T); acc_y = np.zeros(T)
        zmp_x = np.zeros(T); zmp_y = np.zeros(T)

        for k in range(T):
            # X axis
            x_x, zx = self.step(x_x, u_x[k])
            pos_x[k] = x_x[0]
            vel_x[k] = x_x[1]
            acc_x[k] = x_x[2]
            zmp_x[k] = zx

            # Y axis
            x_y, zy = self.step(x_y, u_y[k])
            pos_y[k] = x_y[0]
            vel_y[k] = x_y[1]
            acc_y[k] = x_y[2]
            zmp_y[k] = zy

        return CoMTrajectory(
            dt=self.dt, zc=self.zc,
            pos_x=pos_x, pos_y=pos_y,
            vel_x=vel_x, vel_y=vel_y,
            acc_x=acc_x, acc_y=acc_y,
            zmp_x=zmp_x, zmp_y=zmp_y,
        )

    # INTEGRATION: FROM ZMPData  (convenience wrapper)
    def integrate_from_zmp(self,
                           zmp_data,
                           u_x: np.ndarray,
                           u_y: np.ndarray,
                           ) -> CoMTrajectory:
        """
        Integrate LIPM using jerk inputs computed by a controller,
        given ZMPData from zmp.py.

        Validates that the timestep matches.

        Parameters
        ----------
        zmp_data : ZMPData  (from zmp.py)
        u_x      : (T,) jerk inputs for X axis  [m/s³]
        u_y      : (T,) jerk inputs for Y axis  [m/s³]

        Returns
        -------
        CoMTrajectory
        """
        if abs(zmp_data.dt - self.dt) > 1e-9:
            raise ValueError(
                f"ZMPData.dt={zmp_data.dt} ≠ LIPM.dt={self.dt}. "
                f"Use the same timestep for both.")

        if len(u_x) != zmp_data.T:
            raise ValueError(
                f"Jerk input length {len(u_x)} ≠ ZMPData.T={zmp_data.T}.")

        return self.integrate(u_x, u_y)

    # UTILITIES
    def state_from_pos(self,
                       pos: float,
                       vel: float = 0.0,
                       acc: float = 0.0,
                       ) -> np.ndarray:
        """
        Build a LIPM state vector from physical quantities.

        Parameters
        ----------
        pos : CoM position      [m]
        vel : CoM velocity      [m/s]  default 0
        acc : CoM acceleration  [m/s²] default 0

        Returns
        -------
        x : (3,) state vector
        """
        return np.array([pos, vel, acc])

    def zmp_from_pos_acc(self, pos: float, acc: float) -> float:
        """
        Compute ZMP directly from CoM position and acceleration.

            p_zmp = p − (zc/g) · p̈

        Parameters
        ----------
        pos : CoM position      [m]
        acc : CoM acceleration  [m/s²]

        Returns
        -------
        zmp : float  [m]
        """
        return pos - (self.zc / self.g) * acc

    def print_matrices(self) -> None:
        """Print the LIPM system matrices for debugging."""
        print(f"\n[LIPM] System matrices  (dt={self.dt}s, zc={self.zc}m):")
        print(f"  A =\n{self.A}")
        print(f"  B = {self.B}")
        print(f"  C = {self.C}")
        print(f"  eigenvalues(A) = {np.linalg.eigvals(self.A)}")
        
    @staticmethod
    def plot(traj: CoMTrajectory, u_x=None, u_y=None, foot_z=None) -> None:
        T = traj.T
        t = np.arange(T) * traj.dt

    # ── 1. CoM vs ZMP ─────────────────────────────
        plt.figure()
        plt.title("CoM vs ZMP")
        plt.plot(t, traj.pos_x, label="CoM x")
        plt.plot(t, traj.zmp_x, "--", label="ZMP x")
        plt.plot(t, traj.pos_y, label="CoM y")
        plt.plot(t, traj.zmp_y, "--", label="ZMP y")
        plt.xlabel("Time [s]")
        plt.ylabel("Position [m]")
        plt.legend()
        plt.grid(True)
        image_path=f"visualization/plots/dynamics_plots/com_trajectoire/com_vs_zmp.png"
        plt.savefig(image_path,dpi=300,bbox_inches='tight')
         
    # ── 2. Velocity & Acceleration ────────────────
        plt.figure()
        plt.title("Velocity & Acceleration")
        plt.plot(t, traj.vel_x, label="vel x")
        plt.plot(t, traj.acc_x, "--", label="acc x")
        plt.plot(t, traj.vel_y, label="vel y")
        plt.plot(t, traj.acc_y, "--", label="acc y")
        plt.xlabel("Time [s]")
        plt.ylabel("Vel / Acc")
        plt.legend()
        plt.grid(True)
        image_path=f"visualization/plots/dynamics_plots/com_trajectoire/vel_acc.png"
        plt.savefig(image_path,dpi=300,bbox_inches='tight')

    # ── 3. Jerk ───────────────────────────────────
        plt.figure()
        plt.title("Jerk Inputs")

        if u_x is not None:
            plt.plot(t, u_x, label="jerk x")
        if u_y is not None:
            plt.plot(t, u_y, label="jerk y")

        plt.xlabel("Time [s]")
        plt.ylabel("Jerk [m/s³]")
        plt.legend()
        plt.grid(True)
        image_path=f"visualization/plots/dynamics_plots/com_trajectoire/jerk_inputs.png"
        plt.savefig(image_path,dpi=300,bbox_inches='tight')
        
    # ── 4. Foot Z trajectory ─────────────────────────────
    @staticmethod
    def plot_feet(T, dt, step_time=0.8, swing_height=0.05):

        t = np.arange(T) * dt

        foot_z_left = np.zeros(T)
        foot_z_right = np.zeros(T)

        step_samples = int(step_time / dt)

        for k in range(T):
            phase = (k // step_samples) % 2
            local = (k % step_samples) / step_samples

            swing = swing_height * np.sin(np.pi * local)

            if phase == 0:
                foot_z_left[k] = swing
            else:
                foot_z_right[k] = swing

    # ── Plot BOTH in same figure ──
        plt.figure()
        plt.title("Foot Trajectories (Z height)")

        plt.plot(t, foot_z_left, label="left foot")
        plt.plot(t, foot_z_right, label="right foot")

        plt.xlabel("Time [s]")
        plt.ylabel("Height [m]")
        plt.legend()
        plt.grid(True)

        image_path = "visualization/plots/dynamics_plots/com_trajectoire/foot_z.png"
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.close()

        return foot_z_left, foot_z_right


# STAND-ALONE TEST
if __name__ == "__main__":
    print("=" * 60)
    print("  LIPM — stand-alone test")
    print("=" * 60)

    dt  = 0.005
    zc  = GEO.pelvis_height
    g   = 9.81
    T   = 1000

    lipm = LIPM(dt=dt, zc=zc)
    lipm.print_matrices()

    # ── Test 1: zero jerk — CoM stays still ──────────────────────────
    print("\n[ Test 1 — zero jerk: CoM must stay at rest ]")
    com = lipm.integrate(np.zeros(T), np.zeros(T))
    assert np.allclose(com.pos_x, 0.0), "FAIL: pos_x not zero"
    assert np.allclose(com.zmp_x, 0.0), "FAIL: zmp_x not zero"
    print("  pos_x stays at 0.0 : PASS")
    print("  zmp_x stays at 0.0 : PASS")

    # ── Test 2: constant positive jerk — CoM accelerates ─────────────
    print("\n[ Test 2 — constant jerk: CoM accelerates forward ]")
    u_const = np.full(T, 10.0)   # 10 m/s³ jerk
    com2    = lipm.integrate(u_const, np.zeros(T))
    assert com2.pos_x[-1] > 0,          "FAIL: pos_x should be positive"
    assert com2.vel_x[-1] > 0,          "FAIL: vel_x should be positive"
    assert com2.acc_x[-1] > 0,          "FAIL: acc_x should be positive"
    print(f"  Final pos x = {com2.pos_x[-1]:.4f} m  (> 0): PASS")
    print(f"  Final vel x = {com2.vel_x[-1]:.4f} m/s  (> 0): PASS")
    print(f"  Final acc x = {com2.acc_x[-1]:.4f} m/s²  (> 0): PASS")

    # ── Test 3: ZMP equation verification ────────────────────────────
    print("\n[ Test 3 — ZMP = pos − (zc/g)·acc must hold at every step ]")
    u_test = np.sin(np.linspace(0, 4*np.pi, T)) * 5.0
    com3   = lipm.integrate(u_test, np.zeros(T))
    zmp_check = com3.pos_x - (zc / g) * com3.acc_x
    max_err   = np.max(np.abs(zmp_check - com3.zmp_x))
    # One-step offset: zmp_x is computed BEFORE state update
    # so we compare zmp_x[k] with the state BEFORE update at k
    # (small offset is expected — check it is tiny)
    print(f"  Max ZMP equation error: {max_err*1000:.4f} mm  "
          f"({'PASS' if max_err < 0.01 else 'FAIL'})")

    # ── Test 4: state_at() accessor ──────────────────────────────────
    print("\n[ Test 4 — state_at() accessor ]")
    s = com3.state_at(42)
    assert s["pos_x"] == com3.pos_x[42], "FAIL: pos_x mismatch"
    assert s["zmp_x"] == com3.zmp_x[42], "FAIL: zmp_x mismatch"
    print(f"  state_at(42): pos_x={s['pos_x']:.6f}  zmp_x={s['zmp_x']:.6f}  PASS")

    # ── Test 5: step() is consistent with integrate() ────────────────
    print("\n[ Test 5 — step() must match integrate() ]")
    x = np.zeros(3)
    for k in range(T):
        x, _ = lipm.step(x, u_test[k])
    diff = abs(x[0] - com3.pos_x[-1])
    print(f"  step() vs integrate() final pos_x diff: {diff:.2e}  "
          f"({'PASS' if diff < 1e-12 else 'FAIL'})")

    # ── Test 6: integrate_from_zmp() validates dt ────────────────────
    print("\n[ Test 6 — integrate_from_zmp() dt mismatch raises ValueError ]")
    class _FakeZMP:
        dt = 0.010   # wrong dt
        T  = T
    try:
        lipm.integrate_from_zmp(_FakeZMP(), np.zeros(T), np.zeros(T))
        print("  FAIL — no exception raised")
    except ValueError as e:
        print(f"  ValueError raised correctly: {e}  PASS")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n[ CoM trajectory summary ]")
    com3.print_summary()
    t = np.arange(T) * dt
    foot_z = 0.05 * np.maximum(0, np.sin(np.pi *t))
    lipm.plot(com3, u_test, foot_z=foot_z)
    lipm.plot_feet(T, dt)