"""
zmp.py
======
Zero Moment Point (ZMP) reference generator for a 12-DOF biped. 

Coordinate frames
-----------------
  All positions are in the WORLD frame (z=0 = ground level).
  The pelvis-frame conversion is done later in simulation.py once
  the CoM trajectory is known from lipm.py.
 
ZMP reference construction  (Kajita 2003)
------------------------------------------
  Each footstep is divided into two phases:
 
  Double Support (DS) — duration = step_duration × ds_ratio
    ZMP interpolates linearly from the outgoing (previous swing) foot
    position to the incoming (support) foot position.
    Both feet are in contact.
 
  Single Support (SS) — duration = step_duration × (1 - ds_ratio)
    ZMP stays fixed directly over the support foot.
    Swing foot follows a cubic Bézier arc to its landing target.

"""
from turtle import left, lt, right

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import sys
import os
import matplotlib.pyplot as plt
 
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.robot_config import GEO


@dataclass
class FootStep:
    """
    One foot placement target in the world frame.
 
    Parameters
    ----------
    x, y  : landing position  [m]  (z = 0, ground level)
    side  : "right" | "left"
    """
    x:    float
    y:    float
    side: str
 
    def __post_init__(self):
        if self.side not in ("right", "left"):
            raise ValueError(
                f"FootStep.side must be 'right' or 'left', got '{self.side}'")
 
    def as_array(self) -> np.ndarray:
        """Return [x, y, 0] as a numpy array."""
        return np.array([self.x, self.y, 0.0])
 
    def __repr__(self):
        return f"FootStep(side={self.side}, x={self.x:.3f}, y={self.y:.3f})"

@dataclass
class ZMPData:
    """
    ZMP reference and foot trajectories, indexed by timestep k = 0…T-1.
 
    Produced by ZMPPlanner.plan() and consumed by lipm.py + simulation.py.
 
    Attributes
    ----------
    dt          : control period  [s]
    zmp_x       : ZMP reference x  (T,)  [m]  world frame
    zmp_y       : ZMP reference y  (T,)  [m]  world frame
    foot_pos    : {"right": (T,3), "left": (T,3)}  foot positions  [m]  world frame
    foot_contact: {"right": (T,) bool, "left": (T,) bool}
                  True  = foot is on the ground (support or DS)
                  False = foot is in the air    (swing)
    n_steps     : number of footsteps planned
    step_duration    : [s]
    ds_ratio         : double-support fraction
    swing_height     : [m]
    """
    dt: float
    zmp_x: np.ndarray
    zmp_y: np.ndarray
    foot_pos: dict = field(default_factory=lambda: {"right": None, "left": None})
    foot_contact: dict = field(default_factory=lambda: {"right": None, "left": None})
    n_steps: int = 0
    step_duration: float = 0.8
    ds_ratio: float = 0.2
    swing_height: float = 0.05
    
    
    @property
    def T(self) -> int:
        """Total number of timesteps."""
        return len(self.zmp_x)
 
    @property
    def duration(self) -> float:
        """Total trajectory duration [s]."""
        return self.T * self.dt
 
    def __repr__(self):
        return (f"ZMPData("
                f"steps={self.n_steps}, "
                f"T={self.T}, "
                f"dt={self.dt}s, "
                f"duration={self.duration:.2f}s)")
 
    def print_summary(self) -> None:
        """Print a human-readable summary."""
        print(f"  ZMPData summary:")
        print(f"    footsteps     : {self.n_steps}")
        print(f"    timesteps T   : {self.T}")
        print(f"    duration      : {self.duration:.2f} s")
        print(f"    dt            : {self.dt} s")
        print(f"    step_duration : {self.step_duration} s")
        print(f"    ds_ratio      : {self.ds_ratio}")
        print(f"    swing_height  : {self.swing_height} m")
        print(f"    ZMP x range   : [{self.zmp_x.min():.4f}, "
              f"{self.zmp_x.max():.4f}] m")
        print(f"    ZMP y range   : [{self.zmp_y.min():.4f}, "
              f"{self.zmp_y.max():.4f}] m")
        for side in ("right", "left"):
            z = self.foot_pos[side][:, 2]
            n_air = int((~self.foot_contact[side]).sum())
            print(f"    {side:5s} foot z : [{z.min():.4f}, {z.max():.4f}] m  "
                  f"({n_air} ticks in air)")
            
class ZMPPlanner:
    """
    Converts a list of FootStep targets into a ZMP reference sequence
    and foot trajectories.

    Parameters
    ----------
    dt            : control period          [s]   default 0.005
    step_duration : total time per step     [s]   default 0.8
    ds_ratio      : double-support fraction [0,1] default 0.2
    swing_height  : peak foot clearance     [m]   default 0.04
    """

    def __init__(self,
                 dt:            float = 0.005,
                 step_duration: float = 0.8,
                 ds_ratio:      float = 0.2,
                 swing_height:  float = 0.04):
        if not (0.0 <= ds_ratio < 1.0):
            raise ValueError(f"ds_ratio must be in [0, 1), got {ds_ratio}")

        self.dt            = dt
        self.step_duration = step_duration
        self.ds_ratio      = ds_ratio
        self.swing_height  = swing_height

        # Pre-compute tick counts
        self._n_total = int(round(step_duration / dt))
        self._n_ds    = int(round(self._n_total * ds_ratio))
        self._n_ss    = self._n_total - self._n_ds
        
    def plan(self,
             steps:      List[FootStep],
             init_right: Optional[np.ndarray] = None,
             init_left:  Optional[np.ndarray] = None,
             verbose:    bool = True,
             ) -> ZMPData:
        """
        Build ZMP reference + foot trajectories for the given step list.

        Parameters
        ----------
        steps      : ordered list of FootStep  (built in main.py)
        init_right : initial right-foot world position [x, y, 0]  [m]
                     default: [0, -hip_width, 0]
        init_left  : initial left-foot  world position [x, y, 0]  [m]
                     default: [0, +hip_width, 0]
        verbose    : print planning log

        Returns
        -------
        ZMPData
        """
        if len(steps) == 0:
            raise ValueError("steps list is empty.")

        # Default initial foot positions
        if init_right is None:
            init_right = np.array([0.0, -GEO.hip_width, 0.0])
        if init_left is None:
            init_left  = np.array([0.0,  GEO.hip_width, 0.0])

        init_right = np.asarray(init_right, dtype=float)
        init_left  = np.asarray(init_left,  dtype=float)

        if verbose:
            print(f"[ZMP] Planning {len(steps)} footsteps "
                  f"(step_dur={self.step_duration}s, "
                  f"ds={self.ds_ratio}, "
                  f"swing_h={self.swing_height}m) ...")

        # Allocate output arrays
        T     = self._n_total * len(steps)
        zmp_x = np.zeros(T)
        zmp_y = np.zeros(T)
        fp    = {
            "right": np.zeros((T, 3)),
            "left":  np.zeros((T, 3)),
        }
        fc    = {
            "right": np.ones(T, dtype=bool),
            "left":  np.ones(T, dtype=bool),
        }

        # Current foot positions (updated each step)
        cur = {
            "right": init_right.copy(),
            "left":  init_left.copy(),
        }

        prev_support = None  
        for s_idx, step in enumerate(steps):
            swing   = step.side
            support = "left" if swing == "right" else "right"
            target  = step.as_array()   # [x, y, 0] landing position

            t0 = s_idx * self._n_total
            t1 = t0 + self._n_total
            t_ds_end = t0 + self._n_ds

             # ✅ Determine previous support foot position
            if prev_support is None:
            # First step → assume initial support foot
                prev_support_pos = cur[support].copy()
            else:
                prev_support_pos = prev_support.copy()
            
            # ── ZMP reference ─────────────────────────────────────────
            self._fill_zmp(
                zmp_x, zmp_y,
                t0, t_ds_end, t1,
                cur[swing],
                cur[support],
            )

            # ── Foot trajectories ─────────────────────────────────────
            self._fill_foot_support(fp, fc, support, t0, t1, cur[support])
            self._fill_foot_swing(
                fp, fc, swing,
                t0, t_ds_end, t1,
                cur[swing], target,
            )

            if verbose:
                print(f"  step {s_idx+1:>2}/{len(steps)}  "
                      f"swing={swing:5s}  "
                      f"target=({target[0]:.3f}, {target[1]:.3f})  "
                      f"support @ ({cur[support][0]:.3f}, {cur[support][1]:.3f})")

            # Advance current swing foot to its new position
            prev_support = cur[support].copy()
            cur[swing] = target.copy()

        data = ZMPData(
            dt=self.dt,
            zmp_x=zmp_x,
            zmp_y=zmp_y,
            foot_pos=fp,
            foot_contact=fc,
            n_steps=len(steps),
            step_duration=self.step_duration,
            ds_ratio=self.ds_ratio,
            swing_height=self.swing_height,
        )

        if verbose:
            print(f"[ZMP] Done → {data}")

        return data
    
    def _fill_zmp(self,
                  zmp_x:    np.ndarray,
                  zmp_y:    np.ndarray,
                  t0:       int,
                  t_ds_end: int,
                  t1:       int,
                  p_prev_support:  np.ndarray,
                  p_support:np.ndarray,
                  ) -> None:
        """
        Fill ZMP reference for one footstep.

        DS phase : linear ramp from outgoing swing-foot position
                   to support-foot position.
        SS phase : constant at support-foot position.
        """
        n_ds = t_ds_end - t0

        if n_ds > 0:
            # Linear interpolation  alpha ∈ [0, 1]
            alpha = np.linspace(0.0, 1.0, n_ds)
            zmp_x[t0:t_ds_end] = (1.0 - alpha) * p_prev_support[0] + alpha * p_support[0]
            zmp_y[t0:t_ds_end] = (1.0 - alpha) * p_prev_support[1] + alpha * p_support[1]

        # SS: fixed over support foot
        zmp_x[t_ds_end:t1] = p_support[0]
        zmp_y[t_ds_end:t1] = p_support[1]

    def _fill_foot_support(self,
                           fp:      dict,
                           fc:      dict,
                           side:    str,
                           t0:      int,
                           t1:      int,
                           pos:     np.ndarray,
                           ) -> None:
        """Support foot: stationary for the entire step, always in contact."""
        fp[side][t0:t1] = pos
        fc[side][t0:t1] = True

    def _fill_foot_swing(self,
                         fp:       dict,
                         fc:       dict,
                         side:     str,
                         t0:       int,
                         t_ds_end: int,
                         t1:       int,
                         p_start:  np.ndarray,
                         p_end:    np.ndarray,
                         ) -> None:
        """
        Swing foot trajectory.

        DS phase : foot stays at lift-off position, contact = True.
        SS phase : foot follows cubic Bézier arc,  contact = False.
        """
        n_ss = t1 - t_ds_end

        # DS: stationary at current position, still on ground
        fp[side][t0:t_ds_end] = p_start
        fc[side][t0:t_ds_end] = True

        # SS: Bézier arc, foot is in the air
        if n_ss > 0:
            arc = self._bezier(p_start, p_end, n_ss)
            fp[side][t_ds_end:t1] = arc
            fc[side][t_ds_end:t1] = False

    def _bezier(self,
                p0: np.ndarray,
                p1: np.ndarray,
                n:  int,
                ) -> np.ndarray:
        """
        Cubic Bézier foot trajectory from p0 to p1 with a vertical lift.

        Control points:
          P0 = p0
          P1 = p0 + [0, 0, swing_height]   (rising phase)
          P2 = p1 + [0, 0, swing_height]   (descending phase)
          P3 = p1
          
        Parameters
        ----------
        p0 : (3,) lift-off  position  [m]
        p1 : (3,) touch-down position [m]
        n  : number of samples

        Returns
        -------
        traj : (n, 3)
        """
        h  = np.array([0.0, 0.0, self.swing_height])
        P0 = p0
        P1 = p0 + h
        P2 = p1 + h
        P3 = p1

        t  = np.linspace(0.0, 1.0, n)            # (n,)
        t  = t[:, np.newaxis]                     # (n,1) for broadcasting

        traj = ((1 - t)**3        * P0
              + 3 * (1-t)**2 * t  * P1
              + 3 * (1-t)  * t**2 * P2
              +     t**3          * P3)           # (n, 3)
        return traj
    
    @staticmethod
    def make_forward_steps(n:           int   = 4,
                           step_length: float = 0.10,
                           step_width:  float = None,
                           first_side:  str   = "right",
                           ) -> List[FootStep]:
        """
        Build a straight-ahead step list.

        Parameters
        ----------
        n           : total number of steps
        step_length : forward displacement per step  [m]
        step_width  : lateral foot offset from centre line  [m]
                      default = GEO.hip_width
        first_side  : foot that moves first

        Returns
        -------
        steps : List[FootStep]
        """
        w     = step_width if step_width is not None else GEO.hip_width
        other = "left" if first_side == "right" else "right"
        sides = ([first_side, other] * (n // 2 + 1))[:n]

        x_pos = {"right": 0.0, "left": 0.0}
        steps = []
        for side in sides:
            x_pos[side] += step_length
            y = -w if side == "right" else w
            steps.append(FootStep(x=x_pos[side], y=y, side=side))
        return steps
    
    
    @staticmethod
    def make_turn_steps(n:          int   = 4,
                        step_length:float = 0.08,
                        step_width: float = None,
                        turn_angle: float = 0.1,
                        first_side: str   = "right",
                        ) -> List[FootStep]:
        """
        Build a turning step list (arc walk).

        Parameters
        ----------
        n           : total number of steps
        step_length : forward displacement per step  [m]
        step_width  : lateral foot offset  [m]
        turn_angle  : yaw angle added per step  [rad]
        first_side  : foot that moves first

        Returns
        -------
        steps : List[FootStep]
        """
        w     = step_width if step_width is not None else GEO.hip_width
        other = "left" if first_side == "right" else "right"
        sides = ([first_side, other] * (n // 2 + 1))[:n]

        heading = 0.0
        pos     = {"right": np.array([0.0, -w]),
                   "left":  np.array([0.0,  w])}
        steps   = []

        for step_idx, side in enumerate(sides):
            heading += turn_angle
            dx = step_length * np.cos(heading)
            dy = step_length * np.sin(heading)
            lat = -w if side == "right" else w
            pos[side] += np.array([dx, dy])
            steps.append(FootStep(x=pos[side][0],
                                  y=pos[side][1] + lat,
                                  side=side))
        return steps
    
    @staticmethod
    def plot(data: ZMPData, show=True):
        t = np.arange(data.T) * data.dt

    # 1. ZMP X vs Time
        plt.figure()
        plt.plot(t, data.zmp_x)
        plt.title("ZMP X vs Time")
        plt.xlabel("Time [s]")
        plt.ylabel("ZMP X [m]")
        plt.grid()
        image_path = f"visualization/plots/dynamics/zmp_trajectoire/ZMP_X_VSTIME.png"   
        plt.savefig(image_path, dpi=300, bbox_inches="tight")     


    # 2. ZMP Y vs Time
        plt.figure()
        plt.plot(t, data.zmp_y)
        plt.title("ZMP Y vs Time")
        plt.xlabel("Time [s]")
        plt.ylabel("ZMP Y [m]")
        plt.grid()
        image_path = f"visualization/plots/dynamics/zmp_trajectoire/ZMP_Y_VSTIME.png" 
        plt.savefig(image_path, dpi=300, bbox_inches="tight")  

    # 3. TOP VIEW (CRITICAL)
        plt.figure()
        plt.plot(data.zmp_x, data.zmp_y)

        plt.title("ZMP Top View")
        plt.xlabel("ZMP X [m]")
        plt.ylabel("ZMP Y [m]")
        plt.axis('equal')
        plt.grid()
        plt.legend()
        image_path = f"visualization/plots/dynamics/zmp_trajectoire/TOP_VIEW.png"
        plt.savefig(image_path, dpi=300, bbox_inches="tight")

    # 4. CONTACT PHASE
        plt.figure()
        plt.plot(t, data.foot_contact["right"].astype(int), label="Right foot")
        plt.plot(t, data.foot_contact["left"].astype(int), label="Left foot")
        plt.title("Foot Contact (1 = ground, 0 = swing)")
        plt.xlabel("Time [s]")
        plt.ylabel("Contact")
        plt.legend()
        plt.grid()
        image_path = f"visualization/plots/dynamics/zmp_trajectoire/CONTACT_PHASE.png"
        plt.savefig(image_path,dpi=300,bbox_inches='tight')

        if show:
            plt.show()
        

        
if __name__ == "__main__":
    print("=== ZMP PLANNER BASIC TEST ===")

    planner = ZMPPlanner(
        dt=0.01,
        step_duration=0.5,
        ds_ratio=0.2,
        swing_height=0.05
    )

    steps = planner.make_forward_steps(
        n=6,
        step_length=0.1
    )

    data = planner.plan(steps, verbose=True)
    planner.plot(data, show=True)

    print("\n--- OUTPUT CHECK ---")
    print("ZMP shape:", data.zmp_x.shape)
    print("Right foot shape:", data.foot_pos["right"].shape)
    print("Left foot shape:", data.foot_pos["left"].shape)

    print("\nFirst ZMP points:", data.zmp_x[:5], data.zmp_y[:5])