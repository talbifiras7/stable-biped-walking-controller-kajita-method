"""
zmp.py
======
Zero Moment Point planner — LIPM + Preview Control (Kajita 2003).
"""


import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import os 
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.robot_config import GEO

@dataclass
class FootStep:
    """
    One foot placement target (world frame, ground level z=0).

    Parameters
    ----------
    x, y  : position  [m]
    side  : "right" | "left"
    """
    x:    float
    y:    float
    side: str

    def __post_init__(self):
        assert self.side in ("right", "left"), \
            f"side must be 'right' or 'left', got '{self.side}'"

    def __repr__(self):
        return f"FootStep({self.side}, x={self.x:.3f}, y={self.y:.3f})"


@dataclass
class GaitTrajectory:
    """
    Pre-computed gait data indexed by timestep k = 0 … T-1.

    Attributes
    ----------
    dt            : control period  [s]
    com_x, com_y  : CoM position    (T,)  [m]  world frame
    zmp_ref_x/y   : ZMP reference   (T,)  [m]  world frame
    foot_pos      : {"right": (T,3), "left": (T,3)}  pelvis frame  [m]
    foot_contact  : {"right": (T,) bool, "left": (T,) bool}
    """
    dt:           float
    com_x:        np.ndarray
    com_y:        np.ndarray
    zmp_ref_x:    np.ndarray
    zmp_ref_y:    np.ndarray
    foot_pos:     dict          # pelvis-frame foot positions
    foot_contact: dict          # True = foot on ground

    @property
    def n_steps(self) -> int:
        return len(self.com_x)

    def __repr__(self):
        return (f"GaitTrajectory(T={self.n_steps}, dt={self.dt}s, "
                f"duration={self.n_steps * self.dt:.2f}s)")


class FootStepPlanner:
    """
    Converts a list of FootStep targets into ZMP reference and foot
    trajectories.
    
    
        Parameters
    ----------
    dt              : control period          [s]
    step_duration   : total time per footstep [s]
    double_support  : fraction in DS phase    [0–1]
    swing_height    : peak foot clearance     [m]
    
    """
    def __init__(self,
                 dt:             float = 0.005,
                 step_duration:  float = 0.8,
                 double_support: float = 0.2,
                 swing_height:   float = 0.04):
        self.dt      = dt
        self.sh      = swing_height
        self.n_step  = int(step_duration / dt)
        self.n_ds    = int(step_duration * double_support / dt)
        self.n_ss    = self.n_step - self.n_ds
    