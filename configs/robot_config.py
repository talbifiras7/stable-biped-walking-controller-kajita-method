"""
Shared robot geometry constants and elementary rotation matrix utilities.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class RobotGeometry:
    pelvis_height: float=0.85
    thigh_length:  float=0.40
    shank_length:  float=0.40 
    foot_height:   float=0.05
    hip_width:     float=0.18

GEO = RobotGeometry()


def Rx(a: float) -> np.ndarray:
    """Rotation about X axis by angle a"""
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[1,  0,   0 ],
                     [0,  ca, -sa],
                     [0,  sa,  ca]], dtype=float)

def Ry(a: float) -> np.ndarray:
    """Rotation about Y axis by angle a"""
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ ca, 0, sa],
                     [  0, 1,  0],
                     [-sa, 0, ca]], dtype=float)

def Rz(a: float) -> np.ndarray:
    """Rotation about Z axis by angle a"""
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, -sa, 0],
                     [sa,  ca, 0],
                     [ 0,   0, 1]], dtype=float)

def homogeneous(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build a 4×4 homogeneous transform from R (3×3) and t (3,)."""
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3,  3] = t
    return T