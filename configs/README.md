# Robot Configuration Module

This module defines the humanoid robot geometric parameters and provides reusable mathematical utilities for 3D transformations.

It acts as the shared foundation for the entire project and is used by:

* Forward Kinematics
* Inverse Kinematics
* Dynamics
* Walking Pattern Generation
* Visualization
* Simulation Pipelines

---

# File Structure

```text
robot_config.py
```

---

# Features

* Centralized robot geometry configuration
* Elementary 3D rotation matrices
* Homogeneous transformation matrix generation
* Reusable utilities for robotics computations

---

# Robot Geometry

The robot dimensions are stored using a Python dataclass.

```python
@dataclass
class RobotGeometry:
    pelvis_height: float = 0.85
    thigh_length:  float = 0.40
    shank_length:  float = 0.40
    foot_height:   float = 0.05
    hip_width:     float = 0.18
```

## Geometry Parameters

| Parameter       | Description                   | Value    |
| --------------- | ----------------------------- | -------- |
| `pelvis_height` | Pelvis height from the ground | `0.85 m` |
| `thigh_length`  | Upper leg length              | `0.40 m` |
| `shank_length`  | Lower leg length              | `0.40 m` |
| `foot_height`   | Foot thickness                | `0.05 m` |
| `hip_width`     | Distance between the hips     | `0.18 m` |

A global geometry instance is created for use across the project:

```python
GEO = RobotGeometry()
```

---

# Rotation Matrix Utilities

The module provides standard 3D rotation matrices.

## Rotation Around X-axis

```python
Rx(a)
```

Returns the rotation matrix:

$$
[
R_x(a)=
\begin{bmatrix}
1 & 0 & 0 \\
0 & \cos(a) & -\sin(a) \\
0 & \sin(a) & \cos(a)
\end{bmatrix}
]
$$
---

## Rotation Around Y-axis

```python
Ry(a)
```

Returns the rotation matrix:
$$
[
R_y(a)=
\begin{bmatrix}
\cos(a) & 0 & \sin(a) \\
0 & 1 & 0 \\
-\sin(a) & 0 & \cos(a)
\end{bmatrix}
]
$$
---

## Rotation Around Z-axis

```python
Rz(a)
```

Returns the rotation matrix:
$$
[
R_z(a)=
\begin{bmatrix}
\cos(a) & -\sin(a) & 0 \\
\sin(a) & \cos(a) & 0 \\
0 & 0 & 1
\end{bmatrix}
]
$$
---

# Homogeneous Transformation Matrix

## Function

```python
homogeneous(R, t)
```

Builds a 4×4 homogeneous transformation matrix from:

* A rotation matrix `R`
* A translation vector `t`

## Mathematical Form
$$
[
T =
\begin{bmatrix}
R & t \\
0 & 1
\end{bmatrix}
]
$$
This representation is widely used in:

* Coordinate frame transformations
* Robot pose estimation
* Kinematic chains
* Motion simulation

---

# Dependencies

Install the required dependency:

```bash
pip install numpy
```

---

# Example Usage

```python
import numpy as np
from robot_config import GEO, Rz, homogeneous

R = Rz(np.pi / 4)

translation = np.array([0.1, 0.2, 0.3])

T = homogeneous(R, translation)

print(T)
```

---

# Role in the Project

This module serves as the low-level mathematical backbone of the humanoid robot framework.

By centralizing geometry parameters and transformation utilities, it ensures consistency across all robotics subsystems.
