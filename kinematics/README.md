# Kinematics Module

This module implements the forward and inverse kinematics of a 12-DOF humanoid robot.

It provides the mathematical foundation required to:

- Compute foot positions and orientations
- Estimate robot joint configurations
- Solve pose reconstruction problems
- Generate walking motions
- Support dynamics and control algorithms

The implementation supports both left and right legs using a symmetric kinematic structure.

---

# File Structure

```text
kinematics/
├── forward_kinematics.py
└── inverse_kinematics.py
```

---

# Architecture Overview

## Forward Kinematics (FK)

Computes the end-effector pose from joint angles:

$$
q \rightarrow T_{foot}
$$

### Inputs

- Joint configuration vector

### Outputs

- Foot position
- Foot orientation
- Homogeneous transformation matrix
- Jacobian matrix

---

## Inverse Kinematics (IK)

Computes the required joint angles to reach a desired foot pose:

$$
T_{foot} \rightarrow q
$$

### Inputs

- Desired foot position
- Desired foot orientation

### Outputs

- Joint configuration
- Convergence information
- Position error

---

# Forward Kinematics

## Class

```python
ForwardKinematics(side)
```

Supported sides:

- `right`
- `left`

The implementation mirrors both legs while preserving a consistent coordinate system.

---

## Main Features

### Homogeneous Transform Computation

Computes the complete transformation matrix from pelvis to foot.

```python
compute(q)
```

Returns:

$$
T =
\begin{bmatrix}
R & p \\
0 & 1
\end{bmatrix}
$$

where:

- \(R\) is the rotation matrix
- \(p\) is the foot position vector

---

### Foot Position Extraction

```python
position(q)
```

Returns the 3D foot position:

$$
p =
\begin{bmatrix}
x \\
y \\
z
\end{bmatrix}
$$

---

### Foot Orientation Extraction

```python
rotation(q)
```

Returns the 3×3 foot rotation matrix.

---

### Numerical Jacobian

```python
jacobian(q)
```

Computes the 6×6 numerical Jacobian using finite differences.

The Jacobian maps joint velocities to Cartesian velocities:

$$
\dot{x} = J(q)\dot{q}
$$

where:

$$
\dot{x} =
\begin{bmatrix}
v \\
\omega
\end{bmatrix}
$$

The Jacobian includes:

- Linear velocity components
- Angular velocity components

---

### Visualization

```python
plot_feet(q)
```

Plots:

- Left foot position
- Right foot position
- Pelvis reference frame

Generated figures are automatically saved inside:

```text
visualization/plots/fk/
```

---

# Inverse Kinematics

## Class

```python
InverseKinematics(side)
```

Implements a Jacobian pseudo-inverse iterative solver.

---

# Solver Method

The inverse kinematics solver uses the Newton–Raphson iterative method.

At each iteration:

$$
\Delta q = J^{+} e
$$

where:

- \(J^{+}\) is the pseudo-inverse Jacobian
- \(e\) is the pose error vector

The update rule becomes:

$$
q_{k+1} = q_k + \Delta q
$$

A damping factor is applied for improved numerical stability.

---

# Orientation Error

Orientation tracking is computed using rotation matrix error:

$$
R_{err} = R_{target}R^T
$$

The angular error vector is extracted from the skew-symmetric part of the matrix.

---

# Main Features

## IK Solver

```python
solve(p_foot, R_foot)
```

Returns:

- Joint solution
- Success flag

---

## Convergence Monitoring

The solver tracks the Cartesian position error during optimization.

---

## Convergence Plot

```python
plot_convergence(errors)
```

Generates a convergence curve showing:

- Iteration number
- Position error evolution

Plots are automatically saved inside:

```text
visualization/plots/ik/
```

---

# Mathematical Background

## Forward Kinematics

The robot pose is computed through chained homogeneous transformations:

$$
T = T_1 T_2 T_3 \cdots T_n
$$

Each transformation combines:

- Rotation
- Translation

---

## Inverse Kinematics

The IK problem is solved numerically because the humanoid structure is nonlinear and highly coupled.

The pseudo-inverse Jacobian method iteratively minimizes the pose error.

---

# Dependencies

Install required packages:

```bash
pip install numpy matplotlib
```

---

# Example Usage

## Forward Kinematics

```python
import numpy as np
from kinematics.forward_kinematics import ForwardKinematics

fk = ForwardKinematics("right")

q = np.zeros(6)

T = fk.compute(q)

print(T)
```

---

## Inverse Kinematics

```python
import numpy as np
from kinematics.inverse_kinematics import InverseKinematics

ik = InverseKinematics("right")

p_target = np.array([0.0, -0.1, -0.8])

q_solution, success = ik.solve(p_target)
```

---

# Outputs

The module can generate:

- Foot position plots
- Jacobian matrices
- IK convergence curves
- Transformation matrices
- Joint angle solutions

---

# Role in the Project

The kinematics module is one of the core subsystems of the humanoid walking controller.

It bridges:

- Robot geometry
- Motion generation
- Walking control
- Dynamics
- Simulation
