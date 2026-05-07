# Dynamics Module

This module implements the dynamic walking foundations of the humanoid robot using:

- Linear Inverted Pendulum Model (LIPM)
- Zero Moment Point (ZMP) trajectory planning
- Footstep generation
- Center of Mass (CoM) trajectory integration

The module is responsible for generating dynamically stable walking trajectories that can later be tracked by the robot controller.

---

# File Structure

```text
dynamics/
├── lipm.py
└── zmp.py
```

---

# System Overview

The walking pipeline implemented in this module follows the classical humanoid locomotion architecture:

```text
Footsteps → ZMP Reference → LIPM Dynamics → CoM Trajectory
```

The generated trajectories are later consumed by:

- Preview controllers
- Whole-body controllers
- Inverse kinematics
- Simulation modules

---

# Zero Moment Point (ZMP)

## Overview

The ZMP planner generates:

- Stable ZMP references
- Foot trajectories
- Swing/support phase scheduling
- Contact states

based on a predefined footstep sequence.

The implementation follows the walking strategy introduced in Kajita’s humanoid locomotion framework.

---

# ZMP Architecture

## Double Support Phase (DS)

During double support:

- Both feet are in contact with the ground
- The ZMP smoothly transitions between support feet

The ZMP is linearly interpolated:

$$
p_{zmp}(t) =
(1-\alpha)p_{prev} + \alpha p_{support}
$$

---

## Single Support Phase (SS)

During single support:

- One foot remains fixed on the ground
- The other foot swings toward the next target
- The ZMP remains fixed over the support foot

---

# Foot Swing Trajectory

Swing foot trajectories are generated using cubic Bézier curves.

## Control Points

$$
P_0 = p_{start}
$$

$$
P_1 = p_{start} + h
$$

$$
P_2 = p_{end} + h
$$

$$
P_3 = p_{end}
$$

The Bézier trajectory provides:

- Smooth lift-off
- Continuous motion
- Smooth landing
- Natural humanoid stepping motion

---

# Main Classes

## FootStep

Represents one foot placement target.

```python
FootStep(x, y, side)
```

### Attributes

| Attribute | Description         |
|-----------|---------------------|
| `x`       | Footstep x-position |
| `y`       | Footstep y-position |
| `side`    | `right` or `left`   |

---

## ZMPData

Stores the complete walking reference trajectory.

Contains:

- ZMP references
- Foot trajectories
- Contact states
- Timing information

This structure is consumed by:

- LIPM
- Preview controller
- Simulation

---

## ZMPPlanner

Main trajectory planning class.

```python
ZMPPlanner()
```

Responsible for:

- Building ZMP references
- Generating foot trajectories
- Managing support phases
- Generating walking patterns

---

# Built-in Walking Generators

## Forward Walking

```python
make_forward_steps()
```

Generates straight walking trajectories.

---

## Turning Walking

```python
make_turn_steps()
```

Generates turning and curved walking trajectories.

---

# Linear Inverted Pendulum Model (LIPM)

## Overview

The LIPM approximates the humanoid robot as:

- A point mass
- Moving at constant height
- Supported by massless legs

This simplification enables stable dynamic walking generation.

---

# Continuous-Time Dynamics

The ZMP equation is:

$$
p_{zmp}(t)=
p(t)-\frac{z_c}{g}\ddot{p}(t)
$$

where:

- \(p(t)\) is the CoM position
- \(z_c\) is the CoM height
- \(g\) is gravitational acceleration

---

# State Representation

The LIPM state is defined as:

$$
x =
\begin{bmatrix}
p \\
\dot{p} \\
\ddot{p}
\end{bmatrix}
$$

where:

- Position
- Velocity
- Acceleration

are tracked for both X and Y axes.

---

# Discrete-Time Dynamics

The discretized LIPM system is:

$$
x_{k+1}=Ax_k+Bu_k
$$

with jerk input:

$$
u = \dddot{p}
$$

The discrete-time matrices are:

$$
A=
\begin{bmatrix}
1 & dt & \frac{dt^2}{2} \\
0 & 1 & dt \\
0 & 0 & 1
\end{bmatrix}
$$

$$
B=
\begin{bmatrix}
\frac{dt^3}{6} \\
\frac{dt^2}{2} \\
dt
\end{bmatrix}
$$

---

# CoMTrajectory

The `CoMTrajectory` dataclass stores:

- CoM position
- CoM velocity
- CoM acceleration
- ZMP trajectories
- Timing information

for the complete walking sequence.

---

# Main Features

## LIPM Integration

```python
integrate(u_x, u_y)
```

Integrates the CoM dynamics over time using jerk inputs.

---

## ZMP Computation

```python
compute_zmp(x)
```

Computes the Zero Moment Point from the current CoM state.

---

## State Propagation

```python
step(x, u)
```

Advances the system by one timestep.

---

## Trajectory Visualization

The module automatically generates:

- CoM vs ZMP plots
- Velocity plots
- Acceleration plots
- Jerk plots
- Foot height trajectories
- ZMP top-view plots
- Contact phase diagrams

Plots are saved inside:

```text
visualization/plots/dynamics_plots/
```

---

# Mathematical Background

## Dynamic Stability

Humanoid walking stability is achieved by ensuring that:

- The ZMP remains inside the support polygon
- CoM motion remains dynamically feasible

---

## Jerk-Controlled Dynamics

The controller uses jerk as the control input because it produces:

- Smooth acceleration profiles
- Continuous velocity
- Natural walking motion
- Improved numerical stability

---

# Dependencies

Install required packages:

```bash
pip install numpy matplotlib
```

---

# Example Usage

## ZMP Planning

```python
from dynamics.zmp import ZMPPlanner

planner = ZMPPlanner()

steps = planner.make_forward_steps(
    n=6,
    step_length=0.1
)

zmp_data = planner.plan(steps)
```

---

## LIPM Integration

```python
import numpy as np
from dynamics.lipm import LIPM

lipm = LIPM(dt=0.005)

u_x = np.zeros(1000)
u_y = np.zeros(1000)

traj = lipm.integrate(u_x, u_y)
```

---

# Outputs

The module produces:

- ZMP reference trajectories
- CoM trajectories
- Foot trajectories
- Contact schedules
- Dynamic walking plots
- Walking state data structures

---

# Role in the Project

This module is the dynamic core of the humanoid walking framework.

It generates dynamically stable walking references that are later tracked by:

- Preview control algorithms
- Kinematic solvers
- Whole-body controllers
- Physics simulations