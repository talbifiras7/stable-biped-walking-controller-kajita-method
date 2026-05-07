# Humanoid Walking Controller

A modular humanoid robotics framework for:

- Forward and inverse kinematics
- Dynamic walking generation
- ZMP trajectory planning
- LIPM-based locomotion
- Walking simulation and visualization

The project is designed as a research-oriented humanoid locomotion stack with clean subsystem separation and reusable robotics components.

---

# Project Architecture

```text
Humanoid Walking Controller
│
├── robot_config/     → Geometry and transformation utilities
├── kinematics/       → Forward and inverse kinematics
├── dynamics/         → ZMP + LIPM walking dynamics
├── visualization/    → Plotting and rendering tools
└── simulation/       → Walking simulation pipelines
```

---

# Core Modules

## Robot Configuration

Defines:

- Robot geometry
- Rotation matrices
- Homogeneous transformations
- Coordinate utilities

📄 See:

```text
robot_config/README.md
```

---

## Kinematics

Implements:

- Forward kinematics
- Inverse kinematics
- Jacobian computation
- Pose reconstruction

📄 See:

```text
kinematics/README.md
```

---

## Dynamics

Implements:

- Linear Inverted Pendulum Model (LIPM)
- Zero Moment Point (ZMP)
- Footstep planning
- CoM trajectory generation

📄 See:

```text
dynamics/README.md
```

---

# Walking Pipeline

The humanoid walking pipeline follows:

$$
\text{Footsteps}
\rightarrow
\text{ZMP Planning}
\rightarrow
\text{LIPM Dynamics}
\rightarrow
\text{CoM Trajectory}
\rightarrow
\text{Inverse Kinematics}
\rightarrow
\text{Joint Commands}
$$

---

# Features

- Modular humanoid robotics architecture
- 12-DOF humanoid leg model
- Numerical inverse kinematics
- Jacobian-based solvers
- ZMP trajectory generation
- LIPM dynamic walking
- Walking visualization tools
- Research-oriented implementation

---

# Dependencies

Install required packages:

```bash
pip install numpy matplotlib
```

---

# Quick Example

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

## ZMP Walking Generation

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

# Visualization

The framework can generate:

- Foot trajectory plots
- ZMP trajectories
- CoM trajectories
- IK convergence plots
- Walking phase diagrams
- Dynamic walking visualizations

Generated plots are automatically saved inside:

```text
visualization/plots/
```

---

# Mathematical Foundations

This project is based on classical humanoid robotics methods including:

- Homogeneous transformations
- Numerical Jacobians
- Newton–Raphson inverse kinematics
- Zero Moment Point (ZMP)
- Linear Inverted Pendulum Model (LIPM)
- Jerk-controlled dynamic walking

---

# Future Extensions

Planned additions include:

- Preview control
- MPC walking controller
- Whole-body control
- ROS integration
- Physics engine simulation
- Reinforcement learning locomotion

---

# License

MIT License

---

# References

- Kajita et al. — Introduction to Humanoid Robotics