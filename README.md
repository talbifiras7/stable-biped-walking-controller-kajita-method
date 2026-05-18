# Stable Biped Walking Controller — Kajita Method

A research-oriented humanoid locomotion framework implementing the classical **Kajita walking pipeline** using:

- Zero Moment Point (ZMP) trajectory generation
- Linear Inverted Pendulum Model (LIPM)
- Preview Control
- Forward & Inverse Kinematics
- Walking visualization and simulation

The project is designed to be modular, educational, and extensible for robotics, humanoid locomotion, and control research.

---

# Project Goals

This repository focuses on building the foundations of a stable humanoid walking controller from scratch instead of relying on large robotics frameworks.

Main objectives:

- Understand humanoid locomotion mathematically
- Implement classical walking control algorithms
- Simulate dynamically stable walking
- Build reusable robotics modules
- Prepare the stack for future integration with:
  - MPC
  - Reinforcement Learning
  - Whole-body control
  - ROS / Gazebo
  - Real humanoid hardware

---

# Walking Pipeline

```text
Footstep Planner
        ↓
ZMP Reference Generation
        ↓
Preview Controller
        ↓
LIPM Dynamics
        ↓
CoM Trajectory
        ↓
Inverse Kinematics
        ↓
Joint Motion
        ↓
Visualization / Simulation
```

---

# Repository Structure

```text
stable-biped-walking-controller-kajita-method/
│
├── configs/           # Robot geometry and transformation utilities
├── controler/         # Preview control implementation
├── dynamics/          # ZMP and LIPM dynamics
├── kinematics/        # FK and IK solvers
├── simulation/        # Physics / robot simulation
├── visualization/     # Plots and GIF generation
│
├── system_architecture.png
└── README.md
```

---

# Core Modules

## Configs

Centralized robot parameters and transformation utilities.

Includes:
- Rotation matrices
- Homogeneous transforms
- Robot geometry
- Coordinate conventions

---

## Kinematics

Implements:
- Forward Kinematics (FK)
- Inverse Kinematics (IK)

Supports:
- Foot pose computation
- Joint reconstruction
- Humanoid leg modeling

---

## Dynamics

Implements:
- ZMP trajectory generation
- LIPM dynamics
- CoM integration
- Walking stability foundations

---

## Preview Controller

Implements the Kajita preview control method to track ZMP references and stabilize walking trajectories.

---

## Visualization

Generates:
- CoM vs ZMP plots
- Walking trajectories
- Animated GIFs
- Top-view walking analysis

---

## Simulation

Contains the simulation environment and URDF model used to test locomotion behaviors.

---

# Mathematical Foundations

The project is based on the Linear Inverted Pendulum Model:

x_zmp = x - (z_c / g) * ẍ

and the preview control formulation introduced by:
- Kajita et al.
- Honda humanoid walking research
- Classical ZMP stabilization methods

---

# Current Features

- Modular robotics architecture
- Humanoid walking pipeline
- ZMP generation
- LIPM trajectory integration
- FK / IK implementation
- GIF trajectory visualization
- Research-oriented code organization

---

# Planned Improvements

- Full MPC controller
- ROS2 integration
- Gazebo simulation
- Terrain adaptation
- Footstep optimization
- Reinforcement learning integration
- Real-time dashboard
- Whole-body balancing

---

# Installation

```bash
git clone https://github.com/talbifiras7/stable-biped-walking-controller-kajita-method.git

cd stable-biped-walking-controller-kajita-method

pip install -r requirements.txt
```

---

# Running the Project

Example:

```bash
python visualization/gifs/Visualizer.py
```

---

# Visualization Examples

Generated outputs include:

- Top-view walking GIFs
- CoM trajectories
- ZMP tracking curves
- Walking phase visualization

---
