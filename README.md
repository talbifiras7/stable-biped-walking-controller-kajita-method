# PFA Bipedal Robot Project

## Project Overview

This repository contains a Python implementation of core bipedal robot kinematics, including forward kinematics, inverse kinematics, and a PyBullet-based simulation environment. The work is organized to support a 12-DOF humanoid leg model with symmetric left/right leg handling.

> This README is intended to be updated every time new functionality, modules, or documentation is added to the project.

## Current Structure

- `configs/robot_config.py`
  - shared robot geometry constants and rotation utilities
- `kinematics/forward_kinematics.py`
  - leg forward kinematics for right and left legs
  - analytic computation of transform, position, rotation, and finite-difference Jacobian
- `kinematics/inverse_kinematics.py`
  - Newton-Raphson IK solver for one leg
  - solves foot pose given a target position and orientation
- `simulation/simulation.py`
  - PyBullet visualization and robot simulation entrypoint
  - example leg joint control for both right and left legs
- `visualization/plots/`
  - placeholder directories for FK, IK, Jacobian, and trajectory plots
- `simulation/hum.urdf`
  - humanoid URDF model used by the PyBullet simulation

## What’s Implemented

- `ForwardKinematics` for a single leg with mirrored left/right functionality
- `InverseKinematics` solver using Jacobian pseudo-inverse and pose error correction
- `robot_config` geometry definitions for pelvis, thigh, shank, foot, and hip width
- PyBullet demo simulation with a humanoid URDF and leg joint control mapping

## How to Run

1. Activate your Python environment.
2. Install dependencies such as `numpy`, `matplotlib`, and `pybullet` if needed.
3. Run the simulation:

```bash
python simulation/simulation.py
```

4. Run kinematics self-tests directly:

```bash
python kinematics/forward_kinematics.py
python kinematics/inverse_kinematics.py
```

## Notes and Future Updates

- Add new modules or features here as they are implemented.
- Update the `Current Structure` and `What’s Implemented` sections for every new capability.
- Use the top-level `visualization/plots/` directories to store generated figures for FK, IK, Jacobian, and trajectories.

## Development Guidelines

- Keep leg-side symmetry explicit in kinematics code.
- Keep robot geometry parameterized in `configs/robot_config.py`.
- Add unit tests or example scripts for new kinematics or dynamics functions.
- Document any new simulation controls or URDF changes in this README.
