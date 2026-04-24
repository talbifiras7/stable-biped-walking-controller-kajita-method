# Bipedal Robot Project

##  Overview

This project implements the **core kinematics pipeline of a lower-body bipedal robot**, including forward kinematics, Jacobian computation, inverse kinematics, and simulation in PyBullet.

The system models a **12-DOF humanoid lower body (6 DOF per leg)** with consistent handling of **left/right symmetry** and full **6D foot pose control (position + orientation)**.

---

##  Key Features

* Full **Forward Kinematics (FK)** for both legs
* Explicit **left/right symmetry modeling**
* **6×6 Jacobian** (numerical, finite differences)
* **Inverse Kinematics (IK)** using Jacobian pseudo-inverse
* Control of **foot position and orientation**
* **FK → IK → FK validation pipeline**
* Integration with **PyBullet simulation**

---

## Project Structure

```id="p4j8xk"
PFA/
│
├── configs/
│   └── robot_config.py
│       → Robot geometry (pelvis, thigh, shank, foot)
│       → Rotation matrices (Rx, Ry, Rz)
│       → Homogeneous transformation utilities
│
├── kinematics/
│   ├── forward_kinematics.py
│   │   → Computes pelvis → foot transform
│   │   → Returns position, rotation, Jacobian
│   │   → Handles left/right symmetry
│   │
│   └── inverse_kinematics.py
│       → Newton-Raphson IK solver
│       → Uses Jacobian pseudo-inverse
│       → Supports full pose targets
│
├── simulation/
│   ├── simulation.py
│   │   → PyBullet simulation entry point
│   │   → Applies joint commands to both legs
│   │
│   └── hum.urdf
│       → Humanoid lower-body model
│
├── visualization/
│   └── plots/
│       → FK / IK / Jacobian visual outputs
```

---

## Kinematic Model

### Structure

Each leg is modeled as:

```id="zt3p9f"
Pelvis → Hip (3 DOF) → Knee (1 DOF) → Ankle (2 DOF) → Foot
```

Total system:

```id="l7x6mk"
2 legs × 6 DOF = 12 DOF
```

---

### Forward Kinematics

The full transform is computed as:

```id="b7w9as"
T = T_pelvis→hip
  · R_hip(q0,q1,q2)
  · T_hip→knee
  · R_knee(q3)
  · T_knee→ankle
  · R_ankle(q4,q5)
  · T_ankle→foot
```

Outputs:

* Foot position `p ∈ ℝ³`
* Foot orientation `R ∈ SO(3)`
* Homogeneous transform `T ∈ ℝ⁴ˣ⁴`

---

### Jacobian

A **6×6 Jacobian** is computed numerically:

```id="5rpk4y"
J = [ linear velocity
      angular velocity ]
```

Used for:

* inverse kinematics
* local motion mapping
* singularity analysis

---

### Inverse Kinematics

Solved using **Newton-Raphson with pseudo-inverse**:

```id="xk6m5y"
Δq = J⁺ · e
```

Where:

```id="k2a6xp"
e = [ position error
      orientation error ]
```

Features:

* Iterative convergence
* Full pose tracking
* Stable updates via gain scaling

---

## How to Run

### Simulation

```bash id="g2c7ha"
python simulation/simulation.py
```

---

### 🔹 Kinematics Tests

```bash id="s6nq4j"
python kinematics/forward_kinematics.py
python kinematics/inverse_kinematics.py
```

These tests include:

* Zero pose validation
* Bent-leg configurations
* FK → IK → FK consistency checks

---

## Validation

* ✔ Left/right symmetry consistency
* ✔ Jacobian numerical stability
* ✔ IK accuracy (millimeter-level error)
* ✔ FK/IK round-trip validation

---

## Summary

This project implements the **core mathematical and computational blocks of a bipedal robot lower body**:

* Forward kinematics
* Differential kinematics (Jacobian)
* Inverse kinematics
* Simulation integration

It provides a solid foundation for extending toward **full-body control, balance, and locomotion**.
