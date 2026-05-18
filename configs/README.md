# Configs Module

This module centralizes all robot geometry parameters and mathematical transformation utilities used throughout the project.

It acts as the shared foundation for:
- Kinematics
- Dynamics
- Preview control
- Simulation
- Visualization

---

# File Structure

```text
configs/
├── robot_config.py
└── README.md
```

---

# Responsibilities

The module provides:

- Robot geometric dimensions
- Link lengths
- Coordinate system conventions
- Rotation matrices
- Homogeneous transformation matrices
- Shared robotics math utilities

---

# Coordinate System

The project uses a right-handed coordinate frame:

```text
X → Forward
Y → Lateral
Z → Upward
```

---

# Main Utilities

Typical utilities include:
- Rotation around X-axis
- Rotation around Y-axis
- Rotation around Z-axis
- Translation transforms
- Frame composition

---

# Why This Module Matters

Keeping geometry and transformations centralized ensures:
- Consistency across the project
- Easier debugging
- Cleaner robotics architecture
- Better scalability