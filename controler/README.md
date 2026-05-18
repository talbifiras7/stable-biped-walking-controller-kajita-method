# Preview Controller Module

This module implements the preview control strategy used for stable humanoid walking based on the Kajita method.

The controller tracks a desired ZMP trajectory and computes jerk commands that stabilize the Center of Mass (CoM).

---

# File Structure

```text
controler/
├── preview_controler.py
└── README.md
```

---

# Controller Pipeline

```text
ZMP Reference
        ↓
Preview Controller
        ↓
CoM Jerk Input
        ↓
LIPM Integration
        ↓
Stable Walking Motion
```

---

# Main Responsibilities

The preview controller:
- Predicts future ZMP behavior
- Minimizes tracking error
- Stabilizes the walking trajectory
- Produces smooth CoM motion

---

# State Representation

The controller typically operates on:

x = [c, ċ, c̈]^T

where:
- c is the CoM position
- ċ is velocity
- c̈ is acceleration

---

# Why Preview Control

Preview control is widely used in humanoid robotics because it:
- Anticipates future footsteps
- Produces stable walking
- Smooths dynamic transitions
- Works efficiently with LIPM

---

# References

Inspired by:
- Kajita et al.
- Honda humanoid walking research
- Classical ZMP preview control formulations