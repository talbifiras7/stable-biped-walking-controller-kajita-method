import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__))) 

from kinematics.forward_kinematics import ForwardKinematics
from kinematics.inverse_kinematics import InverseKinematics


class IKFKSimulation:
    def __init__(self):
        self.fk = ForwardKinematics()
        self.ik = InverseKinematics()

    def generate_trajectory(self, t):
        """
        Simple stepping trajectory (2D sagittal plane)
        """
        x = 0.1 * np.sin(2 * np.pi * t)
        z = -0.7 + 0.05 * np.maximum(0, np.sin(2 * np.pi * t))  # lift foot
        return np.array([x, 0.0, z])  # (x, y, z)

    def run(self, steps=100):
        desired_positions = []
        fk_positions = []
        errors = []

        q = np.zeros(6)  # initial guess (adapt to your DOF)

        for i in range(steps):
            t = i / steps

            # 1. Desired position
            target = self.generate_trajectory(t)

            # 2. IK → joint angles
            q = self.ik.solve(target, q)

            # 3. FK → reconstructed position
            pos_fk = self.fk.compute_foot_position(q)

            # 4. Store
            desired_positions.append(target)
            fk_positions.append(pos_fk)
            errors.append(np.linalg.norm(target - pos_fk))

        return np.array(desired_positions), np.array(fk_positions), np.array(errors)


if __name__ == "__main__":
    sim = IKFKSimulation()
    desired, fk, errors = sim.run()

    # === Plot trajectory ===
    plt.figure()
    plt.plot(desired[:, 0], desired[:, 2], label="Desired")
    plt.plot(fk[:, 0], fk[:, 2], '--', label="FK result")
    plt.legend()
    plt.title("IK → FK consistency")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.grid()

    # === Plot error ===
    plt.figure()
    plt.plot(errors)
    plt.title("Reconstruction Error")
    plt.xlabel("Step")
    plt.ylabel("Error (m)")
    plt.grid()

    plt.show()