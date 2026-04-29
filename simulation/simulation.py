import pybullet as p
import pybullet_data
import time
import numpy as np

from kinematics.forward_kinematics import ForwardKinematics


# ----------------------------
# PYBULLET SETUP
# ----------------------------
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

p.loadURDF("plane.urdf")


robot = p.loadURDF("hum.urdf", [0, 0, 1])

p.createConstraint(
    parentBodyUniqueId=robot,
    parentLinkIndex=-1,
    childBodyUniqueId=robot,
    childLinkIndex=-1,
    jointType=p.JOINT_FIXED,
    jointAxis=[0, 0, 0],
    parentFramePosition=[0, 0, 1.0],
    childFramePosition=[0, 0, 1.0]
)


# ----------------------------
# JOINT MAPPING (EDIT THIS)
# ----------------------------
RIGHT_LEG = [0, 1, 2, 3, 4, 5]
LEFT_LEG  = [6, 7, 8, 9, 10, 11]


def set_leg(robot, joints, q):
    for i in range(6):
        p.setJointMotorControl2(
        bodyIndex=robot,
        jointIndex=joints[i],
        controlMode=p.POSITION_CONTROL,
        targetPosition=float(q[i]),
        force=200,
        positionGain=0.2,
        velocityGain=1.0
        )


# ----------------------------
# ONE FK OBJECT ONLY
# ----------------------------
fk = ForwardKinematics("right")  # we will switch side dynamically


t0 = time.time()

while True:

    t = time.time() - t0

    # test motion
    q = np.array([
        0.3 * np.sin(t),
        -0.4,
        0.0,
        0.8,
        -0.4,
        0.3 * np.sin(t)
    ])

    # ----------------------------
    # RIGHT LEG FK CHECK
    # ----------------------------
    fk.side = "right"
    p_r = fk.position(q)

    set_leg(robot, RIGHT_LEG, q)

    # ----------------------------
    # LEFT LEG FK CHECK
    # ----------------------------
    fk.side = "left"
    p_l = fk.position(q)

    set_leg(robot, LEFT_LEG, q)

    # reset (clean state)
    fk.side = "right"

    # ----------------------------
    # DEBUG SYMMETRY
    # ----------------------------
    print("Symmetry error (Y):", p_r[1] + p_l[1])

    # step sim
    p.stepSimulation()
    time.sleep(1. / 240.)