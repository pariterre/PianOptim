import numpy as np


def calculate_joint_energy_transfer(q, qdot, tau, dt):
    # Calculate joint power
    joint_power = tau * qdot

    # Calculate joint work
    joint_work = joint_power * dt

    # Calculate total energy transfer
    total_energy_transfer = np.sum(joint_work, axis=0)

    return joint_power, joint_work, total_energy_transfer
