import numpy as np
import biorbd
import pickle
import matplotlib.pyplot as plt

# Load the musculoskeletal model
model = biorbd.Model("./With.bioMod")

# Load data from a pickle file
with open("/home/alpha/pianoptim/PianOptim/Nov.Codes/Updated_BioModFile_YeadonModel/Results/Felipe_25March/26March-qdot2/Pressed_with_Thorax.pckl", "rb") as file:
    data_1 = pickle.load(file)

# Extract joint positions (q), velocities (qdot), and accelerations (qddot) from the loaded data
array_q_s_1 = [data_1["states_no_intermediate"][2]["q"][i][0] for i in range (12)]
array_qdot_s_1 = [data_1["states_no_intermediate"][2]["qdot"][i][0] for i in range (12)]

time_step = 0.01  # Example value, replace with your actual time step
Pressed_time_step = 0.00605
Struck_time_step = 0.00501

# Initialize an array to store the accelerations
array_qddot_s_1 = np.zeros_like(array_q_s_1)

for i in range(1, len(array_q_s_1) - 1):
    array_qddot_s_1[i] = (array_q_s_1[i + 1] - 2 * array_q_s_1[i] + array_q_s_1[i - 1]) / Pressed_time_step**2

# For the first and last points, you can use forward/backward difference or another method
# For example, using forward difference for the first point and backward difference for the last point
array_qddot_s_1[0] = (array_q_s_1[2] - 2 * array_q_s_1[1] + array_q_s_1[0]) / Pressed_time_step**2
array_qddot_s_1[-1] = (array_q_s_1[-1] - 2 * array_q_s_1[-2] + array_q_s_1[-3]) / Pressed_time_step**2

# Number of DOFs and nodes (assuming they are consistent with the loaded data)
num_dofs = model.nbQ()
num_nodes = len(array_q_s_1)

# Initialize a matrix to store the inverse dynamics results for each node
inverse_dynamics_results = np.zeros((num_nodes, num_dofs))

# Iterate over the nodes to compute inverse dynamics
for i in range(num_dofs):
    Q = array_q_s_1[i]
    Qdot = array_qdot_s_1[i]
    Qddot = array_qddot_s_1[i]

    # Compute the inverse dynamics (generalized forces) for this node
    Tau = model.InverseDynamics(Q, Qdot, Qddot)

    # Store the results
    inverse_dynamics_results[i, :] = Tau.to_array()

# For example, print the generalized forces for the first node
print(f"Generalized forces for Node 1:\n{inverse_dynamics_results[0, :]}")

