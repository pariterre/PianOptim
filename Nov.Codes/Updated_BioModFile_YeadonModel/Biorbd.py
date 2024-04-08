import biorbd
import numpy as np
import pickle
import matplotlib.pyplot as plt

def get_user_input():
    while True:
        all_dof = input("Include thorax? (y/n): ").lower()
        if all_dof in ['y', 'n']:
            all_dof = all_dof == 'y'
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

    while True:
        pressed = input("Show 'Pressed' or 'Struck' condition? (p/s): ").lower()
        if pressed in ['p', 's']:
            pressed = pressed == 'p'
            break
        else:
            print("Invalid input. Please enter 'p' or 's'.")

    return all_dof, pressed


allDOF, pressed = get_user_input()

dirName = "/home/alpha/pianoptim/PianOptim/Nov.Codes/Updated_BioModFile_YeadonModel/Results/Felipe_25March/Version2_2Apr/"

if allDOF:
    saveName = dirName + ("Pressed" if pressed else "Struck") + "_with_Thorax.pckl"
else:
    saveName = dirName + ("Pressed" if pressed else "Struck") + "_without_Thorax.pckl"

biorbd_model_path = "./With.bioMod" if allDOF else "./Without.bioMod"

# Load the musculoskeletal model
model = biorbd.Model("./With.bioMod" if allDOF else "./Without.bioMod")


with open(saveName, "rb") as file:
    new_dict = pickle.load(file)

# Ask the user for the number of Degrees of Freedom (DOFs)
num_dofs = int(input("Please enter the number of Degrees of Freedom (DOFs) in the system: "))

# Ask the user for the number of shooting nodes
num_nodes = int(input("Please enter the number of shooting nodes in the system: "))


# Extract joint positions (q) and velocities (qdot) from the loaded data
array_q_s_1 = [new_dict["states_no_intermediate"][i]["q"] for i in range(len(new_dict["states_no_intermediate"]))]
array_qdot_s_1 = [new_dict["states_no_intermediate"][i]["qdot"] for i in range(len(new_dict["states_no_intermediate"]))]

# List to store the detailed contributions for each DOF and node before summation
detailed_contributions = []

# Iterate over the nodes
for i in range(num_nodes):  # Assuming there are 7 nodes
    q_obj = np.array([array_q_s_1[1][j][i] for j in range(num_dofs)])  # Phase 1, Over 7 DOFs
    qdot_obj = np.array([array_qdot_s_1[1][j][i] for j in range(num_dofs)])  # Phase 1, Over 7 DOFs

    # Computing the Jacobian for the "finger_marker"
    fingertip_marker_name = "finger_marker"
    parent_name = "RightFingers"
    p = biorbd.NodeSegment(0, 0, -0.046782999938)
    update_kin = True
    markers_jacobian = model.markersJacobian(q_obj, parent_name, p, update_kin)

    # Extract the Jacobian matrix as a NumPy array
    jacobian_array = markers_jacobian.to_array()

    # Initialize a matrix to store the contributions for this node
    contributions_matrix = np.zeros((jacobian_array.shape[0], jacobian_array.shape[1]))

    # Calculate the contributions for each DOF
    for dof in range(jacobian_array.shape[1]):  # Iterate over DOFs
        for dim in range(jacobian_array.shape[0]):  # Iterate over dimensions (x, y, z)
            contributions_matrix[dim][dof] = jacobian_array[dim][dof] * qdot_obj[dof]

    # Store the contributions matrix for this node
    detailed_contributions.append(contributions_matrix)

    # Sum the contributions to get the velocity for this node
    fingertip_velocity = np.sum(contributions_matrix, axis=1)
    print(f"Fingertip velocity for Node {i}:\n{fingertip_velocity}")

# Now detailed_contributions contains the individual contributions for each DOF and node before summation
# You can print or process these values as needed
for node_idx, contributions in enumerate(detailed_contributions):
    print(f"Contributions for Node {node_idx}:\n{contributions}")

z_contributions_by_nodes = np.random.randn(num_nodes, num_dofs)

for i in range(num_nodes):

    z_contributions_by_nodes[i] = detailed_contributions[i][2]

num_nodes, num_dofs = z_contributions_by_nodes.shape
# Calculate the final Z velocity for each node by summing the contributions across all DOFs
final_z_velocity_per_node = np.sum(z_contributions_by_nodes, axis=1)

# Assign the DOF names based on the all_dof variable
if allDOF:
    joint_dof_map = {
        "Pelvic": [0],
        "Thoracic": [1, 2],
        "Upper Thoracic": [3, 4],
        "Shoulder": [5, 6, 7],
        "Elbow": [8, 9],
        "Wrist": [10],
        "MCP": [11]
    }
else:
    joint_dof_map = {
        "Shoulder": [0, 1, 2],
        "Elbow": [3, 4],
        "Wrist": [5],
        "MCP": [6]
    }


joint_contributions_by_nodes = {}

# Sum the contributions of the DOFs belonging to the same joint for each node
for node_idx in range(num_nodes):
    joint_contributions = {}
    for joint_name, dof_indices in joint_dof_map.items():
        joint_contribution = sum(z_contributions_by_nodes[node_idx][dof_idx] for dof_idx in dof_indices)
        joint_contributions[joint_name] = joint_contribution
    joint_contributions_by_nodes[node_idx] = joint_contributions

# Calculate the total contribution of all joints for each node
total_contributions_by_nodes = [sum(joint_contributions.values()) for joint_contributions in joint_contributions_by_nodes.values()]

# Create a figure for the time series plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the time series curves of the joint contributions
for joint_name in joint_dof_map.keys():
    joint_contributions = [joint_contributions_by_nodes[node_idx][joint_name] for node_idx in range(num_nodes)]
    ax.plot(range(num_nodes), joint_contributions, label=joint_name)

# Plot the total contributions of all joints for each node
ax.plot(range(num_nodes), total_contributions_by_nodes, label='Total', linestyle='--', linewidth=2)

ax.set_xlabel('Node')
ax.set_ylabel('Z Velocity Contribution')
ax.set_title('Joint Contributions to Z Velocity over Shooting Nodes')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()