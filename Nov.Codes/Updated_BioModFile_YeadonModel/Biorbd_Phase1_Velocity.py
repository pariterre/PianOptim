import biorbd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def get_user_input():
    while True:
        pressed = input("Show 'Pressed' or 'Struck' condition? (p/s): ").lower()
        if pressed in ['p', 's']:
            pressed = pressed == 'p'
            break
        else:
            print("Invalid input. Please enter 'p' or 's'.")

    return pressed

pressed = get_user_input()

dirName  = "/home/alpha/pianoptim/PianOptim/Nov.Codes/Updated_BioModFile_YeadonModel/Updated_Biomod_Distance/Final_Presentation_25APril_124_Q/"

saveName_DT = dirName + ("Pressed" if pressed else "Struck") + "_with_Thorax.pckl"
saveName_ST = dirName + ("Pressed" if pressed else "Struck") + "_without_Thorax.pckl"

biorbd_model_path_DT = "./With.bioMod"
biorbd_model_path_ST = "./Without.bioMod"

# Load the musculoskeletal models
model_DT = biorbd.Model("./With.bioMod")
model_ST = biorbd.Model("./Without.bioMod")

with open(saveName_DT, "rb") as file:
    dict_DT = pickle.load(file)

with open(saveName_ST, "rb") as file:
    dict_ST = pickle.load(file)

# Number of Degrees of Freedom (DOFs)
num_DOFs_DT = model_DT.nbDof()
num_DOFs_ST = model_ST.nbDof()

# Number of Shooting Nodes
if pressed:
    num_nodes = 8
else:
    num_nodes = 7

# Extract joint positions (q) and velocities (qdot) from the loaded data
array_q_s_DT = [dict_DT["states_no_intermediate"][i]["q"] for i in range(len(dict_DT["states_no_intermediate"]))]
array_qdot_s_DT = [dict_DT["states_no_intermediate"][i]["qdot"] for i in range(len(dict_DT["states_no_intermediate"]))]

array_q_s_ST = [dict_ST["states_no_intermediate"][i]["q"] for i in range(len(dict_ST["states_no_intermediate"]))]
array_qdot_s_ST = [dict_ST["states_no_intermediate"][i]["qdot"] for i in range(len(dict_ST["states_no_intermediate"]))]

# List to store the detailed contributions for each DOF and node before summation
detailed_contributions_DT = []
detailed_contributions_ST = []

# Iterate over the nodes for the first file (DT)
for i in range(num_nodes):
    q_obj_DT = np.array([array_q_s_DT[1][j][i] for j in range(num_DOFs_DT)])
    qdot_obj_DT = np.array([array_qdot_s_DT[1][j][i] for j in range(num_DOFs_DT)])

    # Computing the Jacobian for the "finger_marker" in the first file (DT)
    fingertip_marker_name = "finger_marker"
    parent_name = "RightFingers"
    p = biorbd.NodeSegment(0, 0, -0.046782999938)
    update_kin = True
    markers_jacobian_DT = model_DT.markersJacobian(q_obj_DT, parent_name, p, update_kin)

    # Extract the Jacobian matrix as a NumPy array for the first file (DT)
    jacobian_array_DT = markers_jacobian_DT.to_array()

    # Initialize a matrix to store the contributions for this node in the first file (DT)
    contributions_matrix_DT = np.zeros((jacobian_array_DT.shape[0], jacobian_array_DT.shape[1]))

    # Calculate the contributions for each DOF in the first file (DT)
    for dof in range(jacobian_array_DT.shape[1]):
        for dim in range(jacobian_array_DT.shape[0]):
            contributions_matrix_DT[dim][dof] = jacobian_array_DT[dim][dof] * qdot_obj_DT[dof]

    # Store the contributions matrix for this node in the first file (DT)
    detailed_contributions_DT.append(contributions_matrix_DT)

# Iterate over the nodes for the second file (ST)
for i in range(num_nodes):
    q_obj_ST = np.array([array_q_s_ST[1][j][i] for j in range(num_DOFs_ST)])
    qdot_obj_ST = np.array([array_qdot_s_ST[1][j][i] for j in range(num_DOFs_ST)])

    # Computing the Jacobian for the "finger_marker" in the second file (ST)
    fingertip_marker_name = "finger_marker"
    parent_name = "RightFingers"
    p = biorbd.NodeSegment(0, 0, -0.046782999938)
    update_kin = True
    markers_jacobian_ST = model_ST.markersJacobian(q_obj_ST, parent_name, p, update_kin)

    # Extract the Jacobian matrix as a NumPy array for the second file (ST)
    jacobian_array_ST = markers_jacobian_ST.to_array()

    # Initialize a matrix to store the contributions for this node in the second file (ST)
    contributions_matrix_ST = np.zeros((jacobian_array_ST.shape[0], jacobian_array_ST.shape[1]))

    # Calculate the contributions for each DOF in the second file (ST)
    for dof in range(jacobian_array_ST.shape[1]):
        for dim in range(jacobian_array_ST.shape[0]):
            contributions_matrix_ST[dim][dof] = jacobian_array_ST[dim][dof] * qdot_obj_ST[dof]

    # Store the contributions matrix for this node in the second file (ST)
    detailed_contributions_ST.append(contributions_matrix_ST)

z_contributions_by_nodes_DT = np.zeros((num_nodes, num_DOFs_DT))
z_contributions_by_nodes_ST = np.zeros((num_nodes, num_DOFs_ST))

for i in range(num_nodes):
    z_contributions_by_nodes_DT[i] = detailed_contributions_DT[i][2]
    z_contributions_by_nodes_ST[i] = detailed_contributions_ST[i][2]

# Calculate the final Z velocity for each node by summing the contributions across all DOFs
final_z_velocity_per_node_DT = np.sum(z_contributions_by_nodes_DT, axis=1)
final_z_velocity_per_node_ST = np.sum(z_contributions_by_nodes_ST, axis=1)

# Assign the DOF names for the first file (DT)
joint_dof_map_DT = {
    "Pelvic": [0],
    "Thoracic": [1, 2],
    "Upper Thoracic": [3, 4],
    "Shoulder": [5, 6, 7],
    "Elbow": [8, 9],
    "Wrist": [10],
    "MCP": [11]
}

# Assign the DOF names for the second file (ST)
joint_dof_map_ST = {
    "Shoulder": [0, 1, 2],
    "Elbow": [3, 4],
    "Wrist": [5],
    "MCP": [6]
}

joint_contributions_by_nodes_DT = {}
joint_contributions_by_nodes_ST = {}

# Sum the contributions of the DOFs belonging to the same joint for each node in the first file (DT)
for node_idx in range(num_nodes):
    joint_contributions_DT = {}
    for joint_name, dof_indices in joint_dof_map_DT.items():
        joint_contribution_DT = sum(z_contributions_by_nodes_DT[node_idx][dof_idx] for dof_idx in dof_indices)
        joint_contributions_DT[joint_name] = joint_contribution_DT
    joint_contributions_by_nodes_DT[node_idx] = joint_contributions_DT

# Sum the contributions of the DOFs belonging to the same joint for each node in the second file (ST)
for node_idx in range(num_nodes):
    joint_contributions_ST = {}
    for joint_name, dof_indices in joint_dof_map_ST.items():
        joint_contribution_ST = sum(z_contributions_by_nodes_ST[node_idx][dof_idx] for dof_idx in dof_indices)
        joint_contributions_ST[joint_name] = joint_contribution_ST
    joint_contributions_by_nodes_ST[node_idx] = joint_contributions_ST

# Calculate the total contribution of all joints for each node in the first file (DT)
total_contributions_by_nodes_DT = [sum(joint_contributions.values()) for joint_contributions in joint_contributions_by_nodes_DT.values()]

# Calculate the total contribution of all joints for each node in the second file (ST)
total_contributions_by_nodes_ST = [sum(joint_contributions.values()) for joint_contributions in joint_contributions_by_nodes_ST.values()]

num_joints = len(joint_dof_map_DT)
colors = cm.rainbow(np.linspace(0, 1, num_joints))
joint_color_map = dict(zip(joint_dof_map_DT.keys(), colors))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

for joint_name, color in joint_color_map.items():
    joint_contributions_DT = [joint_contributions_by_nodes_DT[node_idx][joint_name] for node_idx in range(num_nodes)]
    ax1.plot(range(num_nodes), joint_contributions_DT, label=joint_name, color=color)

    if joint_name in joint_dof_map_ST:
        joint_contributions_ST = [joint_contributions_by_nodes_ST[node_idx][joint_name] for node_idx in range(num_nodes)]
        ax2.plot(range(num_nodes), joint_contributions_ST, label=joint_name, color=color)

ax1.plot(range(num_nodes), total_contributions_by_nodes_DT, label='Total', linestyle='--', linewidth=2, color='black')
ax2.plot(range(num_nodes), total_contributions_by_nodes_ST, label='Total', linestyle='--', linewidth=2, color='black')

y_min = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
y_max = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
ax1.set_ylim(y_min, y_max)
ax2.set_ylim(y_min, y_max)

ax1.set_xlabel('Node')
ax1.set_ylabel('Z Velocity Contribution')
ax1.set_title("Pressed_Joint Contributions (DT)" if pressed else "Struck_Joint Contributions (DT)")
ax1.legend()
ax1.grid(True)

ax2.set_xlabel('Node')
ax2.set_ylabel('Z Velocity Contribution')
ax2.set_title("Pressed_Joint Contributions (ST)" if pressed else "Struck_Joint Contributions (ST)")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# Get user input for the number of digits after zero for rounding
num_digits = 2

# Create a figure with subplots for the graph and tables
fig = plt.figure(figsize=(16, 12))
ax_graph = fig.add_subplot(3, 1, 1)
ax_table_DT = fig.add_subplot(3, 1, 2)
ax_table_ST = fig.add_subplot(3, 1, 3)

# Plot the final velocity for each node
ax_graph.plot(range(num_nodes), final_z_velocity_per_node_DT, label='Final Velocity (DT)')
ax_graph.plot(range(num_nodes), final_z_velocity_per_node_ST, label='Final Velocity (ST)')
ax_graph.set_xlabel('Node', fontsize=16)
ax_graph.set_ylabel('Final Z Velocity', fontsize=16)
ax_graph.set_title('Final Z Velocity per Node', fontsize=16)
ax_graph.legend()
ax_graph.grid(True)

# Create a table to display the joint contributions for each node in the first file (DT)
table_data_DT = []
for node_idx in range(num_nodes):
    row_data_DT = [node_idx]
    for joint_name in joint_dof_map_DT.keys():
        row_data_DT.append(round(joint_contributions_by_nodes_DT[node_idx][joint_name], num_digits))
    row_data_DT.append(round(total_contributions_by_nodes_DT[node_idx], num_digits))
    table_data_DT.append(row_data_DT)

columns_DT = ['Node'] + list(joint_dof_map_DT.keys()) + ['Total']
ax_table_DT.axis('tight')
ax_table_DT.axis('off')
table_DT = ax_table_DT.table(cellText=table_data_DT, colLabels=columns_DT, loc='center', cellLoc='center')
table_DT.auto_set_font_size(False)
table_DT.set_fontsize(10)
table_DT.scale(1.2, 1.2)
ax_table_DT.set_title("Table 1: Joint Contributions (DT)")

# Create a table to display the joint contributions for each node in the second file (ST)
table_data_ST = []
for node_idx in range(num_nodes):
    row_data_ST = [node_idx]
    for joint_name in joint_dof_map_ST.keys():
        row_data_ST.append(round(joint_contributions_by_nodes_ST[node_idx][joint_name], num_digits))
    row_data_ST.append(round(total_contributions_by_nodes_ST[node_idx], num_digits))
    table_data_ST.append(row_data_ST)

columns_ST = ['Node'] + list(joint_dof_map_ST.keys()) + ['Total']
ax_table_ST.axis('tight')
ax_table_ST.axis('off')
table_ST = ax_table_ST.table(cellText=table_data_ST, colLabels=columns_ST, loc='center', cellLoc='center')
table_ST.auto_set_font_size(False)
table_ST.set_fontsize(10)
table_ST.scale(1.2, 1.2)
ax_table_ST.set_title("Table 2: Joint Contributions (ST)")
ax_table_ST.axis('tight')

# Create a figure with subplots for each joint
fig, (ax_shoulder, ax_wrist, ax_elbow) = plt.subplots(3, 1, figsize=(10, 12))

# Plot the joint contributions for shoulder
if 'Shoulder' in joint_dof_map_DT:
    shoulder_contributions_DT = [joint_contributions_by_nodes_DT[node_idx]['Shoulder'] for node_idx in range(num_nodes)]
    ax_shoulder.plot(range(num_nodes), shoulder_contributions_DT, label='Shoulder (DT)', linestyle='-', color="red")

if 'Shoulder' in joint_dof_map_ST:
    shoulder_contributions_ST = [joint_contributions_by_nodes_ST[node_idx]['Shoulder'] for node_idx in range(num_nodes)]
    ax_shoulder.plot(range(num_nodes), shoulder_contributions_ST, label='Shoulder (ST)', linestyle='--', color="blue")

ax_shoulder.set_xlabel('Node', fontsize=14)
ax_shoulder.set_ylabel('Z Velocity Contribution', fontsize=14)
ax_shoulder.set_title('Shoulder Contributions', fontsize=14)
ax_shoulder.legend()
ax_shoulder.grid(True)

# Plot the joint contributions for wrist
if 'Wrist' in joint_dof_map_DT:
    wrist_contributions_DT = [joint_contributions_by_nodes_DT[node_idx]['Wrist'] for node_idx in range(num_nodes)]
    ax_wrist.plot(range(num_nodes), wrist_contributions_DT, label='Wrist (DT)', linestyle='-', color="red")

if 'Wrist' in joint_dof_map_ST:
    wrist_contributions_ST = [joint_contributions_by_nodes_ST[node_idx]['Wrist'] for node_idx in range(num_nodes)]
    ax_wrist.plot(range(num_nodes), wrist_contributions_ST, label='Wrist (ST)', linestyle='--', color="blue")

ax_wrist.set_xlabel('Node', fontsize=14)
ax_wrist.set_ylabel('Z Velocity Contribution', fontsize=14)
ax_wrist.set_title('Wrist Contributions', fontsize=14)
ax_wrist.legend()
ax_wrist.grid(True)

# Plot the joint contributions for elbow
if 'Elbow' in joint_dof_map_DT:
    elbow_contributions_DT = [joint_contributions_by_nodes_DT[node_idx]['Elbow'] for node_idx in range(num_nodes)]
    ax_elbow.plot(range(num_nodes), elbow_contributions_DT, label='Elbow (DT)', linestyle='-', color="red")

if 'Elbow' in joint_dof_map_ST:
    elbow_contributions_ST = [joint_contributions_by_nodes_ST[node_idx]['Elbow'] for node_idx in range(num_nodes)]
    ax_elbow.plot(range(num_nodes), elbow_contributions_ST, label='Elbow (ST)', linestyle='--', color="blue")

ax_elbow.set_xlabel('Node', fontsize=14)
ax_elbow.set_ylabel('Z Velocity Contribution', fontsize=14)
ax_elbow.set_title('Elbow Contributions', fontsize=14)
ax_elbow.legend()
ax_elbow.grid(True)


plt.tight_layout()
plt.show()

