import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
def degrees(radians):
    return np.degrees(radians)


dirName = "/home/alpha/Desktop/New_results_19Jan2024/"
typeTouch = "Struck" #"Pressed" #

# Load data_1
with open(dirName + typeTouch + "_with_Thorax.pckl",
          "rb") as file:
    data_1 = pickle.load(file)

with open(dirName + typeTouch + "_without_Thorax.pckl",
          "rb") as file:
    data_2 = pickle.load(file)

# Process specific points for data_1 and data_2
specific_points_s_1 = [sum(data_1["phase_time"][: i + 1]) for i in range(len(data_1["phase_time"]))]
specific_points_s_2 = [sum(data_2["phase_time"][: i + 1]) for i in range(len(data_2["phase_time"]))]

# Labels for data_1 and data_2
label_1 = "with"
label_2 = "without"

# Processing data_1 and data_2 for q, qdot, tau
# For data_1
array_q_s_1 = [data_1["states_no_intermediate"][i]["q"] for i in range(len(data_1["states_no_intermediate"]))]
array_qdot_s_1 = [data_1["states_no_intermediate"][i]["qdot"] for i in range(len(data_1["states_no_intermediate"]))]
array_tau_s_1 = [data_1["controls"][i]["tau"] for i in range(len(data_1["controls"]))]

# Replace NaN values in tau arrays for data_1
for i in range(len(array_tau_s_1) - 1):
    array_tau_s_1[i][:, -1] = array_tau_s_1[i + 1][:, 0]
array_tau_s_1[-1][:, -1] = array_tau_s_1[-1][:, -2]

# For data_2
array_q_s_2 = [data_2["states_no_intermediate"][i]["q"] for i in range(len(data_2["states_no_intermediate"]))]
array_qdot_s_2 = [data_2["states_no_intermediate"][i]["qdot"] for i in range(len(data_2["states_no_intermediate"]))]
array_tau_s_2 = [data_2["controls"][i]["tau"] for i in range(len(data_2["controls"]))]

# Replace NaN values in tau arrays for data_2
for i in range(len(array_tau_s_2) - 1):
    array_tau_s_2[i][:, -1] = array_tau_s_2[i + 1][:, 0]
array_tau_s_2[-1][:, -1] = array_tau_s_2[-1][:, -2]

# Concatenate arrays for q, qdot, tau for both data sets
concatenated_array_q_s_1 = degrees(np.concatenate(array_q_s_1, axis=1))
concatenated_array_qdot_s_1 = degrees(np.concatenate(array_qdot_s_1, axis=1))
concatenated_array_tau_s_1 = np.concatenate(array_tau_s_1, axis=1)

concatenated_array_q_s_2 = degrees(np.concatenate(array_q_s_2, axis=1))
concatenated_array_qdot_s_2 = degrees(np.concatenate(array_qdot_s_2, axis=1))
concatenated_array_tau_s_2 = np.concatenate(array_tau_s_2, axis=1)

# Generate time array for plotting for both data sets
time_arrays_1 = [
    np.linspace(specific_points_s_1[i], specific_points_s_1[i + 1], len(array_q_s_1[i][0]))
    for i in range(len(specific_points_s_1) - 1)
]
concatenated_array_time_s_1 = np.concatenate(time_arrays_1)

time_arrays_2 = [
    np.linspace(specific_points_s_2[i], specific_points_s_2[i + 1], len(array_q_s_2[i][0]))
    for i in range(len(specific_points_s_2) - 1)
]
concatenated_array_time_s_2 = np.concatenate(time_arrays_2)

# Plotting
Name = [
    "Pelvic Tilt, Ant/Post",
    "Thorax, Left/Right",
    "Thorax, Flex/Ext",
    "Right Shoulder, Abd/Add",
    "Right Shoulder, Int/Ext",
    "Right Shoulder, Flex/Ext",
    "Elbow, Flex/Ext",
    "Elbow, Pron/Supin",
    "Wrist, Flex/Ext",
    "MCP, Flex/Ext",
]
#
# # Function to display plots for all phases for each group of joints
# def display_all_phases_per_group_of_joints(data1, data2, num_phases, num_joints_data1, num_joints_data2, title_prefix):
#     offset = 3  # Offset for data2 indices
#
#     for start_joint_data1 in range(0, num_joints_data1, 3):
#         fig, axs = plt.subplots(1, 3, figsize=(20, 6))  # Adjust figsize as needed
#         fig.suptitle(f'{title_prefix} Joints {start_joint_data1} to {min(start_joint_data1 + 2, num_joints_data1 - 1)}')
#
#         for i, joint_data1 in enumerate(range(start_joint_data1, min(start_joint_data1 + 3, num_joints_data1))):
#             ax = axs[i]
#             joint_data2 = joint_data1 - offset  # Adjusting the joint index for data2
#
#             for phase in range(num_phases):
#                 # Plot for data1
#                 tau_data1 = data1["controls"][phase]["tau"][joint_data1]
#                 theta_data1 = degrees(data1["states_no_intermediate"][phase]["q"][joint_data1])
#                 ax.plot(theta_data1, tau_data1, label=f'With Phase {phase}')
#
#                 # Plot for data2 if index is within range
#                 if 0 <= joint_data2 < num_joints_data2:
#                     tau_data2 = data2["controls"][phase]["tau"][joint_data2]
#                     theta_data2 = degrees(data2["states_no_intermediate"][phase]["q"][joint_data2])
#                     ax.plot(theta_data2, tau_data2, label=f'Without Phase {phase}', linestyle='--')
#
#             ax.set_title(Name[joint_data1])
#             ax.set_xlabel('Theta (degrees)')
#             ax.set_ylabel('Tau (Nm)')
#             ax.legend()
#             ax.grid(True)
#
#         plt.tight_layout()
#         plt.show()
#
# # Assuming data_1 and data_2 are already loaded as per your snippet
# phases = ["Preparation", "Key Descend", "Key Bed", "Key Release", "Return to Neutral"]
# num_phases = len(phases)
# num_joints_data1 = len(data_1["states_no_intermediate"][0]["q"])  # Number of joints in data1
# num_joints_data2 = len(data_2["states_no_intermediate"][0]["q"])  # Number of joints in data2
#
# # Displaying plots for data_1 and data_2
# display_all_phases_per_group_of_joints(data_1, data_2, num_phases, num_joints_data1, num_joints_data2, 'Comparative Analysis of Data_1 and Data_2')

# Selected indices for data1 and their corresponding indices in data2
selected_indices_data1 = [9,8,6,5]
selected_indices_data2 = [6,5,3,2]

# Function to display plots for selected phases for each group of joints
def display_selected_phases_per_group_of_joints(data1, data2, num_phases, selected_indices_data1, selected_indices_data2, title_prefix):
    for idx1, idx2 in zip(selected_indices_data1, selected_indices_data2):
        fig, ax = plt.subplots(figsize=(6, 6))  # Adjust figsize as needed

        for phase in range(num_phases):
            # Plot for data1
            tau_data1 = data1["controls"][phase]["tau"][idx1]
            theta_data1 = degrees(data1["states_no_intermediate"][phase]["q"][idx1])
            ax.plot(theta_data1, tau_data1, label=f'With Phase {phase}')

            # Plot for data2 if index is within range
            if 0 <= idx2 < len(data2["states_no_intermediate"][0]["q"]):
                tau_data2 = data2["controls"][phase]["tau"][idx2]
                theta_data2 = degrees(data2["states_no_intermediate"][phase]["q"][idx2])
                ax.plot(theta_data2, tau_data2, label=f'Without Phase {phase}', linestyle='--')

        joint_name = Name[idx1] if idx1 < len(Name) else "Unknown Joint"
        ax.set_title(joint_name)
        ax.set_xlabel('Theta (degrees)')
        ax.set_ylabel('Tau (Nm)')
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.show()

# Phases and other assumptions
phases = ["Preparation", "Key Descend", "Key Bed", "Key Release", "Return to Neutral"]
num_phases = len(phases)

# Displaying plots for the selected indices in data_1 and data_2
display_selected_phases_per_group_of_joints(data_1, data_2, num_phases, selected_indices_data1, selected_indices_data2, 'Comparative Analysis for Selected Joints')

