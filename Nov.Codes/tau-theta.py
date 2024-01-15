import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
def degrees(radians):
    return np.degrees(radians)


dirName = "/home/alpha/Desktop/5Dec/"
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
    "Pelvic Tilt, Anterior (-) and Posterior (+) Rotation",
    "Thorax, Left (+) and Right (-) Rotation",
    "Thorax, Flexion (-) and Extension (+)",
    "Right Shoulder, Abduction (-) and Adduction (+)",
    "Right Shoulder, Internal (+) and External (-) Rotation",
    "Right Shoulder, Flexion (+) and Extension (-)",
    "Elbow, Flexion (+) and Extension (-)",
    "Elbow, Pronation (+) and Supination (-)",
    "Wrist, Flexion (-) and Extension (+)",
    "MCP, Flexion (+) and Extension (-)",
]


# Function to display plots for all phases for each group of joints
def display_all_phases_per_group_of_joints(data, num_phases, num_joints, title_prefix):
    for start_joint in range(0, num_joints, 3):
        fig, axs = plt.subplots(1, 3, figsize=(20, 6))  # Adjust figsize as needed
        fig.suptitle(f'{title_prefix} Joints {start_joint} to {min(start_joint + 2, num_joints - 1)}')

        for i, joint in enumerate(range(start_joint, min(start_joint + 3, num_joints))):
            ax = axs[i]
            for phase in range(num_phases):
                tau = data["controls"][phase]["tau"][joint]
                theta = degrees(data["states_no_intermediate"][phase]["q"][joint])

                ax.plot(theta, tau, label=f'Phase {phase}')

            ax.set_title(f'Joint {joint}')
            ax.set_xlabel('Theta (degrees)')
            ax.set_ylabel('Tau (Nm)')
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.show()


# Assuming data_1 and data_2 are already loaded as per your snippet
phases = ["Preparation", "Key Descend", "Key Bed", "Key Release", "Return to Neutral"]
num_phases = len(phases)
num_joints = len(data_1["states_no_intermediate"][0]["q"])  # Assuming same number of joints in each phase

# Displaying plots for data_1 and data_2
display_all_phases_per_group_of_joints(data_1, num_phases, num_joints, 'Data_1')

num_phases = len(phases)
num_joints = len(data_2["states_no_intermediate"][0]["q"])  # Assuming same number of joints in each phase

# Displaying plots for data_1 and data_2
display_all_phases_per_group_of_joints(data_2, num_phases, num_joints, 'Data_2')