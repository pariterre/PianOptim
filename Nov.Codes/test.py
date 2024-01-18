import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator
from scipy.interpolate import interp1d


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
label_1 = "With"
label_2 = "Without"

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


# # Creating finger arrays
# array1 = concatenated_array_tau_s_2[6,:]  # shorter array with 75 elements
# hand_pressed = concatenated_array_tau_s_1[6,:]  # longer array with 81 elements
#
# # Creating an interpolation function for the shorter array
# interp_function = interp1d(np.linspace(0, 1, len(array1)), array1)
#
# # Using the interpolation function to create a new array with the length of the longer array
# interpolated_array_hand_struck = interp_function(np.linspace(0, 1, len(hand_pressed)))
#
# # Creating wrist arrays
# array1 = concatenated_array_tau_s_2[5,:]  # shorter array with 75 elements
# wrist_pressed = concatenated_array_tau_s_1[5,:]  # longer array with 81 elements
#
# # Creating an interpolation function for the shorter array
# interp_function = interp1d(np.linspace(0, 1, len(array1)), array1)
#
# # Using the interpolation function to create a new array with the length of the longer array
# interpolated_array_wrist_struck = interp_function(np.linspace(0, 1, len(wrist_pressed)))


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
#
# Handle NaN values in tau arrays
fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(15, 30))  # Adjust figsize as needed

i_to_j_mapping = {-4: -1, -3: -2, -2: -4, -1: -5}

for i in range(-4, 0):  # Iterate over each DOF
    # Determine the corresponding j value for this i
    j = i_to_j_mapping[i]

    axs[i, 0].plot(concatenated_array_time_s_1, concatenated_array_q_s_1[j, :], color="red", label=label_1)
    axs[i, 0].plot(concatenated_array_time_s_2, concatenated_array_q_s_2[j, :], color="blue", linestyle="--", label=label_2)
    axs[i, 0].set_title(Name[j])
    axs[i, 0].set_ylabel("θ (deg)")
    axs[i, 0].fill_betweenx(axs[i, 0].get_ylim(), 0.3, 0.4, color='gray', alpha=0.2)

    # Plot qdot for this DOF
    axs[i, 1].plot(concatenated_array_time_s_1, concatenated_array_qdot_s_1[j, :], color="red", label=label_1)
    axs[i, 1].plot(concatenated_array_time_s_2, concatenated_array_qdot_s_2[j, :], color="blue", linestyle="--", label=label_2)
    axs[i, 1].set_title(Name[j])
    axs[i, 1].set_ylabel(r"$\dot{\theta}$ (deg/sec)")
    handles, labels = axs[i, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize='small', bbox_to_anchor=(0.210,0.21))
    axs[i, 1].fill_betweenx(axs[i, 1].get_ylim(), 0.3, 0.4, color='gray', alpha=0.2)

    # Plot tau for this DOF
    axs[i, 2].step(concatenated_array_time_s_1, concatenated_array_tau_s_1[j, :], color="red", label=label_1)
    axs[i, 2].step(concatenated_array_time_s_2, concatenated_array_tau_s_2[j, :], color="blue", linestyle="--", label=label_2)
    axs[i, 2].set_title(Name[j])
    axs[i, 2].set_ylabel(r"$\tau$ (N/m)")
    axs[i, 2].fill_betweenx(axs[i, 2].get_ylim(), 0.3, 0.4, color='gray', alpha=0.2)

# Set common properties for all subplots
for ax in axs.flat:
    ax.grid(True)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(which="minor", linestyle=":", linewidth="0.2", color="gray")

    # Add vertical lines for specific points in data_1
    for point in specific_points_s_1:
        ax.axvline(x=point, color="k", linestyle=":")

    for point in specific_points_s_2:
        ax.axvline(x=point, color="k", linestyle=":")

plt.tight_layout()
plt.show()