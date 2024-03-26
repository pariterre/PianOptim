import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator
from scipy.interpolate import interp1d

def degrees(radians):
    return np.degrees(radians)

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


dirName = "/home/alpha/pianoptim/PianOptim/Nov.Codes/Updated_BioModFile_YeadonModel/Results/Felipe_25March/26March-qdot2/"


saveName = dirName + ("Pressed" if pressed else "Struck") + "_with_Thorax.pckl"
with open(saveName, "rb") as file:
    data_1 = pickle.load(file)


saveName = dirName + ("Pressed" if pressed else "Struck") + "_without_Thorax.pckl"
with open(saveName, "rb") as file:
    data_2 = pickle.load(file)


# Process specific points for data_1 and data_2
specific_points_s_1 = [sum(data_1["phase_time"][: i + 1]) for i in range(len(data_1["phase_time"]))]
specific_points_s_2 = [sum(data_2["phase_time"][: i + 1]) for i in range(len(data_2["phase_time"]))]

# Labels for data_1 and data_2
label_1 = ("Pressed_" if pressed else "Struck_")+ "with"
label_2 = ("Pressed_" if pressed else "Struck_")+ "without"

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
    "Pelvic Tilt, Anterior (+) and Posterior (-) Rotation",
    "Thoracic, Flexion (+) and Extension (-)",
    "Thoracic, Left (+) and Right (-) Rotation",
    "Upper Thoracic (Rib Cage), Flexion (+) and Extension (-)",
    "Upper Thoracic (Rib Cage), Left (+) and Right (-) Rotation",
    "Right Shoulder, Flexion (-) and Extension (+)",
    "Right Shoulder, Abduction (+) and Adduction (-)",
    "Right Shoulder, Internal (+) and External (-) Rotation",
    "Elbow, Flexion (-) and Extension (+)",
    "Elbow, Left (+) and Right (-) Rotation",
    "Wrist, Flexion (-) and Extension (+)",
    "MCP, Flexion (-) and Extension (+)",
]

for i in range(-12, 0):
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10, 8))

    # Plot for q
    axs[0].plot(concatenated_array_time_s_1, concatenated_array_q_s_1[i, :], color="red", label=label_1)
    if i >= -7:  # Check if data2 has this index
        axs[0].plot(concatenated_array_time_s_2, concatenated_array_q_s_2[i, :], color="blue", linestyle="--", label=label_2)
    axs[0].fill_betweenx(axs[0].get_ylim(), 0.3, 0.4, color='gray', alpha=0.2)
    axs[0].set_title(Name[i])
    axs[0].set_ylabel("θ (deg)")
    axs[0].legend()

    # Plot for qdot
    axs[1].plot(concatenated_array_time_s_1, concatenated_array_qdot_s_1[i, :], color="red", label=label_1)
    if i >= -7:  # Check if data2 has this index
        axs[1].plot(concatenated_array_time_s_2, concatenated_array_qdot_s_2[i, :], color="blue", linestyle="--", label=label_2)
    axs[1].fill_betweenx(axs[1].get_ylim(), 0.3, 0.4, color='gray', alpha=0.2)
    axs[1].set_ylabel(r"$\dot{\theta}$ (deg/sec)")
    axs[1].legend()

    # Plot for tau
    axs[2].step(concatenated_array_time_s_1, concatenated_array_tau_s_1[i, :], color="red", label=label_1)
    if i >= -7:  # Check if data2 has this index
        axs[2].step(concatenated_array_time_s_2, concatenated_array_tau_s_2[i, :], color="blue", linestyle="--", label=label_2)
    axs[2].fill_betweenx(axs[2].get_ylim(), 0.3, 0.4, color='gray', alpha=0.2)
    axs[2].set_ylabel(r"$\tau$ (N/m)")
    axs[2].set_xlabel("Time (sec)")
    axs[2].legend()

    # Set common properties for all subplots
    for ax in axs:
        ax.grid(True)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(which="minor", linestyle=":", linewidth="0.2", color="gray")

        # Add vertical lines for specific points in data_1 and data_2
        for point in specific_points_s_1:
            ax.axvline(x=point, color="k", linestyle=":")
        if i >= -7:
            for point in specific_points_s_2:
                ax.axvline(x=point, color="k", linestyle=":")

    plt.tight_layout()
plt.show()

# First part: Using squared values
I1_squared = (np.trapz(concatenated_array_tau_s_1[-1, :]**2, x=concatenated_array_time_s_1) +
             np.trapz(concatenated_array_tau_s_1[-2, :]**2, x=concatenated_array_time_s_1))
I2_squared = (np.trapz(concatenated_array_tau_s_2[-1, :]**2, x=concatenated_array_time_s_2) +
             np.trapz(concatenated_array_tau_s_2[-2, :]**2, x=concatenated_array_time_s_2))

print(I1_squared, I2_squared, (I1_squared-I2_squared)/I1_squared*100)

# Second part: Using absolute values
abs_concatenated_array_tau_s_1 = np.abs(concatenated_array_tau_s_1)
abs_concatenated_array_tau_s_2 = np.abs(concatenated_array_tau_s_2)

I1_absolute_Values = (np.trapz(abs_concatenated_array_tau_s_1[-1, :], x=concatenated_array_time_s_1) +
                      np.trapz(abs_concatenated_array_tau_s_1[-2, :], x=concatenated_array_time_s_1))
I2_absolute_Values = (np.trapz(abs_concatenated_array_tau_s_2[-1, :], x=concatenated_array_time_s_2) +
                      np.trapz(abs_concatenated_array_tau_s_2[-2, :], x=concatenated_array_time_s_2))

print(I1_absolute_Values, I2_absolute_Values, (I1_absolute_Values - I2_absolute_Values) / I1_absolute_Values * 100)


