import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator
from scipy.interpolate import interp1d


def degrees(radians):
    return np.degrees(radians)

dirName = "/home/alpha/Desktop/5Dec/"
typeTouch = "Struck" #"Pressed" #

# Load data_1
with open(dirName + typeTouch + "_with_Thorax.pckl",
          "rb") as file:
    data_1 = pickle.load(file)

with open(dirName + typeTouch + "_with_Thorax_100.pckl",
          "rb") as file:
    data_2 = pickle.load(file)

# Process specific points for data_1 and data_2
specific_points_s_1 = [sum(data_1["phase_time"][: i + 1]) for i in range(len(data_1["phase_time"]))]
specific_points_s_2 = [sum(data_2["phase_time"][: i + 1]) for i in range(len(data_2["phase_time"]))]

# Labels for data_1 and data_2
label_1 = "with"
label_2 = "with_100"

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

I1=np.zeros(5)
I2=np.zeros(5)
percentage_diff=np.zeros(5)

for i in range(0,5):
	I1[i] = (np.trapz(array_tau_s_1[i][-1, :]**2, x=time_arrays_1[i]) +
	      np.trapz(array_tau_s_1[i][-2, :]**2, x=time_arrays_1[i]))
	I2[i] = (np.trapz(array_tau_s_2[i][-1, :]**2, x=time_arrays_2[i]) +
	      np.trapz(array_tau_s_2[i][-2, :]**2, x=time_arrays_2[i]))

	percentage_diff[i]=((I1[i]-I2[i])/I2[i]*100)

# Transposing data for plotting
phases = np.arange(len(I1))  # Assuming each row is a different phase
# Creating subplots
fig, axe = plt.subplots(1, 1, figsize=(10, 8))

# Bar graph for values 1 and 2 per each phase
axe.bar(phases - 0.2, I1, 0.4, label='With_Thorax')
axe.bar(phases + 0.2, I2, 0.4, label='Without_Thorax')
axe.set_title('Analysis of Tau² Reduction Across Different Phases_'+ typeTouch)
axe.set_xlabel('Phases')
axe.set_ylabel('Values')
axe.legend()

# Adjust layout and show plot
plt.tight_layout()
plt.show()
