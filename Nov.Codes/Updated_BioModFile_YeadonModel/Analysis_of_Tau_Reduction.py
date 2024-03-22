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

dirName = ("/home/alpha/pianoptim/PianOptim/Nov.Codes/Updated_BioModFile_YeadonModel/Results/Updated_Profile_W200/")


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
label_1 = ("Pressed_" if pressed else "Struck_")+ "With"
label_2 = ("Pressed_" if pressed else "Struck_")+ "Without"

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

percentage_diff_squared=np.zeros(5)

I1_squared=np.zeros(5)
I2_squared=np.zeros(5)

I1_absolute_Values=np.zeros(5)
I2_absolute_Values=np.zeros(5)

percentage_diff_absolute_Value=np.zeros(5)

for i in range(0,5):

    I1_squared[i] = (np.trapz(array_tau_s_1[i][-1, :] ** 2, x=time_arrays_1[i]) +
                     np.trapz(array_tau_s_1[i][-2, :] ** 2, x=time_arrays_1[i]))

    I2_squared[i] = (np.trapz(array_tau_s_2[i][-1, :] ** 2, x=time_arrays_2[i]) +
                     np.trapz(array_tau_s_2[i][-2, :] ** 2, x=time_arrays_2[i]))

    percentage_diff_squared[i]=((I1_squared[i]-I2_squared[i])/I2_squared[i]*100)


# Second part: Using absolute values
for i in range(0,5):

    I1_absolute_Values[i] = (np.trapz(np.abs(array_tau_s_1[i][-1, :]), x=time_arrays_1[i]) +
                          np.trapz(np.abs(array_tau_s_1[i][-2, :]), x=time_arrays_1[i]))

    I2_absolute_Values[i] = (np.trapz(np.abs(array_tau_s_2[i][-1, :]), x=time_arrays_2[i]) +
                          np.trapz(np.abs(array_tau_s_2[i][-2, :]), x=time_arrays_2[i]))

    percentage_diff_absolute_Value[i]=((I1_absolute_Values[i]-I2_absolute_Values[i])/I2_absolute_Values[i]*100)



phase = ["Preparation", "Key Descend", "Key Bed", "Key Release", "Return to Neutral"]

# Transposing data for plotting
phases = np.arange(len(I1_absolute_Values))  # Assuming each row is a different phase
# Creating subplots
fig, axe = plt.subplots(1, 1, figsize=(10, 8))

# Bar graph for values 1 and 2 per each phase
axe.bar(phases - 0.2, I1_absolute_Values, 0.4, label='DT Strategy')
axe.bar(phases + 0.2, I2_absolute_Values, 0.4, label='ST Strategy')
axe.set_title('Analysis of Tau Reduction Across Different Phases_'+ ("Pressed" if pressed else "Struck"))
axe.set_xlabel('Phases')
plt.xticks(phases, phase)
axe.set_ylabel('Area Under Curve (AUC) of tau for the distal joints (N.m)')
axe.legend()
axe.grid(True)

# Adjust layout and show plot
plt.tight_layout()
plt.show()

