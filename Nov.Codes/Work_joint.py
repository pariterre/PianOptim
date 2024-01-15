import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator
from scipy.interpolate import interp1d

def degrees(radians):
    return np.degrees(radians)

def calculate_work(tau, delta_q):
    return np.sum(tau * delta_q)

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
#
num_joints = len(concatenated_array_q_s_1)
# Initialize the array to store work for each joint in each phase
w_0 = np.zeros((10, 1))
w_1 = np.zeros((10, 1))
w_2 = np.zeros((10, 1))
w_3 = np.zeros((10, 1))
w_4 = np.zeros((10, 1))

# Calculate work for each joint in each phase
for phase in range(5):
    for joint in range(num_joints):
        delta_q = np.diff(array_q_s_1[phase][joint])
        tau = array_tau_s_1[phase][joint][:-1]

        if phase == 0:
            w_0[joint] = calculate_work(tau, delta_q)
        elif phase == 1:
            w_1[joint] = calculate_work(tau, delta_q)
        elif phase == 2:
            w_2[joint] = calculate_work(tau, delta_q)
        elif phase == 3:
            w_3[joint] = calculate_work(tau, delta_q)
        elif phase == 4:
            w_4[joint] = calculate_work(tau, delta_q)

# Initialize arrays to store work for each joint in each phase for data_2
w_0_data2 = np.zeros((7, 1))
w_1_data2 = np.zeros((7, 1))
w_2_data2 = np.zeros((7, 1))
w_3_data2 = np.zeros((7, 1))
w_4_data2 = np.zeros((7, 1))

# Calculate work for each joint in each phase for data_2
for phase in range(5):
    for joint in range(7):  # Adjusted for 7 DOFs in data_2
        delta_q = np.diff(array_q_s_2[phase][joint])
        tau = array_tau_s_2[phase][joint][:-1]

        if phase == 0:
            w_0_data2[joint] = calculate_work(tau, delta_q)
        elif phase == 1:
            w_1_data2[joint] = calculate_work(tau, delta_q)
        elif phase == 2:
            w_2_data2[joint] = calculate_work(tau, delta_q)
        elif phase == 3:
            w_3_data2[joint] = calculate_work(tau, delta_q)
        elif phase == 4:
            w_4_data2[joint] = calculate_work(tau, delta_q)
# Assuming the work arrays for data_1 have been correctly calculated and stored in w_0, w_1, w_2, w_3, w_4

# Create a list of work arrays for data_1 for easier access in the loop
work_arrays_data1 = [w_0, w_1, w_2, w_3, w_4]

# Create a list of work arrays for data_2 for easier access in the loop
work_arrays_data2 = [w_0_data2, w_1_data2, w_2_data2, w_3_data2, w_4_data2]

# Plotting work for each joint in each phase, aligning joint_3 of data_1 with joint_0 of data_2
# for phase in range(5):
#     plt.figure()
#
#     # Plot data_1
#     work_array_data1 = work_arrays_data1[phase].flatten()
#     plt.bar(np.arange(3, 10), work_array_data1[3:], width=0.4, label='with_Thorax')
#
#     # Plot data_2
#     work_array_data2 = work_arrays_data2[phase].flatten()
#     plt.bar(np.arange(3, 10) + 0.4, work_array_data2, width=0.4, label='without_Thorax')
#
#     plt.xlabel('Joint')
#     plt.ylabel('Work (Nm)')
#     plt.title(f'Total Work Done by Each Joint in Phase {phase}')
#     plt.xticks(np.arange(3, 10) + 0.2, ['Joint ' + str(i) for i in range(3, 10)])
#     plt.legend()
# plt.show()


# # Plotting work for each joint across all phases
# for joint in range(len(Name)):
#     plt.figure(figsize=(10, 6))
#     work_per_joint = [work_array[joint] for work_array in work_arrays_data1]
#
#     plt.bar(phases, [w[0] for w in work_per_joint], color='skyblue')
#
#     plt.xlabel('Phase')
#     plt.ylabel('Work (Nm)')
#     plt.title(f'Work Done by {Name[joint]} in Each Phase')
#     plt.xticks(rotation=45)
#     plt.grid(axis='y')
#
# plt.show()


phases = ["Preparation", "Key Descend", "Key Bed", "Key Release", "Return to Neutral"]
num_phases = len(phases)

# Offset for data_2 joint indices
data2_joint_offset = 3

for joint in range(len(Name)):
    plt.figure(figsize=(10, 6))

    # Extracting work for the joint across all phases for data_1
    work_data1 = [work_arrays_data1[phase][joint][0] for phase in range(num_phases)]

    # Extracting work for the corresponding joint in data_2, if it exists
    if joint >= data2_joint_offset:
        joint_data2 = joint - data2_joint_offset
        work_data2 = [work_arrays_data2[phase][joint_data2][0] for phase in range(num_phases)]
    else:
        work_data2 = [0] * num_phases

    # Creating the bar chart
    x = np.arange(num_phases)  # the label locations
    width = 0.35  # the width of the bars

    bars1 = plt.bar(x - width/2, work_data1, width, label='(with Thorax & Pelvic)')
    bars2 = plt.bar(x + width/2, work_data2, width, label='(without Thorax & Pelvic)')

    # Adding annotations to the bars
    for bar in bars1:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

    for bar in bars2:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

    plt.xlabel('Phase')
    plt.ylabel('Work (N.m.deg)')
    plt.title(f'Work Done by {Name[joint]} in Each Phase')
    plt.xticks(x, phases)
    plt.legend()
    plt.grid(axis='y')

plt.show()

# Initialize the array to store overall work for each joint
overall_work_data1 = np.zeros((num_joints, 1))
overall_work_data2 = np.zeros((7, 1))  # Adjusted for 7 DOFs in data_2

# Calculate overall work for each joint
for joint in range(num_joints):
    for phase in range(5):
        delta_q = np.diff(array_q_s_1[phase][joint])
        tau = array_tau_s_1[phase][joint][:-1]
        overall_work_data1[joint] += calculate_work(tau, delta_q)

for joint in range(7):  # Adjusted for 7 DOFs in data_2
    for phase in range(5):
        delta_q = np.diff(array_q_s_2[phase][joint])
        tau = array_tau_s_2[phase][joint][:-1]
        overall_work_data2[joint] += calculate_work(tau, delta_q)

# Print overall work for each joint in data_1 and data_2
print("Overall work for each joint in data_1 (with Thorax & Pelvic):", overall_work_data1.flatten())
print("Overall work for each joint in data_2 (without Thorax & Pelvic):", overall_work_data2.flatten())


