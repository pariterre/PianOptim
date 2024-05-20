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

dirName = "/home/alpha/Desktop/Class/"
typeTouch = "Pressed" #"Struck" #

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
label_1 = typeTouch+ " Touch DT Strategy"
label_2 = typeTouch+ "Touch ST Strategy"

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

def calculate_dimensionless_jerk(q, dt):
    # Compute velocity using central difference (first derivative of position)
    velocity = np.zeros(len(q))
    velocity[1:-1] = (q[2:] - q[:-2]) / (2 * dt)

    # Compute acceleration using central difference (first derivative of velocity)
    acceleration = np.zeros(len(velocity))
    acceleration[1:-1] = (velocity[2:] - velocity[:-2]) / (2 * dt)

    # Compute jerk using central difference (first derivative of acceleration)
    jerk = np.zeros(len(acceleration))
    jerk[1:-1] = (acceleration[2:] - acceleration[:-2]) / (2 * dt)

    # Calculate sum of squared jerks times dt^5
    jerk_sum_squares_dt5 = np.sum(jerk[1:-1]**2) * dt**5  # Exclude first and last point

    # Calculate total duration T and amplitude of the movement
    T = (len(q) - 1) * dt  # Adjusted to account for central difference
    amplitude = np.max(q) - np.min(q)

    # Calculate dimensionless jerk using Hogan's formula
    dimensionless_jerk = jerk_sum_squares_dt5 / (2 * amplitude**2 * T**2)**(2/3)
    return dimensionless_jerk

# Define the joints of interest (by their indices)
joints_of_interest = [5, 6, 8, 9]  # Indices for shoulder, elbow, wrist, and MCP

# Define the total number of phases
num_phases = 5  # Update this if the number of phases is different
# Define the offset for joint indices between data_1 and data_2
data2_joint_offset = 3  # Set this to the correct value based on your data
dt = 0.01
# Rest of your code
for joint in joints_of_interest:
    print(f"Dimensionless Jerk for {Name[joint]}:")
    for phase in range(num_phases):
        q_data1 = array_q_s_1[phase][joint]
        q_data2 = array_q_s_2[phase][joint - data2_joint_offset] if joint >= data2_joint_offset else None

        dj_data1 = calculate_dimensionless_jerk(q_data1, dt)
        print(f"  Data 1 - Phase {phase}: {dj_data1}")

        if q_data2 is not None:
            dj_data2 = calculate_dimensionless_jerk(q_data2, dt)
            print(f"  Data 2 - Phase {phase}: {dj_data2}")
    print()

phases = ["Preparation", "Key Descend", "Key Bed", "Key Release", "Return to Neutral"]

# Calculate Dimensionless Jerk for each joint of interest in each phase
dimensionless_jerk_results = {}
for joint in joints_of_interest:
    jerk_values_data1 = []
    jerk_values_data2 = []
    for phase in range(num_phases):
        q_data1 = array_q_s_1[phase][joint]
        dj_data1 = calculate_dimensionless_jerk(q_data1, dt)
        jerk_values_data1.append(dj_data1)

        if joint >= data2_joint_offset:
            q_data2 = array_q_s_2[phase][joint - data2_joint_offset]
            dj_data2 = calculate_dimensionless_jerk(q_data2, dt)
            jerk_values_data2.append(dj_data2)
        else:
            jerk_values_data2.append(0)  # Append zero if joint is not in data_2

    dimensionless_jerk_results[Name[joint]] = (jerk_values_data1, jerk_values_data2)

# Plotting Dimensionless Jerk for each joint
for joint_name, (jerks_data1, jerks_data2) in dimensionless_jerk_results.items():
    plt.figure(figsize=(10, 6))
    x = np.arange(num_phases)
    width = 0.35

    plt.bar(x - width/2, jerks_data1, width, label='DT (with Thorax & Pelvic)')
    plt.bar(x + width/2, jerks_data2, width, label='ST (without Thorax & Pelvic)')

    plt.xlabel('Phase')
    plt.ylabel('Dimensionless Jerk')
    plt.title(f'Dimensionless Jerk for {joint_name} across Phases')
    plt.xticks(x, phases)
    plt.legend()
    plt.grid(axis='y')
plt.show()

# Convert Dimensionless Jerk results into a DataFrame
df_dimensionless_jerk_data1 = pd.DataFrame(index=phases)
df_dimensionless_jerk_data2 = pd.DataFrame(index=phases)

for joint_name, (jerks_data1, jerks_data2) in dimensionless_jerk_results.items():
    df_dimensionless_jerk_data1[joint_name] = jerks_data1
    df_dimensionless_jerk_data2[joint_name] = jerks_data2

# Transpose DataFrames for a better layout (joints as columns, phases as rows)
df_dimensionless_jerk_data1 = df_dimensionless_jerk_data1.T
df_dimensionless_jerk_data2 = df_dimensionless_jerk_data2.T

with pd.ExcelWriter(f"{dirName}dimensionless_jerk_results_{typeTouch}.xlsx") as writer:
    df_dimensionless_jerk_data1.to_excel(writer, sheet_name='With Thorax & Pelvic')
    df_dimensionless_jerk_data2.to_excel(writer, sheet_name='Without Thorax & Pelvic')
