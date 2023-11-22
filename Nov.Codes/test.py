import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load data_1
with open("/home/alpha/Desktop/22Nov._Updated_BioMod/Pressed_with_Thorax.pckl", "rb") as file:
    data_1 = pickle.load(file)

specific_points_s_1 = [sum(data_1["phase_time"][: i + 1]) for i in range(len(data_1["phase_time"]))]

# Load data_2
with open("/home/alpha/Desktop/22Nov._Updated_BioMod/Pressed_without_Thorax.pckl", "rb") as file:
    data_2 = pickle.load(file)

specific_points_s_2 = [sum(data_2["phase_time"][: i + 1]) for i in range(len(data_2["phase_time"]))]

# Extract relevant data_1
array_q_s_1 = [data_1["states_no_intermediate"][i]["q"] for i in range(len(data_1["states_no_intermediate"]))]
array_qdot_s_1 = [data_1["states_no_intermediate"][i]["qdot"] for i in range(len(data_1["states_no_intermediate"]))]

time_s_1 = [
    np.linspace(specific_points_s_1[i], specific_points_s_1[i + 1], len(array_q_s_1[i][0]))
    for i in range(len(specific_points_s_1) - 1)
]

time_1= np.concatenate(time_s_1)

# Extract relevant data_2
array_q_s_2 = [data_2["states_no_intermediate"][i]["q"] for i in range(len(data_2["states_no_intermediate"]))]
array_qdot_s_2 = [data_2["states_no_intermediate"][i]["qdot"] for i in range(len(data_2["states_no_intermediate"]))]

time_s_2 = [
    np.linspace(specific_points_s_2[i], specific_points_s_2[i + 1], len(array_q_s_2[i][0]))
    for i in range(len(specific_points_s_2) - 1)
]

time_2= np.concatenate(time_s_2)

# Concatenate arrays_1
concatenated_array_q_s_1 = np.degrees(np.concatenate(array_q_s_1, axis=1))
concatenated_array_qdot_s_1 = np.degrees(np.concatenate(array_qdot_s_1, axis=1))

# Calculate acceleration (second derivative of position with respect to time)
qddot_s_1 = np.diff(concatenated_array_qdot_s_1[-2,:])

# Calculate jerk values (derivative of acceleration with respect to time)
jerk_s_1 = np.diff(qddot_s_1)

# Print or use jerk values as needed
print("Jerk values:", jerk_s_1)

# Concatenate arrays_2
concatenated_array_q_s_2 = np.degrees(np.concatenate(array_q_s_2, axis=1))
concatenated_array_qdot_s_2 = np.degrees(np.concatenate(array_qdot_s_2, axis=1))

# Calculate acceleration (second derivative of position with respect to time)
qddot_s_2 = np.diff(concatenated_array_qdot_s_2[-2,:])

# Calculate jerk values (derivative of acceleration with respect to time)
jerk_s_2 = np.diff(qddot_s_2)

# Print or use jerk values as needed
print("Jerk values:", jerk_s_2)

# Plot jerk graph
plt.plot(jerk_s_1, label='with')
plt.plot(jerk_s_2, label='without')

plt.xlabel('Nodes')
plt.ylabel('Jerk')
plt.title('Jerk on Wrist joint')
plt.legend()
plt.show()