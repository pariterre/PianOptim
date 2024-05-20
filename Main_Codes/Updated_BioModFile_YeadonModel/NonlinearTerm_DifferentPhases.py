import biorbd
import numpy as np
import pickle
import matplotlib.pyplot as plt

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
num_nodes_DT = (30, 7, 9, 10, 10)
num_nodes_ST = (30, 6, 9, 10, 10)


phase_descriptions = [
    "Phase 0: Preparation",
    "Phase 1: Key Descend",
    "Phase 2: Key Bed",
    "Phase 3: Key Release (Upward)",
    "Phase 4: Return to Neutral (Downward)"
]


# Extract joint positions (q) and velocities (qdot) from the loaded data
array_q_s_DT = [dict_DT["states_no_intermediate"][i]["q"] for i in range(len(dict_DT["states_no_intermediate"]))]
array_qdot_s_DT = [dict_DT["states_no_intermediate"][i]["qdot"] for i in range(len(dict_DT["states_no_intermediate"]))]
array_tau_s_DT = [dict_DT["controls"][2]["tau"] for i in range(len(dict_DT["states_no_intermediate"]))]

array_q_s_ST = [dict_ST["states_no_intermediate"][i]["q"] for i in range(len(dict_ST["states_no_intermediate"]))]
array_qdot_s_ST = [dict_ST["states_no_intermediate"][i]["qdot"] for i in range(len(dict_ST["states_no_intermediate"]))]
array_tau_s_ST = [dict_ST["controls"][2]["tau"] for i in range(len(dict_ST["states_no_intermediate"]))]

# Define the joint maps for DT and ST scenarios
joint_map_DT = {
    "Pelvic Tilt": [0],
    "Thoracic": [1, 2],
    "Upper Thoracic (Rib Cage)": [3, 4],
    "Shoulder": [5, 6, 7],
    "Elbow": [8, 9],
    "Wrist": [10],
    "MCP": [11]
}

joint_map_ST = {
    "Shoulder": [0, 1, 2],
    "Elbow": [3, 4],
    "Wrist": [5],
    "MCP": [6]
}

# Initialize dictionaries to store the Nonlinear_Effects per joint for each phase
Nonlinear_Effects_per_joint_DT = {joint: [[] for _ in range(len(num_nodes_DT))] for joint in joint_map_DT}
Nonlinear_Effects_per_joint_ST = {joint: [[] for _ in range(len(num_nodes_ST))] for joint in joint_map_ST}

for k in range(len(num_nodes_DT)):
    for i in range(num_nodes_DT[k]):
        Q = np.array([array_q_s_DT[k][j][i] for j in range(num_DOFs_DT)])
        Qdot = np.array([array_qdot_s_DT[k][j][i] for j in range(num_DOFs_DT)])

        # Compute the nonlinear effects for all DOFs in the DT scenario
        nonlinear_effects_DT = model_DT.NonLinearEffect(Q, Qdot).to_array()

        # Compute the nonlinear effects per joint for the DT scenario
        for joint, dof_indices in joint_map_DT.items():
            nonlinear_effects_joint = sum(nonlinear_effects_DT[dof_index] for dof_index in dof_indices)
            Nonlinear_Effects_per_joint_DT[joint][k].append(nonlinear_effects_joint)

for k in range(len(num_nodes_ST)):
    for i in range(num_nodes_ST[k]):
        Q = np.array([array_q_s_ST[k][j][i] for j in range(num_DOFs_ST)])
        Qdot = np.array([array_qdot_s_ST[k][j][i] for j in range(num_DOFs_ST)])

        # Compute the nonlinear effects for all DOFs in the ST scenario
        nonlinear_effects_ST = model_ST.NonLinearEffect(Q, Qdot).to_array()

        # Compute the nonlinear effects per joint for the ST scenario
        for joint, dof_indices in joint_map_ST.items():
            nonlinear_effects_joint = sum(nonlinear_effects_ST[dof_index] for dof_index in dof_indices)
            Nonlinear_Effects_per_joint_ST[joint][k].append(nonlinear_effects_joint)

# # Plot the Nonlinear_Effects per joint for each phase
num_phases = len(num_nodes_DT)
# fig, axs = plt.subplots(num_phases, 2, figsize=(12, 4 * num_phases), sharex=True)
# fig.suptitle('Nonlinear Effects per Joint for Each Phase'+("_Pressed" if pressed else "_Struck"), fontsize=16)
#
# colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
# linestyles = ['-', '--', '-.', ':', '-', '--', '-.']
#
# for k in range(num_phases):
#     for joint, values in Nonlinear_Effects_per_joint_DT.items():
#         axs[k, 0].plot(values[k], label=joint, color=colors[list(Nonlinear_Effects_per_joint_DT.keys()).index(joint)], linestyle=linestyles[list(Nonlinear_Effects_per_joint_DT.keys()).index(joint)])
#     axs[k, 0].set_title(f'Phase {k+1} (DT)')
#     axs[k, 0].set_xlabel('Shooting Nodes')
#     axs[k, 0].set_ylabel('Nonlinear Effects')
#     axs[k, 0].legend(loc='upper right')
#     axs[k, 0].grid(True)
#
#     for joint, values in Nonlinear_Effects_per_joint_ST.items():
#         axs[k, 1].plot(values[k], label=joint, color=colors[list(Nonlinear_Effects_per_joint_ST.keys()).index(joint)], linestyle=linestyles[list(Nonlinear_Effects_per_joint_ST.keys()).index(joint)])
#     axs[k, 1].set_title(f'Phase {k+1} (ST)')
#     axs[k, 1].set_xlabel('Shooting Nodes')
#     axs[k, 1].set_ylabel('Nonlinear Effects')
#     axs[0, 1].legend(loc='upper right')
#     axs[k, 1].grid(True)
#
# plt.tight_layout()
# plt.subplots_adjust(top=0.95)
# plt.show()

fig, axs = plt.subplots(num_phases, 2, figsize=(12, 4 * num_phases), sharex=True)
fig.suptitle('Relative Contributions of Joints to Nonlinear Effects' + ("_Pressed" if pressed else "_Struck"), fontsize=14)

colors_DT = ['blue', 'green', 'red', 'cyan', 'magenta', 'purple', 'brown']
colors_ST = ['cyan', 'magenta', 'purple', 'brown']

# Create dummy legend entries
legend_entries_DT = [plt.Line2D([0], [0], color=color, label=joint, linestyle='-') for joint, color in zip(joint_map_DT, colors_DT)]
legend_entries_ST = [plt.Line2D([0], [0], color=color, label=joint, linestyle='-') for joint, color in zip(joint_map_ST, colors_ST)]

for k in range(num_phases):
    for i in range(num_nodes_DT[k]):
        total_effect_DT = sum(Nonlinear_Effects_per_joint_DT[joint][k][i] for joint in joint_map_DT)
        bottom = 0
        for joint, color in zip(joint_map_DT, colors_DT):
            effect = Nonlinear_Effects_per_joint_DT[joint][k][i]
            axs[k, 0].bar(i, effect / total_effect_DT, bottom=bottom, color=color)
            bottom += effect / total_effect_DT
    axs[k, 0].set_title(phase_descriptions[k] + ' (DT)')
    axs[k, 0].set_xlabel('Shooting Nodes', fontsize=12)
    axs[2, 0].set_ylabel('Relative Contribution [RC]', fontsize=12)

    for i in range(num_nodes_ST[k]):
        total_effect_ST = sum(Nonlinear_Effects_per_joint_ST[joint][k][i] for joint in joint_map_ST)
        bottom = 0
        for joint, color in zip(joint_map_ST, colors_ST):
            effect = Nonlinear_Effects_per_joint_ST[joint][k][i]
            axs[k, 1].bar(i, effect / total_effect_ST, bottom=bottom, color=color)
            bottom += effect / total_effect_ST
    axs[k, 1].set_title(phase_descriptions[k] + ' (ST)')
    axs[k, 1].set_xlabel('Shooting Nodes', fontsize=12)
    axs[2, 1].set_ylabel('Relative Contribution [RC]', fontsize=12)

# Add legends outside the subplots
fig.legend(legend_entries_DT, [entry.get_label() for entry in legend_entries_DT], loc='center', bbox_to_anchor=(0.35, 0.45), ncol=1, title='DT')
fig.legend(legend_entries_ST, [entry.get_label() for entry in legend_entries_ST], loc='center', bbox_to_anchor=(0.95, 0.45), ncol=1, title='ST')

plt.subplots_adjust(top=0.928)
plt.subplots_adjust(bottom=0.061)
plt.tight_layout()

plt.show()

