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

dirName = "/home/alpha/pianoptim/PianOptim/Nov.Codes/Updated_BioModFile_YeadonModel/Updated_Biomod_Distance/XXX/"

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
if pressed:
    num_nodes = 9
else:
    num_nodes = 9

# Extract joint positions (q) and velocities (qdot) from the loaded data
array_q_s_DT = [dict_DT["states_no_intermediate"][i]["q"] for i in range(len(dict_DT["states_no_intermediate"]))]
array_qdot_s_DT = [dict_DT["states_no_intermediate"][i]["qdot"] for i in range(len(dict_DT["states_no_intermediate"]))]
array_tau_s_DT = [dict_DT["controls"][2]["tau"] for i in range(len(dict_DT["states_no_intermediate"]))]

array_q_s_ST = [dict_ST["states_no_intermediate"][i]["q"] for i in range(len(dict_ST["states_no_intermediate"]))]
array_qdot_s_ST = [dict_ST["states_no_intermediate"][i]["qdot"] for i in range(len(dict_ST["states_no_intermediate"]))]
array_tau_s_ST = [dict_ST["controls"][2]["tau"] for i in range(len(dict_ST["states_no_intermediate"]))]


# Initialize lists to store the time series data for plotting
C_T_lambda_values_DT_series = []
Nonlinear_Effects_DT_series = []
Mass_Qddot_values_DT_series = []
Tau_values_DT_series = []

C_T_lambda_values_ST_series = []
Nonlinear_Effects_ST_series = []
Mass_Qddot_values_ST_series = []
Tau_values_ST_series = []

for i in range(num_nodes):

    # Extract the relevant values for the DT scenarios
    # Choose a position/velocity/acceleration to compute dynamics from
    Q = np.array([array_q_s_DT[2][j][i] for j in range(num_DOFs_DT)])
    Qdot = np.array([array_qdot_s_DT[2][j][i] for j in range(num_DOFs_DT)])

    # Extract external Forces
    f_ext = np.array([dict_DT["Contact_Force"][i][0][0], dict_DT["Contact_Force"][i][1][0], dict_DT["Contact_Force"][i][2][0]])
    Tau = np.array([dict_DT["controls"][2]["tau"][j][i] for j in range(num_DOFs_DT)])

    # Compute the generalized accelerations using forward dynamics
    Qddot = model_DT.ForwardDynamicsConstraintsDirect(Q, Qdot, Tau).to_array()
    Qddot_m = Qddot.reshape(num_DOFs_DT, 1)

    # Compute the mass matrix
    Mass = model_DT.massMatrix(Q).to_array()

    # Compute the mass matrix multiplied by the generalized accelerations
    Mass_Qddot = np.dot(Mass, Qddot_m)

    # Compute the nonlinear effects
    nonlinear_effects = model_DT.NonLinearEffect(Q, Qdot).to_array().reshape(num_DOFs_DT, 1)

    # Compute the Jacobian for the contact point on the "RightFingers" segment
    parent_name = "RightFingers"
    contact_point = np.array([0.0015, 0.0015, -0.046782999938])
    p = biorbd.NodeSegment(contact_point[0], contact_point[1], contact_point[2])
    update_kin = True
    contact_jacobian = model_DT.markersJacobian(Q, parent_name, p, update_kin)

    # Extract the Jacobian matrix as a NumPy array
    jacobian_array = contact_jacobian.to_array()
    J_contact = np.array(jacobian_array).T

    # Define the contact force vector
    F_contact = np.array([dict_DT["Contact_Force"][i][0][0], dict_DT["Contact_Force"][i][1][0], dict_DT["Contact_Force"][i][2][0]]).reshape(3, 1)

    # Compute C^T lambda for the contact force
    C_T_lambda_contact = np.dot(J_contact, F_contact)

    # Extract the relevant tau values for the wrist and finger
    tau_wrist = Tau[-2]
    tau_finger = Tau[-1]

    # Extract the contributions from different terms for the wrist and finger
    C_T_lambda_wrist = C_T_lambda_contact[-2]
    C_T_lambda_finger = C_T_lambda_contact[-1]

    Nonlinear_Effects_wrist = nonlinear_effects[-2]
    Nonlinear_Effects_finger = nonlinear_effects[-1]

    # Extract the relevant values for the wrist and finger
    Mass_Qddot_wrist = Mass_Qddot[-2]
    Mass_Qddot_finger = Mass_Qddot[-1]

    # Create a stacked bar plot for each shooting node
    labels = ['Wrist', 'Finger']
    C_T_lambda_values_DT = [-C_T_lambda_wrist[0], -C_T_lambda_finger[0]]
    Nonlinear_Effects_DT = [Nonlinear_Effects_wrist[0], Nonlinear_Effects_finger[0]]
    Mass_Qddot_values_DT = [Mass_Qddot_wrist[0], Mass_Qddot_finger[0]]
    Tau_values_DT = [tau_wrist, tau_finger]

    # Extract the relevant values for the ST scenarios
    # Choose a position/velocity/acceleration to compute dynamics from
    Q = np.array([array_q_s_ST[2][j][i] for j in range(num_DOFs_ST)])
    Qdot = np.array([array_qdot_s_ST[2][j][i] for j in range(num_DOFs_ST)])

    # Extract external Forces
    f_ext = np.array([dict_ST["Contact_Force"][i][0][0], dict_ST["Contact_Force"][i][1][0], dict_ST["Contact_Force"][i][2][0]])
    Tau = np.array([dict_ST["controls"][2]["tau"][j][i] for j in range(num_DOFs_ST)])

    # Compute the generalized accelerations using forward dynamics
    Qddot = model_ST.ForwardDynamicsConstraintsDirect(Q, Qdot, Tau).to_array()
    Qddot_m = Qddot.reshape(num_DOFs_ST, 1)

    # Compute the mass matrix
    Mass = model_ST.massMatrix(Q).to_array()

    # Compute the mass matrix multiplied by the generalized accelerations
    Mass_Qddot = np.dot(Mass, Qddot_m)

    # Compute the nonlinear effects
    nonlinear_effects = model_ST.NonLinearEffect(Q, Qdot).to_array().reshape(num_DOFs_ST, 1)

    # Compute the Jacobian for the contact point on the "RightFingers" segment
    parent_name = "RightFingers"
    contact_point = np.array([0.0015, 0.0015, -0.046782999938])
    p = biorbd.NodeSegment(contact_point[0], contact_point[1], contact_point[2])
    update_kin = True
    contact_jacobian = model_ST.markersJacobian(Q, parent_name, p, update_kin)

    # Extract the Jacobian matrix as a NumPy array
    jacobian_array = contact_jacobian.to_array()
    J_contact = np.array(jacobian_array).T

    # Define the contact force vector
    F_contact = np.array([dict_ST["Contact_Force"][i][0][0], dict_ST["Contact_Force"][i][1][0], dict_ST["Contact_Force"][i][2][0]]).reshape(3, 1)

    # Compute C^T lambda for the contact force
    C_T_lambda_contact = np.dot(J_contact, F_contact)

    # Extract the relevant tau values for the wrist and finger
    tau_wrist = Tau[-2]
    tau_finger = Tau[-1]

    # Extract the contributions from different terms for the wrist and finger
    C_T_lambda_wrist = C_T_lambda_contact[-2]
    C_T_lambda_finger = C_T_lambda_contact[-1]

    Nonlinear_Effects_wrist = nonlinear_effects[-2]
    Nonlinear_Effects_finger = nonlinear_effects[-1]

    # Extract the relevant values for the wrist and finger
    Mass_Qddot_wrist = Mass_Qddot[-2]
    Mass_Qddot_finger = Mass_Qddot[-1]

    # Create a stacked bar plot for each shooting node
    labels = ['Wrist', 'Finger']
    C_T_lambda_values_ST = [-C_T_lambda_wrist[0], -C_T_lambda_finger[0]]
    Nonlinear_Effects_ST = [Nonlinear_Effects_wrist[0], Nonlinear_Effects_finger[0]]
    Mass_Qddot_values_ST = [Mass_Qddot_wrist[0], Mass_Qddot_finger[0]]
    Tau_values_ST = [tau_wrist, tau_finger]

# Append the values for this node to the time series lists
    C_T_lambda_values_DT_series.append(C_T_lambda_values_DT)
    Nonlinear_Effects_DT_series.append(Nonlinear_Effects_DT)
    Mass_Qddot_values_DT_series.append(Mass_Qddot_values_DT)
    Tau_values_DT_series.append(Tau_values_DT)

    C_T_lambda_values_ST_series.append(C_T_lambda_values_ST)
    Nonlinear_Effects_ST_series.append(Nonlinear_Effects_ST)
    Mass_Qddot_values_ST_series.append(Mass_Qddot_values_ST)
    Tau_values_ST_series.append(Tau_values_ST)

# Initialize lists to store the Wrist and MCP values separately for DT and ST    (C_T_lambda)
Wrist_C_T_lambda_DT = [node[-2] for node in C_T_lambda_values_DT_series]
MCP_C_T_lambda_DT = [node[-1] for node in C_T_lambda_values_DT_series]

Wrist_C_T_lambda_ST = [node[-2] for node in C_T_lambda_values_ST_series]
MCP_C_T_lambda_ST = [node[-1] for node in C_T_lambda_values_ST_series]

# Initialize lists to store the Wrist and MCP values separately for DT and ST    (Nonlinear_Effects)
Wrist_Nonlinear_Effects_DT = [node[-2] for node in Nonlinear_Effects_DT_series]
MCP_Nonlinear_Effects_DT = [node[-1] for node in Nonlinear_Effects_DT_series]

Wrist_Nonlinear_Effects_ST = [node[-2] for node in Nonlinear_Effects_ST_series]
MCP_Nonlinear_Effects_ST = [node[-1] for node in Nonlinear_Effects_ST_series]

# Initialize lists to store the Wrist and MCP values separately for DT and ST    (Mass_Qddot)
Wrist_Mass_Qddot_DT = [node[-2] for node in Mass_Qddot_values_DT_series]
MCP_Mass_Qddot_DT = [node[-1] for node in Mass_Qddot_values_DT_series]

Wrist_Mass_Qddot_ST = [node[-2] for node in Mass_Qddot_values_ST_series]
MCP_Mass_Qddot_ST = [node[-1] for node in Mass_Qddot_values_ST_series]

# Initialize lists to store the Wrist and MCP values separately for DT and ST    (Tau_values)
Wrist_Tau_DT = [node[-2] for node in Tau_values_DT_series]
MCP_Tau_DT = [node[-1] for node in Tau_values_DT_series]

Wrist_Tau_ST = [node[-2] for node in Tau_values_ST_series]
MCP_Tau_ST = [node[-1] for node in Tau_values_ST_series]

# Number of nodes is 9 as per the user input
num_nodes = 9
nodes = list(range(num_nodes))

# Create a figure with four subplots arranged in 2x2
fig, axs = plt.subplots(2, 2, figsize=(16, 12))

# Plotting C_T_lambda values for Wrist and MCP joints
axs[0, 0].plot(nodes, Wrist_C_T_lambda_DT, label='Wrist (DT)', marker='o')
axs[0, 0].plot(nodes, MCP_C_T_lambda_DT, label='MCP (DT)', marker='s')
axs[0, 0].plot(nodes, Wrist_C_T_lambda_ST, label='Wrist (ST)', marker='^')
axs[0, 0].plot(nodes, MCP_C_T_lambda_ST, label='MCP (ST)', marker='x')
axs[0, 0].set_title('-C_T_lambda Contributions')
axs[0, 0].set_xlabel('Node')
axs[0, 0].set_ylabel('C_T_lambda Value')
axs[0, 0].legend()
axs[0, 0].grid(True)

# Plotting Nonlinear_Effects values for Wrist and MCP joints
axs[0, 1].plot(nodes, Wrist_Nonlinear_Effects_DT, label='Wrist (DT)', marker='o')
axs[0, 1].plot(nodes, MCP_Nonlinear_Effects_DT, label='MCP (DT)', marker='s')
axs[0, 1].plot(nodes, Wrist_Nonlinear_Effects_ST, label='Wrist (ST)', marker='^')
axs[0, 1].plot(nodes, MCP_Nonlinear_Effects_ST, label='MCP (ST)', marker='x')
axs[0, 1].set_title('Nonlinear Effects Contributions')
axs[0, 1].set_xlabel('Node')
axs[0, 1].set_ylabel('Nonlinear Effects Value')
axs[0, 1].legend()
axs[0, 1].grid(True)

# Plotting Mass_Qddot values for Wrist and MCP joints
axs[1, 0].plot(nodes, Wrist_Mass_Qddot_DT, label='Wrist (DT)', marker='o')
axs[1, 0].plot(nodes, MCP_Mass_Qddot_DT, label='MCP (DT)', marker='s')
axs[1, 0].plot(nodes, Wrist_Mass_Qddot_ST, label='Wrist (ST)', marker='^')
axs[1, 0].plot(nodes, MCP_Mass_Qddot_ST, label='MCP (ST)', marker='x')
axs[1, 0].set_title('Mass_Qddot Contributions')
axs[1, 0].set_xlabel('Node')
axs[1, 0].set_ylabel('Mass_Qddot Value')
axs[1, 0].legend()
axs[1, 0].grid(True)

# Plotting Tau_values for Wrist and MCP joints
axs[1, 1].plot(nodes, Wrist_Tau_DT, label='Wrist (DT)', marker='o')
axs[1, 1].plot(nodes, MCP_Tau_DT, label='MCP (DT)', marker='s')
axs[1, 1].plot(nodes, Wrist_Tau_ST, label='Wrist (ST)', marker='^')
axs[1, 1].plot(nodes, MCP_Tau_ST, label='MCP (ST)', marker='x')
axs[1, 1].set_title('Tau Values Contributions')
axs[1, 1].set_xlabel('Node')
axs[1, 1].set_ylabel('Tau Value')
axs[1, 1].legend()
axs[1, 1].grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()

# Display the plot
plt.show()

# Function to round the values to 4 decimal places
def round_values(values):
    return [round(value, 4) for value in values]

# Function to prepare table data
def prepare_table_data(metrics_DT, metrics_ST, metric_labels):
    table_data = []
    for label, dt_values, st_values in zip(metric_labels, metrics_DT, metrics_ST):
        dt_values_rounded = round_values(dt_values)
        st_values_rounded = round_values(st_values)
        table_data.append([label + ' (DT)'] + dt_values_rounded)
        table_data.append([label + ' (ST)'] + st_values_rounded)
    return table_data

# Metric labels
metric_labels = ['-C_T_lambda', 'NL Effects', 'Mass Qddot', 'Tau']

# Prepare data for the Wrist table
Wrist_metrics_DT = [Wrist_C_T_lambda_DT, Wrist_Nonlinear_Effects_DT, Wrist_Mass_Qddot_DT, Wrist_Tau_DT]
Wrist_metrics_ST = [Wrist_C_T_lambda_ST, Wrist_Nonlinear_Effects_ST, Wrist_Mass_Qddot_ST, Wrist_Tau_ST]
table_data_Wrist = prepare_table_data(Wrist_metrics_DT, Wrist_metrics_ST, metric_labels)

# Prepare data for the MCP table
MCP_metrics_DT = [MCP_C_T_lambda_DT, MCP_Nonlinear_Effects_DT, MCP_Mass_Qddot_DT, MCP_Tau_DT]
MCP_metrics_ST = [MCP_C_T_lambda_ST, MCP_Nonlinear_Effects_ST, MCP_Mass_Qddot_ST, MCP_Tau_ST]
table_data_MCP = prepare_table_data(MCP_metrics_DT, MCP_metrics_ST, metric_labels)

# Create the Wrist table figure
fig_wrist, axs_wrist = plt.subplots(4, 1, figsize=(16, 12))  # 2x2 subplots for Wrist

fig_wrist.suptitle('Wrist Metrics for DT and ST Scenarios')
for i, ax in enumerate(axs_wrist.flatten()):
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=table_data_Wrist[i*2:(i+1)*2], colLabels=['Metric'] + [f'Node {j+1}' for j in range(num_nodes)], loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.5, 1.2)

# Create the MCP table figure
fig_mcp, axs_mcp = plt.subplots(4, 1, figsize=(16, 12))  # 2x2 subplots for MCP
fig_mcp.suptitle('MCP Metrics for DT and ST Scenarios')
for i, ax in enumerate(axs_mcp.flatten()):
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=table_data_MCP[i*2:(i+1)*2], colLabels=['Metric'] + [f'Node {j+1}' for j in range(num_nodes)], loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.5, 1.2)

plt.show()