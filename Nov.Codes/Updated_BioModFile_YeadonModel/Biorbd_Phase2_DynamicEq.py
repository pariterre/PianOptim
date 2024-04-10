import biorbd
import numpy as np
import pickle
import matplotlib.pyplot as plt

def get_user_input():
    while True:
        all_dof = input("Include thorax? (y/n): ").lower()
        if all_dof in ['y', 'n']:
            all_dof = all_dof == 'y'
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

    while True:
        pressed = input("Show 'Pressed' or 'Struck' condition? (p/s): ").lower()
        if pressed in ['p', 's']:
            pressed = pressed == 'p'
            break
        else:
            print("Invalid input. Please enter 'p' or 's'.")

    return all_dof, pressed


allDOF, pressed = get_user_input()

dirName = "/home/alpha/pianoptim/PianOptim/Nov.Codes/Updated_BioModFile_YeadonModel/Results/Felipe_25March/X/"

if allDOF:
    saveName = dirName + ("Pressed" if pressed else "Struck") + "_with_Thorax.pckl"
else:
    saveName = dirName + ("Pressed" if pressed else "Struck") + "_without_Thorax.pckl"

biorbd_model_path = "./With.bioMod" if allDOF else "./Without.bioMod"

# Load the musculoskeletal model
model = biorbd.Model("./With.bioMod" if allDOF else "./Without.bioMod")


with open(saveName, "rb") as file:
    new_dict = pickle.load(file)

# Ask the user for the number of Degrees of Freedom (DOFs)
num_dofs = model.nbDof()

if pressed:
    num_nodes = 9
else:
    num_nodes = 9

# Extract joint positions (q) and velocities (qdot) from the loaded data
array_q_s = [new_dict["states_no_intermediate"][i]["q"] for i in range(len(new_dict["states_no_intermediate"]))]
array_qdot_s = [new_dict["states_no_intermediate"][i]["qdot"] for i in range(len(new_dict["states_no_intermediate"]))]
array_tau_s = [new_dict["controls"][2]["tau"] for i in range(len(new_dict["states_no_intermediate"]))]

# Create subplots for each shooting node
fig, axs = plt.subplots(num_nodes, 1, figsize=(10, 4*num_nodes))

for i in range(num_nodes):
    # Choose a position/velocity/acceleration to compute dynamics from
    Q = np.array([array_q_s[2][j][i] for j in range(num_dofs)])
    Qdot = np.array([array_qdot_s[2][j][i] for j in range(num_dofs)])

    # Extract external Forces
    f_ext = np.array([new_dict["Contact_Force"][i][0][0], new_dict["Contact_Force"][i][1][0], new_dict["Contact_Force"][i][2][0]])
    Tau = np.array([new_dict["controls"][2]["tau"][j][i] for j in range(num_dofs)])

    # Compute the generalized accelerations using forward dynamics
    Qddot = model.ForwardDynamicsConstraintsDirect(Q, Qdot, Tau).to_array()
    Qddot_m = Qddot.reshape(7, 1)

    # Compute the mass matrix
    Mass = model.massMatrix(Q).to_array()

    # Compute the mass matrix multiplied by the generalized accelerations
    Mass_Qddot = np.dot(Mass, Qddot_m)

    # Compute the nonlinear effects
    nonlinear_effects = model.NonLinearEffect(Q, Qdot).to_array().reshape(7, 1)

    # Compute the Jacobian for the contact point on the "RightFingers" segment
    parent_name = "RightFingers"
    contact_point = np.array([0.0015, 0.0015, -0.046782999938])
    p = biorbd.NodeSegment(contact_point[0], contact_point[1], contact_point[2])
    update_kin = True
    contact_jacobian = model.markersJacobian(Q, parent_name, p, update_kin)

    # Extract the Jacobian matrix as a NumPy array
    jacobian_array = contact_jacobian.to_array()
    J_contact = np.array(jacobian_array).T

    # Define the contact force vector
    F_contact = np.array([0, 0, 54]).reshape(3, 1)

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
    C_T_lambda_values = [-C_T_lambda_wrist[0], -C_T_lambda_finger[0]]
    Nonlinear_Effects = [Nonlinear_Effects_wrist[0], Nonlinear_Effects_finger[0]]
    Mass_Qddot_values = [Mass_Qddot_wrist[0], Mass_Qddot_finger[0]]
    Tau_values = [tau_wrist, tau_finger]

    width = 0.15
    x = np.arange(len(labels))

    axs[i].bar(x - 2 * width, C_T_lambda_values, width, label='C^T lambda', linewidth=1)
    axs[i].bar(x - width, Tau_values, width, label='Tau', linewidth=1)
    axs[i].bar(x, Nonlinear_Effects, width, label='Nonlinear Effects', linewidth=1)
    axs[i].bar(x + width, Mass_Qddot_values, width, label='Mass * Qddot', linewidth=1)

    axs[i].set_ylabel('Torque (N.m)')
    axs[i].set_title(f'Contributions to Tau Values - Node {i}')
    axs[i].set_xticks(x)
    axs[i].set_xticklabels(labels)
    axs[i].legend()

plt.tight_layout()
plt.show()


