import biorbd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# def get_user_input():
#     while True:
#         all_dof = input("Include thorax? (y/n): ").lower()
#         if all_dof in ['y', 'n']:
#             all_dof = all_dof == 'y'
#             break
#         else:
#             print("Invalid input. Please enter 'y' or 'n'.")
#
#     while True:
#         pressed = input("Show 'Pressed' or 'Struck' condition? (p/s): ").lower()
#         if pressed in ['p', 's']:
#             pressed = pressed == 'p'
#             break
#         else:
#             print("Invalid input. Please enter 'p' or 's'.")
#
#     return all_dof, pressed
#
#
# allDOF, pressed = get_user_input()

dirName = "/home/alpha/pianoptim/PianOptim/Nov.Codes/Updated_BioModFile_YeadonModel/Results/Felipe_25March/Version2_2Apr/Struck_without_Thorax.pckl"
#
# if allDOF:
#     saveName = dirName + ("Pressed" if pressed else "Struck") + "_with_Thorax.pckl"
# else:
#     saveName = dirName + ("Pressed" if pressed else "Struck") + "_without_Thorax.pckl"
#
# biorbd_model_path = "./With.bioMod" if allDOF else "./Without.bioMod"

# Load the musculoskeletal model
model = biorbd.Model("./Without.bioMod")
print("Model loaded successfully.")

with open(dirName, "rb") as file:
    new_dict = pickle.load(file)

# Extract joint positions (q) and velocities (qdot) from the loaded data
array_q_s_1 = [new_dict["states_no_intermediate"][i]["q"] for i in range(len(new_dict["states_no_intermediate"]))]
array_qdot_s_1 = [new_dict["states_no_intermediate"][i]["qdot"] for i in range(len(new_dict["states_no_intermediate"]))]


# Calculate the time step (dt)
specific_points_s_1 = [sum(new_dict["phase_time"][: i + 1]) for i in range(len(new_dict["phase_time"]))]
time_arrays_1 = [np.linspace(specific_points_s_1[2], specific_points_s_1[3], 10)]
concatenated_array_time_s_1 = np.concatenate(time_arrays_1)
dt = concatenated_array_time_s_1[1] - concatenated_array_time_s_1[0]


# Initialize a list to store the accelerations for each phase
array_qddot_s_1 = []

num_nodes = 10
num_dofs = 7

qddot_phase = np.zeros((num_dofs, num_nodes))

# Calculate qddot using central difference for each phase
for i in range(1, num_nodes - 1):
    q_obj = np.array([array_q_s_1[2][j][i] for j in range(num_dofs)])
    q_prev = np.array([array_q_s_1[2][j][i-1] for j in range(num_dofs)])
    q_next = np.array([array_q_s_1[2][j][i+1] for j in range(num_dofs)])
    qddot_phase[:, i] = (q_next - 2 * q_obj + q_prev) / dt**2

# Handle the first and last nodes using forward/backward difference
q_first = np.array([array_q_s_1[2][j][0] for j in range(num_dofs)])
q_second = np.array([array_q_s_1[2][j][1] for j in range(num_dofs)])
q_last = np.array([array_q_s_1[2][j][-1] for j in range(num_dofs)])
q_second_last = np.array([array_q_s_1[2][j][-2] for j in range(num_dofs)])

qddot_phase[:, 0] = (q_second - 2 * q_first + q_second) / dt**2
qddot_phase[:, -1] = (q_last - 2 * q_second_last + q_second_last) / dt**2

array_qddot_s_1.append(qddot_phase)

nq = model.nbQ()
nqdot = model.nbQdot()
nqddot = model.nbQddot()

print(f"Number of generalized coordinates (nq): {nq}")
print(f"Number of generalized velocities (nqdot): {nqdot}")
print(f"Number of generalized accelerations (nqddot): {nqddot}")

# Choose a position/velocity/acceleration to compute dynamics from
Q = np.array([array_q_s_1[2][i][0] for i in range(num_dofs)])
Qdot = np.array([array_qdot_s_1[2][i][0] for i in range(num_dofs)])
Qddot_1 =np.array([array_qddot_s_1[0][i][0] for i in range(num_dofs)])

array_tau_s_1 = [new_dict["controls"][2]["tau"]]
array_tau_s= np.array([array_tau_s_1[0][i][0] for i in range(7)])
Qddot_2 = model.ForwardDynamics(Q, Qdot, array_tau_s)

# Define the external forces
f_ext = []

wrist_joint_id = model.getDofIndex("RightPalm", "RotX")
finger_joint_id = model.getDofIndex("RightFingers", "RotX")

print(f"Wrist joint ID: {wrist_joint_id}")
print(f"Finger joint ID: {finger_joint_id}")

force_vector = np.array([0, 0, 54, 0, 0, 0])
force_point = np.array([0.0015, 0.0015, -0.046782999938])

# Create an ExternalForceSet object
f_ext = biorbd.ExternalForceSet(model)

# Create a SpatialVector for the external force and point of application
spatial_force = biorbd.SpatialVector(force_vector)

# Assuming fingertip_marker_name is the name of the marker on the segment where the force is applied
fingertip_marker_name = "finger_marker"
parent_name = "RightFingers"
p = biorbd.NodeSegment(force_point)

# Add the SpatialVector to the ExternalForceSet for the segment where the force is applied
f_ext.addInSegmentReferenceFrame(parent_name, spatial_force, p)
# Proceed with the inverse dynamics
Tau = model.InverseDynamics(Q, Qdot, Qddot_2, f_ext)

# Print the generalized forces (torques) to the console
print("Generalized forces (torques):")
print(Tau.to_array())


array_tau_s_1 = [new_dict["controls"][2]["tau"]]
array_tau_s= np.array([array_tau_s_1[0][i][0] for i in range(7)])
Qddot_2 = model.ForwardDynamics(Q, Qdot, array_tau_s)

# Print them to the console
print("Generalized accelerations (Using central difference):")
print(Qddot_1)

print("Generalized accelerations (Using Forward Dynamics):")
print(Qddot_2.to_array())
D=Qddot_2.to_array()
D_m=np.array(D)
print(D_m)
T=model.massMatrix(Q)
Mass=(T.to_array())
print(np.array(Mass))

result = np.dot(Mass, D_m)


nonlinear_effects = model.NonLinearEffect(Q, Qdot, f_ext)
X=(nonlinear_effects.to_array())
X_m=np.array(X)

# Computing the Jacobian for the "finger_marker"
fingertip_marker_name = "finger_marker"
parent_name = "RightFingers"
p = biorbd.NodeSegment(force_point)
update_kin = True
markers_jacobian = model.markersJacobian(q_obj, parent_name, p, update_kin)

# Extract the Jacobian matrix as a NumPy array
jacobian_array = markers_jacobian.to_array()
J_m=np.array(jacobian_array).T
F=np.array([0, 0, 54]).T
result_F = np.dot(J_m, F)
print(result+X)
print(result_F+array_tau_s)
print(jacobian_array)