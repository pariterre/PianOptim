import biorbd
import numpy as np
import pickle

dirName = "/home/alpha/pianoptim/PianOptim/Nov.Codes/Updated_BioModFile_YeadonModel/Results/Felipe_25March/X/Struck_without_Thorax_2.pckl"

# Load the musculoskeletal model
model = biorbd.Model("./Without.bioMod")

with open(dirName, "rb") as file:
    new_dict = pickle.load(file)

# Extract joint positions (q) and velocities (qdot) from the loaded data
array_q_s_1 = [new_dict["states_no_intermediate"][i]["q"] for i in range(len(new_dict["states_no_intermediate"]))]
array_qdot_s_1 = [new_dict["states_no_intermediate"][i]["qdot"] for i in range(len(new_dict["states_no_intermediate"]))]

num_nodes = 10
num_dofs = 7

# Choose a position/velocity/acceleration to compute dynamics from
Q = np.array([array_q_s_1[2][i][0] for i in range(num_dofs)])
Qdot = np.array([array_qdot_s_1[2][i][0] for i in range(num_dofs)])

array_tau_s_1 = [new_dict["controls"][2]["tau"]]
array_tau_s = np.array([array_tau_s_1[0][i][0] for i in range(7)])
Qddot = model.ForwardDynamics(Q, Qdot, array_tau_s)
#
# Define the external forces
f_ext = biorbd.ExternalForceSet(model)

# force_vector = np.array([new_dict["Contact_Force"][0][0][0], new_dict["Contact_Force"][0][1][0], new_dict["Contact_Force"][0][2][0], 0, 0, 0])
force_vector = np.array([0, 0, 54, 0, 0, 0])
force_point = np.array([0.0015, 0.0015, -0.046782999938])

# Create a SpatialVector for the external force and point of application
spatial_force = biorbd.SpatialVector(force_vector)

# # Assuming fingertip_marker_name is the name of the marker on the segment where the force is applied
# parent_name = "RightPalm"
# p = biorbd.NodeSegment(force_point)
#
# # Add the SpatialVector to the ExternalForceSet for the segment where the force is applied
# f_ext.addInSegmentReferenceFrame(parent_name, spatial_force, p)

# Compute the mass matrix
Mass = model.massMatrix(Q).to_array()
# Compute the generalized accelerations using forward dynamics
Qddot_2_m = Qddot.to_array().reshape(7, 1)

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
# F_contact = np.array([new_dict["Contact_Force"][3][0][0], new_dict["Contact_Force"][3][1][0], new_dict["Contact_Force"][3][2][0]]).reshape(3, 1)
F_contact = np.array([0,0,54]).reshape(3, 1)

# Compute C^T lambda for the contact force
C_T_lambda_contact = np.dot(J_contact, F_contact)

# Compute the final result: m.massmatrix(q) * qddot + m.NLEffects(q,qdot) + C^T lambda
# result = np.dot(Mass, Qddot_2_m) + nonlinear_effects + C_T_lambda_contact
Tau = model.InverseDynamics(Q, Qdot, Qddot, f_ext)

print("C^T lambda:")
print(C_T_lambda_contact)
Tau_2 = np.dot(Mass, Qddot_2_m) + nonlinear_effects
print("m.massmatrix(q) * qddot + m.NLEffects(q,qdot):")
print(Tau_2)
print("Tau (InverseDynamics)")
print(Tau.to_array().reshape(7, 1))
print("nonlinear_effects:")
print(nonlinear_effects)

print(array_tau_s.reshape(7,1)-Tau)
