# Comment on the axes used in the biomechanical model
    # Note: This comment explains that the axes used in this model are different from
    # the standard biomechanics axes. Specifically, it mentions how the axes are redefined:
    # Y becomes X, Z becomes Y, and X becomes Z.

# A brief of the Phases and Joint indices

# Joint indices in the biomechanical model:
    # 0. Pelvic Tilt, Anterior (-) and Posterior (+) Rotation
    # 1. Thorax, Left (+) and Right (-) Rotation
    # 2. Thorax, Flexion (-) and Extension (+)
    # 3. Right Shoulder, Abduction (-) and Adduction (+)
    # 4. Right Shoulder, Internal (+) and External (-) Rotation
    # 5. Right Shoulder, Flexion (+) and Extension (-)
    # 6. Elbow, Flexion (+) and Extension (-)
    # 7. Elbow, Pronation (+) and Supination (-)
    # 8. Wrist, Flexion (-) and Extension (+)
    # 9. MCP, Flexion (+) and Extension (-)

    # Note: The signs (+/-) indicate the direction of the movement for each joint.

# Description of movement phases:
    # Phase 0: Preparation - Getting the fingers in position.
    # Phase 1: Key Descend - The downward motion of the fingers pressing the keys.
    # Phase 2: Key Bed - The phase where the keys are fully pressed and meet the bottom.
    # Phase 3: Key Release (Upward) - Releasing the keys and moving the hand upward.
    # Phase 4: Return to Neutral (Downward) - Bringing the fingers back to a neutral position, ready for the next action.

# Importing necessary libraries and functions
from casadi import MX, acos, dot, pi, Function  # CasADi is a symbolic framework for numeric optimization
import time
import numpy as np
import biorbd_casadi as biorbd  # BioRBD CasADi is a library for biomechanical analysis
import pickle  # Standard Python module for serializing and de-serializing Python object structures

# Importing various classes and functions from the bioptim library
from bioptim import (
    BiorbdModel,
    PenaltyController,
    ObjectiveList,
    PhaseTransitionFcn,
    DynamicsList,
    ConstraintFcn,
    BoundsList,
    InitialGuessList,
    CostType,
    PhaseTransitionList,
    Node,
    OptimalControlProgram,
    DynamicsFcn,
    ObjectiveFcn,
    ConstraintList,
    OdeSolver,
    Solver,
    MultinodeObjectiveList,
    Axis,
)

# Function to minimize the difference between control parameters of two following phases in an optimization problem.
def minimize_difference(controllers: list[PenaltyController, PenaltyController]):
    pre, post = controllers
    return pre.controls.cx_end - post.controls.cx

#  Custom function to track the relative position of the fifth finger to the principal finger
# Description:This function computes the Y-axis difference between two finger markers.
def custom_func_track_finger_5_on_the_right_of_principal_finger(controller: PenaltyController) -> MX:
    # Get the index of the marker on the principal finger
    finger_marker_idx = biorbd.marker_index(controller.model.model, "finger_marker")
    # Extract the position of the principal finger marker
    markers = controller.mx_to_cx("markers", controller.model.markers, controller.states["q"])
    finger_marker = markers[:, finger_marker_idx]

    # Get the index of the marker on the fifth finger
    finger_marker_5_idx = biorbd.marker_index(controller.model.model, "finger_marker_5")
    # Extract the position of the fifth finger marker
    markers_5 = controller.mx_to_cx("markers_5", controller.model.markers, controller.states["q"])
    finger_marker_5 = markers_5[:, finger_marker_5_idx]
    # Compute the difference in the Y-axis
    markers_diff_key2 = finger_marker[1] - finger_marker_5[1]

    return markers_diff_key2

# Function to track the vertical distance of a specified finger marker from a piano key bed.
def custom_func_track_principal_finger_and_finger5_above_bed_key(controller: PenaltyController, marker: str) -> MX:
    biorbd_model = controller.model
    finger_marker_idx = biorbd.marker_index(biorbd_model.model, marker)
    markers = controller.mx_to_cx("markers", biorbd_model.markers, controller.states["q"])
    finger_marker = markers[:, finger_marker_idx]

    markers_diff_key3 = finger_marker[2] - (0.07808863830566405 - 0.02)

    return markers_diff_key3

# Function to calculate the angle between a local axis of a segment and a global axis.
# It is useful for understanding the orientation of a segment relative to a global reference frame.
def custom_func_track_principal_finger_pi_in_two_global_axis(controller: PenaltyController, segment: str) -> MX:
    rotation_matrix_index = biorbd.segment_index(controller.model.model, segment)
    q = controller.states["q"].mx
    # global JCS gives the local matrix according to the global matrix
    principal_finger_axis= controller.model.model.globalJCS(q, rotation_matrix_index).to_mx()  # x finger = y global
    y = MX.zeros(4)
    y[:4] = np.array([0, 1, 0, 1])
    # @ x : pour avoir l'orientation du vecteur x du jcs local exprimé dans le global
    # @ produit matriciel
    principal_finger_y = principal_finger_axis @ y
    principal_finger_y = principal_finger_y[:3, :]

    global_y = MX.zeros(3)
    global_y[:3] = np.array([0, 1, 0])

    teta = acos(dot(principal_finger_y, global_y[:3]))
    output_casadi = controller.mx_to_cx("scal_prod", teta, controller.states["q"])

    return output_casadi

  # Prepare an Optimal Control Program (OCP) for a biomechanical model.
def prepare_ocp(
    biorbd_model_path: str = "/home/alpha/Desktop/PianOptim/2_Mathilde_2022/2__final_models_piano/1___final_model___squeletum_hand_finger_1_key_4_phases_/bioMod/Squeletum_hand_finger_3D_2_keys_octave_LA.bioMod",
    ode_solver: OdeSolver = OdeSolver.COLLOCATION(polynomial_degree=4),
) -> OptimalControlProgram:

    # Loading the biomechanical model for each phase of the movement
    biorbd_model = (
        BiorbdModel(biorbd_model_path),
        BiorbdModel(biorbd_model_path),
        BiorbdModel(biorbd_model_path),
        BiorbdModel(biorbd_model_path),
        BiorbdModel(biorbd_model_path),

    )

    # Defining the number of shooting points and time for each phase
    n_shooting = (30, 7, 9, 17, 18)
    phase_time = (0.3, 0.044, 0.051, 0.17, 0.18)

    # Defining the control limits for the torque
    tau_min, tau_max, tau_init = -200, 200, 0

    # Defining the velocity profile based on motion capture data
    vel_push_array2 = [
        [
            0,
            -0.113772161006927,
            -0.180575996580578,
            -0.270097219830468,
            -0.347421549388341,
            -0.290588704744975,
            -0.0996376128423782,
            0,
        ]
    ]

    # Defining a force profile
    Froce = [30, 26, 24, 20, 16, 12, 8, 4, 0]

    # Setting initial conditions for each phase
    pi_sur_2_phase_0 = np.full((1, n_shooting[0] + 1), pi / 2)
    pi_sur_2_phase_1 = np.full((1, n_shooting[1] + 1), pi / 2)
    pi_sur_2_phase_2 = np.full((1, n_shooting[2] + 1), pi / 2)
    pi_sur_2_phase_3 = np.full((1, n_shooting[3] + 1), pi / 2)
    pi_sur_2_phase_4 = np.full((1, n_shooting[4] + 1), pi / 2)

    # Objectives
    # Loop over each phase
    # Add an objective function to minimize the torques (tau) for specific indexes
    # The weight given to this objective is 100
    objective_functions = ObjectiveList()
    for i in [0, 1, 2, 3, 4]:
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", phase=i, weight=100, index=[0, 1, 2, 5, 3, 4, 6, 7]
        )

    # The weight given to this objective is 10000
    for i in [0, 1, 2, 3, 4]:
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", phase=i, weight=10000, index=[8, 9], derivative=True
        )

    # Adding objective functions "Regularization Term"
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", phase=0, weight=0.0001, index=[0, 1, 2, 3, 4, 5, 6, 7]
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", phase=1, weight=0.0001, index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", phase=2, weight=0.0001, index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", phase=3, weight=0.0001, index=[0, 1, 2, 3, 4, 5, 6, 7]
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", phase=4, weight=0.0001, index=[0, 1, 2, 3, 4, 5, 6, 7]
    )

    # To block rotation (Movement Adjustment)
    for i in [0, 1, 2, 3, 4]:
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", phase=i, weight=100000, index=[3, 7])

    # To Track Marker Velocity Profile in phase 1
    objective_functions.add(
        ObjectiveFcn.Mayer.TRACK_MARKERS_VELOCITY,
        target=vel_push_array2,
        node=Node.ALL,
        phase=1,
        marker_index=4,
        weight=10000,
    )

    # To keep the hand/index perpendicular of the key piano
    objective_functions.add(
        custom_func_track_principal_finger_pi_in_two_global_axis,
        custom_type=ObjectiveFcn.Lagrange,
        node=Node.ALL,
        phase=0,
        weight=1000,
        quadratic=True,
        target=pi_sur_2_phase_0,
        segment="2proxph_2mcp_flexion",
    )
    objective_functions.add(
        custom_func_track_principal_finger_pi_in_two_global_axis,
        custom_type=ObjectiveFcn.Lagrange,
        node=Node.ALL,
        phase=1,
        weight=100000,
        quadratic=True,
        target=pi_sur_2_phase_1,
        segment="2proxph_2mcp_flexion",
    )
    objective_functions.add(
        custom_func_track_principal_finger_pi_in_two_global_axis,
        custom_type=ObjectiveFcn.Lagrange,
        node=Node.ALL,
        phase=2,
        weight=100000,
        quadratic=True,
        target=pi_sur_2_phase_2,
        segment="2proxph_2mcp_flexion",
    )
    objective_functions.add(
        custom_func_track_principal_finger_pi_in_two_global_axis,
        custom_type=ObjectiveFcn.Lagrange,
        node=Node.ALL,
        phase=3,
        weight=1000,
        quadratic=True,
        target=pi_sur_2_phase_3,
        segment="2proxph_2mcp_flexion",
    )

    objective_functions.add(
        custom_func_track_principal_finger_pi_in_two_global_axis,
        custom_type=ObjectiveFcn.Lagrange,
        node=Node.ALL,
        phase=4,
        weight=100000,
        quadratic=True,
        target=pi_sur_2_phase_4,
        segment="2proxph_2mcp_flexion",
    )

    objective_functions.add(
        custom_func_track_principal_finger_pi_in_two_global_axis,
        custom_type=ObjectiveFcn.Lagrange,
        node=Node.ALL,
        phase=0,
        weight=1000,
        quadratic=True,
        target=pi_sur_2_phase_0,
        segment="secondmc",
    )
    objective_functions.add(
        custom_func_track_principal_finger_pi_in_two_global_axis,
        custom_type=ObjectiveFcn.Lagrange,
        node=Node.ALL,
        phase=1,
        weight=100000,
        quadratic=True,
        target=pi_sur_2_phase_1,
        segment="secondmc",
    )
    objective_functions.add(
        custom_func_track_principal_finger_pi_in_two_global_axis,
        custom_type=ObjectiveFcn.Lagrange,
        node=Node.ALL,
        phase=2,
        weight=100000,
        quadratic=True,
        target=pi_sur_2_phase_2,
        segment="secondmc",
    )

    objective_functions.add(
        custom_func_track_principal_finger_pi_in_two_global_axis,
        custom_type=ObjectiveFcn.Lagrange,
        node=Node.ALL,
        phase=3,
        weight=100000,
        quadratic=True,
        target=pi_sur_2_phase_3,
        segment="secondmc",
    )

    objective_functions.add(
        custom_func_track_principal_finger_pi_in_two_global_axis,
        custom_type=ObjectiveFcn.Lagrange,
        node=Node.ALL,
        phase=4,
        weight=100000,
        quadratic=True,
        target=pi_sur_2_phase_4,
        segment="secondmc",
    )

    # To avoid the "noise/tremor" caused by the objective function
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", phase=0, weight=1000, index=[8, 9], derivative=True
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", phase=3, weight=1000, index=[8, 9], derivative=True
    )

    # To Track Contact Forces Profile (Vertical)
    objective_functions.add(
        ObjectiveFcn.Lagrange.TRACK_CONTACT_FORCES,
        target=Froce,
        node=Node.ALL_SHOOTING,
        contact_index=2,
        phase=2,
        weight=10000,
    )

    Mul_Node_Obj = MultinodeObjectiveList()
    # To minimize the difference between 0 and 1 (Phase Transition)
    Mul_Node_Obj.add(
        minimize_difference,
        custom_type=ObjectiveFcn.Mayer,
        weight=1000,
        nodes_phase=(0, 1),
        nodes=(Node.END, Node.START),
        quadratic=True,
    )

    Mul_Node_Obj.add(
        minimize_difference,
        custom_type=ObjectiveFcn.Mayer,
        weight=1000,
        nodes_phase=(1, 2),
        nodes=(Node.END, Node.START),
        quadratic=True,
    )

    Mul_Node_Obj.add(
        minimize_difference,
        custom_type=ObjectiveFcn.Mayer,
        weight=1000,
        nodes_phase=(2, 3),
        nodes=(Node.END, Node.START),
        quadratic=True,
    )

    Mul_Node_Obj.add(
        minimize_difference,
        custom_type=ObjectiveFcn.Mayer,
        weight=1000,
        nodes_phase=(3, 4),
        nodes=(Node.END, Node.START),
        quadratic=True,
    )

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=0)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=1)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True, phase=2)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=3)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=4)

    # Constraints
    constraints = ConstraintList()

    # Superimposing Markers in phase 0
    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        node=Node.ALL,
        first_marker="finger_marker",
        second_marker="high_square",
        phase=0,
    )

    # Superimposing Markers in phase 1
    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        node=Node.END,
        first_marker="finger_marker",
        second_marker="low_square",
        phase=1,
    )

    # Contact Forces Ranges for other Axes
    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES, node=Node.ALL, contact_index=0, min_bound=-5, max_bound=5, phase=2
    )
    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES, node=Node.ALL, contact_index=1, min_bound=-5, max_bound=5, phase=2
    )

    # Superimposing Markers in phase 3
    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        node=Node.END,
        first_marker="MCP_contact_finger",
        second_marker="phase_3_upward",
        phase=3,
    )

    # Superimposing Markers in phase 4
    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        node=Node.END,
        first_marker="finger_marker",
        second_marker="high_square",
        phase=4,
    )

    # To keep the index and the small finger above the bed key.
    constraints.add(
        custom_func_track_principal_finger_and_finger5_above_bed_key,
        node=Node.ALL,
        marker="finger_marker",
        min_bound=0,
        max_bound=10000,
        phase=0,
    )
    constraints.add(
        custom_func_track_principal_finger_and_finger5_above_bed_key,
        node=Node.ALL,
        marker="finger_marker",
        min_bound=0,
        max_bound=10000,
        phase=1,
    )
    constraints.add(
        custom_func_track_principal_finger_and_finger5_above_bed_key,
        node=Node.ALL,
        marker="finger_marker",
        min_bound=0,
        max_bound=10000,
        phase=2,
    )
    constraints.add(
        custom_func_track_principal_finger_and_finger5_above_bed_key,
        node=Node.ALL,
        marker="finger_marker",
        min_bound=0,
        max_bound=10000,
        phase=3,
    )

    constraints.add(
        custom_func_track_principal_finger_and_finger5_above_bed_key,
        node=Node.ALL,
        marker="finger_marker",
        min_bound=0,
        max_bound=10000,
        phase=4,
    )

    constraints.add(
        custom_func_track_principal_finger_and_finger5_above_bed_key,
        node=Node.ALL,
        marker="finger_marker_5",
        min_bound=0,
        max_bound=10000,
        phase=0,
    )
    constraints.add(
        custom_func_track_principal_finger_and_finger5_above_bed_key,
        node=Node.ALL,
        marker="finger_marker_5",
        min_bound=0,
        max_bound=10000,
        phase=1,
    )
    constraints.add(
        custom_func_track_principal_finger_and_finger5_above_bed_key,
        node=Node.ALL,
        marker="finger_marker_5",
        min_bound=0,
        max_bound=10000,
        phase=2,
    )
    constraints.add(
        custom_func_track_principal_finger_and_finger5_above_bed_key,
        node=Node.ALL,
        marker="finger_marker_5",
        min_bound=0,
        max_bound=10000,
        phase=3,
    )

    constraints.add(
        custom_func_track_principal_finger_and_finger5_above_bed_key,
        node=Node.ALL,
        marker="finger_marker_5",
        min_bound=0,
        max_bound=10000,
        phase=4,
    )

    # To keep the small finger on the right of the principal finger.
    constraints.add(
        custom_func_track_finger_5_on_the_right_of_principal_finger,
        node=Node.ALL,
        min_bound=0.00001,
        max_bound=10000,
        phase=0,
    )
    constraints.add(
        custom_func_track_finger_5_on_the_right_of_principal_finger,
        node=Node.ALL,
        min_bound=0.00001,
        max_bound=10000,
        phase=1,
    )
    constraints.add(
        custom_func_track_finger_5_on_the_right_of_principal_finger,
        node=Node.ALL,
        min_bound=0.00001,
        max_bound=10000,
        phase=2,
    )
    constraints.add(
        custom_func_track_finger_5_on_the_right_of_principal_finger,
        node=Node.ALL,
        min_bound=0.00001,
        max_bound=10000,
        phase=3,
    )

    constraints.add(
        custom_func_track_finger_5_on_the_right_of_principal_finger,
        node=Node.ALL,
        min_bound=0.00001,
        max_bound=10000,
        phase=4,
    )


    phase_transition = PhaseTransitionList()
    phase_transition.add(PhaseTransitionFcn.IMPACT, phase_pre_idx=1)

    # Defining bounds for state variables
    x_bounds = BoundsList()
    x_bounds.add("q", bounds=biorbd_model[0].bounds_from_ranges("q"), phase=0)
    x_bounds.add("qdot", bounds=biorbd_model[0].bounds_from_ranges("qdot"), phase=0)

    x_bounds.add("q", bounds=biorbd_model[1].bounds_from_ranges("q"), phase=1)
    x_bounds.add("qdot", bounds=biorbd_model[1].bounds_from_ranges("qdot"), phase=1)

    x_bounds.add("q", bounds=biorbd_model[2].bounds_from_ranges("q"), phase=2)
    x_bounds.add("qdot", bounds=biorbd_model[2].bounds_from_ranges("qdot"), phase=2)

    x_bounds.add("q", bounds=biorbd_model[3].bounds_from_ranges("q"), phase=3)
    x_bounds.add("qdot", bounds=biorbd_model[3].bounds_from_ranges("qdot"), phase=3)

    x_bounds.add("q", bounds=biorbd_model[4].bounds_from_ranges("q"), phase=4)
    x_bounds.add("qdot", bounds=biorbd_model[4].bounds_from_ranges("qdot"), phase=4)

    #Initial Posture for Each Phase of the Movement
    # This section defines the starting, intermediate, and final postures for the biomechanical model in each phase of the movement.
    x_bounds[0]["q"][[0], 0] = -0.1
    x_bounds[0]["q"][[2], 0] = 0.1

    x_bounds[4]["q"][[0], 2] = -0.1
    x_bounds[4]["q"][[2], 2] = 0.1

    # Initial guess
    x_init = InitialGuessList()

    x_init.add("q", [0] * biorbd_model[0].nb_q, phase=0)
    x_init.add("qdot", [0] * biorbd_model[0].nb_q, phase=0)

    x_init.add("q", [0] * biorbd_model[0].nb_q, phase=1)
    x_init.add("qdot", [0] * biorbd_model[0].nb_q, phase=1)

    x_init.add("q", [0] * biorbd_model[0].nb_q, phase=2)
    x_init.add("qdot", [0] * biorbd_model[0].nb_q, phase=2)

    x_init.add("q", [0] * biorbd_model[0].nb_q, phase=3)
    x_init.add("qdot", [0] * biorbd_model[0].nb_q, phase=3)

    x_init.add("q", [0] * biorbd_model[0].nb_q, phase=4)
    x_init.add("qdot", [0] * biorbd_model[0].nb_q, phase=4)

    # Setting initial guess joint angles for the start of each phase
    for i in range(5):
        x_init[i]["q"][4, 0] = 0.08
        x_init[i]["q"][5, 0] = 0.67
        x_init[i]["q"][6, 0] = 1.11
        x_init[i]["q"][7, 0] = 1.48
        x_init[i]["q"][9, 0] = 0.17

    # Defining bounds for control variables for each phase
    u_bounds = BoundsList()

    u_bounds.add("tau", min_bound=[tau_min] * biorbd_model[0].nb_tau, max_bound=[tau_max] * biorbd_model[0].nb_tau,
                 phase=0)
    u_bounds.add("tau", min_bound=[tau_min] * biorbd_model[1].nb_tau, max_bound=[tau_max] * biorbd_model[1].nb_tau,
                 phase=1)
    u_bounds.add("tau", min_bound=[tau_min] * biorbd_model[2].nb_tau, max_bound=[tau_max] * biorbd_model[2].nb_tau,
                 phase=2)
    u_bounds.add("tau", min_bound=[tau_min] * biorbd_model[3].nb_tau, max_bound=[tau_max] * biorbd_model[3].nb_tau,
                 phase=3)
    u_bounds.add("tau", min_bound=[tau_min] * biorbd_model[4].nb_tau, max_bound=[tau_max] * biorbd_model[4].nb_tau,
                 phase=4)

    # Initializing the torques to tau_init

    u_init = InitialGuessList()
    u_init.add("tau", [tau_init] * biorbd_model[0].nb_tau, phase=0)
    u_init.add("tau", [tau_init] * biorbd_model[1].nb_tau, phase=1)
    u_init.add("tau", [tau_init] * biorbd_model[2].nb_tau, phase=2)
    u_init.add("tau", [tau_init] * biorbd_model[3].nb_tau, phase=3)
    u_init.add("tau", [tau_init] * biorbd_model[4].nb_tau, phase=4)

    # Creating an Optimal Control Program
    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        phase_time,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        phase_transitions=phase_transition,
        ode_solver=ode_solver,
    )


def main():
    """
    Defines a multiphase ocp and animate the results
    """

    ocp = prepare_ocp()
    # ocp.add_plot_penalty(CostType.ALL)

    # Solving the Optimal Control Program , Configure the solver (IPOPT) for the optimization
    solv = Solver.IPOPT(show_online_optim=False)
    solv.set_maximum_iterations(1000000)
    solv.set_linear_solver("ma57")
    tic = time.time()
    sol = ocp.solve(solv)

    # Defining symbolic variables for joint angles, rates of change, and torques
    q_sym = MX.sym('q_sym', 10, 1)
    qdot_sym = MX.sym('qdot_sym', 10, 1)
    tau_sym = MX.sym('tau_sym', 10, 1)

    # Calculating contact forces using forward dynamics
    Calculaing_Force = Function("Temp", [q_sym, qdot_sym, tau_sym], [
        ocp.nlp[2].model.contact_forces_from_constrained_forward_dynamics(q_sym, qdot_sym, tau_sym)])

    # A matrix to store force values
    rows = 9
    cols = 3

    F = [[0] * cols for _ in range(rows)]

    for i in range(0, 9):
        F[i] = Calculaing_Force(sol.states[2]["q"][:, i], sol.states[2]["qdot"][:, i],
                                sol.controls[2]['tau'][:, i])

    F_array = np.array(F)

    # Create a dictionary to store all the relevant data
    data = dict(
        states=sol.states,
        states_no_intermediate=sol.states_no_intermediate,
        controls=sol.controls,
        parameters=sol.parameters,
        iterations=sol.iterations,
        cost=np.array(sol.cost)[0][0],
        # detailed_cost=sol.detailed_cost,
        # real_time_to_optimize=sol.real_time_to_optimize,
        param_scaling=[nlp.parameters.scaling for nlp in ocp.nlp],
        phase_time=sol.phase_time,
        Time=sol.time,
        Force_Values=F_array,

    )

    # Save the data to a .pckl file
    with open(
            "/home/alpha/Desktop/Nov. 14/Pressed_with_Thorax_AllObjRemoved.pckl","wb") as file:
        pickle.dump(data, file)


#  Important Note on Bioptim Version 3.1.0 Compatibility
    # In Bioptim version 3.1.0, some of the following features or functions may not work as expected:
    # - sol.print_cost() may not function properly to display the cost details of the solution.
    # - ocp.print(to_console=False, to_graph=False) may not provide the expected detailed information
    # - sol.graphs(show_bounds=True) may not be available for generating and displaying online graphs related to the solution.
    # - sol.animate() may not be functional for visualizing the movement in an animation.


# Displaying information about the optimization results

    # print("Tesults saved")
    # print("Temps de resolution : ", time.time() - tic, "s")
    #
    # sol.print_cost()
    # ocp.print(to_console=False, to_graph=False)
    # # sol.graphs(show_bounds=True)
    # sol.animate(show_floor=False, show_global_center_of_mass=False, show_segments_center_of_mass=False, show_global_ref_frame=True, show_local_ref_frame=False, show_markers=False, n_frames=250,)


if __name__ == "__main__":
    main()