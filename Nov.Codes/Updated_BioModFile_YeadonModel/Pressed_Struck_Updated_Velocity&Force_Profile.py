from casadi import MX, acos, dot, pi, fmin, fmax
import time
import numpy as np
import biorbd_casadi as biorbd
import pickle
import os

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

# Joint indices in the biomechanical model:
#0. Pelvic Tilt: Anterior (+) Tilt / Posterior (-) Tilt
#1. Thoracic Flexion/Extension: Flexion (+) / Extension (-)
#2. Thoracic Rotation: Left (+) / Right (-)
#3. Upper Thoracic (Rib Cage) Flexion/Extension: Flexion (+) / Extension (-)
#4. Upper Thoracic (Rib Cage) Rotation: Left (+) / Right (-)
#5|0 Shoulder Flexion/Extension: Flexion (-) / Extension (+)
#6|1 Shoulder Abduction/Adduction: Abduction (+) / Adduction (-)
#7|2 Shoulder Internal/External Rotation: Internal (+) / External (-)
#8|3 Elbow Flexion/Extension: Flexion (-) / Extension (+)
#9|4 Forearm Pronation/Supination: Pronation (Left +) / Supination (Right -)
#10|5 Wrist Flexion/Extension: Flexion (-) / Extension (+)
#11|6 Metacarpophalangeal (MCP) Flexion/Extension: Flexion (-) / Extension (+)

# Description of movement phases:
# Phase 0: Preparation - Getting the fingers in position.
# Phase 1: Key Descend - The downward motion of the fingers pressing the keys.
# Phase 2: Key Bed - The phase where the keys are fully pressed and meet the bottom.
# Phase 3: Key Release (Upward) - Releasing the keys and moving the hand upward.
# Phase 4: Return to Neutral (Downward) - Bringing the fingers back to a neutral position, ready for the next action.


def minimize_difference(controllers: list[PenaltyController, PenaltyController]):
    pre, post = controllers
    return pre.controls.cx_end - post.controls.cx

def custom_func_right_relative_To_PrincipalTracker(controller: PenaltyController) -> MX:
    finger_marker_idx = biorbd.marker_index(controller.model.model, "finger_marker")
    markers = controller.mx_to_cx("markers", controller.model.markers, controller.states["q"])
    finger_marker = markers[:, finger_marker_idx]

    finger_marker_5_idx = biorbd.marker_index(controller.model.model, "finger_marker_RightPinky")
    markers_RightPinky = controller.mx_to_cx("markers_RightPinky", controller.model.markers, controller.states["q"])
    finger_marker_RightPinky = markers_RightPinky[:, finger_marker_5_idx]

    markers_diff = finger_marker[2] - finger_marker_RightPinky[2]

    return markers_diff

def custom_func_trackPrincipalAndPinkyFingerAboveKey(controller: PenaltyController, marker: str) -> MX:
    biorbd_model = controller.model
    finger_marker_idx = biorbd.marker_index(biorbd_model.model, marker)
    markers = controller.mx_to_cx("markers", biorbd_model.markers, controller.states["q"])
    finger_marker = markers[:, finger_marker_idx]

    markers_diff_key = finger_marker[2] - (0.10)

    return markers_diff_key

def prepare_ocp(allDOF, pressed, ode_solver) -> OptimalControlProgram:
    if allDOF:
        biorbd_model_path = "./With.bioMod"
        dof_wrist_finger = [10, 11]
        wrist= [10]
        finger= [11]
        Shoulder= [5]
        all_dof_except_wrist_finger=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    else:
        biorbd_model_path = "./Without.bioMod"
        dof_wrist_finger = [5, 6]
        all_dof_except_wrist_finger = [0, 1, 2, 3, 4]
        Shoulder= [0]
        wrist= [5]
        finger= [6]

    all_phases = [0, 1, 2, 3, 4]

    biorbd_model = (
        BiorbdModel(biorbd_model_path),
        BiorbdModel(biorbd_model_path),
        BiorbdModel(biorbd_model_path),
        BiorbdModel(biorbd_model_path),
        BiorbdModel(biorbd_model_path),
    )

    if pressed:
        # Profiles found thanks to the motion capture datas.
        # vel_push_array = [0.0, -0.756, -1.120, -1.210, -0.873, -0.450, -0.058,]
        vel_push_array = [0.0, -0.114, -0.181, -0.270, -0.347, -0.291, -0.100, ]
        n_shooting = (30, 7, 9, 10, 10)
        phase_time = (0.3, 0.024, 0.0605, 0.15, 0.15)
        Force_Profile = [57, 50, 43, 35, 26, 17, 8, 4, 0]


    else:
        # vel_push_array = [-1.444, -1.343, -1.052, -0.252, -0.196, -0.014,]
        vel_push_array = [-0.698, -0.475, -0.368, -0.357, -0.368, -0.278, ]
        n_shooting = (30, 6, 9, 10, 10)
        phase_time = (0.3, 0.020, 0.0501, 0.15, 0.15)
        Force_Profile = [54, 47, 41, 35, 28, 18, 10, 4, 0]

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=0)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=1)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True, phase=2)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=3)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=4)

    # Objectives
    # Minimize Torques
    objective_functions = ObjectiveList()
    for phase in all_phases:

        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", phase=phase, weight=1, index=all_dof_except_wrist_finger
        )
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_POWER, key_control="tau", phase=phase, weight=200, index=dof_wrist_finger
        )
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", phase=phase, weight=0.0001, index=dof_wrist_finger #all_dof_except_wrist_finger
        )
    # #
    # for phase in [0, 1]:
    #     objective_functions.add(
    #             ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", phase=phase, weight=10, index=wrist
    #     )

    # Constraints
    constraints = ConstraintList()
    if pressed:
        constraints.add(
            ConstraintFcn.SUPERIMPOSE_MARKERS,
            phase=0, node=Node.ALL,
            first_marker="finger_marker",
            second_marker="Key1_Top",
        )

    else:
        constraints.add(
            ConstraintFcn.SUPERIMPOSE_MARKERS,
            phase=0, node=Node.START,
            first_marker="finger_marker",
            second_marker="Key1_Top",
        )
        constraints.add(
            ConstraintFcn.SUPERIMPOSE_MARKERS,
            phase=0, node=Node.END,
            first_marker="finger_marker",
            second_marker="Key1_Top",
        )

    #Stricly constrained velocity at start
    constraints.add(
        ConstraintFcn.TRACK_MARKERS_VELOCITY,
        phase=1, node=Node.START,
        marker_index=0,
        target=vel_push_array[0]
    )

    for node in range(1, len(vel_push_array)):
        constraints.add(
            ConstraintFcn.TRACK_MARKERS_VELOCITY,
            phase=1, node=node,
            marker_index=0, axes=Axis.Z,
            min_bound=-0.93, max_bound=0.93,
            target=vel_push_array[node],
        )

    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        phase=1, node=Node.END,
        first_marker="finger_marker",
        second_marker="key1_base",
    )

    # if pressed==False:
    #
    #     if allDOF==False:
    #
    #         # finger_marker
    #         x_base = -0.16104042053222656
    #         y_base = -0.5356114196777344 + 0.025
    #
    #         # Calculating min_bound and max_bound with adjustments
    #         min_bound = np.array([x_base - 0.05, y_base - 0.05])
    #         max_bound = np.array([x_base + 0.05, y_base + 0.05])
    #
    #         constraints.add(
    #             ConstraintFcn.TRACK_MARKERS,
    #             phase=0,
    #             node=Node.INTERMEDIATES,
    #             marker_index=0,
    #             axes=[Axis.X, Axis.Y],
    #             min_bound=np.array([-0.22, -0.56]),
    #             max_bound=np.array([-0.11, -0.46]),
    #         )

    for node in range(n_shooting[2]):
        for idx in [0, 1]:
            constraints.add(
                ConstraintFcn.TRACK_CONTACT_FORCES,
                phase=2, node=node,
                contact_index=idx,
                min_bound=-Force_Profile[node] / 3, max_bound=Force_Profile[node] / 3,
                quadratic=False,
            )

        constraints.add(
            ConstraintFcn.TRACK_CONTACT_FORCES,
            phase=2, node=node,
            contact_index=2,
            target=Force_Profile[node],
            quadratic=False,
            # min_bound=-0.1, max_bound=0.1,
        )

    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        phase=3, node=Node.END,
        first_marker="MCP_marker",
        second_marker="key1_above",
    )

    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        phase=4, node=Node.END,
        first_marker="finger_marker",
        second_marker="Key1_Top",
    )

    for phase in all_phases:
        # # To block ulna rotation before the key pressing.
        # constraints.add(
        #     ConstraintFcn.TRACK_STATE,
        #     phase=phase, node=Node.ALL,
        #     key="qdot",
        #     index=all_dof_except_wrist_finger[-2],  # prosupination
        #     min_bound=-1, max_bound=1,
        #     quadratic=False,
        # )

        # To keep the pinky finger on the right of the principal finger.
        constraints.add(
            custom_func_right_relative_To_PrincipalTracker,
            phase=phase, node=Node.ALL,
            min_bound=-0.0001,
            max_bound=0.006,
        )

     # To keep the index and the small finger above the bed key.
        constraints.add(
            custom_func_trackPrincipalAndPinkyFingerAboveKey,
            phase=phase, node=Node.ALL,
            marker="finger_marker",
            min_bound=0,
            max_bound=np.inf,
        )

    phase_transition = PhaseTransitionList()
    phase_transition.add(PhaseTransitionFcn.IMPACT, phase_pre_idx=1)

    # States: bounds and Initial guess
    x_init = InitialGuessList()
    x_bounds = BoundsList()

    for phase in all_phases:
        x_bounds.add("q", bounds=biorbd_model[phase].bounds_from_ranges("q"), phase=phase)
        x_bounds.add("qdot", bounds=biorbd_model[phase].bounds_from_ranges("qdot"), phase=phase)

        x_init.add("q", [0] * biorbd_model[phase].nb_q, phase=phase)
        x_init.add("qdot", [0] * biorbd_model[phase].nb_q, phase=phase)


        # This section targets the angular velocity bounds for various degrees of freedom (DOFs) in a biomechanical model,
        # based on experimental datasets.
        # These bounds are crucial for accurately simulating the kinematics of pianist movement and are aligned with
        # the specified bounds for each joint: +/- 3 rad/s for Pelvis, Thorax, and Shoulder, +/- 4 or 5 rad/s for the Elbow,
        # and +/- 15 rad/s for the Wrist and Finger.

        if allDOF:

            x_bounds[phase]["qdot"].min[[0, 1, 2, 3, 4, 5, 6, 7], :] = -3
            x_bounds[phase]["qdot"].max[[0, 1, 2, 3, 4, 5, 6, 7], :] = 3

            x_bounds[phase]["qdot"].min[[8, 9], :] = -4
            x_bounds[phase]["qdot"].max[[8, 9], :] = 4

            x_bounds[phase]["qdot"].min[[10, 11], :] = -15
            x_bounds[phase]["qdot"].max[[10, 11], :] = 15

        else:

            x_bounds[phase]["qdot"].min[[0, 1, 2], :] = -3
            x_bounds[phase]["qdot"].max[[0, 1, 2], :] = 3

            x_bounds[phase]["qdot"].min[[3, 4], :] = -4
            x_bounds[phase]["qdot"].max[[3, 4], :] = 4

            x_bounds[phase]["qdot"].min[[5, 6], :] = -15
            x_bounds[phase]["qdot"].max[[5, 6], :] = 15

    # Set the initial node value for the pelvis in the first phase to 0.0.
    # The first 0 in x_bounds[0] specifies the phase,
    # the [[0], 0] targets the pelvis (index 0) at the initial node.

    if allDOF:

        x_bounds[0]["q"][[0], 0] = 0
        x_bounds[0]["q"][[1], 0] = 0

        x_bounds[4]["q"][[0], 2] = 0
        x_bounds[4]["q"][[1], 2] = 0

    # Define control path constraint and initial guess
    tau_min, tau_max, tau_init = -100, 100, 0
    u_bounds = BoundsList()
    u_init = InitialGuessList()
    for phase in all_phases:
        u_bounds.add(
            "tau",
            min_bound=[tau_min] * biorbd_model[phase].nb_tau,
            max_bound=[tau_max] * biorbd_model[phase].nb_tau,
            phase=phase,
        )
        u_init.add("tau", [tau_init] * biorbd_model[phase].nb_tau, phase=phase)

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

    print("Welcome to the Piano Optimization Program!")
    mode = input("Do you want to generate all conditions together (enter 'all') or just one by one (enter 'one')? ")

    polynomial_degree = 4
    baseDirName = "/home/alpha/pianoptim/PianOptim/Nov.Codes/Updated_BioModFile_YeadonModel/Results/Felipe_25March"
    resultsDirName = input("Please enter the folder name e.g. Version_1: ")



    dirName = os.path.join(baseDirName, resultsDirName)

    # Create the directory if it doesn't exist
    if not os.path.exists(dirName):
        os.makedirs(dirName)

    conditions = [
        {"allDOF": True, "pressed": True, "description": "Pressed with Thorax"},
        {"allDOF": True, "pressed": False, "description": "Struck with Thorax"},
        {"allDOF": False, "pressed": True, "description": "Pressed without Thorax"},
        {"allDOF": False, "pressed": False, "description": "Struck without Thorax"},
    ]

    if mode.lower() == 'all':
        for condition in conditions:
            process_condition(condition, polynomial_degree, dirName)
    elif mode.lower() == 'one':
        print("Select a condition to generate:")
        for i, condition in enumerate(conditions, start=1):
            print(f"{i}: {condition['description']}")
        choice = int(input("Enter the number of the condition you want to generate: ")) - 1
        if 0 <= choice < len(conditions):
            process_condition(conditions[choice], polynomial_degree, dirName)
        else:
            print("Invalid choice. Please restart the program and select a valid number.")
    else:
        print("Invalid mode selected. Please restart the program and enter 'all' or 'one'.")

def process_condition(condition, polynomial_degree, dirName):
    allDOF = condition["allDOF"]
    pressed = condition["pressed"]

    saveName = os.path.join(dirName, f"{'Pressed' if pressed else 'Struck'}_{'with' if allDOF else 'without'}_Thorax.pckl")

    # Assume the prepare_ocp function and related setup are defined elsewhere in the script
    ocp = prepare_ocp(allDOF=allDOF, pressed=pressed, ode_solver=OdeSolver.COLLOCATION(polynomial_degree=polynomial_degree))
    ocp.add_plot_penalty(CostType.ALL)

    solv = Solver.IPOPT(show_online_optim=False)
    solv.set_maximum_iterations(1000)
    solv.set_linear_solver("ma57")

    sol = ocp.solve(solv)

    data = dict(
        states=sol.states,
        states_no_intermediate=sol.states_no_intermediate,
        controls=sol.controls,
        parameters=sol.parameters,
        iterations=sol.iterations,
        cost=np.array(sol.cost)[0][0],
        param_scaling=[nlp.parameters.scaling for nlp in ocp.nlp],
        phase_time=sol.phase_time,
        Time=sol.time,
    )

    with open(saveName, "wb") as file:
        pickle.dump(data, file)
    print(f"Saved results to {saveName}")

if __name__ == "__main__":
    main()
