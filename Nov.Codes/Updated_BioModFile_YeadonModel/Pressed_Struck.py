from casadi import MX, acos, dot, pi, Function
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
# TO DO : Add the joint indices in the biomechanical model.

# Description of movement phases:
# Phase 0: Preparation - Getting the fingers in position.
# Phase 1: Key Descend - The downward motion of the fingers pressing the keys.
# Phase 2: Key Bed - The phase where the keys are fully pressed and meet the bottom.
# Phase 3: Key Release (Upward) - Releasing the keys and moving the hand upward.
# Phase 4: Return to Neutral (Downward) - Bringing the fingers back to a neutral position, ready for the next action.


def minimize_difference(controllers: list[PenaltyController, PenaltyController]):
    pre, post = controllers
    return pre.controls.cx_end - post.controls.cx

def prepare_ocp(allDOF, pressed, ode_solver) -> OptimalControlProgram:
    if allDOF:
        biorbd_model_path = "./With.bioMod"
        dof_wrist_finger = [7, 8]
        all_dof_except_wrist_finger = [0, 1, 2, 3, 4, 5, 6]

    else:
        biorbd_model_path = "./Without.bioMod"
        dof_wrist_finger = [4, 5]
        all_dof_except_wrist_finger = [0, 1, 2, 3]

    all_phases = [0, 1, 2, 3, 4]

    biorbd_model = (
        BiorbdModel(biorbd_model_path),
        BiorbdModel(biorbd_model_path),
        BiorbdModel(biorbd_model_path),
        BiorbdModel(biorbd_model_path),
        BiorbdModel(biorbd_model_path),
    )

    if pressed:
        # Velocity profile found thanks to the motion capture datas.
        vel_push_array = [0.0, -0.114, -0.181, -0.270, -0.347, -0.291, -0.100, ]
        n_shooting = (30, 7, 9, 10, 10)
        phase_time = (0.3, 0.044, 0.051, 0.15, 0.15)

    else:
        vel_push_array = [-0.698, -0.475, -0.368, -0.357, -0.368, -0.278, ]
        n_shooting = (30, 6, 9, 10, 10)
        phase_time = (0.3, 0.027, 0.058, 0.15, 0.15)

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
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", phase=phase, weight=100, index=dof_wrist_finger
        )
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", phase=phase, weight=0.0001, index=dof_wrist_finger #all_dof_except_wrist_finger
        )

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
            min_bound=-0.01, max_bound=0.01,
            target=vel_push_array[node],
        )

    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        phase=1, node=Node.END,
        first_marker="finger_marker",
        second_marker="key1_base",
    )

    ForceProfile = [30, 28, 24, 20, 16, 12, 8, 4, 0]

    for node in range(n_shooting[2]):
        for idx in [0, 1]:
            constraints.add(
                ConstraintFcn.TRACK_CONTACT_FORCES,
                phase=2, node=node,
                contact_index=idx,
                min_bound=-ForceProfile[node] / 3, max_bound=ForceProfile[node] / 3,
                quadratic=False,
            )

        constraints.add(
            ConstraintFcn.TRACK_CONTACT_FORCES,
            phase=2, node=node,
            contact_index=2,
            target=ForceProfile[node],
            quadratic=False,
            # min_bound=-0.1, max_bound=0.1,
        )

    # constraints.add(
    #     ConstraintFcn.SUPERIMPOSE_MARKERS,
    #     phase=3, node=Node.END,
    #     first_marker="MCP_marker",
    #     second_marker="key1_above",
    # )

    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        phase=4, node=Node.END,
        first_marker="finger_marker",
        second_marker="Key1_Top",
    )

    for phase in all_phases:
        # To block ulna rotation before the key pressing.
        constraints.add(
            ConstraintFcn.TRACK_STATE,
            phase=phase, node=Node.ALL,
            key="qdot",
            index=all_dof_except_wrist_finger[-2],  # prosupination
            min_bound=-1, max_bound=1,
            quadratic=False,
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

            x_bounds[phase]["qdot"].min[[0, 1, 2, 3, 4, 5], :] = -3
            x_bounds[phase]["qdot"].max[[0, 1, 2, 3, 4, 5], :] = 3

            x_bounds[phase]["qdot"].min[[6], :] = -4
            x_bounds[phase]["qdot"].max[[6], :] = 4

            x_bounds[phase]["qdot"].min[[7, 8], :] = -15
            x_bounds[phase]["qdot"].max[[7, 8], :] = 15

        else:

            x_bounds[phase]["qdot"].min[[0, 1, 2], :] = -3
            x_bounds[phase]["qdot"].max[[0, 1, 2], :] = 3

            x_bounds[phase]["qdot"].min[[3], :] = -4
            x_bounds[phase]["qdot"].max[[3], :] = 4

            x_bounds[phase]["qdot"].min[[4, 5], :] = -15
            x_bounds[phase]["qdot"].max[[4, 5], :] = 15

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
    """
    Defines a multiphase ocp and animate the results
    """
    print(os.getcwd())
    polynomial_degree = 4
    allDOF = False #True means all DOF, False means no thorax
    pressed = True #False means Struck
    dirName = "/home/alpha/pianoptim/PianOptim/Nov.Codes/Updated_BioModFile_YeadonModel/Results/"

    if allDOF:
        saveName = dirName + ("Pressed" if pressed else "Struck") + "_with_Thorax.pckl"
        nq = 9
    else:
        saveName = dirName + ("Pressed" if pressed else "Struck") +"_without_Thorax.pckl"
        nq = 6

    ocp = prepare_ocp(allDOF=allDOF, pressed=pressed, ode_solver=OdeSolver.COLLOCATION(polynomial_degree=polynomial_degree))

    ocp.add_plot_penalty(CostType.ALL)

    # # --- Solve the program --- # #
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
        # detailed_cost=sol.detailed_cost,
        # real_time_to_optimize=sol.real_time_to_optimize,
        param_scaling=[nlp.parameters.scaling for nlp in ocp.nlp],
        phase_time=sol.phase_time,
        Time=sol.time,
    )

    with open(saveName, "wb") as file:
        pickle.dump(data, file)

if __name__ == "__main__":
    main()
