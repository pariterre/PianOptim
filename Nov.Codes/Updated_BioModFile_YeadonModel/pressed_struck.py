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

def minimize_difference(controllers: list[PenaltyController, PenaltyController]):
    pre, post = controllers
    return pre.controls.cx_end - post.controls.cx

def prepare_ocp(allDOF, pressed, ode_solver) -> OptimalControlProgram:
    if allDOF:
        biorbd_model_path = "./With.bioMod"
        dof_wrist_finger = [9, 10]
        all_dof_except_wrist_finger = [0, 1, 2, 3, 4, 5, 6, 7, 8]

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
        n_shooting = (30, 7, 9, 10, 10)
        phase_time = (0.3, 0.044, 0.051, 0.15, 0.15)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=0)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=1)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=2)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=3)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=4)

    # Objectives
    # Minimize Torques
    objective_functions = ObjectiveList()
    # for phase in all_phases:
    #     objective_functions.add(
    #         ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", phase=phase, weight=1, index=all_dof_except_wrist_finger
    #     )
    #     objective_functions.add(
    #         ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", phase=phase, weight=100, index=dof_wrist_finger
    #     )
    #     objective_functions.add(
    #         ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", phase=phase, weight=0.0001, index=dof_wrist_finger #all_dof_except_wrist_finger
    #     )

    # Constraints
    constraints = ConstraintList()
    for phase in all_phases:
        constraints.add(
            ConstraintFcn.SUPERIMPOSE_MARKERS,
            phase=phase, node=Node.ALL,
            first_marker="finger_marker",
            second_marker="Key1_Top",
        )
        # To block ulna rotation before the key pressing.
        constraints.add(
            ConstraintFcn.TRACK_STATE,
            phase=phase, node=Node.ALL,
            key="q",
            index=all_dof_except_wrist_finger[-2],  # prosupination
            min_bound=-1, max_bound=1,
            quadratic=False,
        )

    # States: bounds and Initial guess
    x_init = InitialGuessList()
    x_bounds = BoundsList()

    for phase in all_phases:
        x_bounds.add("q", bounds=biorbd_model[phase].bounds_from_ranges("q"), phase=phase)
        x_bounds.add("qdot", bounds=biorbd_model[phase].bounds_from_ranges("qdot"), phase=phase)

        x_init.add("q", [0] * biorbd_model[phase].nb_q, phase=phase)
        x_init.add("qdot", [0] * biorbd_model[phase].nb_q, phase=phase)


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
        ode_solver=ode_solver,
    )


def main():
    """
    Defines a multiphase ocp and animate the results
    """
    print(os.getcwd())
    polynomial_degree = 4
    allDOF = True
    pressed = True  #False means Struck
    dirName = "/home/alpha/pianoptim/PianOptim/Nov.Codes/Updated_BioModFile_YeadonModel/Results/"

    if allDOF:
        saveName = dirName + ("Pressed" if pressed else "Struck") + "_with_Thorax_dongnothing.pckl"
        nq = 11

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
