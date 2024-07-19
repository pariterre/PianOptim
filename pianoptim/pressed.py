import numpy as np

from bioptim import (
    BiorbdModel,
    ObjectiveList,
    PhaseTransitionFcn,
    DynamicsList,
    ConstraintFcn,
    BoundsList,
    CostType,
    PhaseTransitionList,
    Node,
    OptimalControlProgram,
    DynamicsFcn,
    ObjectiveFcn,
    ConstraintList,
    OdeSolver,
    Solver,
    Axis,
)

# Joint indices in the biomechanical model:
# 0. Pelvic Tilt: Anterior (+) Tilt / Posterior (-) Tilt
# 1. Thoracic Flexion/Extension: Flexion (+) / Extension (-)
# 2. Thoracic Rotation: Left (+) / Right (-)
# 3. Upper Thoracic (Rib Cage) Flexion/Extension: Flexion (+) / Extension (-)
# 4. Upper Thoracic (Rib Cage) Rotation: Left (+) / Right (-)
# 5|0 Shoulder Flexion/Extension: Flexion (-) / Extension (+)
# 6|1 Shoulder Abduction/Adduction: Abduction (+) / Adduction (-)
# 7|2 Shoulder Internal/External Rotation: Internal (+) / External (-)
# 8|3 Elbow Flexion/Extension: Flexion (-) / Extension (+)
# 9|4 Forearm Pronation/Supination: Pronation (Left +) / Supination (Right -)
# 10|5 Wrist Flexion/Extension: Flexion (-) / Extension (+)
# 11|6 Metacarpophalangeal (MCP) Flexion/Extension: Flexion (-) / Extension (+)

# Description of movement phases:
# Phase 0: Preparation - Getting the fingers in position.
# Phase 1: Key Descend - The downward motion of the fingers pressing the keys.
# Phase 2: Key Bed - The phase where the keys are fully pressed and meet the bottom.
# Phase 3: Key Release (Upward) - Releasing the keys and moving the hand upward.
# Phase 4: Return to Neutral (Downward) - Bringing the fingers back to a neutral position, ready for the next action.


def prepare_ocp(
    model_path: str,
    n_shootings: tuple[int, ...],
    phase_times: tuple[float, ...],
    block_trunk: bool,
    ode_solver: OdeSolver,
) -> OptimalControlProgram:
    trunk_dof = range(6)

    n_phases = 2
    press_phase = 1

    # Control the inputs
    if len(n_shootings) != n_phases:
        raise ValueError(
            f"The number of phases ({n_phases}) and "
            f"the number of shooting points ({len(n_shootings)}) must be the same"
        )
    if len(phase_times) != n_phases:
        raise ValueError(
            f"The number of phases ({n_phases}) and " f"the number of phase times ({len(phase_times)}) must be the same"
        )

    # Load the dynamic model
    biorbd_models = [BiorbdModel(model_path) for _ in range(n_phases)]

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=0)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True, phase=press_phase)

    # Declare the penalty functions so it produces a piano press movement
    objective_functions = ObjectiveList()
    constraints = ConstraintList()

    # Constraints

    # Rest the finger on the key for the whole preparation phase
    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        phase=0,
        node=Node.START,
        first_marker="finger_marker",
        second_marker="Key1_Top",
        target=np.array([0, 0, -0.1]),
    )
    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        phase=0,
        node=Node.END,
        first_marker="finger_marker",
        second_marker="Key1_Top",
    )

    # # The key should be as the bottom a mid of the press phase
    # # TODO: this assumption may be wrong, separate down stroke and up stroke?
    # objective_functions.add(
    #     ObjectiveFcn.Mayer.TRACK_MARKERS,
    #     phase=press_phase,
    #     node=Node.MID,
    #     marker_index="finger_marker",
    #     target=np.array([[-np.inf]]),
    #     axes=Axis.Z,
    # )

    # Minimize the generalized forces to convexify the problem
    for phase in range(n_phases):
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_POWER, key_control="tau", phase=phase, weight=1)

    phase_transition = PhaseTransitionList()
    phase_transition.add(PhaseTransitionFcn.IMPACT, phase_pre_idx=press_phase - 1)

    # Declare the constraints on the states and controls
    x_bounds = BoundsList()
    u_bounds = BoundsList()
    tau_min, tau_max = -100, 100

    for phase in range(n_phases):
        model = biorbd_models[phase]

        x_bounds.add("q", bounds=model.bounds_from_ranges("q"), phase=phase)
        x_bounds.add("qdot", bounds=model.bounds_from_ranges("qdot"), phase=phase)

        if block_trunk:
            x_bounds[phase]["q"][trunk_dof, :] = 0
            x_bounds[phase]["qdot"][trunk_dof, :] = 0

        u_bounds.add("tau", min_bound=[tau_min] * model.nb_tau, max_bound=[tau_max] * model.nb_tau, phase=phase)

    # TODO Fix the first frame to a known position

    # Prepare the optimal control program
    return OptimalControlProgram(
        bio_model=biorbd_models,
        dynamics=dynamics,
        n_shooting=n_shootings,
        phase_time=phase_times,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        phase_transitions=phase_transition,
        ode_solver=ode_solver,
        use_sx=True,
        n_threads=6,
    )


def main():
    model_path = "./models/pianist.bioMod"
    block_trunk = False
    n_shooting = (20, 8)  # , 9, 10, 10)
    phase_time = (0.3, 0.044)  # , 0.051, 0.15, 0.15)
    ode_solver = OdeSolver.COLLOCATION()

    ocp = prepare_ocp(
        model_path=model_path,
        block_trunk=block_trunk,
        n_shootings=n_shooting,
        phase_times=phase_time,
        ode_solver=ode_solver,
    )
    ocp.add_plot_penalty(CostType.ALL)

    solv = Solver.IPOPT(show_online_optim=True)
    solv.set_maximum_iterations(1000)
    # solv.set_linear_solver("ma57")

    sol = ocp.solve(solv)
    sol.animate()


if __name__ == "__main__":
    main()
