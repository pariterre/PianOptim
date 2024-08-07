from enum import Enum, auto

from bioptim import (
    ObjectiveList,
    DynamicsList,
    ConstraintFcn,
    BoundsList,
    InitialGuessList,
    CostType,
    Node,
    OptimalControlProgram,
    DynamicsFcn,
    ObjectiveFcn,
    ConstraintList,
    OdeSolver,
    Solver,
    Axis,
    ShowOnlineType,
)
import numpy as np

from utils.pianist import Pianist
from utils.dynamics import PianistDyanmics


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


class ZeroPosition(Enum):
    Q = auto()
    MARKER = auto()


def prepare_ocp(
    model_path: str,
    n_phases: int,
    n_shootings: tuple[int, ...],
    phase_times: tuple[float, ...],
    block_trunk: bool,
    zero_position: ZeroPosition,
    ode_solver: OdeSolver,
) -> OptimalControlProgram:

    trunk_dof = range(6)

    prep_phase = 0
    down_phase = 1
    up_phase = 2
    end_phase = 3

    tau_min, tau_max = -40, 40

    # Control the inputs
    if len(n_shootings) < n_phases:
        raise ValueError(
            f"The number of shooting points ({len(n_shootings)}) must be greater than or equal to "
            f"the number of phases ({n_phases})"
        )
    n_shootings = n_shootings[:n_phases]

    if len(phase_times) < n_phases:
        raise ValueError(
            f"The number of phase times ({len(phase_times)}) must be greater than or equal to "
            f"the number of phases ({n_phases})"
        )
    phase_times = phase_times[:n_phases]

    pianist_models: list[Pianist] = []
    dynamics = DynamicsList()
    x_bounds = BoundsList()
    x_init = InitialGuessList()
    u_bounds = BoundsList()
    objective_functions = ObjectiveList()
    constraints = ConstraintList()

    # Load and constraints the dynamic model
    for phase in range(n_phases):
        model = Pianist(model_path)
        pianist_models.append(model)

        if phase == down_phase:
            dynamics.add(PianistDyanmics.configure_forward_dynamics_with_external_forces, phase=phase)
        elif phase == up_phase:
            dynamics.add(PianistDyanmics.configure_forward_dynamics_with_external_forces, phase=phase)
        else:
            dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=phase)

        x_bounds.add("q", bounds=model.bounds_from_ranges("q"), phase=phase)
        x_init.add("q", model.q_for_hand_over_keyboard, phase=phase)
        x_bounds.add("qdot", bounds=model.bounds_from_ranges("qdot"), phase=phase)
        u_bounds.add("tau", min_bound=[tau_min] * model.nb_tau, max_bound=[tau_max] * model.nb_tau, phase=phase)

        # Minimization to convexify the problem
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", phase=phase, weight=0.01)

    # The first and last frames are at rest
    x_bounds[0]["qdot"][:, 0] = 0
    x_bounds[-1]["qdot"][:, -1] = 0

    # The trunk should not move if requested
    if block_trunk:
        for phase in range(n_phases):
            x_bounds[phase]["qdot"][trunk_dof, :] = 0

    # Start with the finger on the key
    if zero_position == ZeroPosition.MARKER:
        constraints.add(
            ConstraintFcn.SUPERIMPOSE_MARKERS,
            phase=prep_phase,
            node=Node.START,
            first_marker="finger_marker",
            second_marker="Key1_Top",
            target=np.array([0, 0, -0.1]),
        )
    elif zero_position == ZeroPosition.Q:
        x_bounds[prep_phase]["q"][:, 0] = model.q_for_hand_over_keyboard
    else:
        raise ValueError("Invalid initial position")

    # The finger should be at the top of the key at the end of the first phase and not moving in the horizontal plane
    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        phase=prep_phase,
        node=Node.END,
        first_marker="finger_marker",
        second_marker="Key1_Top",
    )
    constraints.add(
        ConstraintFcn.TRACK_MARKERS_VELOCITY,
        phase=prep_phase,
        node=Node.END,
        axes=[Axis.X, Axis.Y],
        marker_index="finger_marker",
    )

    # The key should be at its bottom at the end of the press phase
    if n_phases > down_phase:
        objective_functions.add(
            ObjectiveFcn.Mayer.TRACK_MARKERS,
            phase=down_phase,
            node=Node.END,
            marker_index="finger_marker",
            quadratic=False,
            axes=Axis.Z,
            weight=1000,
        )

    # The key should be fully lifted at the end of the up phase (press_phase
    if n_phases > up_phase:
        # Lift the finger from the key bed up to the top of the key
        constraints.add(
            ConstraintFcn.SUPERIMPOSE_MARKERS,
            phase=up_phase,
            node=Node.END,
            axes=Axis.Z,
            first_marker="finger_marker",
            second_marker="Key1_Top",
        )

    if n_phases > end_phase:
        if zero_position == ZeroPosition.MARKER:
            constraints.add(
                ConstraintFcn.SUPERIMPOSE_MARKERS,
                phase=end_phase,
                node=Node.END,
                first_marker="finger_marker",
                second_marker="Key1_Top",
                target=np.array([0, 0, -0.1]),
            )
        elif zero_position == ZeroPosition.Q:
            x_bounds[end_phase]["q"][:, -1] = model.q_for_hand_over_keyboard
        else:
            raise ValueError("Invalid final position")

    # Add time minimization
    for phase in [prep_phase, down_phase, up_phase, end_phase]:
        objective_functions.add(
            ObjectiveFcn.Mayer.MINIMIZE_TIME, phase=phase, min_bound=0, max_bound=phase_times[phase], weight=1000
        )

    # Prepare the optimal control program
    return OptimalControlProgram(
        bio_model=pianist_models,
        dynamics=dynamics,
        n_shooting=n_shootings,
        phase_time=phase_times,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        objective_functions=objective_functions,
        constraints=constraints,
        ode_solver=ode_solver,
        use_sx=False,
        n_threads=6,
    )


def main():
    model_path = "./models/pianist.bioMod"
    block_trunk = False
    n_phases = 4
    n_shooting = (100, 50, 50, 100)
    phase_time = (0.1, 0.05, 0.05, 0.1)
    zero_position = ZeroPosition.MARKER
    ode_solver = OdeSolver.RK4(n_integration_steps=1)

    ocp = prepare_ocp(
        model_path=model_path,
        n_phases=n_phases,
        block_trunk=block_trunk,
        n_shootings=n_shooting,
        phase_times=phase_time,
        zero_position=zero_position,
        ode_solver=ode_solver,
    )
    ocp.add_plot_penalty(CostType.ALL)

    solv = Solver.IPOPT(
        show_online_optim=True,
        show_options={"type": ShowOnlineType.SERVER},
    )
    solv.set_maximum_iterations(500)  # TODO This should not be necessary
    # solv.set_linear_solver("ma57")

    sol = ocp.solve(solv)
    # sol.graphs()
    sol.animate()


if __name__ == "__main__":
    main()
