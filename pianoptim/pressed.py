from enum import Enum, auto

from casadi import MX, SX, vertcat
import numpy as np

from bioptim import (
    BiorbdModel,
    ObjectiveList,
    PenaltyController,
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
    NonLinearProgram,
    ConfigureProblem,
    DynamicsEvaluation,
    DynamicsFunctions,
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


class ZeroPosition(Enum):
    Q = auto()
    MARKER = auto()


class Pianist(BiorbdModel):
    q_for_hand_over_keyboard = np.array(
        [
            0.01727668684896449,
            -0.010361215519040973,
            -0.031655228839994644,
            -0.005791243213463487,
            -0.03353834947298232,
            -0.4970400708065864,
            0.1921422881472987,
            0.5402828037880408,
            -0.9409433651537537,
            -0.012335229401281565,
            0.03744995282765959,
            0.7450306555722724,
        ]
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, segments_to_apply_external_forces=["RightFingers"], **kwargs)


def configure_forward_dynamics_with_external_forces(
    ocp: OptimalControlProgram, nlp: NonLinearProgram, numerical_data_timeseries=None
):
    """
    Tell the program which variables are states and controls.
    The user is expected to use the ConfigureProblem.configure_xxx functions.

    Parameters
    ----------
    ocp: OptimalControlProgram
        A reference to the ocp
    nlp: NonLinearProgram
        A reference to the phase
    """

    ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)
    ConfigureProblem.configure_dynamics_function(ocp, nlp, forward_dynamics_with_external_forces)


def forward_dynamics_with_external_forces(
    time: MX | SX,
    states: MX | SX,
    controls: MX | SX,
    parameters: MX | SX,
    algebraic_states: MX | SX,
    numerical_timeseries: MX | SX,
    nlp: NonLinearProgram,
) -> DynamicsEvaluation:
    """
    The custom dynamics function that provides the derivative of the states: dxdt = f(x, u, p)

    Parameters
    ----------
    time: MX | SX
        The time of the system
    states: MX | SX
        The state of the system
    controls: MX | SX
        The controls of the system
    parameters: MX | SX
        The parameters acting on the system
    algebraic_states: MX | SX
        The algebraic states of the system
    nlp: NonLinearProgram
        A reference to the phase
    my_additional_factor: int
        An example of an extra parameter sent by the user

    Returns
    -------
    The derivative of the states in the tuple[MX | SX] format
    """

    q = DynamicsFunctions.get(nlp.states["q"], states)
    qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
    tau = DynamicsFunctions.get(nlp.controls["tau"], controls)
    translational_force = compute_key_reaction_forces(nlp.model, q)

    # You can directly call biorbd function (as for ddq) or call bioptim accessor (as for dq)
    dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)
    ddq = nlp.model.forward_dynamics(q, qdot, tau, translational_forces=translational_force)

    # the user has to choose if want to return the explicit dynamics dx/dt = f(x,u,p)
    # as the first argument of DynamicsEvaluation or
    # the implicit dynamics f(x,u,p,xdot)=0 as the second argument
    # which may be useful for IRK or COLLOCATION integrators
    return DynamicsEvaluation(dxdt=vertcat(dq, ddq), defects=None)


def compute_key_reaction_forces(model: Pianist, q: MX | SX):
    """
    Compute the external forces based on the position of the finger. The force is an exponential function based on the
    depth of the finger in the key. The force is null if the finger is over the key, the force slowly increases
    during the course of the movement as the finger presses the key, and finally increases drastically as it reaches
    the bottom of the key.

    Parameters
    ----------
    q: MX | SX
        The generalized coordinates of the system

    Returns
    -------
    The external forces in the tuple[MX | SX] format
    """
    finger = model.marker(q, model.marker_index("finger_marker"))
    key_top = model.marker(q, model.marker_index("Key1_Top"))
    key_bottom = model.marker(q, model.marker_index("key1_base"))

    finger_penetration = key_top[2] - finger[2]
    key_penetration_bed = key_top[2] - key_bottom[2]
    force_at_bed = 1
    force_increate_rate = 5e4

    x = 0  # -10 * (finger[0] - key_top[0])
    y = 0  # -10 * (finger[1] - key_top[1])
    z = force_at_bed * np.exp(force_increate_rate * (finger_penetration - key_penetration_bed))
    px = key_bottom[0]
    py = key_bottom[1]
    pz = key_bottom[2]
    return vertcat(x, y, z, px, py, pz)


def minimum_finger_position(controller: PenaltyController, allowed_key_bed_penetration: float = 0.005) -> MX | SX:
    """
    Compute the position of the finger at the bottom of the key. Allow for a small penetration.
    This constraint is not formally necessary as upward force on the body should prevent from going down too much.
    It however helps the solver to converge by not going to position that produces large forces and get confused by it.

    Parameters
    ----------
    q: MX | SX
        The generalized coordinates of the system

    Returns
    -------
    The position of the finger in the tuple[MX | SX] format
    """
    model = controller.model
    q = controller.q.mx

    finger = model.marker(q, model.marker_index("finger_marker"))
    key_bottom = model.marker(q, model.marker_index("key1_base"))
    penetration = finger[2] - (key_bottom[2] - allowed_key_bed_penetration)

    return controller.mx_to_cx("penetration", penetration, controller.states["q"])


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
    press_phase = 1

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

    # Load the dynamic model
    pianist_models: list[Pianist] = []
    dynamics = DynamicsList()
    for i in range(n_phases):
        pianist_models.append(Pianist(model_path))
        if i == press_phase:
            dynamics.add(configure_forward_dynamics_with_external_forces, phase=i)
        elif i == press_phase + 1:
            dynamics.add(configure_forward_dynamics_with_external_forces, phase=i)
        else:
            dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=i)

    # Declare the penalty functions so it produces a piano press movement
    objective_functions = ObjectiveList()
    constraints = ConstraintList()

    # Constraints
    # Rest the finger on the key for the whole preparation phase
    if zero_position == ZeroPosition.MARKER:
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

    if n_phases > press_phase:
        # The key should be at the key bed when the press phase ends
        objective_functions.add(
            ObjectiveFcn.Mayer.TRACK_MARKERS,
            phase=press_phase,
            node=Node.END,
            marker_index="finger_marker",
            quadratic=False,
            axes=Axis.Z,
            weight=1000,
        )

        # Do not allow the finger to go through the key bed
        constraints.add(
            minimum_finger_position,
            phase=press_phase,
            node=Node.ALL,
            min_bound=0,
            max_bound=np.inf,
            allowed_key_bed_penetration=0.005,
        )

    if n_phases > press_phase + 1:
        # Lift the finger from the key bed up to the top of the key
        constraints.add(
            ConstraintFcn.SUPERIMPOSE_MARKERS,
            phase=press_phase + 1,
            node=Node.END,
            first_marker="finger_marker",
            second_marker="Key1_Top",
        )

        # Do not allow the finger to go through the key bed
        constraints.add(
            minimum_finger_position,
            phase=press_phase + 1,
            node=Node.ALL,
            min_bound=0,
            max_bound=np.inf,
            allowed_key_bed_penetration=0.005,
        )

    if zero_position == ZeroPosition.MARKER and n_phases > press_phase + 2:
        constraints.add(
            ConstraintFcn.SUPERIMPOSE_MARKERS,
            phase=n_phases - 1,
            node=Node.END,
            first_marker="finger_marker",
            second_marker="Key1_Top",
            target=np.array([0, 0, -0.1]),
        )

    # Minimize the generalized forces to convexify the problem
    for phase in range(n_phases):
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_POWER, key_control="tau", phase=phase, weight=1)

    # Declare the constraints on the states and controls
    x_bounds = BoundsList()
    x_init = InitialGuessList()
    u_bounds = BoundsList()
    tau_min, tau_max = -100, 100

    for phase in range(n_phases):
        model = pianist_models[phase]

        x_bounds.add("q", bounds=model.bounds_from_ranges("q"), phase=phase)
        x_bounds.add("qdot", bounds=model.bounds_from_ranges("qdot"), phase=phase)

        if phase == 0:
            if zero_position == ZeroPosition.Q:
                # The first frame is prescribed
                x_bounds[phase]["q"][:, 0] = model.q_for_hand_over_keyboard
            # The first frame is at rest
            x_bounds[phase]["qdot"][:, 0] = 0
        if phase == n_phases - 1:
            if zero_position == ZeroPosition.Q and n_phases > press_phase + 2:
                # The last frame is prescribed if we are simulating the return to neutral phase
                x_bounds[phase]["q"][:, -1] = model.q_for_hand_over_keyboard
            # The last frame is at rest
            x_bounds[phase]["qdot"][:, -1] = 0

        x_init.add("q", model.q_for_hand_over_keyboard, phase=phase)

        if block_trunk:
            x_bounds[phase]["q"][trunk_dof, :] = 0
            x_bounds[phase]["qdot"][trunk_dof, :] = 0

        u_bounds.add("tau", min_bound=[tau_min] * model.nb_tau, max_bound=[tau_max] * model.nb_tau, phase=phase)

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
        use_sx=True,
        n_threads=6,
    )


def main():
    model_path = "./models/pianist.bioMod"
    block_trunk = False
    n_phases = 4
    n_shooting = (40, 30, 30, 40)
    phase_time = (0.1, 0.05, 0.05, 0.1)
    zero_position = ZeroPosition.Q
    ode_solver = OdeSolver.COLLOCATION()

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

    solv = Solver.IPOPT(show_online_optim=False)
    solv.set_maximum_iterations(500)  # TODO This should be be necessary
    # solv.set_linear_solver("ma57")

    sol = ocp.solve(solv)
    sol.graphs()
    sol.animate()


if __name__ == "__main__":
    main()
