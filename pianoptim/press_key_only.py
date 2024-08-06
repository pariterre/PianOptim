from bioptim import (
    ObjectiveList,
    DynamicsList,
    ConstraintFcn,
    BoundsList,
    InitialGuessList,
    CostType,
    Node,
    OptimalControlProgram,
    ObjectiveFcn,
    ConstraintList,
    OdeSolver,
    Solver,
    Axis,
    OnlineOptim,
    PlotType,
)

from utils.pianist import Pianist
from utils.dynamics import PianistDyanmics


def prepare_ocp(
    model_path: str,
    n_shootings: int,
    min_phase_times: float,
    max_phase_times: float,
    ode_solver: OdeSolver,
) -> OptimalControlProgram:

    dynamics = DynamicsList()
    x_bounds = BoundsList()
    x_init = InitialGuessList()
    u_bounds = BoundsList()
    objective_functions = ObjectiveList()
    constraints = ConstraintList()

    # Load and constraints the dynamic model
    model = Pianist(model_path)

    dynamics.add(PianistDyanmics.configure_forward_dynamics_with_external_forces)

    x_bounds.add("q", bounds=model.bounds_from_ranges("q"))
    x_init.add("q", model.q_hand_on_keyboard)

    x_bounds.add("qdot", bounds=model.bounds_from_ranges("qdot"))
    u_bounds.add("tau", bounds=model.joint_torque_bounds)

    # Minimization to convexify the problem
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=0.001)

    # Some of the phases should be done as fast as possible
    if min_phase_times != max_phase_times:
        objective_functions.add(
            ObjectiveFcn.Mayer.MINIMIZE_TIME,
            min_bound=min_phase_times,
            max_bound=max_phase_times,
            weight=100,
        )

    # Start and end with the finger on the key at top position without any velocity
    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        phase=0,
        node=Node.START,
        first_marker="finger_marker",
        second_marker="Key1_Top",
    )
    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        phase=0,
        node=Node.END,
        first_marker="finger_marker",
        second_marker="Key1_Top",
        axes=Axis.Z,
    )
    x_bounds["qdot"][:, 0] = 0

    # The key should be fully pressed as long as possible
    objective_functions.add(
        ObjectiveFcn.Lagrange.TRACK_MARKERS,
        marker_index="finger_marker",
        axes=Axis.Z,
        quadratic=False,
        weight=-1000,
    )

    # Prepare the optimal control program
    ocp = OptimalControlProgram(
        bio_model=model,
        dynamics=dynamics,
        n_shooting=n_shootings,
        phase_time=(min_phase_times + max_phase_times) / 2,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        objective_functions=objective_functions,
        constraints=constraints,
        ode_solver=ode_solver,
        use_sx=False,
        n_threads=1,
    )

    # Add a graph that shows the finger height
    ocp.add_plot(
        "Finger height",
        lambda t0, phases_dt, node_idx, x, u, p, a, d: model.compute_marker_from_dm(x[: model.nb_q], "finger_marker")[
            2, :
        ]
        * 1000,
        plot_type=PlotType.INTEGRATED,
    )
    ocp.add_plot_penalty(CostType.ALL)

    return ocp


def main():
    model_path = "./models/pianist.bioMod"
    n_shooting = 100
    min_phase_time = 0.02
    max_phase_time = 0.02
    ode_solver = OdeSolver.RK4(n_integration_steps=1)

    ocp = prepare_ocp(
        model_path=model_path,
        n_shootings=n_shooting,
        min_phase_times=min_phase_time,
        max_phase_times=max_phase_time,
        ode_solver=ode_solver,
    )

    solv = Solver.IPOPT(
        online_optim=OnlineOptim.MULTIPROCESS_SERVER,
        show_options={"show_bounds": False, "automatically_organize": False},
    )
    solv.set_maximum_iterations(500)  # TODO This should not be necessary
    # solv.set_linear_solver("ma57")

    sol = ocp.solve(solv)
    # sol.graphs()
    sol.animate()


if __name__ == "__main__":
    main()
