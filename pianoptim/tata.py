from bioptim import (
    OptimalControlProgram,
    DynamicsFcn,
    Dynamics,
    BoundsList,
    ObjectiveFcn,
    Objective,
    Solver,
    BiorbdModel,
    ShowOnlineType,
)


def prepare_ocp(biorbd_model_path: str, final_time: float, n_shooting: int) -> OptimalControlProgram:

    bio_model = BiorbdModel(biorbd_model_path)

    # Add objective functions
    objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")

    # Dynamics
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN)

    # Path bounds
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["q"][:, [0, -1]] = 0  # Start and end at 0...
    x_bounds["q"][1, -1] = 3.14  # ...but end with pendulum 180 degrees rotated
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:, [0, -1]] = 0  # Start and end without any velocity

    # Define control path bounds
    n_tau = bio_model.nb_tau
    u_bounds = BoundsList()
    u_bounds["tau"] = [-100] * n_tau, [100] * n_tau  # Limit the strength of the pendulum to (-100 to 100)...
    u_bounds["tau"][1, :] = 0  # ...but remove the capability to actively rotate

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
    )


def main():
    # --- Prepare the ocp --- #
    ocp = prepare_ocp(biorbd_model_path="models/pendulum.bioMod", final_time=1, n_shooting=400)
    ocp.solve(
        Solver.IPOPT(
            show_online_optim=True,
            show_options={"type": ShowOnlineType.TCP},
            # show_online_optim=True
        )
    )


if __name__ == "__main__":
    main()
