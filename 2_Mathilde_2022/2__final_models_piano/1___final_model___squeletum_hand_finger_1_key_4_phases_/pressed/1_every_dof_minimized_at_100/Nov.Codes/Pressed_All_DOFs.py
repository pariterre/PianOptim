"""
 !! Les axes du modèle ne sont pas les mêmes que ceux généralement utilisés en biomécanique : x axe de flexion, y supination/pronation, z vertical
 ici on a : Y -» X , Z-» Y et X -» Z
 """
from casadi import MX, acos, dot, pi, Function
import time
import numpy as np
import biorbd_casadi as biorbd
import pickle

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
#
# def minimize_difference(all_pn: PenaltyNode):
#     return all_pn[0].nlp.controls.cx_end - all_pn[1].nlp.controls.cx
#
def minimize_difference(controllers: list[PenaltyController, PenaltyController]):
    pre, post = controllers
    return pre.controls.cx_end - post.controls.cx

def custom_func_track_finger_5_on_the_right_of_principal_finger(controller: PenaltyController) -> MX:
    finger_marker_idx = biorbd.marker_index(controller.model.model, "finger_marker")
    markers = controller.mx_to_cx("markers", controller.model.markers, controller.states["q"])
    finger_marker = markers[:, finger_marker_idx]

    finger_marker_5_idx = biorbd.marker_index(controller.model.model, "finger_marker_5")
    markers_5 = controller.mx_to_cx("markers_5", controller.model.markers, controller.states["q"])
    finger_marker_5 = markers_5[:, finger_marker_5_idx]

    markers_diff_key2 = finger_marker[1] - finger_marker_5[1]

    return markers_diff_key2

def custom_func_track_principal_finger_and_finger5_above_bed_key(controller: PenaltyController, marker: str) -> MX:
    biorbd_model = controller.model
    finger_marker_idx = biorbd.marker_index(biorbd_model.model, marker)
    markers = controller.mx_to_cx("markers", biorbd_model.markers, controller.states["q"])
    finger_marker = markers[:, finger_marker_idx]

    markers_diff_key3 = finger_marker[2] - (0.07808863830566405 - 0.02)

    return markers_diff_key3

def custom_func_track_principal_finger_pi_in_two_global_axis(controller: PenaltyController, segment: str) -> MX:
    rotation_matrix_index = biorbd.segment_index(controller.model.model, segment)
    q = controller.states["q"].mx
    # global JCS gives the local matrix according to the global matrix
    principal_finger_axis = controller.model.model.globalJCS(q, rotation_matrix_index).to_mx()  # x finger = y global
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

def prepare_ocp(
    biorbd_model_path: str = "../../../bioMod/Squeletum_hand_finger_3D_2_keys_octave_LA.bioMod",
    ode_solver: OdeSolver = OdeSolver.COLLOCATION(polynomial_degree=4),
) -> OptimalControlProgram:


    biorbd_model = (
        BiorbdModel(biorbd_model_path),
        BiorbdModel(biorbd_model_path),
        BiorbdModel(biorbd_model_path),
        BiorbdModel(biorbd_model_path),
        BiorbdModel(biorbd_model_path),
    )

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=0)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=1)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True, phase=2)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=3)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=4)

    # Average of N frames by phase ; Average of phases time ; both measured with the motion capture datas.
    n_shooting = (30, 7, 9, 17, 18)
    phase_time = (0.3, 0.044, 0.051, 0.17, 0.18)
    tau_min, tau_max, tau_init = -200, 200, 0
    # Velocity profile found thanks to the motion capture datas.


    ForceProfile = [30, 26, 24, 20, 16, 12, 8, 4, 0]

    # Objectives
    # Minimize Torques generated into articulations
    objective_functions = ObjectiveList()
    for phase in [0, 1, 2, 3, 4]:
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", phase=phase, weight=100, index=[0, 1, 2, 5, 3, 4, 6, 7]
        )
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", phase=phase, weight=10000, index=[8, 9]
        )
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", phase=phase, weight=0.0001, index=[0, 1, 2, 3, 4, 5, 6, 7]
        )

    for phase in [1, 2]:
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", phase=phase, weight=0.0001, index=[8, 9]
        )


    # To block ulna rotation before the key pressing.
    for i in [0, 1, 2, 3, 4]:
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", phase=i, weight=100000, index=[3, 7])



    # To keep the hand/index perpendicular of the key piano all long the attack.
    for phase in [0, 1, 2, 3, 4]:
        objective_functions.add(
            custom_func_track_principal_finger_pi_in_two_global_axis,
            custom_type=ObjectiveFcn.Lagrange,
            node=Node.ALL,
            phase=phase,
            weight=100000,
            quadratic=True,
            target=np.full((1, n_shooting[phase] + 1), pi / 2),
            segment="secondmc",
        )

        objective_functions.add(
            custom_func_track_principal_finger_pi_in_two_global_axis,
            custom_type=ObjectiveFcn.Lagrange,
            node=Node.ALL,
            phase=phase,
            weight=10000,
            quadratic=True,
            target=np.full((1, n_shooting[phase] + 1), pi / 2),
            segment="2proxph_2mcp_flexion",
        )

    # Constraints
    constraints = ConstraintList()

    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        phase=0, node=Node.ALL,
        first_marker="finger_marker",
        second_marker="high_square",
    )

    vel_push_array2 = [
        [ -0.114, -0.181, -0.270, -0.347, -0.291, -0.100,] #MB: remove first and last
    ]

    constraints.add(
        ConstraintFcn.TRACK_MARKERS_VELOCITY,
        phase=1, node=Node.INTERMEDIATES,
        target=vel_push_array2,
        marker_index=4,
        min_bound=-0.01, max_bound=0.01,
    )

    # No finger's tip velocity at the end of phase 1
    constraints.add(
        ConstraintFcn.TRACK_MARKERS_VELOCITY,
        phase=1, node=Node.END,
        marker_index=4,
    )
    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        phase=1, node=Node.END,
        first_marker="finger_marker",
        second_marker="low_square",
    )

    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        phase=2, node=Node.ALL,
        contact_index=0,
        min_bound=-5, max_bound=5,
    )
    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        phase=2, node=Node.ALL,
        contact_index=1,
        min_bound=-5, max_bound=5,
    )

    for node in range(n_shooting[2]):
        constraints.add(
            ConstraintFcn.TRACK_CONTACT_FORCES,
            phase=2, node=node,
            contact_index=2,
            target=ForceProfile[node],
            #min_bound=-0.1, max_bound=0.1,
        )

    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        node=Node.END,
        first_marker="MCP_contact_finger",
        second_marker="phase_3_upward",
        phase=3,
    )

    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        node=Node.END,
        first_marker="finger_marker",
        second_marker="high_square",
        phase=4,
    )

    # To keep the index and the small finger above the bed key.
    # To keep the small finger on the right of the principal finger.
    for phase in [0, 1, 2, 3, 4]:
        constraints.add(
            custom_func_track_principal_finger_and_finger5_above_bed_key,
            node=Node.ALL,
            marker="finger_marker",
            min_bound=0,
            max_bound=np.inf,
            phase=phase,
        )

        constraints.add(
            custom_func_track_principal_finger_and_finger5_above_bed_key,
            node=Node.ALL,
            marker="finger_marker_5",
            min_bound=0,
            max_bound=np.inf,
            phase=phase,
        )

        constraints.add(
            custom_func_track_finger_5_on_the_right_of_principal_finger,
            node=Node.ALL,
            min_bound=0.00001,
            max_bound=np.inf,
            phase=phase,
        )

    # To keep the hand/index perpendicular of the key piano all long the attack.
    for phase in [0, 1, 2, 3, 4]:
        constraints.add(
            custom_func_track_principal_finger_pi_in_two_global_axis,
            phase=phase, node=Node.ALL,
            target=np.full((1, n_shooting[phase] + 1), pi / 2),
            min_bound=-np.pi/24, max_bound=np.pi/24,
            segment="secondmc",
            quadratic=False,
        )

        constraints.add(
            custom_func_track_principal_finger_pi_in_two_global_axis,
            phase=phase, node=Node.ALL,
            target=np.full((1, n_shooting[phase] + 1), pi / 2),
            min_bound=-np.pi / 24, max_bound=np.pi / 24,
            segment="2proxph_2mcp_flexion",
            quadratic=False,
        )



    phase_transition = PhaseTransitionList()
    phase_transition.add(PhaseTransitionFcn.IMPACT, phase_pre_idx=1)

    # States: bounds and Initial guess
    x_init = InitialGuessList()
    x_bounds = BoundsList()
    for phase in [0, 1, 2, 3, 4]:
        x_bounds.add("q", bounds=biorbd_model[phase].bounds_from_ranges("q"), phase=phase)
        x_bounds.add("qdot", bounds=biorbd_model[phase].bounds_from_ranges("qdot"), phase=phase)

        x_init.add("q", [0] * biorbd_model[phase].nb_q, phase=phase)
        x_init.add("qdot", [0] * biorbd_model[phase].nb_q, phase=phase)

        x_init[phase]["q"][4, 0] = 0.08
        x_init[phase]["q"][5, 0] = 0.67
        x_init[phase]["q"][6, 0] = 1.11
        x_init[phase]["q"][7, 0] = 1.48
        x_init[phase]["q"][9, 0] = 0.17

    x_bounds[0]["q"][[0], 0] = -0.1
    x_bounds[0]["q"][[2], 0] = 0.1

    x_bounds[4]["q"][[0], 2] = -0.1
    x_bounds[4]["q"][[2], 2] = 0.1


    # Define control path constraint and initial guess
    u_bounds = BoundsList()
    u_init = InitialGuessList()
    for phase in [0, 1, 2, 3, 4]:
        u_bounds.add("tau",
                     min_bound=[tau_min] * biorbd_model[phase].nb_tau,
                     max_bound=[tau_max] * biorbd_model[phase].nb_tau,
                     phase=phase)
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
    polynomial_degree = 4
    ocp = prepare_ocp()
    ocp.add_plot_penalty(CostType.ALL)

    # # --- Solve the program --- # #

    solv = Solver.IPOPT(show_online_optim=False)
    solv.set_maximum_iterations(1000)
    solv.set_linear_solver("ma57")
    tic = time.time()
    sol = ocp.solve(solv)

    # # --- Download datas on a .pckl file --- #
    q_sym = MX.sym('q_sym', 10, 1)
    qdot_sym = MX.sym('qdot_sym', 10, 1)
    tau_sym = MX.sym('tau_sym', 10, 1)

    phase = 2
    Contact_Force = Function("Contact_Force", [q_sym, qdot_sym, tau_sym], [
        ocp.nlp[phase].model.contact_forces_from_constrained_forward_dynamics(q_sym, qdot_sym, tau_sym)])

    rows = 9
    cols = 3
    F = [[0] * cols for _ in range(rows)]

    for i in range(0, 9):
        idx = i * polynomial_degree
        F[i] = Contact_Force(sol.states[phase]["q"][:, idx],
                             sol.states[phase]["qdot"][:, idx],
                             sol.controls[phase]['tau'][:, i])

    F_array = np.array(F)

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

    with open(
            "/home/alpha/Desktop/Nov. 14/Pressed_with_Thorax.pckl",
            "wb") as file:
        pickle.dump(data, file)
    #
    # print("Tesults saved")
    # print("Temps de resolution : ", time.time() - tic, "s")
    #
    # sol.print_cost()
    # ocp.print(to_console=False, to_graph=False)
    # # sol.graphs(show_bounds=True)
    # sol.animate(show_floor=False, show_global_center_of_mass=False, show_segments_center_of_mass=False, show_global_ref_frame=True, show_local_ref_frame=False, show_markers=False, n_frames=250,)


if __name__ == "__main__":
    main()