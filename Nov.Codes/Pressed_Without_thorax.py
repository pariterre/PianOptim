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

def prepare_ocp(
    biorbd_model_path: str = "/home/alpha/pianoptim/PianOptim/2_Mathilde_2022/2__final_models_piano/1___final_model___squeletum_hand_finger_1_key_4_phases_/bioMod/Squeletum_hand_finger_3D_2_keys_octave_LA.bioMod",
    ode_solver: OdeSolver = OdeSolver.COLLOCATION(polynomial_degree=4),
) -> OptimalControlProgram:


    biorbd_model = (
        BiorbdModel(biorbd_model_path),
        BiorbdModel(biorbd_model_path),
        BiorbdModel(biorbd_model_path),
        BiorbdModel(biorbd_model_path),
        BiorbdModel(biorbd_model_path),

    )

    # Average of N frames by phase ; Average of phases time ; both measured with the motion capture datas.
    n_shooting = (30, 7, 9, 17, 18)
    phase_time = (0.3, 0.044, 0.051, 0.17, 0.18)
    tau_min, tau_max, tau_init = -200, 200, 0
    # Velocity profile found thanks to the motion capture datas.
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

    Froce = [30, 26, 24, 20, 16, 12, 8, 4, 0]

    pi_sur_2_phase_0 = np.full((1, n_shooting[0] + 1), pi / 2)
    pi_sur_2_phase_1 = np.full((1, n_shooting[1] + 1), pi / 2)
    pi_sur_2_phase_2 = np.full((1, n_shooting[2] + 1), pi / 2)
    pi_sur_2_phase_3 = np.full((1, n_shooting[3] + 1), pi / 2)
    pi_sur_2_phase_4 = np.full((1, n_shooting[4] + 1), pi / 2)

    # Objectives
    # Minimize Torques generated into articulations
    objective_functions = ObjectiveList()
    for i in [0, 1, 2, 3, 4]:
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", phase=i, weight=100, index=[0, 1, 2, 3, 4]
        )

    for i in [0, 1, 2, 3, 4]:
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", phase=i, weight=10000, index=[5, 6], derivative=True
        )

    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", phase=0, weight=0.0001, index=[0, 1, 2, 3, 4]
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", phase=1, weight=0.0001, index=[0, 1, 2, 3, 4, 5, 6]
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", phase=2, weight=0.0001, index=[0, 1, 2, 3, 4, 5, 6]
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", phase=3, weight=0.0001, index=[0, 1, 2, 3, 4]
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", phase=4, weight=0.0001, index=[0, 1, 2, 3, 4]
    )

    # To block ulna rotation before the key pressing.
    for i in [0, 1, 2, 3, 4]:
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", phase=i, weight=100000, index=[0, 4])

    objective_functions.add(
        ObjectiveFcn.Mayer.TRACK_MARKERS_VELOCITY,
        target=vel_push_array2,
        node=Node.ALL,
        phase=1,
        marker_index=4,
        weight=10000,
    )

    # To keep the hand/index perpendicular of the key piano all long the attack.
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

    # To avoid the apparition of "noise" caused by the objective function just before.
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", phase=0, weight=1000, index=[5, 6], derivative=True
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", phase=3, weight=1000, index=[5, 6], derivative=True
    )

    objective_functions.add(
        ObjectiveFcn.Lagrange.TRACK_CONTACT_FORCES,
        target=Froce,
        node=Node.ALL_SHOOTING,
        contact_index=2,
        phase=2,
        weight=10000,
    )

    Mul_Node_Obj = MultinodeObjectiveList()
    # To minimize the difference between 0 and 1
    Mul_Node_Obj.add(
        minimize_difference,
        custom_type=ObjectiveFcn.Mayer,
        weight=1000,
        nodes_phase=(0, 1),
        nodes=(Node.END, Node.START),
        quadratic=True,
    )
    # # To minimize the difference between 0 and 1
    Mul_Node_Obj.add(
        minimize_difference,
        custom_type=ObjectiveFcn.Mayer,
        weight=1000,
        nodes_phase=(1, 2),
        nodes=(Node.END, Node.START),
        quadratic=True,
    )
    # # To minimize the difference between 2 and 3
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

    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        node=Node.ALL,
        first_marker="finger_marker",
        second_marker="high_square",
        phase=0,
    )
    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        node=Node.END,
        first_marker="finger_marker",
        second_marker="low_square",
        phase=1,
    )
    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES, node=Node.ALL, contact_index=0, min_bound=-5, max_bound=5, phase=2
    )
    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES, node=Node.ALL, contact_index=1, min_bound=-5, max_bound=5, phase=2
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
    #
    # x_bounds[0]["q"][[0], 0] = -0.1
    # x_bounds[0]["q"][[2], 0] = 0.1
    #
    # x_bounds[4]["q"][[0], 2] = -0.1
    # x_bounds[4]["q"][[2], 2] = 0.1
    x_bounds[4]["q"][[5], 2] = -0.25


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

    for i in range(5):
        x_init[i]["q"][1, 0] = 0.08
        x_init[i]["q"][2, 0] = 0.67
        x_init[i]["q"][3, 0] = 1.11
        x_init[i]["q"][4, 0] = 1.48
        x_init[i]["q"][6, 0] = 0.17

    # Define control path constraint
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

    u_init = InitialGuessList()
    u_init.add("tau", [tau_init] * biorbd_model[0].nb_tau, phase=0)
    u_init.add("tau", [tau_init] * biorbd_model[1].nb_tau, phase=1)
    u_init.add("tau", [tau_init] * biorbd_model[2].nb_tau, phase=2)
    u_init.add("tau", [tau_init] * biorbd_model[3].nb_tau, phase=3)
    u_init.add("tau", [tau_init] * biorbd_model[4].nb_tau, phase=4)

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
    ocp.add_plot_penalty(CostType.ALL)

    # # --- Solve the program --- # #

    solv = Solver.IPOPT(show_online_optim=False)
    solv.set_maximum_iterations(1000000)
    solv.set_linear_solver("ma57")
    tic = time.time()
    sol = ocp.solve(solv)

    # # --- Download datas on a .pckl file --- #
    q_sym = MX.sym('q_sym', 7, 1)
    qdot_sym = MX.sym('qdot_sym', 7, 1)
    tau_sym = MX.sym('tau_sym', 7, 1)
    Calculaing_Force = Function("Temp", [q_sym, qdot_sym, tau_sym], [
        ocp.nlp[2].model.contact_forces_from_constrained_forward_dynamics(q_sym, qdot_sym, tau_sym)])

    rows = 9
    cols = 3
    F = [[0] * cols for _ in range(rows)]

    for i in range(0, 9):
        F[i] = Calculaing_Force(sol.states[2]["q"][:, i], sol.states[2]["qdot"][:, i],
                                sol.controls[2]['tau'][:, i])

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
            "/home/alpha/Desktop/Nov. 14/Pressed_without_Thorax.pckl",
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