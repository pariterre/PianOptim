from bioptim import OptimalControlProgram, NonLinearProgram, ConfigureProblem, DynamicsEvaluation, DynamicsFunctions
from casadi import MX, SX, vertcat

from .pianist import Pianist


class PianistDyanmics:
    @staticmethod
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
        ConfigureProblem.configure_dynamics_function(ocp, nlp, PianistDyanmics.forward_dynamics_with_external_forces)

    @staticmethod
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

        model: Pianist = nlp.model

        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        tau = DynamicsFunctions.get(nlp.controls["tau"], controls)
        translational_force = nlp.model.compute_key_reaction_forces(q)

        # You can directly call biorbd function (as for ddq) or call bioptim accessor (as for dq)
        dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)
        ddq = model.constrained_forward_dynamics(q, qdot, tau, translational_forces=translational_force)

        # the user has to choose if want to return the explicit dynamics dx/dt = f(x,u,p)
        # as the first argument of DynamicsEvaluation or
        # the implicit dynamics f(x,u,p,xdot)=0 as the second argument
        # which may be useful for IRK or COLLOCATION integrators
        return DynamicsEvaluation(dxdt=vertcat(dq, ddq), defects=None)
