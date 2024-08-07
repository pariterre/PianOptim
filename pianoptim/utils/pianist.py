from functools import cached_property

from bioptim import BiorbdModel, Bounds
from casadi import MX, SX, vertcat, if_else, nlpsol, DM, Function
import numpy as np


class Pianist(BiorbdModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, segments_to_apply_external_forces=["RightFingers"], **kwargs)

    @property
    def trunk_dof(self) -> list[int]:
        """
        Returns the index of the degrees of freedom of the trunk
        """
        return [0, 1, 2, 3]

    @property
    def hand_dof(self) -> list[int]:
        """
        Returns the index of the degrees of freedom of the hand
        """
        return [10, 11]

    @cached_property
    def q_hand_on_keyboard(self) -> np.array:
        """
        This runs the inverse kinematics to get the body position for the hand over the keyboard
        """
        q = MX.sym("q", self.nb_q, 1)

        target = self.marker(q, self.marker_names.index("Key1_Top"))
        finger = self.marker(q, self.marker_names.index("finger_marker"))

        s = nlpsol("sol", "ipopt", {"x": q, "g": finger - target}, {"ipopt.print_level": 0})
        return np.array(s(x0=np.zeros(self.nb_q), lbg=np.zeros(3), ubg=np.zeros(3))["x"])[:, 0]

    @cached_property
    def q_hand_above_keyboard(self) -> np.array:
        """
        This runs the inverse kinematics to get the body position for the hand over the keyboard
        """
        q = MX.sym("q", self.nb_q, 1)

        target = self.marker(q, self.marker_names.index("key1_above"))
        finger = self.marker(q, self.marker_names.index("finger_marker"))

        s = nlpsol("sol", "ipopt", {"x": q, "g": finger - target}, {"ipopt.print_level": 0})
        return np.array(s(x0=np.zeros(self.nb_q), lbg=np.zeros(3), ubg=np.zeros(3))["x"])[:, 0]

    @property
    def joint_torque_bounds(self) -> Bounds:
        return Bounds(min_bound=[-40] * self.nb_tau, max_bound=[40] * self.nb_tau, key="tau")

    def compute_marker_from_dm(self, q: DM, marker_name: str) -> DM:
        """
        Compute the position of a marker given the generalized coordinates

        Parameters
        ----------
        q: DM
            The generalized coordinates of the system
        marker_name: str
            The name of the marker to compute

        Returns
        -------
        The position of the marker
        """
        q_sym = MX.sym("q", self.nb_q, 1)
        func = Function("marker", [q_sym], [self.marker(q_sym, self.marker_names.index(marker_name))])
        return func(q)

    def compute_key_reaction_forces(self, q: MX | SX):
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
        finger = self.marker(q, self.marker_index("finger_marker"))
        key_top = self.marker(q, self.marker_index("Key1_Top"))
        key_bottom = self.marker(q, self.marker_index("key1_base"))

        finger_penetration = np.mean(key_top[2] - key_bottom[2]) - finger[2]
        max_penetration = key_top[2] - key_bottom[2]
        force_at_bed = 1
        force_increate_rate = 5e4
        max_force = 10

        x = 0  # This is done via contact
        y = 0  # This is done via contact
        # z = force_at_bed * np.exp(force_increate_rate * (finger_penetration - max_penetration))
        z = if_else(
            finger_penetration < 0,  # Not penetrating
            0,
            if_else(
                finger_penetration < max_penetration,
                finger_penetration / max_penetration * max_force,
                max_force,
            ),
        )  # Temporary until we get actual data
        px = finger[0]
        py = finger[1]
        pz = finger[2]
        return vertcat(x, y, z, px, py, pz)
