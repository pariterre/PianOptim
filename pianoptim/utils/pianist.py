from bioptim import BiorbdModel
from casadi import MX, SX, vertcat, if_else
import numpy as np


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

        finger_penetration = key_top[2] - finger[2]
        key_penetration_bed = key_top[2] - key_bottom[2]
        force_at_bed = 1
        force_increate_rate = 5e4

        x = 0  # This is done via contact
        y = 0  # This is done via contact
        # z = force_at_bed * np.exp(force_increate_rate * (finger_penetration - key_penetration_bed))
        z = if_else(finger[2] >= key_bottom[2], 0, 30)  # Temporary until we get actual data
        px = finger[0]
        py = finger[1]
        pz = finger[2]
        return vertcat(x, y, z, px, py, pz)
