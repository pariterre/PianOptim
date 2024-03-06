import bioviz
import numpy as np
from matplotlib import pyplot as plt
import pickle

allDOF = True  # True means all DOF, False means no thorax
pressed = True  # False means Struck
dirName = "/home/alpha/pianoptim/PianOptim/Nov.Codes/Updated_BioModFile_YeadonModel/Results/"

if allDOF:
    saveName = dirName + ("Pressed" if pressed else "Struck") + "_with_Thorax.pckl"
else:
    saveName = dirName + ("Pressed" if pressed else "Struck") + "_without_Thorax.pckl"

biorbd_model_path = "With.bioMod" if allDOF else "Without.bioMod"

with open(saveName, "rb") as file:
    new_dict = pickle.load(file)

b = bioviz.Viz(
    biorbd_model_path,
    markers_size=0.005,
    contacts_size=0.010,
    show_floor=False,
    show_segments_center_of_mass=True,
    show_global_ref_frame=True,
    show_global_center_of_mass=True,
    show_global_jcs=True,
    show_markers=True,
    n_frames=100,
    show_local_ref_frame=False,
)

all_q = np.hstack(
    (
        new_dict["states"][0]["q"],
        new_dict["states"][1]["q"],
        new_dict["states"][2]["q"],
        new_dict["states"][3]["q"],
        new_dict["states"][4]["q"],
    )
)

b.load_movement(all_q)
b.exec()
plt.show()
