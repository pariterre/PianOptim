import rerun as rr
import numpy as np
from matplotlib import pyplot as plt
import pickle

rr.init("my_visualization_session")
rr.spawn()
def get_user_input():
    while True:
        all_dof = input("Include thorax? (y/n): ").lower()
        if all_dof in ['y', 'n']:
            all_dof = all_dof == 'y'
            biorbd_model_path = "./With.bioMod" if all_dof else "./Without.bioMod"
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

    while True:
        pressed = input("Show 'Pressed' or 'Struck' condition? (p/s): ").lower()
        if pressed in ['p', 's']:
            pressed = pressed == 'p'
            break
        else:
            print("Invalid input. Please enter 'p' or 's'.")

    return biorbd_model_path, pressed

biorbd_model_path, pressed = get_user_input()
dirName = "/home/alpha/pianoptim/PianOptim/Nov.Codes/Updated_BioModFile_YeadonModel/Results/"

saveName = dirName + ("Pressed" if pressed else "Struck") + "_with_Thorax.pckl" if biorbd_model_path.endswith("With.bioMod") else dirName + ("Pressed" if pressed else "Struck") + "_without_Thorax.pckl"

with open(saveName, "rb") as file:
    new_dict = pickle.load(file)

all_q = np.hstack(
    (
        new_dict["states"][0]["q"],
        new_dict["states"][1]["q"],
        new_dict["states"][2]["q"],
        new_dict["states"][3]["q"],
        new_dict["states"][4]["q"],
    )
)

positions = all_q.reshape(-1, 3)
rr.log("movement_points", rr.Points3D(positions))

