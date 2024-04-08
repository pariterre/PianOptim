import bioviz
import numpy as np
from matplotlib import pyplot as plt
import pickle

def get_user_input():

    while True:
        all_dof = input("Include thorax? (y/n): ").lower()
        if all_dof in ['y', 'n']:
            all_dof = all_dof == 'y'
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

    return all_dof, pressed

allDOF, pressed = get_user_input()

dirName = "/home/alpha/pianoptim/PianOptim/Nov.Codes/Updated_BioModFile_YeadonModel/Results/Felipe_25March/Power_200/"


if allDOF:
    saveName = dirName + ("Pressed" if pressed else "Struck") + "_with_Thorax.pckl"
else:
    saveName = dirName + ("Pressed" if pressed else "Struck") + "_without_Thorax.pckl"

biorbd_model_path = "./With.bioMod" if allDOF else "./Without.bioMod"

with open(saveName, "rb") as file:
    new_dict = pickle.load(file)
def print_all_camera_parameters(biorbd_viz: bioviz.Viz):
    print("Camera roll: ", biorbd_viz.get_camera_roll())
    print("Camera zoom: ", biorbd_viz.get_camera_zoom())
    print("Camera position: ", biorbd_viz.get_camera_position())
    print("Camera focus point: ", biorbd_viz.get_camera_focus_point())

b = bioviz.Viz(
    biorbd_model_path,
    markers_size=0.005,
    contacts_size=0.0005,
    show_floor=False,
    show_segments_center_of_mass=True,
    show_global_ref_frame=True,
    show_global_center_of_mass=True,
    show_global_jcs=True,
    show_markers=True,
    n_frames=50,
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

print_all_camera_parameters(b)

b.load_movement(all_q)
b.exec()
plt.show()


# import bioviz
# import numpy as np
# from matplotlib import pyplot as plt
# import pickle
#
# def get_user_input():
#
#     while True:
#         all_dof = input("Include thorax? (y/n): ").lower()
#         if all_dof in ['y', 'n']:
#             all_dof = all_dof == 'y'
#             break
#         else:
#             print("Invalid input. Please enter 'y' or 'n'.")
#
#     while True:
#         pressed = input("Show 'Pressed' or 'Struck' condition? (p/s): ").lower()
#         if pressed in ['p', 's']:
#             pressed = pressed == 'p'
#             break
#         else:
#             print("Invalid input. Please enter 'p' or 's'.")
#
#     return all_dof, pressed
#
# def print_all_camera_parameters(biorbd_viz: bioviz.Viz):
#     print("Camera roll: ", biorbd_viz.get_camera_roll())
#     print("Camera zoom: ", biorbd_viz.get_camera_zoom())
#     print("Camera position: ", biorbd_viz.get_camera_position())
#     print("Camera focus point: ", biorbd_viz.get_camera_focus_point())
#
# allDOF, pressed = get_user_input()
# dirName = "/home/alpha/pianoptim/PianOptim/Nov.Codes/Updated_BioModFile_YeadonModel/Results/Updated_Profile_Good_Version_W1000/"
#
# if allDOF:
#     saveName = dirName + ("Pressed" if pressed else "Struck") + "_with_Thorax.pckl"
# else:
#     saveName = dirName + ("Pressed" if pressed else "Struck") + "_without_Thorax.pckl"
#
# biorbd_model_path = "./With.bioMod" if allDOF else "./Without.bioMod"
#
# with open(saveName, "rb") as file:
#     new_dict = pickle.load(file)
#
# b = bioviz.Viz(
#     biorbd_model_path,
#     markers_size=0.005,
#     contacts_size=0.010,
#     show_floor=False,
#     show_segments_center_of_mass=True,
#     show_global_ref_frame=True,
#     show_global_center_of_mass=True,
#     show_global_jcs=True,
#     show_markers=True,
#     n_frames=100,
#     show_local_ref_frame=False,
# )
#
# all_q = np.hstack(
#     (
#         new_dict["states"][0]["q"],
#         new_dict["states"][1]["q"],
#         new_dict["states"][2]["q"],
#         new_dict["states"][3]["q"],
#         new_dict["states"][4]["q"],
#     )
# )
#
# # Set the desired camera parameters
# b.set_camera_roll(-89.99999999999999)
# b.set_camera_zoom(1.5076399213631315)
# b.set_camera_position(2.467479199276459, -0.17774303257465363, 0.39589890092611313)
# b.set_camera_focus_point(-0.09527020156383514, -0.17774303257465363, 0.39589890092611313)
#
# # Load movement data
# b.load_movement(all_q)
#
# # Print camera parameters
# print_all_camera_parameters(b)
#
# # Save visualization as a video
# b.start_recording("side_view.ogv")
# num_frames = 330
# for frame in range(num_frames):
#     b.movement_slider[0].setValue(frame)
#     b.add_frame()
# b.stop_recording()
#
# # Close the visualization window
# b.quit()

