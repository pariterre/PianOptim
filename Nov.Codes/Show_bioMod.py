import bioviz

model_path = "/home/alpha/pianoptim/PianOptim/2_Mathilde_2022/2__final_models_piano/1___final_model___squeletum_hand_finger_1_key_4_phases_/bioMod/Squeletum_hand_finger_3D_2_keys_octave_LA.bioMod"

b = bioviz.Viz(
    model_path,
    markers_size=0.005,
    contacts_size=0.00150,
    show_floor=False,
    show_segments_center_of_mass=False,
    show_global_ref_frame=True,
    show_local_ref_frame=False,
    show_markers=True,
)
b.exec()
