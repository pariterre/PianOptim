import bioviz

model_path = "./With.bioMod"

b = bioviz.Viz(
    model_path,
    markers_size=0.0025,
    contacts_size=0.00550,
    show_floor=False,
    show_segments_center_of_mass=True,
    show_mass_center=False,
    show_gravity_vector=True,
    show_global_ref_frame=False,
    show_local_ref_frame=True,
    show_markers=True,
    show_meshes=True,
)
b.exec()
