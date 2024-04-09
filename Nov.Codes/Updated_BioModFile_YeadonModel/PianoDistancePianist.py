import ezc3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Load the C3D file
c3d_path = "/home/alpha/Desktop/Exp.Piano/Pressed_With.c3d"
c3d = ezc3d.c3d(c3d_path)

# Extract marker data
markers = c3d['data']['points']  # Shape: (4, n_markers, n_frames)
n_frames = markers.shape[2]
n_markers = markers.shape[1]

# # Prepare the plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # Plot settings
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
#
# # Initialize scatter plot
# scat = ax.scatter(markers[0, :, 0], markers[1, :, 0], markers[2, :, 0])
#
# # Update function for animation
# def update(frame):
#     scat._offsets3d = (markers[0, :, frame], markers[1, :, frame], markers[2, :, frame])
#     return scat,
#
# # Create animation
# ani = FuncAnimation(fig, update, frames=n_frames, interval=2, blit=False)
#
# plt.show()


# Get the header information
header = c3d['header']

# Get the parameter information
parameters = c3d['parameters']

# Get the marker names from the parameters
marker_names = parameters['POINT']['LABELS']['value']

# Print the marker names
print("Marker Names:")
for name in marker_names:
    print(name)


desired_markers = ["Piano:Piano_AigDown", "Piano:Piano_GraDown", "004:PSISr", "004:PSISl"]

for frame in range(markers.shape[2]):
    print(f"Frame {frame + 1}:")
    for marker_name in desired_markers:
        marker_index = marker_names.index(marker_name)
        x = markers[0, marker_index, frame]
        y = markers[1, marker_index, frame]
        z = markers[2, marker_index, frame]
        print(f"{marker_name}: X={x}, Y={y}, Z={z}")
    print()

# Select the desired markers
desired_markers = ["Piano:Piano_AigDown", "Piano:Piano_GraDown", "004:PSISr", "004:PSISl"]
desired_indices = [marker_names.index(marker) for marker in desired_markers]

# Extract the data for the desired markers
selected_markers = markers[:3, desired_indices, :]
num_frames = selected_markers.shape[2]
num_markers = selected_markers.shape[1]

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initialize the plot with empty lines
lines = [ax.plot([], [], [], 'o')[0] for _ in range(num_markers)]

# Set plot limits and labels
ax.set_xlim(np.min(selected_markers[0, :, :]), np.max(selected_markers[0, :, :]))
ax.set_ylim(np.min(selected_markers[1, :, :]), np.max(selected_markers[1, :, :]))
ax.set_zlim(np.min(selected_markers[2, :, :]), np.max(selected_markers[2, :, :]))
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Update function for animation
def update(frame):
    for i, line in enumerate(lines):
        line.set_data(selected_markers[0, i, frame], selected_markers[1, i, frame])
        line.set_3d_properties(selected_markers[2, i, frame])
    return lines

# Create the animation
ani = FuncAnimation(fig, update, frames=range(num_frames), interval=50, blit=True)

# Display the animation
plt.show()
