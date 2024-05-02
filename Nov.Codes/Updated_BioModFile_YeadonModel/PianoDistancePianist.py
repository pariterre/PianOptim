import ezc3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Load the C3D file
c3d_path = "/home/alpha/Desktop/Exp.Piano/Struck_Without.c3d"
c3d = ezc3d.c3d(c3d_path)

# Extract marker data
markers = c3d['data']['points']  # Shape: (4, n_markers, n_frames)
n_frames = markers.shape[2]
n_markers = markers.shape[1]

# Prepare the animation
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot settings
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Initialize scatter plot
scat = ax.scatter(markers[0, :, 0], markers[1, :, 0], markers[2, :, 0])

# Update function for animation
def update(frame):
    scat._offsets3d = (markers[0, :, frame], markers[1, :, frame], markers[2, :, frame])
    return scat,

# Create animation
ani = FuncAnimation(fig, update, frames=n_frames, interval=2, blit=False)

plt.show()


# The header information
header = c3d['header']

# The parameter information
parameters = c3d['parameters']

# The marker names from the parameters
marker_names = parameters['POINT']['LABELS']['value']

# Print the marker names
print("Marker Names:")
for name in marker_names:
    print(name)


# desired_markers = ["Piano:REF_audio_bof_vicon", "004:PSISr", "004:PSISl"]
#
# for frame in range(markers.shape[2]):
#     print(f"Frame {frame + 1}:")
#     for marker_name in desired_markers:
#         marker_index = marker_names.index(marker_name)
#         x = markers[0, marker_index, frame]
#         y = markers[1, marker_index, frame]
#         z = markers[2, marker_index, frame]
#         print(f"{marker_name}: X={x}, Y={y}, Z={z}")
#     print()
#

# Select the desired markers
desired_markers = ["Piano:REF_audio_bof_vicon", "004:PSISr", "004:PSISl"]
marker_indices = [marker_names.index(marker) for marker in desired_markers]

# Extract the data for the desired markers
marker_data = markers[:3, marker_indices, :]

# Create subplots for each marker
fig, axs = plt.subplots(len(desired_markers), 3, figsize=(12, 16))

# Plot coordinates for each marker
for i, marker in enumerate(desired_markers):
    # Plot X coordinates
    axs[i, 0].plot(range(n_frames), marker_data[0, i, :], label=marker)
    axs[i, 0].set_xlabel('Frame')
    axs[i, 0].set_ylabel('X')
    axs[i, 0].legend()

    # Plot Y coordinates
    axs[i, 1].plot(range(n_frames), marker_data[1, i, :], label=marker)
    axs[i, 1].set_xlabel('Frame')
    axs[i, 1].set_ylabel('Y')
    axs[i, 1].legend()

    # Plot Z coordinates
    axs[i, 2].plot(range(n_frames), marker_data[2, i, :], label=marker)
    axs[i, 2].set_xlabel('Frame')
    axs[i, 2].set_ylabel('Z (mm)')
    axs[i, 2].legend()

# Adjust spacing between subplots
plt.tight_layout()

# Display the plot
plt.show()

print(np.mean(marker_data[2][0,1000:7000]))    #Removing the outliers      Piano Ref
print(np.mean(marker_data[2][1,1000:7000]))    #"004:PSISr"
print(np.mean(marker_data[2][2,1000:7000]))    #"004:PSISl"


# Calculate the distance between the PSIS marker and Piano_Ref
distance = np.mean(marker_data[2][0,1000:7000]) - (np.mean(marker_data[2][1,1000:7000]) + np.mean(marker_data[2][2,1000:7000]))/2
print(f"Vertical distance is: {distance} mm")


