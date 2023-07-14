import matplotlib.pyplot as plt
import numpy as np

def calculate_weighted_centroid(x, y):
    # Calculate the area of the graph
    area = sum((y[i] + y[i + 1]) * (x[i + 1] - x[i]) / 2 for i in range(len(x) - 1))

    # Calculate the weighted centroid
    weighted_centroid = sum(y[i] * y[i] * (x[i + 1] - x[i]) for i in range(len(x) - 1)) / area

    return weighted_centroid


# Example data for x and y coordinates of the tau graph
x = [0,1,2]

# Example tau values (corresponding to each time point)
y = [0,10,0]


# Call the function to calculate the weighted centroid
result = calculate_weighted_centroid(x, y)

# Plotting the graph
plt.plot(x, y, 'b-', label='Tau Graph')
plt.fill_between(x, y, color='skyblue', alpha=0.5, label='Area under Graph')
plt.xlabel('Time')
plt.ylabel('Torque')
plt.legend()

# Plotting the centroid point
plt.plot(1, result, 'ro', label='Weighted Centroid')

# Show the plot
plt.show()