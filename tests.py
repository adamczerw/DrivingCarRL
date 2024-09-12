import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Example sequence of matrices
matrices = [
    np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
    np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]]),
    np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]]),
    np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]]),
    np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
]

# Create a figure and axis
fig, ax = plt.subplots()

# Function to update the plot
def update(frame):
    ax.clear()
    ax.imshow(frame, cmap='gray', vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=matrices, interval=500)

# Display the animation
plt.show()