import numpy as np
import matplotlib.pyplot as plt

def is_valid_line(line):
    """ Check if the line is valid (does not contain 'MISSING'). """
    return 'MISSING' not in line

def read_and_filter_data(file_path):
    """ Read data from file and filter out invalid lines. """
    valid_data = []
    with open(file_path, 'r') as file:
        for line in file:
            stripped_line = line.strip()
            if is_valid_line(stripped_line):
                try:
                    # Convert each valid line to a numpy array after splitting.
                    data_line = np.array(stripped_line.split(','), dtype=float)
                    valid_data.append(data_line)
                except ValueError:
                    # If conversion fails (e.g., because of an incomplete line), skip the line.
                    continue
    return np.array(valid_data)

# Specify the path to your data file
data_file = '../data/relative_movements_translation_aruco_small_closer.txt'

# Call the read and filter data function
data = read_and_filter_data(data_file)

# Continue with your analysis only if data is not empty
if data.size > 0:
    # Extract timestamps, robot poses, and estimated object positions
    timestamps = data[:, 0] - data[0, 0]  # Adjust timestamps to start from zero
    robot_poses = data[:, 1:4]
    object_positions = data[:, 13:16]

    # Compute Euclidean distances
    robot_distances = np.linalg.norm(robot_poses - robot_poses[0], axis=1)
    object_distances = np.linalg.norm(object_positions - object_positions[0], axis=1)

    # Plot trajectories against time
    plt.plot(timestamps, robot_distances, label='Robot distance moved')
    plt.plot(timestamps, object_distances, label='Object estimated distance moved')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Euclidean Distance')
    plt.legend()
    plt.show()
else:
    print("No valid data to plot.")
