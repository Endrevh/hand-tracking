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

def count_occurrences_in_file(file_path, word):
    total_lines = 0
    number_of_occurrences = 0
    with open(file_path, 'r') as file:
        for line in file:
            total_lines += 1
            if word in line:
                number_of_occurrences += 1
    return total_lines, number_of_occurrences            

# Specify the path to the data file
data_file_1 = '../data/relative_movements_translation_aruco_small.txt'
data_file_2 = '../data/relative_movements_translation_aruco_big.txt'

# Call the read and filter data function
data_1 = read_and_filter_data(data_file_1)
data_2 = read_and_filter_data(data_file_2)

# Count occurrences of "MISSING" in data files and print detection success rate
data_1_total, data_1_occurrences = count_occurrences_in_file(data_file_1, "MISSING")
data_2_total, data_2_occurrences = count_occurrences_in_file(data_file_2, "MISSING")

data_1_success_rate = 1 - float(data_1_occurrences / data_1_total)
data_2_success_rate = 1 - float(data_2_occurrences / data_2_total)
print(f"Detection success rate in data 1: {data_1_success_rate}")
print(f"Detection success rate in data 2: {data_2_success_rate}")

# Continue with analysis only if data is not empty
if data_1.size > 0:
    # Extract timestamps, robot poses, and estimated object positions
    timestamps_1 = data_1[:, 0] - data_1[0, 0]  # Adjust timestamps to start from zero
    timestamps_2 = data_2[:, 0] - data_2[0, 0]
    robot_poses = data_1[:, 1:4]
    object_positions_1 = data_1[:, 13:16]
    object_positions_2 = data_2[:, 13:16]

    # Compute Euclidean distances
    robot_distances = np.linalg.norm(robot_poses - robot_poses[0], axis=1)
    object_distances_1 = np.linalg.norm(object_positions_1 - object_positions_1[0], axis=1)
    object_distances_2 = np.linalg.norm(object_positions_2 - object_positions_2[0], axis=1)

    # Plot trajectories against time
    plt.plot(timestamps_1, robot_distances, label='End-effector')
    plt.plot(timestamps_1, object_distances_1, label='Small ArUco marker')
    plt.plot(timestamps_2, object_distances_2, label='Large ArUco marker')
    plt.xlabel('Time [s]')
    plt.ylabel('Euclidean distance [m]')
    plt.title('Estimated distance moved by marker vs. end-effector')
    plt.legend()
    plt.show()
else:
    print("No valid data to plot.")
