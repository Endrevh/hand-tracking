import numpy as np
import matplotlib.pyplot as plt

def read_hand_eye_transformation(file_path):
    # Read the hand-eye transformation matrix from the file
    return np.loadtxt(file_path)

def compute_robot_transformation_matrix(data_point):
    # Extract position and rotation (axis-angle) for the hand-frame
    position = np.array(data_point[1:4])
    rotation_vector = np.array(data_point[4:7])

    # Convert rotation vector to matrix
    rotation_matrix = rotation_vector_to_matrix(rotation_vector)

    # Form the transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = position

    return transformation_matrix

def rotation_vector_to_matrix(rotation_vector):
    """
    Convert an axis-angle rotation vector to a rotation matrix.
    Uses Rodrigues' rotation formula.
    """
    angle = np.linalg.norm(rotation_vector)
    if angle == 0:
        return np.eye(3)

    axis = rotation_vector / angle
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    cross_product_matrix = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])

    rotation_matrix = cos_angle * np.eye(3) + sin_angle * cross_product_matrix + (1 - cos_angle) * np.outer(axis, axis)
    return rotation_matrix

def read_data_file(file_path):
    timestamps = []
    robot_transformation_matrices = []
    marker_transformation_matrices = []
    with open(file_path, 'r') as file:
        for line in file:
            data_point = [float(value) for value in line.split(',')]

            timestamps.append(data_point[0])

            robot_transformation_matrix = compute_robot_transformation_matrix(data_point)
            robot_transformation_matrices.append(robot_transformation_matrix)

            marker_translation_vector = np.array(data_point[13:16])
            marker_rotation_vector = np.array(data_point[16:19])
            marker_rotation_matrix = rotation_vector_to_matrix(marker_rotation_vector)
            marker_transformation_matrix = np.eye(4)
            marker_transformation_matrix[:3, :3] = marker_rotation_matrix
            marker_transformation_matrix[:3, 3] = marker_translation_vector
            marker_transformation_matrices.append(marker_transformation_matrix)
    return timestamps, robot_transformation_matrices, marker_transformation_matrices

def calculate_frobenius_trajectory(hand_eye_matrix, robot_transformations, marker_transformations, reference_index):
    frobenius_norms = []
    n = len(robot_transformations)
    initial_robot_transformation_matrix = robot_transformations[reference_index]
    initial_marker_transformation_matrix = marker_transformations[reference_index]
    for i in range(reference_index + 1, n):
        current_robot_transformation_matrix = robot_transformations[i]
        relative_robot_transformation_matrix = np.linalg.inv(initial_robot_transformation_matrix) @ current_robot_transformation_matrix
        current_marker_transformation_matrix = marker_transformations[i]
        identity_matrix_candidate = np.linalg.inv(hand_eye_matrix @ initial_marker_transformation_matrix) @ relative_robot_transformation_matrix @ hand_eye_matrix @ current_marker_transformation_matrix
        # Calculate frobenius norm of the difference between the candidate and the identity matrix
        frobenius_norm = np.linalg.norm(identity_matrix_candidate - np.eye(4))
        frobenius_norms.append(frobenius_norm)
    return frobenius_norms

data_file_path_big = '../data/relative_movements_rotation_big.txt'
data_file_path_small = '../data/relative_movements_rotation_small.txt'

timestamps_big, robot_transformations_big, marker_transformations_big = read_data_file(data_file_path_big)
timestamps_small, robot_transformations_small, marker_transformations_small = read_data_file(data_file_path_small)

# Adjust timestamps to start from 0
timestamps_big = np.array(timestamps_big)
timestamps_big -= timestamps_big[0]

timestamps_small = np.array(timestamps_small)
timestamps_small -= timestamps_small[0]

reference_index = 0
hand_eye_matrix = read_hand_eye_transformation('../calibration/hand_eye_transformation.txt')

# Calculate frobenius norms for big dataset
frobenius_norms_big = calculate_frobenius_trajectory(hand_eye_matrix, robot_transformations_big, marker_transformations_big, reference_index)

# Calculate frobenius norms for small dataset
frobenius_norms_small = calculate_frobenius_trajectory(hand_eye_matrix, robot_transformations_small, marker_transformations_small, reference_index)

# Calculate mean of frobenius norms
mean_frobenius_norm_big = np.mean(frobenius_norms_big)
print(f"Mean frobenius norm big: {mean_frobenius_norm_big}")
mean_frobenius_norm_small = np.mean(frobenius_norms_small)
print(f"Mean frobenius norm small: {mean_frobenius_norm_small}")

# Plot frobenius norms against time
plt.plot(timestamps_big[(reference_index+1):], frobenius_norms_big)
plt.plot(timestamps_small[(reference_index+1):], frobenius_norms_small)
plt.xlabel('Time [s]')
plt.ylabel('Frobenius norm [m]')
plt.grid()
plt.show()

# Compute the relative transformation between the first and last robot pose for both marker sizes
initial_robot_transformation_small = robot_transformations_small[0]
initial_marker_transformation_small = marker_transformations_small[0]
final_marker_transformation_small = marker_transformations_small[-1]
relative_robot_transformation_small = np.linalg.inv(initial_robot_transformation_small) @ robot_transformations_small[-1]

initial_robot_transformation_big = robot_transformations_big[0]
initial_marker_transformation_big = marker_transformations_big[0]
final_marker_transformation_big = marker_transformations_big[-1]
relative_robot_transformation_big = np.linalg.inv(initial_robot_transformation_big) @ robot_transformations_big[-1]

identity_matrix_candidate_small = np.linalg.inv(hand_eye_matrix @ initial_marker_transformation_small) @ relative_robot_transformation_small @ hand_eye_matrix @ final_marker_transformation_small
identity_matrix_candidate_big = np.linalg.inv(hand_eye_matrix @ initial_marker_transformation_big) @ relative_robot_transformation_big @ hand_eye_matrix @ final_marker_transformation_big

print(f"Identity matrix candidate small:\n{identity_matrix_candidate_small}")
print(f"Identity matrix candidate big:\n{identity_matrix_candidate_big}")