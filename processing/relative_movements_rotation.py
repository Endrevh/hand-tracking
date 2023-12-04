import numpy as np

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

data_file_path = '../data/relative_movements_rotation_big.txt'
hand_eye_matrix = read_hand_eye_transformation('../calibration/hand_eye_transformation.txt')

robot_transformation_matrices = []
marker_transformation_matrices = []

with open(data_file_path, 'r') as file:
    for line in file:
        data_point = [float(value) for value in line.split(',')]
        robot_transformation_matrix = compute_robot_transformation_matrix(data_point)
        robot_transformation_matrices.append(robot_transformation_matrix)

        marker_translation_vector = np.array(data_point[13:16])
        marker_rotation_vector = np.array(data_point[16:19])
        marker_rotation_matrix = rotation_vector_to_matrix(marker_rotation_vector)
        marker_transformation_matrix = np.eye(4)
        marker_transformation_matrix[:3, :3] = marker_rotation_matrix
        marker_transformation_matrix[:3, 3] = marker_translation_vector
        marker_transformation_matrices.append(marker_transformation_matrix)
        
# Compute the relative transformation between the firts and last robot pose
initial_robot_transformation_matrix = robot_transformation_matrices[0]
final_robot_transformation_matrix = robot_transformation_matrices[-1]
relative_robot_transformation_matrix = np.linalg.inv(initial_robot_transformation_matrix) @ final_robot_transformation_matrix

initial_marker_transformation_matrix = marker_transformation_matrices[0]
final_marker_transformation_matrix = marker_transformation_matrices[-1]

identity_matrix_candidate = np.linalg.inv(hand_eye_matrix @ initial_marker_transformation_matrix) @ relative_robot_transformation_matrix @ hand_eye_matrix @ final_marker_transformation_matrix

print(f"Identity matrix candidate:\n{identity_matrix_candidate}")