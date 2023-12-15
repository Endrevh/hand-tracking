import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

# Specify the dimensions of the chessboard
chessboard_size = (18, 29)
square_size = 0.01  # 10mm in meters

# Prepare the object points for the chessboard (0,0,0), (1,0,0), (2,0,0) ... etc.
objp = np.zeros(((chessboard_size[1]-1) * (chessboard_size[0]-1), 3), np.float32)
objp[:, :2] = np.mgrid[0:(chessboard_size[0]-1), 0:(chessboard_size[1]-1)].T.reshape(-1, 2)
objp *= square_size

# Termination criteria for the iterative algorithm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Arrays to store object points and image points from all images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
image_shape = None  # We'll set this after reading the first image

# Name of folder where images and poses are located
path_to_data = "calibration_data2"

# Load the robot poses
robot_poses = []
with open(f'{path_to_data}/extrinsic_calibration_poses.txt', 'r') as f:
    for line in f:
        #print(line)
        #line = line[1:-1]
        pose = np.array(eval(line.strip()))
        robot_poses.append(pose)

# Process the images and find the chessboard corners
images = glob.glob(f'{path_to_data}/image_*.png')

for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Set the image shape here after reading the first image
    if image_shape is None:
        image_shape = gray.shape[::-1]

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (chessboard_size[0]-1, chessboard_size[1]-1), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (chessboard_size[0]-1, chessboard_size[1]-1), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(5000)

cv2.destroyAllWindows()

# Perform the camera calibration to get camera matrix, distortion coefficients, and rotation and translation vectors from camera to chessboard
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_shape, None, None)

# Save the camera matrix and distortion coefficients to a file
print("Camera matrix: \n", mtx)
print("Distortion coefficients: \n", dist)

np.savetxt('camera_matrix.txt', mtx)
np.savetxt('dist_coefficients.txt', dist)

# Initialize empty transformation matrix
hand_eye_transformation_matrix = np.eye(4)

# Perform hand-eye calibration
if len(objpoints) == len(robot_poses):
    # Convert the robot poses to rotation matrices and translation vectors
    robot_rotation_matrices = []
    robot_translation_vectors = []
    for pose in robot_poses:
        R, _ = cv2.Rodrigues(np.array(pose[3:]))  # Convert rotation vector to rotation matrix
        t = np.array(pose[:3]).reshape((3, 1))    # Extract translation vector
        robot_rotation_matrices.append(R)
        robot_translation_vectors.append(t)

    # Perform hand-eye calibration 
    r_matrix, tvec = cv2.calibrateHandEye(robot_rotation_matrices, robot_translation_vectors,
                                              rvecs, tvecs, 
                                              mtx, dist)

    # Form the transformation matrix
    hand_eye_transformation_matrix[:3, :3] = r_matrix
    hand_eye_transformation_matrix[:3, 3] = tvec.flatten()

    print("Transformation Matrix:\n", hand_eye_transformation_matrix)
else:
    print("Mismatch in the number of object points and robot poses")

# Save the transformation matrix to file
np.savetxt('hand_eye_transformation.txt', hand_eye_transformation_matrix)

# Calculate re-projection error
error_accumulated = 0
errors = []
for i in range(len(objpoints)):
    imgpoints_reprojected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints_reprojected, cv2.NORM_L2)/len(imgpoints_reprojected)
    error_accumulated += error
    errors.append(error)

# Calculate mean error
mean_error = error_accumulated/len(objpoints)
print( "total error: {}".format(mean_error))

# Calculate standard deviation
std_dev = np.std(errors)
k = 2 # Number of standard deviations away from mean considered as outliers
print("Standard deviation: {}".format(std_dev))

# Scatter plot of re-projection errors
plt.figure(figsize=(10, 6))
# Generate a list of indices to use as the x-axis for the scatter plot
indices = list(i for i in range(1, len(errors) + 1))
plt.scatter(indices, errors)
#plt.title('Scatter plot of re-projection errors over all corners')
plt.xlabel('Image number')
plt.ylabel('Re-projection error [pixels]')
plt.axhline(y=mean_error, color='b', linestyle='--')
plt.axhline(y=mean_error + k*std_dev, color='r', linestyle='--')
plt.axhline(y=mean_error - k*std_dev, color='r', linestyle='--')
# Add legend for the mean and standard deviation lines
plt.legend(['Single image error', 'Mean error', 'Mean ± 3*std'])
plt.xticks(indices)
plt.show()