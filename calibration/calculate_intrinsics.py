import numpy as np
import cv2
import glob

# Define the chessboard size and squares' size
chessboard_size = (18, 29)
square_size = 0.01  # 10mm in meters

# Arrays to store object points and image points from all the images
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0), ....,(6,5,0)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Termination criteria for the iterative algorithm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Load images for calibration
images = glob.glob('./extrinsic_data/*.jpg')  # The path to where the images are stored

for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # If found, add object points, image points
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

# Perform the camera calibration to get the camera matrix and distortion coefficients
ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Save the camera matrix and distortion coefficients to a file
np.savetxt('camera_matrix.txt', mtx)
np.savetxt('dist_coefficients.txt', dist)

# Optionally, you can save the camera matrix and distortion coefficients with more metadata as a .npz file
np.savez('camera_parameters.npz', mtx=mtx, dist=dist)

print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist)