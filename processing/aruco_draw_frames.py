import cv2
import cv2.aruco as aruco
import numpy as np
import os

def my_estimatePoseSingleMarker(corner, marker_size, mtx, distortion):
    '''
    This will estimate the rvec and tvec for the marker corner detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    '''
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)

    nada, R, t = cv2.solvePnP(marker_points, corner, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
    rvec = np.array([R[0][0],R[1][0],R[2][0]])
    tvec = np.array([t[0][0],t[1][0],t[2][0]])
    trash = nada
    return rvec, tvec, trash

def load_calibration_data(calibration_folder):
    with open(os.path.join(calibration_folder, "camera_matrix.txt"), 'r') as file:
        camera_matrix = np.loadtxt(file, delimiter=' ')
    with open(os.path.join(calibration_folder, "dist_coefficients.txt"), 'r') as file:
        dist_coeffs = np.loadtxt(file, delimiter=' ')
    
    return camera_matrix, dist_coeffs


# Load the image
image = cv2.imread('../data/image_start_rotation_big.png')

# Configure ArUco marker detector

# Small marker
# aruco_dictionary = cv2.aruco.DICT_6X6_100
# marker_length = 0.028

# Big marker
aruco_dictionary = cv2.aruco.DICT_5X5_100
marker_length = 0.10

aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dictionary)
aruco_parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_parameters)

# Load the camera calibration data
cameraMatrix, distCoeffs = load_calibration_data("../calibration")

# Convert image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect markers
corners, ids, rejected = detector.detectMarkers(gray_image)

# Draw detected markers and axes
if ids is not None:
    for corner, id in zip(corners, ids):
        rvec, tvec, _ = my_estimatePoseSingleMarker(corner, marker_length, cameraMatrix, distCoeffs)
        aruco.drawDetectedMarkers(image, corners)
        cv2.drawFrameAxes(image, cameraMatrix, distCoeffs, rvec, tvec, 0.15, 2)

# Display the result
cv2.imshow('ArUco Marker', image)
cv2.waitKey(0)
cv2.destroyAllWindows()