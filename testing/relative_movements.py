import cv2
import mediapipe as mp
import pyrealsense2 as rs
import rtde_receive
import rtde_control
import numpy as np
import os
import time
  

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

# Load calibration data
def load_calibration_data(calibration_folder):
    with open(os.path.join(calibration_folder, "camera_matrix.txt"), 'r') as file:
        camera_matrix = np.loadtxt(file, delimiter=' ')
    with open(os.path.join(calibration_folder, "dist_coefficients.txt"), 'r') as file:
        dist_coeffs = np.loadtxt(file, delimiter=' ')
    with open(os.path.join(calibration_folder, "hand_eye_transformation.txt"), 'r') as file:
        hand_eye_transformation = np.loadtxt(file, delimiter=' ')
    
    return camera_matrix, dist_coeffs, hand_eye_transformation

# Initialize camera
def initialize_camera():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Start streaming
    pipeline.start(config)
    return pipeline

# Initialize robot
def initialize_robot(robot_ip_address, initial_pose):

    # Connect to robot
    rtde_c = rtde_control.RTDEControlInterface(robot_ip_address)
    rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip_address)
    
    # Move to a start position
    speed = 0.2
    acc = 0.2
    rtde_c.moveL(initial_pose, speed, acc, False)
    
    return rtde_c, rtde_r

# Tracking function for ArUco marker
def track_aruco(color_image, camera_matrix, dist_coefficients, aruco_dictionary, marker_length):
    # Dictionary of ArUco markers
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dictionary)
    aruco_parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_parameters)

    # Convert image to grayscale
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    
    # Detect markers
    corners, ids, rejected = detector.detectMarkers(gray_image)
    #corners, ids = cv2.aruco.detectMarkers(gray_image, aruco_dict, parameters=aruco_parameters)
    
    # If at least one marker detected
    if ids is not None:
        # TODO: Choose how to handle multiple markers
        rvec, tvec, trash = my_estimatePoseSingleMarker(corners[0], marker_length, camera_matrix, dist_coefficients)
        #print(rvec)
        # TODO: Convert tvec to the desired coordinate frame if necessary
        return tvec.flatten(), rvec
    else:
        # No marker detected
        return None, None

# Tracking function for human hand using MediaPipe
def track_hand(color_image, depth_frame, camera_matrix, dist_coefficients):
    # TODO: Initialize MediaPipe hand tracking
    # For example:
    # mp_hands = mp.solutions.hands
    # hands = mp_hands.Hands()
    
    # TODO: Perform hand tracking using MediaPipe on the color_image
    # TODO: Get the hand keypoint coordinates
    # TODO: Estimate depth using the depth_frame at the keypoint coordinates
    
    # Return the hand position (x, y, z) in the camera frame
    # Replace with actual hand tracking logic
    # return np.array([x, y, z])
    return None

# Main function
def main():
    calibration_folder = "../calibration"
    camera_matrix, dist_coefficients, hand_eye_transformation = load_calibration_data(calibration_folder)
    
    pipeline = initialize_camera()

    robot_ip_address = "192.168.0.90"
    initial_robot_pose = [0.214, 0.298, 0.829, 0.0179, -0.0120, -5.4805] #used in translation-test
    #initial_robot_pose = [0.649, 0.547, 1.066, 0.0071, 0.8116, 1.1933] #used in rotation test
    rtde_c, rtde_r = initialize_robot(robot_ip_address, initial_robot_pose)
    print("Finished initializing robot")

    # Align the depth frame to color frame
    align_to = rs.stream.color
    align = rs.align(align_to)

    # ArUco parameters
    # For big marker:
    #aruco_dictionary = cv2.aruco.DICT_5X5_100
    #marker_length = 0.1
    # For small marker:
    aruco_dictionary = cv2.aruco.DICT_6X6_100
    marker_length = 0.028

    # Open file where data will be saved
    data_file = open('../data/relative_movements_translation_small_with_images_closer.txt', 'w')

    # Start robot movement
    velocity = 0.2
    acceleration = 0.2
    blend = 0.01
    # Used in translation-test:
    pose_1 = [0.414, 0.298, 0.829, 0.0179, -0.0120, -5.4805, velocity, acceleration, blend]
    pose_2 = [0.414, 0.498, 0.829, 0.0179, -0.0120, -5.4805, velocity, acceleration, blend]
    pose_3 = [0.414, 0.498, 1.029, 0.0179, -0.0120, -5.4805, velocity, acceleration, 0]
    path = [pose_1, pose_2, pose_3]

    #pose_1 = [0.360, 0.507, 0.994, 0.5417, -0.3235, -5.4589, velocity, acceleration, blend]
    #pose_2 = [0.084, 0.542, 1.016, 3.2768, -0.7741, -4.9144, velocity, acceleration, 0]
    #path = [pose_1, pose_2]
    
    rtde_c.moveL(path, True)

    print("Før sleep")

    # Let robot start moving before evaluating robot movement condition
    time.sleep(0.1)
    
    print("Etter sleep")


    tcp_speed = rtde_r.getActualTCPSpeed()
    tcp_speed_norm = np.linalg.norm(np.array(tcp_speed))

    first = True

    try:
        while tcp_speed_norm > 0.001:  # Robot movement condition
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue
            
            print("Iteration")
            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # Perform tracking
            tvec, rvec = track_aruco(color_image, camera_matrix, dist_coefficients, aruco_dictionary, marker_length)
            if tvec is None or rvec is None:
                data_file.write("MISSING\n")
                continue
            print("Etter aruco track")

            if first is True:
                img_name = os.path.join("../data", f"image_start_translation_small_closer.png")
                cv2.imwrite(img_name, color_image)
                first = False
            # Or for hand tracking:
            # object_position = track_hand(color_image, depth_image, camera_matrix, dist_coefficients)
            
            # Get robot pose and speed
            tcp_pose = rtde_r.getActualTCPPose()
            tcp_speed = rtde_r.getActualTCPSpeed()
            tcp_speed_norm = np.linalg.norm(np.array(tcp_speed))

            # Save data to file
            timestamp = rtde_r.getTimestamp()

            line = (f"{timestamp},"
            f"{tcp_pose[0]},{tcp_pose[1]},{tcp_pose[2]},"
            f"{tcp_pose[3]},{tcp_pose[4]},{tcp_pose[5]},"
            f"{tcp_speed[0]},{tcp_speed[1]},{tcp_speed[2]},"
            f"{tcp_speed[3]},{tcp_speed[4]},{tcp_speed[5]},"
            f"{tvec[0]},{tvec[1]},{tvec[2]},"
            f"{rvec[0]},{rvec[1]},{rvec[2]}\n")
        
            data_file.write(line)

            #time.sleep(0.1)  # Add appropriate delay if necessary
    finally:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        img_name = os.path.join("../data", f"image_end_translation_small_closer.png")
        cv2.imwrite(img_name, color_image)
        
        pipeline.stop()
        # Clean up and close any other necessary connections
        rtde_c.disconnect()
        rtde_r.disconnect()
        data_file.close()

# Entry point of the script
if __name__ == "__main__":
    main()
