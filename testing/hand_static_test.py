import cv2
import mediapipe as mp
import pyrealsense2 as rs
import numpy as np
import os
import time

# Configure depth and color streams from RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming from RealSense
pipeline.start(config)

# Align depth frames to color frames
align_to = rs.stream.color
align = rs.align(align_to)

# Setup MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

# Get camera matrix and distortion coefficients from file
camera_matrix_file = '../calibration/camera_matrix.txt'
dist_coeff_file = '../calibration/dist_coefficients.txt'

# Open file where data will be recorded    
data_file = open('../data/trash.txt', 'w')

# Load intrinsic parameters from file
camera_matrix = np.loadtxt(camera_matrix_file, delimiter=' ')
dist_coeffs = np.loadtxt(dist_coeff_file, delimiter=' ')

first = True

time.sleep(3.0)

start = time.time()

timestamp_seconds = time.time()

try:
    while (timestamp_seconds - start) < 1.0:
        # Wait for frames from the RealSense device
        frames = pipeline.wait_for_frames()
        #print("new frame")
        aligned_frames = align.process(frames)
        
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not aligned_depth_frame or not color_frame:
            continue
        
        
        # Convert images to numpy arrays
        aligned_depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply camera matrix and distortion coefficients
        #color_image = cv2.undistort(color_image, camera_matrix, dist_coeffs)

        # Process the color image with MediaPipe
        results = hands.process(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

        # Draw hand landmarks and get depth info
        if results.multi_hand_landmarks:
            #print("Number of hands detected: ", len(results.multi_hand_landmarks))
            for hand_landmarks in results.multi_hand_landmarks:
                # Convert from pixel to camera coordinates for all landmarks
                landmarks_camera_coordinates = []
                for landmark in hand_landmarks.landmark:
                    
                    x_pixel, y_pixel = int(landmark.x * color_image.shape[1]), int(landmark.y * color_image.shape[0])

                    # Sanity check 
                    x_pixel = np.clip(x_pixel, 0, aligned_depth_image.shape[1] - 1)
                    y_pixel = np.clip(y_pixel, 0, aligned_depth_image.shape[0] - 1)

                    z_camera = aligned_depth_frame.get_distance(x_pixel, y_pixel)

                    #print(z_camera)

                    x_camera = (x_pixel - camera_matrix[0,2]) * z_camera / camera_matrix[0,0]
                    y_camera = (y_pixel - camera_matrix[1,2]) * z_camera / camera_matrix[1,1]

                    landmarks_camera_coordinates.append([x_camera, y_camera, z_camera])
                
                # Prepare timestamp and save to file
                timestamp_seconds = time.time()
                timestamp_milliseconds = int((timestamp_seconds-start) * 1000)

                data_file.write(f"{timestamp_milliseconds},")
                for i, sublist in enumerate(landmarks_camera_coordinates):
                    # Convert each item in the sublist to a string and join with commas
                    line = ','.join(map(str, sublist))
                    # Write to file, add a comma except for the last sublist
                    data_file.write(line + (',' if i < len(landmarks_camera_coordinates) - 1 else ''))
                data_file.write("\n")
                # Save first image to file
                if first is True:
                    mp_drawing.draw_landmarks(color_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    img_name = os.path.join("../data", f"noise_test_image_3.png")
                    cv2.imwrite(img_name, color_image)
                    first = False

        # Display the image
        #cv2.imshow('MediaPipe with RealSense', color_image)
        #if cv2.waitKey(1) & 0xFF == ord('q'):

finally:
    # Stop streaming
    pipeline.stop()
