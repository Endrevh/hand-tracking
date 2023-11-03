import cv2
import mediapipe as mp
import pyrealsense2 as rs
import numpy as np

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

try:
    while True:
        # Wait for frames from the RealSense device
        frames = pipeline.wait_for_frames()

        aligned_frames = align.process(frames)
        
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not aligned_depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        aligned_depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Process the color image with MediaPipe
        results = hands.process(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

        # Draw hand landmarks and get depth info
        if results.multi_hand_landmarks:
            print("Number of hands detected: ", len(results.multi_hand_landmarks))
            for hand_landmarks in results.multi_hand_landmarks:
                #middle_finger_base = hand_landmarks.landmark[9]
                wrist = hand_landmarks.landmark[0]
                #x_middle_finger, y_middle_finger = int(middle_finger_base.x * color_image.shape[1]), int(middle_finger_base.y * color_image.shape[0])
                x_wrist, y_wrist = int(wrist.x * color_image.shape[1]), int(wrist.y * color_image.shape[0])
                
                # Sanity check 
                x_wrist = np.clip(x_wrist, 0, aligned_depth_image.shape[1] - 1)
                y_wrist = np.clip(y_wrist, 0, aligned_depth_image.shape[0] - 1)

                print(f"x_wrist: {x_wrist}, y_wrist: {y_wrist}")
                z_wrist = aligned_depth_frame.get_distance(x_wrist,y_wrist)

                #print("Depth to base of middle finger: ", z)
                print("Depth to wrist: ", z_wrist)
                #    #depth = depth_frame.get_distance(x, y)
                #    #print(f"Landmark at (x={x}, y={y}), Depth: {depth} meters")
                
                #for lm in hand_landmarks.landmark:
                #    x, y = int(lm.x * color_image.shape[1]), int(lm.y * color_image.shape[0])
                #    #depth = depth_frame.get_distance(x, y)
                #    #print(f"Landmark at (x={x}, y={y}), Depth: {depth} meters")
                mp_drawing.draw_landmarks(color_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display the image
        cv2.imshow('MediaPipe with RealSense', color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
