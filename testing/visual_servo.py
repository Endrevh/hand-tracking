import cv2
import mediapipe as mp
import pyrealsense2 as rs
import rtde_receive
import rtde_control
import signal
import numpy as np
import os
import threading
import time
import sched
from enum import Enum
from datetime import datetime
from queue import Queue

class TrackingState(Enum):
    SEARCHING = 1
    TRACKING = 2

class Handedness(Enum):
    LEFT = 1
    RIGHT = 2

# Initialize camera
def initialize_camera(frequency):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, frequency)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, frequency)
    
    try:
        # Start the pipeline
        pipeline.start(config)
        print("RealSense camera initialized successfully.")
        return pipeline
    except Exception as e:
        print(f"Error initializing RealSense camera: {e}")
        return None

# Initialize robot
def initialize_robot(robot_ip_address):

    # Connect to robot
    rtde_c = rtde_control.RTDEControlInterface(robot_ip_address)
    rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip_address)
    
    return rtde_c, rtde_r

# Setup hand detection in MediaPipe
def initialize_hands(detection_confidence):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=detection_confidence)
    return hands

# Check if palm is facing towards camera by cross product
def check_hand_direction(landmark_coordinates):
    base_landmark = np.array(landmark_coordinates[0])
    first_direction = np.array(landmark_coordinates[1])
    second_direction = np.array(landmark_coordinates[2])

    # Calculate vectors
    vector1 = second_direction - base_landmark
    vector2 = first_direction - base_landmark

    # Calculate the cross product, only use x and y components of vector in the cross product
    cross_product = np.cross(vector1[0:2], vector2[0:2])

    # Determine the sign of the cross product
    if cross_product > 0:
        return True
    else:
        print("Palm not facing towards camera")
        return False

# Tracking function for human hand using MediaPipe
def track_hand(hand_detector, color_image_rgb, depth_frame, camera_matrix, coordinates_at_index, indices_for_hand_direction, annotate_image):

    mp_drawing = mp.solutions.drawing_utils
    
    # Process the color image with MediaPipe
    results = hand_detector.process(color_image_rgb)

    landmarks_pixel_coordinates = []

    if results.multi_hand_landmarks:
        
        # Check handedness
        handedness = Handedness.RIGHT
        if "Right" in str(results.multi_handedness[0].classification): #opposite because of flipped image
            handedness = Handedness.LEFT
            print("Left hand detected")

        # Convert to pixel coordinates for selected landmarks
        landmarks = results.multi_hand_landmarks[0]   
        for index in indices_for_hand_direction:
            landmark = landmarks.landmark[index]
            x_pixel, y_pixel = int(landmark.x * color_image_rgb.shape[1]), int(landmark.y * color_image_rgb.shape[0])

            # Sanity check 
            x_pixel = np.clip(x_pixel, 0, color_image_rgb.shape[1] - 1)
            y_pixel = np.clip(y_pixel, 0, color_image_rgb.shape[0] - 1)
            landmarks_pixel_coordinates.append([x_pixel, y_pixel])
        
        # Check if palm is facing towards camera
        palm_towards_camera = check_hand_direction(landmarks_pixel_coordinates)
        
        if annotate_image:
            # Draw landmarks on image
            mp_hands = mp.solutions.hands
            mp_drawing.draw_landmarks(color_image_rgb, landmarks, mp_hands.HAND_CONNECTIONS)

        # Convert to camera coordinates for only one landmark
        landmark = landmarks.landmark[coordinates_at_index]
        x_pixel, y_pixel = int(landmark.x * color_image_rgb.shape[1]), int(landmark.y * color_image_rgb.shape[0])

        # Sanity check
        x_pixel = np.clip(x_pixel, 0, color_image_rgb.shape[1] - 1)
        y_pixel = np.clip(y_pixel, 0, color_image_rgb.shape[0] - 1)
        z_camera = depth_frame.get_distance(x_pixel, y_pixel)
        x_camera = (x_pixel - camera_matrix[0,2]) * z_camera / camera_matrix[0,0]
        y_camera = (y_pixel - camera_matrix[1,2]) * z_camera / camera_matrix[1,1]

        print(f"Depth at wrist: {z_camera}")
        camera_coordinates = np.array([x_camera, y_camera, z_camera])

        if 0.0 in camera_coordinates:
            return None, None, None, None

        return camera_coordinates, color_image_rgb, handedness, palm_towards_camera
    else:
        print("No hand detected at all")
        return None, None, None, None

# Control robot towards desired position using servo
def servoL(rtde_c, target_pose, current_pose, time, lookahead_time, scaling_factor):
    acceleration = 0.1  # these two inputs get ignored in the current version of ur_rtde
    speed = 0.1
    gain = 100  # minimum tolerated value is 100

    actual_target_position = np.array(target_pose)
    current_position = np.array(current_pose)

    intermediate_target_position = current_position + (actual_target_position - current_position) * scaling_factor

    intermediate_target_pose = intermediate_target_position.tolist()

    rtde_c.servoL(intermediate_target_pose, acceleration, speed, time, lookahead_time, gain)

# Load calibration data
def load_calibration_data(calibration_folder):
    with open(os.path.join(calibration_folder, "camera_matrix.txt"), 'r') as file:
        camera_matrix = np.loadtxt(file, delimiter=' ')
    with open(os.path.join(calibration_folder, "dist_coefficients.txt"), 'r') as file:
        dist_coeffs = np.loadtxt(file, delimiter=' ')
    with open(os.path.join(calibration_folder, "hand_eye_transformation.txt"), 'r') as file:
        hand_eye_transformation = np.loadtxt(file, delimiter=' ')
    
    return camera_matrix, dist_coeffs, hand_eye_transformation

# Convert rotation vector to rotation matrix
def axis_angle_to_rotation_matrix(axis_angle):
    angle = np.linalg.norm(axis_angle)
    axis = axis_angle / angle if angle != 0 else np.zeros(3)
    skew_matrix = np.array([[0, -axis[2], axis[1]],
                            [axis[2], 0, -axis[0]],
                            [-axis[1], axis[0], 0]])
    rotation_matrix = np.eye(3) + np.sin(angle) * skew_matrix + (1 - np.cos(angle)) * np.dot(skew_matrix, skew_matrix)
    return rotation_matrix

# Function to transform a 3D vector using a 4x4 transformation matrix
def transform_vector(vector, transformation_matrix):
    homogeneous_vector = np.hstack((vector, 1))  # Convert 3D vector to homogeneous coordinates
    transformed_homogeneous_vector = np.dot(transformation_matrix, homogeneous_vector)  # Apply transformation
    transformed_vector = transformed_homogeneous_vector[:3]  # Convert back to 3D vector
    return transformed_vector

# Define the event
controller_ready_event = threading.Event()
running_flag = True

# Signal handler for Ctrl+C
def signal_handler(sig, frame):
    global running_flag
    print("Ctrl+C received. Stopping threads...")
    running_flag = False

def main():
    # load camera calibration from file
    calibration_folder = "../calibration"
    camera_matrix, dist_coefficients, T_flange_eye = load_calibration_data(calibration_folder)

    # initialize robot
    robot_ip_address = "192.168.0.90"
    rtde_c, rtde_r = initialize_robot(robot_ip_address)
    print("Finished initializing robot")    

    # Initialize hand detection in MediaPipe
    detection_confidence = 0.01
    hand_detector = initialize_hands(detection_confidence)

    # Initialize tracking_state
    tracking_state = TrackingState.SEARCHING

    # Initialize desired tracking offset
    tracking_offset_setpoint = 0.0

    frequency = 15
    dt = 1.0 / frequency

    # initialize camera
    pipeline = initialize_camera(frequency)

    # Align the depth frame to color frame
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Register the signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    timestamp = datetime.now().timestamp()

    time.sleep(5.0)

    counter = 0

    data_recorded = []

    while running_flag:
        counter += 1
        #if controller_ready_event.wait(0):
            #controller_ready_event.clear()
            #print("Main Thread: Controller Ready is True")
            
        frames = pipeline.wait_for_frames()
        print("-------------------------------")
        # Save timestamp
        #print(f"Diff: {datetime.now().timestamp() - timestamp}")
        timestamp = datetime.now().timestamp()

        aligned_frames = align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # Convert image to numpy array
        color_image = np.asanyarray(color_frame.get_data())
        
        color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        id_wrist = 0
        id_index_finger = 5
        id_pinky_finger = 17
        indices_for_hand_direction = [id_wrist, id_index_finger, id_pinky_finger]
        
        save_image = False
        if counter == 50:
            save_image = True

        hand_coordinates_eye, color_frame_annotated, handedness, palm_towards_camera = track_hand(hand_detector, color_image_rgb, depth_frame,
                                                                                                  camera_matrix, id_wrist, indices_for_hand_direction, save_image)
        if counter == 50:
            cv2.imwrite("../data/visual_servo_mediapipe_annotated_orange_glove_50.png", cv2.cvtColor(color_frame_annotated,cv2.COLOR_RGB2BGR))

        if hand_coordinates_eye is not None and handedness == Handedness.RIGHT and palm_towards_camera: # hand successfully detected
            print("detected")

            # Get current pose and speed
            flange_pose = rtde_r.getActualTCPPose()
            flange_speed = rtde_r.getActualTCPSpeed()

            flange_tvec = np.array(flange_pose[0:3])
            flange_rvec = np.array(flange_pose[3:6])

            # Build rotation matrix from axis angle vector
            R_base_flange = axis_angle_to_rotation_matrix(flange_rvec)

            # Build transformation matrix
            T_base_flange = np.eye(4)
            T_base_flange[:3, :3] = R_base_flange
            T_base_flange[:3, 3] = flange_tvec

            # Multiply T_base_flange with T_flange_eye to get T_base_eye
            T_base_eye = np.dot(T_base_flange, T_flange_eye)

            # Move hand coordinates to base frame
            hand_coordinates_base = transform_vector(hand_coordinates_eye, T_base_eye)

            if tracking_state == TrackingState.SEARCHING:
                tracking_offset_setpoint = hand_coordinates_base - flange_tvec
                tracking_state = TrackingState.TRACKING
            
            elif tracking_state == TrackingState.TRACKING:
                flange_tvec_desired = hand_coordinates_base - tracking_offset_setpoint
                pose_desired = np.hstack((flange_tvec_desired,flange_rvec))  # keep rotation vector unchanged

                # Save data to file
                timestamp = datetime.now().timestamp()

                line = (f"{timestamp},"
                f"{flange_pose[0]},{flange_pose[1]},{flange_pose[2]},"
                f"{flange_pose[3]},{flange_pose[4]},{flange_pose[5]},"
                f"{flange_speed[0]},{flange_speed[1]},{flange_speed[2]},"
                f"{flange_speed[3]},{flange_speed[4]},{flange_speed[5]},"
                f"{hand_coordinates_base[0]},{hand_coordinates_base[1]},{hand_coordinates_base[2]},"
                f"{pose_desired[0]},{pose_desired[1]},{pose_desired[2]},"
                f"{pose_desired[3]},{pose_desired[4]},{pose_desired[5]},"
                f"{tracking_offset_setpoint[0]},{tracking_offset_setpoint[1]},{tracking_offset_setpoint[2]}\n")

                data_recorded.append(line)

                lookahead_time = 0.1
                scaling_factor = 0.5

                # Only block the robot for remaining time of dt
                elapsed = datetime.now().timestamp() - timestamp
                servotime = dt - elapsed

                servoL(rtde_c, pose_desired, flange_pose, servotime, lookahead_time, scaling_factor)
                #control_queue.put((rtde_c, pose_desired, flange_pose, servotime, lookahead_time, scaling_factor))
                #print(f"Desired pose: {pose_desired}")
                #print(f"Current pose: {flange_pose}")
                #print(f"Hand pose: {hand_coordinates_base}")

                #print(f"Setpoint: {tracking_offset_setpoint}")
                #print(f"dt {dt}")
                #print(f"elapsed {elapsed}")
                #print(f"servotime {servotime}")

        else:
            #print("Not detected")
            data_recorded.append("NOT DETECTED\n")
            if tracking_state == TrackingState.SEARCHING:
                continue
            elif tracking_state == TrackingState.TRACKING:
                print("Stopping")
                #rtde_c.speedStop(1.0)
                print("Stopped")
                tracking_state = TrackingState.SEARCHING


    # Open data file and save data
    with open(f'../data/visual_servo_mediapipe_orange_glove_50.txt', 'w') as file:
        for line in data_recorded:
            file.write(line)

    # Wait for threads to finish
    #scheduler_thread.join()


    # Remember to double check correct use of RGB and BGR in OpenCV

# Entry point of the script
if __name__ == "__main__":
    main()