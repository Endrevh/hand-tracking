import cv2
import mediapipe as mp
import pyrealsense2 as rs
import rtde_receive
import rtde_control
from enum import Enum

class TrackingState(Enum):
    SEARCHING = 1
    TRACKING = 2

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

# Tracking function for human hand using MediaPipe. Use average of all landmarks
def track_hand(color_image_rgb, depth_frame, camera_matrix):
    # Setup MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.01)
    
    # Process the color image with MediaPipe
    results = hands.process(color_image_rgb)

    landmarks_camera_coordinates = []
    if results.multi_hand_landmarks:
        # Convert from pixel to camera coordinates for all landmarks
        landmarks = results.multi_hand_landmarks[0]
        for landmark in landmarks.landmark:
            
            x_pixel, y_pixel = int(landmark.x * color_image_rgb.shape[1]), int(landmark.y * color_image_rgb.shape[0])

            # Sanity check 
            x_pixel = np.clip(x_pixel, 0, color_image_rgb.shape[1] - 1)
            y_pixel = np.clip(y_pixel, 0, color_image_rgb.shape[0] - 1)

            z_camera = depth_frame.get_distance(x_pixel, y_pixel)
            
            # Get x,y in camera coordinates using estimated depth and camera parameters
            x_camera = (x_pixel - camera_matrix[0,2]) * z_camera / camera_matrix[0,0]
            y_camera = (y_pixel - camera_matrix[1,2]) * z_camera / camera_matrix[1,1]

            landmarks_camera_coordinates.append([x_camera, y_camera, z_camera])

        print("detected")

        #Find average
        num_coordinates = len(coordinates_list)
        sum_x = sum(coord[0] for coord in coordinates_list)
        sum_y = sum(coord[1] for coord in coordinates_list)
        sum_z = sum(coord[2] for coord in coordinates_list)

        average_x = sum_x / num_coordinates
        average_y = sum_y / num_coordinates
        average_z = sum_z / num_coordinates

        return np.array(average_x, average_y, average_z)
    else:
        print("Not detected")
        return None, None

# Control robot towards desired position using servo
def servoL(rtde_c, target_pose, current_pose, time, lookahead_time, scaling_factor):
    acceleration = 0.1  # these inputs get ignored in the current version of ur_rtde
    speed = 0.1
    gain = 100  # minimum tolerated value is 100

    actual_target_position = np.array(target_pose)
    current_position = np.array(current_pose)

    intermediate_target_position = current_position + (actual_target_position - current_position) * scaling_factor

    intermediate_target_pose = intermediate_target_position.tolist()

    rtde_c.servoL(intermediate_target_pose, acceleration, speed, time, lookahead_time, gain)

def main():
    #Pseudocode:

    # initialize camera
    # initialize robot
    # load hand-eye transformation matrix from file
    # load camera matrix and distortion coefficients from file
    # Set tracking_state to SEARCHING

    # Look at algorithm 6 in specialization project for details on the while-loop


    # Remember to double check correct use of RGB and BGR in OpenCV





# Entry point of the script
if __name__ == "__main__":
    main()