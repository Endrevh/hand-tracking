import cv2
import pyrealsense2 as rs
import numpy as np
import rtde_receive
import os

# Connect to the camera
pipeline = rs.pipeline()
config = rs.config()

# Enable color stream
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start the camera stream
pipeline.start(config)

# Connect to the UR10 CB3 robot
robot_ip = "192.168.0.90"  # Replace with correct UR10 robot IP address
rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)

# Directory where the images and poses will be saved
save_directory = "calibration_data2"
os.makedirs(save_directory, exist_ok=True)

# Function to save the color image from the camera
def save_color_image(frame, iteration):
    img_name = os.path.join(save_directory, f"image_{iteration:02}.png")
    cv2.imwrite(img_name, frame)
    print(f"Image saved: {img_name}")

# Function to save the TCP pose
def save_tcp_pose(pose):
    file_name = os.path.join(save_directory, f"extrinsic_calibration_poses.txt")
    with open(file_name, 'a') as file:
        file.write(str(pose)+"\n")
    print(f"TCP pose saved")

# Main loop
for i in range(20):
    # Wait for Enter press
    input(f"Press Enter to capture. Number of images captured: {i}")
    
    # Capture frame-by-frame
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    
    if not color_frame:
        continue  # If no frame is captured, continue to the next iteration
    
    # Convert images to numpy arrays
    color_image = np.asanyarray(color_frame.get_data())
    
    # Save color image
    save_color_image(color_image, i+1)
    
    # Get TCP pose from the robot
    tcp_pose = rtde_r.getActualTCPPose()
    
    # Save TCP pose
    save_tcp_pose(tcp_pose)

# When everything done, release the pipeline
pipeline.stop()

print("Collection completed!")
