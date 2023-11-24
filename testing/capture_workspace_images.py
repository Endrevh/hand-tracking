import cv2
import pyrealsense2 as rs
import numpy as np
import os
import time


# Connect to the camera
pipeline = rs.pipeline()
config = rs.config()

# Enable color stream
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start the camera stream
pipeline.start(config)

# Directory where the images and poses will be saved
save_directory = "../data/workspace_images"
os.makedirs(save_directory, exist_ok=True)

# Function to save the color image from the camera
def save_color_image(frame, iteration):
    img_name = os.path.join(save_directory, f"{iteration:02}.png")
    cv2.imwrite(img_name, frame)
    print(f"Image saved: {img_name}")

# Main loop
for i in range(50,60):
    # Wait for Enter press
    input(f"Press Enter to capture. Number of images captured: {i}")
    
    time.sleep(2.0)

    # Capture frame-by-frame
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    
    if not color_frame:
        continue  # If no frame is captured, continue to the next iteration
    
    # Convert images to numpy arrays
    color_image = np.asanyarray(color_frame.get_data())
    
    # Save color image
    save_color_image(color_image, i+1)
    

# When everything done, release the pipeline
pipeline.stop()

print("Collection completed!")
