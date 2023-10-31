import mediapipe as mp
import cv2

print("Current mediapipe version is " + mp.__version__)

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True,
                       max_num_hands=2,
                       min_detection_confidence=0.4)

# Load image.
image_path = 'images/CR7.jpg'  # Update with your image path.
image = cv2.imread(image_path)
if image is None:
    raise ValueError("Image not found")

# Convert the image color to RGB.
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image and detect hands.
results = hands.process(image_rgb)

print("Number of hands detected:", len(results.multi_hand_landmarks))

# Draw hand landmarks.
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        #print("Hand landmarks:", hand_landmarks)
        #print("\nLandmarks in image coordinates:")
        for id, lm in enumerate(hand_landmarks.landmark):
            h, w, _ = image.shape  # height, width of the image
            x, y = int(lm.x * w), int(lm.y * h)
            #print(f"  - Landmark {id}: x={x}, y={y}")

        # Draw landmarks and connections.
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        

# Show image.
cv2.imshow('Hand Tracking', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Release resources
hands.close()
