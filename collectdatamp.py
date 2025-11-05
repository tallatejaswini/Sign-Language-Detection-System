import cv2
import mediapipe as mp
import numpy as np
import os
import csv
import time

print("=" * 60)
print("SIGN LANGUAGE DATA COLLECTION (with MediaPipe)")
print("=" * 60)

# Setup folders
if not os.path.exists('dataset'):
    os.makedirs('dataset')

gesture_name = input("\nEnter gesture name (e.g., hello, thanks, ok, peace): ").strip().lower()
gesture_folder = f'dataset/{gesture_name}'

if not os.path.exists(gesture_folder):
    os.makedirs(gesture_folder)

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ ERROR: Cannot access camera.")
    exit()

print("\nINSTRUCTIONS:")
print("  - Keep your hand in the green box.")
print("  - Press SPACE to capture frame.")
print("  - Press 'Q' to quit.")
print("=" * 60)

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Process frame with MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # Draw rectangle
    cv2.rectangle(frame, (50, 50), (550, 450), (0, 255, 0), 2)
    cv2.putText(frame, f"Gesture: {gesture_name.upper()}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Images Captured: {count}", (10, 470),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Collect Hand Landmarks", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                file_path = os.path.join(gesture_folder, f'{gesture_name}_{count}.csv')
                with open(file_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(landmarks)
                print(f"✓ Saved: {file_path}")
                count += 1
        else:
            print("⚠️ No hand detected. Try again.")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\n✓ Data collection complete! Total samples: {count}")
