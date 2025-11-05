import cv2
import mediapipe as mp
import numpy as np
import pickle

# Load model
with open('sign_language_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('gestures.pkl', 'rb') as f:
    gestures = pickle.load(f)

print("=" * 60)
print("REAL-TIME SIGN LANGUAGE DETECTION (MediaPipe)")
print("=" * 60)
print(f"Loaded gestures: {', '.join(gestures)}")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot access camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    gesture_text = "No hand detected"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            features = scaler.transform([landmarks])
            pred = model.predict(features)[0]
            proba = model.predict_proba(features)[0]
            confidence = np.max(proba) * 100
            gesture_text = f"{pred.upper()} ({confidence:.1f}%)"

    cv2.putText(frame, gesture_text, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    cv2.imshow("Sign Language Detection (MediaPipe)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\n✓ Detection ended.")
