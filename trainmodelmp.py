import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("=" * 60)
print("TRAINING SIGN LANGUAGE MODEL (MediaPipe Features)")
print("=" * 60)

def load_data(dataset_path='dataset'):
    X, y = [], []
    gestures = []

    for gesture_name in os.listdir(dataset_path):
        gesture_path = os.path.join(dataset_path, gesture_name)
        if os.path.isdir(gesture_path):
            gestures.append(gesture_name)
            for file in os.listdir(gesture_path):
                if file.endswith('.csv'):
                    data = np.loadtxt(os.path.join(gesture_path, file), delimiter=',')
                    X.append(data)
                    y.append(gesture_name)
    return np.array(X), np.array(y), gestures

X, y, gestures = load_data()

if len(X) == 0:
    print("❌ No data found! Run collect_data.py first.")
    exit()

print(f"Loaded {len(X)} samples from {len(gestures)} gestures.")
print(f"Gestures: {', '.join(gestures)}")

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# Train
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print(f"Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save
with open('sign_language_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('gestures.pkl', 'wb') as f:
    pickle.dump(gestures, f)

print("\n✓ Model, scaler, and gestures saved successfully!")
