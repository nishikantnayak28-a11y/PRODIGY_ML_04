import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Dataset path
dataset_path = "leapGestRecog"

# Gesture labels
gesture_names = [
    "palm",
    "l",
    "fist",
    "fist_moved",
    "thumb",
    "index",
    "ok",
    "palm_moved",
    "c",
    "down"
]

X = []
y = []

# Maximum images per gesture
max_images_per_class = 300

print("Loading images...")

for person in os.listdir(dataset_path):

    person_path = os.path.join(dataset_path, person)

    if not os.path.isdir(person_path):
        continue

    for gesture_folder in os.listdir(person_path):

        gesture_path = os.path.join(person_path, gesture_folder)

        label = int(gesture_folder.split('_')[0]) - 1

        count = 0

        for image_name in os.listdir(gesture_path):

            if count >= max_images_per_class:
                break

            image_path = os.path.join(gesture_path, image_name)

            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            img = cv2.resize(img, (64, 64))

            X.append(img.flatten())
            y.append(label)

            count += 1

print("Images Loaded Successfully!")

X = np.array(X)
y = np.array(y)

print("Dataset Shape:", X.shape)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training SVM Model...")

model = SVC(kernel='linear')

model.fit(X_train, y_train)

print("Training Complete!")

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "gesture_svm_model.pkl")

print("\nModel Saved Successfully!")
