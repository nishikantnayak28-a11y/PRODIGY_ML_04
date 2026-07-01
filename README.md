# Hand Gesture Recognition using Support Vector Machine (SVM)

## 📌 Overview

This project is part of my Machine Learning Internship at Prodigy Infotech.

The objective of this project is to develop a Hand Gesture Recognition system capable of accurately identifying different hand gestures from image data. The model uses image processing techniques and a Support Vector Machine (SVM) classifier to recognize gestures with high accuracy.

---

## 🎯 Objective

To build a machine learning model that can classify different hand gestures from images and enable gesture-based human-computer interaction systems.

---

## 📂 Dataset

Dataset Used: LeapGestRecog Dataset

Dataset Information:

- Total Images Used: 20,000
- Number of Gesture Classes: 10
- Image Type: Grayscale Hand Gesture Images

Gesture Classes:

1. Palm
2. L
3. Fist
4. Fist Moved
5. Thumb
6. Index
7. OK
8. Palm Moved
9. C
10. Down

⚠️ Dataset Notice

The LeapGestRecog dataset has not been uploaded to this repository because of its large size and GitHub file size limitations. Please download the dataset from the official Kaggle source and place the extracted dataset folder in the project directory before executing the program.

---

## 🛠️ Technologies Used

- Python
- NumPy
- OpenCV
- Scikit-Learn
- Joblib

---

## 📊 Data Preprocessing

The following preprocessing steps were performed:

1. Image Loading
2. Grayscale Image Processing
3. Image Resizing (64 × 64)
4. Feature Vector Creation
5. Label Encoding
6. Train-Test Split

---

## 🤖 Model Used

### Support Vector Machine (SVM)

Support Vector Machine is a supervised machine learning algorithm used for classification tasks.

Model Configuration:

- Algorithm: SVM (Support Vector Classifier)
- Kernel: Linear Kernel
- Train-Test Split: 80:20
- Random State: 42

---

## 📈 Model Performance

| Metric   | Value  |
| -------- | ------ |
| Accuracy | 100.0% |

### Classification Results

The model achieved perfect classification performance across all gesture categories.

- Precision: 1.00
- Recall: 1.00
- F1-Score: 1.00

This demonstrates the effectiveness of SVM for recognizing hand gestures in a controlled image dataset.

---

## 🚀 How to Run

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Project

```bash
python hand_gesture_recognition.py
```

---

## 🏆 Internship Information

Task: Hand Gesture Recognition

Internship: Prodigy Infotech Machine Learning Internship

Task Code: PRODIGY_ML_04

---

## 👨‍💻 Author

Ashutosh Agrawal

B.Tech CSE Student
Government College of Engineering, Kalahandi

Machine Learning Enthusiast | Aspiring Software Engineer
