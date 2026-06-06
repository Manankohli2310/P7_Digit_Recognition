# ✍️ Handwritten Digit Recognition using SVM and MNIST Dataset

## 📌 Project Overview

This project is a Machine Learning-based Handwritten Digit Recognition System that predicts digits from 0 to 9 using the MNIST dataset and a Support Vector Machine (SVM) classifier. Users can draw a digit on an interactive Streamlit canvas, and the trained model predicts the digit in real time.

The system uses image preprocessing techniques to convert user drawings into MNIST-style 28×28 grayscale images before making predictions. This improves the model's ability to recognize handwritten digits accurately.

---

## 🎯 Objectives

* Recognize handwritten digits from 0–9.
* Train a high-accuracy SVM classifier using the MNIST dataset.
* Build an interactive web application using Streamlit.
* Demonstrate image preprocessing and machine learning classification techniques.
* Provide real-time digit prediction through a user-friendly interface.

---

## 🧠 Algorithm Used

### Support Vector Machine (SVM)

Support Vector Machine (SVM) is a supervised machine learning algorithm used for classification tasks. It works by finding the optimal decision boundary (hyperplane) that separates different classes with the maximum margin.

For this project:

* Kernel: RBF (Radial Basis Function)
* Regularization Parameter (C): 5
* Probability Estimation Enabled
* Multi-Class Classification: One-vs-One Strategy

SVM was chosen because it provides excellent performance on high-dimensional image data such as MNIST.

---

## 📊 Dataset Information

Dataset: MNIST Handwritten Digits Dataset

Features:

* 70,000 handwritten digit images
* Image size: 28 × 28 pixels
* Total features per image: 784
* Classes: 10 (Digits 0–9)

Training Subset Used:

* 30,000 samples

This reduced training time while maintaining high classification accuracy.

---

## 🛠️ Technologies Used

* Python
* Scikit-Learn
* NumPy
* Streamlit
* Pillow (PIL)
* Joblib

---

## 📂 Project Structure

Handwritten_Digit_Recognition_SVM/

├── model/

│ ├── svm_model.pkl

│ └── scaler.pkl

├── app.py

├── train_model.py

├── requirements.txt

├── README.md

└── assets/

---

## ⚙️ Image Preprocessing Pipeline

The drawn digit undergoes the following preprocessing steps:

1. Convert image to grayscale.
2. Apply thresholding.
3. Detect and crop digit region.
4. Resize digit to 20×20 pixels.
5. Center digit on a 28×28 canvas.
6. Flatten image into 784 features.
7. Apply StandardScaler transformation.
8. Predict digit using SVM.

This preprocessing pipeline makes user drawings resemble MNIST images, improving prediction accuracy.

---

## 📈 Model Performance

### Accuracy Score

96.35%

### Classification Performance

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 96.35% |
| Precision | High   |
| Recall    | High   |
| F1 Score  | High   |

The model performs exceptionally well across most digit classes, with minor confusion between visually similar digits such as 7 and 8.

---

## 💻 Streamlit Features

* Interactive Drawing Canvas
* Real-Time Digit Prediction
* Confidence Score Display
* Improved MNIST-Style Preprocessing
* Clean and Responsive User Interface



## 🌐 Live Streamlit Application

Streamlit App:

[Add Your Streamlit Deployment Link Here]

Example:

https://your-app-name.streamlit.app

---

## 🎥 Project Demonstration

YouTube Demo:

[Add Your YouTube Video Link Here]

Example:

https://www.youtube.com/watch?v=YOUR_VIDEO_ID

---

## 📷 Application Workflow

User Draws Digit

↓

Image Preprocessing

↓

Feature Scaling

↓

SVM Classification

↓

Predicted Digit

↓

Confidence Score

---

## 🔮 Future Enhancements

* Upload handwritten digit images.
* Display Top-3 predictions.
* Confusion matrix visualization inside app.
* Model comparison with KNN and Random Forest.
* Deep Learning implementation using CNN.
* Mobile-friendly responsive interface.

---

## 🎓 Educational Concepts Covered

* Machine Learning Classification
* Support Vector Machines (SVM)
* Image Processing
* Feature Scaling
* Model Evaluation
* Streamlit Deployment
* MNIST Dataset Handling

---

## 📜 Conclusion

This project demonstrates the successful implementation of a Handwritten Digit Recognition System using Support Vector Machine (SVM) and the MNIST dataset. By combining machine learning with image preprocessing and a Streamlit-based user interface, the system achieves an accuracy of 96.35% while providing real-time handwritten digit prediction. The project serves as an excellent example of practical image classification using classical machine learning techniques.
