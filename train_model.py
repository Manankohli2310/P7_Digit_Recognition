# train_model.py

import os
import joblib

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# Create Model Folder
# ==========================================

os.makedirs("model", exist_ok=True)

# ==========================================
# Load Dataset
# ==========================================

print("=" * 50)
print("Downloading MNIST Dataset...")
print("=" * 50)

mnist = fetch_openml(
    "mnist_784",
    version=1,
    as_frame=False
)

X = mnist.data
y = mnist.target.astype(int)

print(f"\nOriginal Dataset Shape: {X.shape}")

# ==========================================
# Use Subset for Faster Training
# ==========================================

X = X[:30000]
y = y[:30000]

print(f"Training Dataset Shape: {X.shape}")

# ==========================================
# Train Test Split
# ==========================================

print("\nSplitting Dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training Samples : {len(X_train)}")
print(f"Testing Samples  : {len(X_test)}")

# ==========================================
# Feature Scaling
# ==========================================

print("\nScaling Features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# Build SVM Model
# ==========================================

print("\nTraining SVM Model...")
print("This may take several minutes...")

svm_model = SVC(
    kernel="rbf",
    C=5,
    gamma="scale",
    probability=True,
    random_state=42
)

svm_model.fit(
    X_train_scaled,
    y_train
)

print("\nTraining Completed Successfully!")

# ==========================================
# Prediction
# ==========================================

print("\nMaking Predictions...")

y_pred = svm_model.predict(
    X_test_scaled
)

# ==========================================
# Evaluation
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)

print(f"\nAccuracy : {accuracy * 100:.2f}%")

print("\nClassification Report:\n")
print(
    classification_report(
        y_test,
        y_pred
    )
)

print("\nConfusion Matrix:\n")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

# ==========================================
# Save Model
# ==========================================

print("\nSaving Model...")

joblib.dump(
    svm_model,
    "model/svm_model.pkl"
)

joblib.dump(
    scaler,
    "model/scaler.pkl"
)

print("\nModel Saved Successfully!")

print("\nSaved Files:")
print("model/svm_model.pkl")
print("model/scaler.pkl")

print("\nProject Training Complete!")