# Biometrics-ML: Writer-Independent Handwriting Verification

A scalable handwriting verification system that combines **Deep Learning** and **Machine Learning** for text-independent writer authentication.

The framework extracts deep handwriting embeddings using **AlexNet**, trains person-specific classifiers, and verifies handwritten samples using majority voting over segmented handwriting patches.

---

## Features

- Handwriting image preprocessing
- Adaptive handwriting segmentation
- AlexNet-based deep feature extraction
- Person-specific binary classification
- Multiple ML classifiers:
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (SVM)
  - XGBoost
  - Decision Tree
  - K-Nearest Neighbors
  - Gaussian Naive Bayes
  - Multi-Layer Perceptron (MLP)
- PCA-based dimensionality reduction
- Hyperparameter tuning using GridSearchCV
- Confidence score generation
- Desktop and Web-based verification interfaces

---

## Project Pipeline

```
Handwriting Image
        │
        ▼
Image Preprocessing
(Grayscale → Denoising → Thresholding)
        │
        ▼
Handwriting Segmentation
(Connected Components + Sliding Windows)
        │
        ▼
AlexNet Feature Extraction
        │
        ▼
Feature Embeddings
        │
        ▼
PCA + Feature Scaling
        │
        ▼
Machine Learning Classifier
        │
        ▼
Segment-wise Prediction
        │
        ▼
Majority Voting
        │
        ▼
Writer Verification + Confidence Score
```

---

## Technologies Used

- Python
- OpenCV
- PyTorch
- TensorFlow/Keras
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Flask (Web Interface)

---

## Repository Structure

```
Biometrics-ML/
│
├── dataset/
├── preprocessing/
├── segmentation/
├── feature_extraction/
├── training/
├── models/
├── web_app/
├── desktop_app/
├── utils/
├── notebooks/
├── requirements.txt
└── README.md
```

---

## Machine Learning Models Evaluated

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- XGBoost
- Decision Tree
- KNN
- Gaussian Naive Bayes
- Multi-Layer Perceptron (MLP)

---

## Performance

The proposed framework achieved:

- **91.60% Test Accuracy**
- **95.38% Recall** (XGBoost)
- Robust writer verification using segment-level majority voting
- Real-time verification on both desktop and web platforms

---

## Applications

- Writer Authentication
- Document Verification
- Examination Security
- Forensic Handwriting Analysis
- Behavioral Biometrics
- Access Control Systems

---

## Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/<repository-name>.git
```

Move into the project

```bash
cd <repository-name>
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

Train the models

```bash
python train.py
```

Run verification

```bash
python app.py
```

or

```bash
python desktop_app.py
```

---

## Future Work

- Vision Transformer (ViT)-based embeddings
- ResNet feature extraction
- Dynamic handwriting analysis
- Open-set writer verification
- Larger multilingual handwriting datasets
- Cloud deployment

---

## Contributors

- **Ankur De**
  - 📧 ankurde2005@gmail.com

- **Pavithra S**
  - 📧 paviz2601@gmail.com

- **Jiya Sachdeva**
  - 📧 jiyabangalore123@gmail.com

---

## Acknowledgement

We sincerely thank **Dr. Peeta Basa Pati**, Department of Computer Science & Engineering, Amrita School of Computing, Bengaluru, for his valuable guidance and continuous support throughout this project.

---

## License

This project is intended for academic and research purposes.