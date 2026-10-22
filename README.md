# Heart Disease Prediction Using Deep Learning

A deep learning-based image classification project that uses a **Convolutional Neural Network (CNN)** to classify chest X-ray images into **HEART** and **NORMAL** categories.

The project explores the use of CNN-based medical image analysis with image preprocessing, data augmentation, regularization, adaptive learning-rate optimization, and comprehensive model evaluation.

> **Note:** This project was developed for academic and research purposes. It is not intended to replace professional medical diagnosis or clinical decision-making.

## Project Overview

The goal of this project is to explore how deep learning and image classification techniques can be applied to chest X-ray images for heart-condition classification.

The pipeline includes:

- Chest X-ray image preprocessing
- Image normalization and augmentation
- CNN-based feature extraction and classification
- Batch normalization and dropout regularization
- Adam optimization
- Early stopping
- Adaptive learning-rate reduction
- Test-set evaluation
- Single-image prediction

The original academic project reported an accuracy of approximately **85.2%**. Actual results may vary depending on the dataset, split, environment, and training configuration.

## Model Architecture

The model uses a sequential CNN architecture for binary image classification.

```text
Input Image
150 × 150 × 3
      │
      ▼
Conv2D — 32 Filters
Batch Normalization
Max Pooling
Dropout
      │
      ▼
Conv2D — 64 Filters
Batch Normalization
Max Pooling
Dropout
      │
      ▼
Conv2D — 128 Filters
Batch Normalization
Max Pooling
Dropout
      │
      ▼
Flatten
      │
      ▼
Dense — 256 Units
Batch Normalization
Dropout
      │
      ▼
Dense — 1 Unit
Sigmoid
      │
      ▼
HEART / NORMAL
```

The convolutional layers learn image features, while pooling reduces spatial dimensions. Batch normalization helps stabilize training, and dropout is used to reduce overfitting.

The final sigmoid layer performs binary classification.

## Training Optimizations

The training pipeline includes several techniques to improve generalization and training stability.

### Adam Optimizer

Adam is used to update model parameters with adaptive learning rates during training.

### Early Stopping

Training is stopped when validation loss stops improving, helping prevent unnecessary training and overfitting.

### ReduceLROnPlateau

The learning rate is automatically reduced when validation loss reaches a plateau, allowing smaller parameter updates as training progresses.

### Model Checkpointing

The model with the best validation loss is saved as:

```text
models/heart_disease_cnn.keras
```

## Data Preprocessing

All chest X-ray images are resized to:

```text
150 × 150 × 3
```

Pixel values are normalized to the range:

```text
0–1
```

Training augmentation includes:

- Rotation
- Width shifting
- Height shifting
- Shearing
- Zooming
- Horizontal flipping

A portion of the training dataset is reserved for validation.

## Dataset Structure

The expected dataset structure is:

```text
data/
├── train/
│   ├── HEART/
│   │   ├── image1.jpg
│   │   └── ...
│   │
│   └── NORMAL/
│       ├── image1.jpg
│       └── ...
│
└── test/
    ├── HEART/
    │   ├── image1.jpg
    │   └── ...
    │
    └── NORMAL/
        ├── image1.jpg
        └── ...
```

The dataset itself is not included in the repository.

Additional dataset setup instructions are available in `data/README.md`.

## Class Mapping

Keras automatically generates numeric labels from the class directories during training.

The mapping is saved to:

```text
models/class_indices.json
```

For example, depending on the directory names, the mapping may look like:

```json
{
    "HEART": 0,
    "NORMAL": 1
}
```

The evaluation and prediction scripts read this file automatically rather than assuming class indices.

## Evaluation Metrics

The model is evaluated on a separate test dataset using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC Curve
- ROC-AUC

These metrics provide a broader assessment than accuracy alone, particularly for binary classification.

## Tech Stack

- **Python**
- **TensorFlow**
- **Keras**
- **NumPy**
- **scikit-learn**
- **Matplotlib**
- **Pillow**

## Project Structure

```text
heart-disease-xray-cnn/
│
├── train_model.py
├── evaluate.py
├── predict.py
├── requirements.txt
├── .gitignore
├── README.md
│
├── data/
│   └── README.md
│
└── models/
    ├── heart_disease_cnn.keras
    └── class_indices.json
```

The `models` files are generated during training and are excluded from version control.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/PramodSreejaManthri910/heart-disease-xray-cnn.git
cd heart-disease-xray-cnn
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Train the CNN

First, place the dataset in the structure described above.

Then run:

```bash
python train_model.py
```

The training pipeline will:

1. Load the chest X-ray dataset
2. Normalize and augment the training images
3. Create training and validation generators
4. Build the CNN
5. Train using Adam
6. Apply EarlyStopping and ReduceLROnPlateau
7. Save the best-performing model
8. Save the class mapping
9. Evaluate the best model on the test dataset

The generated files are:

```text
models/heart_disease_cnn.keras
models/class_indices.json
```

## Evaluate the Model

Run:

```bash
python evaluate.py
```

The evaluation script calculates:

```text
Accuracy
Precision
Recall
F1 Score
ROC-AUC
```

It also displays:

- Classification report
- Confusion matrix
- ROC curve

## Predict a New Chest X-Ray

To classify an individual chest X-ray:

```bash
python predict.py path/to/xray.jpg
```

Example output:

```text
Heart Disease Prediction Result
----------------------------------------
Predicted Class : Heart
Model Score     : 0.2147
Confidence      : 78.53%
```

The exact class interpretation is determined using the saved `class_indices.json` file.

## Workflow

```text
Chest X-Ray Dataset
        │
        ▼
Image Preprocessing
        │
        ▼
Data Augmentation
        │
        ▼
CNN Feature Extraction
        │
        ▼
Binary Classification
        │
        ▼
Model Evaluation
        │
        ├──────────────► Accuracy
        ├──────────────► Precision
        ├──────────────► Recall
        ├──────────────► F1 Score
        └──────────────► ROC-AUC
        │
        ▼
New Chest X-Ray
        │
        ▼
HEART / NORMAL Prediction
```

## Reported Project Result

The original academic implementation reported approximately:

**85.2% classification accuracy**

This value represents the result reported during the original project experimentation and should not be interpreted as guaranteed performance for different datasets or populations.

## Future Improvements

Potential extensions include:

- Transfer learning with ResNet, EfficientNet, or DenseNet
- Larger and more diverse medical imaging datasets
- Explainable AI techniques such as Grad-CAM
- Hyperparameter optimization
- Cross-validation experiments
- Model calibration and threshold analysis
- Web-based inference interface
- External validation on independent datasets

## Disclaimer

This repository is an **academic deep learning project** demonstrating CNN-based chest X-ray image classification.

It is **not a validated medical device or clinical diagnostic system**. Predictions generated by the model should not be used for medical diagnosis, treatment decisions, or patient care.

## Author

**Pramod Sreeja Manthri**

GitHub: [PramodSreejaManthri910](https://github.com/PramodSreejaManthri910)
