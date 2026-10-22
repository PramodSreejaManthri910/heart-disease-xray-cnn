import json
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# --------------------------------------------------
# Configuration
# --------------------------------------------------

IMG_SIZE = (150, 150)
BATCH_SIZE = 32

TEST_DIR = os.path.join(
    "data",
    "test",
)

MODEL_PATH = os.path.join(
    "models",
    "heart_disease_cnn.keras",
)

CLASS_MAPPING_PATH = os.path.join(
    "models",
    "class_indices.json",
)


# --------------------------------------------------
# Validate Required Files
# --------------------------------------------------

def validate_paths():
    """Check that the model, class mapping, and test data exist."""

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. "
            "Run train_model.py first."
        )

    if not os.path.isfile(CLASS_MAPPING_PATH):
        raise FileNotFoundError(
            f"Class mapping not found: {CLASS_MAPPING_PATH}. "
            "Run train_model.py first."
        )

    if not os.path.isdir(TEST_DIR):
        raise FileNotFoundError(
            f"Test directory not found: {TEST_DIR}"
        )


# --------------------------------------------------
# Load Class Mapping
# --------------------------------------------------

def load_class_mapping():
    """Load the class indices generated during training."""

    with open(
        CLASS_MAPPING_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        class_indices = json.load(file)

    return {
        int(index): class_name
        for class_name, index in class_indices.items()
    }


# --------------------------------------------------
# Test Data
# --------------------------------------------------

def create_test_generator():
    """Create normalized test data without shuffling."""

    test_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0
    )

    return test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )


# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names,
):
    """Display the confusion matrix."""

    matrix = confusion_matrix(
        y_true,
        y_pred,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names,
    )

    display.plot(
        values_format="d"
    )

    plt.title(
        "Heart Disease Classification - Confusion Matrix"
    )

    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# ROC Curve
# --------------------------------------------------

def plot_roc_curve(
    y_true,
    probabilities,
    auc_score,
):
    """Display the ROC curve."""

    false_positive_rate, true_positive_rate, _ = (
        roc_curve(
            y_true,
            probabilities,
        )
    )

    plt.figure(figsize=(7, 6))

    plt.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"ROC-AUC = {auc_score:.4f}",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier",
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "Heart Disease Classification - ROC Curve"
    )

    plt.legend()
    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def main():

    validate_paths()

    class_mapping = load_class_mapping()

    print(
        f"Class mapping: {class_mapping}"
    )

    test_generator = create_test_generator()

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print(
        "\nRunning predictions on test data..."
    )

    probabilities = (
        model.predict(
            test_generator,
            verbose=1,
        )
        .ravel()
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    actual_labels = (
        test_generator.classes
    )

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    accuracy = accuracy_score(
        actual_labels,
        predictions,
    )

    precision = precision_score(
        actual_labels,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        actual_labels,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        actual_labels,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        actual_labels,
        probabilities,
    )

    print("\nEvaluation Results")
    print("=" * 40)

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    print(
        f"ROC-AUC   : {roc_auc:.4f}"
    )

    # --------------------------------------------------
    # Classification Report
    # --------------------------------------------------

    ordered_indices = sorted(
        class_mapping.keys()
    )

    class_names = [
        class_mapping[index]
        .replace("_", " ")
        .title()
        for index in ordered_indices
    ]

    print(
        "\nClassification Report"
    )

    print(
        classification_report(
            actual_labels,
            predictions,
            target_names=class_names,
            zero_division=0,
        )
    )

    # --------------------------------------------------
    # Visual Evaluation
    # --------------------------------------------------

    plot_confusion_matrix(
        actual_labels,
        predictions,
        class_names,
    )

    plot_roc_curve(
        actual_labels,
        probabilities,
        roc_auc,
    )


if __name__ == "__main__":
    main()
