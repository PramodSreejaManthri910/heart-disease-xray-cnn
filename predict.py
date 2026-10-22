import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image


# --------------------------------------------------
# Configuration
# --------------------------------------------------

IMG_SIZE = (150, 150)

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

def validate_files(image_path):
    """Check that the model, class mapping, and image exist."""

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

    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )


# --------------------------------------------------
# Load Class Mapping
# --------------------------------------------------

def load_class_mapping():
    """Load the class labels generated during training."""

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
# Image Preprocessing
# --------------------------------------------------

def preprocess_image(image_path):
    """Load and preprocess a chest X-ray image."""

    image = Image.open(
        image_path
    ).convert("RGB")

    image = image.resize(
        IMG_SIZE
    )

    image_array = np.asarray(
        image,
        dtype=np.float32,
    )

    # Match training preprocessing
    image_array /= 255.0

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0,
    )

    return image, image_array


# --------------------------------------------------
# Prediction
# --------------------------------------------------

def predict_image(
    model,
    image_array,
    class_mapping,
):
    """Predict the class of a chest X-ray image."""

    probability = float(
        model.predict(
            image_array,
            verbose=0,
        )[0][0]
    )

    predicted_index = (
        1 if probability >= 0.5 else 0
    )

    predicted_class = class_mapping[
        predicted_index
    ]

    confidence = (
        probability
        if predicted_index == 1
        else 1.0 - probability
    )

    predicted_class = (
        predicted_class
        .replace("_", " ")
        .title()
    )

    return (
        predicted_class,
        probability,
        confidence,
    )


# --------------------------------------------------
# Display Prediction
# --------------------------------------------------

def display_result(
    image,
    predicted_class,
    probability,
    confidence,
):
    """Print and visualize the prediction."""

    print(
        "\nHeart Disease Prediction Result"
    )

    print("-" * 40)

    print(
        f"Predicted Class : {predicted_class}"
    )

    print(
        f"Model Score     : {probability:.4f}"
    )

    print(
        f"Confidence      : "
        f"{confidence * 100:.2f}%"
    )

    plt.figure(
        figsize=(6, 6)
    )

    plt.imshow(
        image,
        cmap="gray",
    )

    plt.title(
        f"{predicted_class} | "
        f"Confidence: {confidence * 100:.2f}%"
    )

    plt.axis("off")
    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Classify a chest X-ray image using "
            "the trained CNN model."
        )
    )

    parser.add_argument(
        "image",
        help="Path to the chest X-ray image",
    )

    args = parser.parse_args()

    validate_files(
        args.image
    )

    print(
        "Loading heart disease CNN model..."
    )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    class_mapping = load_class_mapping()

    print(
        f"Class mapping: {class_mapping}"
    )

    image, image_array = preprocess_image(
        args.image
    )

    (
        predicted_class,
        probability,
        confidence,
    ) = predict_image(
        model,
        image_array,
        class_mapping,
    )

    display_result(
        image,
        predicted_class,
        probability,
        confidence,
    )


if __name__ == "__main__":
    main()
