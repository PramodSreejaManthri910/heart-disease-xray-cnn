import json
import os

import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# --------------------------------------------------
# Configuration
# --------------------------------------------------

IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4

TRAIN_DIR = os.path.join(
    "data",
    "train",
)

TEST_DIR = os.path.join(
    "data",
    "test",
)

MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "heart_disease_cnn.keras",
)

CLASS_MAPPING_PATH = os.path.join(
    MODEL_DIR,
    "class_indices.json",
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True,
)


# --------------------------------------------------
# Validate Dataset
# --------------------------------------------------

def validate_dataset():
    """Check that the expected dataset directories exist."""

    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(
            f"Training directory not found: {TRAIN_DIR}"
        )

    if not os.path.isdir(TEST_DIR):
        raise FileNotFoundError(
            f"Testing directory not found: {TEST_DIR}"
        )


# --------------------------------------------------
# Data Generators
# --------------------------------------------------

def create_data_generators():
    """
    Create augmented training data and normalized
    validation/test data generators.
    """

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2,
    )

    evaluation_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0
    )

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="training",
        shuffle=True,
        seed=42,
    )

    validation_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation",
        shuffle=False,
        seed=42,
    )

    test_generator = evaluation_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )

    return (
        train_generator,
        validation_generator,
        test_generator,
    )


# --------------------------------------------------
# Save Class Mapping
# --------------------------------------------------

def save_class_mapping(class_indices):
    """Save the class labels assigned by Keras."""

    with open(
        CLASS_MAPPING_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            class_indices,
            file,
            indent=4,
        )

    print(
        f"\nClass mapping: {class_indices}"
    )

    print(
        f"Class mapping saved to: "
        f"{CLASS_MAPPING_PATH}"
    )


# --------------------------------------------------
# CNN Architecture
# --------------------------------------------------

def build_model():
    """
    Build the CNN architecture used for binary
    heart-condition classification from chest X-rays.
    """

    model = Sequential(
        [
            tf.keras.Input(
                shape=(150, 150, 3)
            ),

            # Convolution Block 1
            Conv2D(
                32,
                (3, 3),
                padding="same",
                activation="relu",
            ),
            BatchNormalization(),
            MaxPooling2D(
                pool_size=(2, 2)
            ),
            Dropout(0.3),

            # Convolution Block 2
            Conv2D(
                64,
                (3, 3),
                padding="same",
                activation="relu",
            ),
            BatchNormalization(),
            MaxPooling2D(
                pool_size=(2, 2)
            ),
            Dropout(0.3),

            # Convolution Block 3
            Conv2D(
                128,
                (3, 3),
                padding="same",
                activation="relu",
            ),
            BatchNormalization(),
            MaxPooling2D(
                pool_size=(2, 2)
            ),
            Dropout(0.3),

            # Fully Connected Layers
            Flatten(),

            Dense(
                256,
                activation="relu",
            ),
            BatchNormalization(),
            Dropout(0.5),

            # Binary Classification
            Dense(
                1,
                activation="sigmoid",
            ),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(
                name="precision"
            ),
            tf.keras.metrics.Recall(
                name="recall"
            ),
            tf.keras.metrics.AUC(
                name="auc"
            ),
        ],
    )

    return model


# --------------------------------------------------
# Training Callbacks
# --------------------------------------------------

def create_callbacks():
    """Create callbacks for optimized CNN training."""

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_learning_rate = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.1,
        patience=5,
        min_lr=1e-7,
        verbose=1,
    )

    model_checkpoint = ModelCheckpoint(
        MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )

    return [
        early_stopping,
        reduce_learning_rate,
        model_checkpoint,
    ]


# --------------------------------------------------
# Main Training Pipeline
# --------------------------------------------------

def main():

    validate_dataset()

    (
        train_generator,
        validation_generator,
        test_generator,
    ) = create_data_generators()

    save_class_mapping(
        train_generator.class_indices
    )

    print(
        "\nTraining samples:",
        train_generator.samples,
    )

    print(
        "Validation samples:",
        validation_generator.samples,
    )

    print(
        "Test samples:",
        test_generator.samples,
    )

    print(
        "Image shape:",
        (*IMG_SIZE, 3),
    )

    model = build_model()

    model.summary()

    callbacks = create_callbacks()

    # --------------------------------------------------
    # Train
    # --------------------------------------------------

    model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # --------------------------------------------------
    # Load Best Model
    # --------------------------------------------------

    best_model = tf.keras.models.load_model(
        MODEL_PATH
    )

    # --------------------------------------------------
    # Test Evaluation
    # --------------------------------------------------

    print("\nEvaluating best model on test data...")

    test_results = best_model.evaluate(
        test_generator,
        verbose=1,
        return_dict=True,
    )

    print("\nTest Results")
    print("-" * 35)

    for metric_name, value in test_results.items():
        print(
            f"{metric_name}: {value:.4f}"
        )

    print(
        f"\nBest model saved to: {MODEL_PATH}"
    )


if __name__ == "__main__":
    main()
