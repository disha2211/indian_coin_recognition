
import json
from pathlib import Path

import numpy as np

from app.config import PROCESSED_DIR
from app.services.afe_service import apply_clahe
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DCNN_DIRNAME = "dcnn"
DCNN_IMAGES_FILENAME = "afe_images.npy"
DCNN_METADATA_FILENAME = "dcnn_input_metadata.json"


def prepare_dcnn_images(dataset_id, force_reprocess=False):
    """
    Apply the existing AFE CLAHE enhancement to every
    preprocessed RGB image and save grayscale images
    as a disk-backed NumPy array.
    """

    processed_dir = Path(PROCESSED_DIR) / dataset_id

    source_path = processed_dir / "images.npy"

    dcnn_dir = processed_dir / DCNN_DIRNAME
    dcnn_dir.mkdir(parents=True, exist_ok=True)

    output_path = dcnn_dir / DCNN_IMAGES_FILENAME
    metadata_path = dcnn_dir / DCNN_METADATA_FILENAME

    if not source_path.exists():
        raise FileNotFoundError(
            f"Preprocessed images not found: {source_path}"
        )

    images = np.load(source_path, mmap_mode="r")

    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(
            "Expected RGB image array with shape (N, H, W, 3)."
        )

    if images.dtype != np.float32:
        raise ValueError("Expected float32 preprocessed images.")

    if not np.isfinite(images).all():
        raise ValueError("Input images contain NaN or infinity.")

    if images.min() < 0 or images.max() > 1:
        raise ValueError("Input image values must be between 0 and 1.")

    image_count, height, width, _ = images.shape

    # Reuse an existing valid enhanced image array.
    if (
        not force_reprocess
        and output_path.exists()
        and metadata_path.exists()
    ):
        try:
            with open(metadata_path, "r", encoding="utf-8") as file:
                metadata = json.load(file)

            existing = np.load(output_path, mmap_mode="r")

            expected_shape = (image_count, height, width, 1)

            if (
                metadata.get("dataset_id") == dataset_id
                and metadata.get("image_count") == image_count
                and metadata.get("input_shape") == list(images.shape)
                and metadata.get("output_shape") == list(expected_shape)
                and existing.shape == expected_shape
                and existing.dtype == np.float32
            ):
                return {
                    **metadata,
                    "status": "reused",
                }

        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass

    # Write directly to disk instead of creating a large RAM array.
    temp_path = dcnn_dir / "afe_images.tmp.npy"

    enhanced_images = np.lib.format.open_memmap(
        temp_path,
        mode="w+",
        dtype=np.float32,
        shape=(image_count, height, width, 1),
    )

    try:
        for index in range(image_count):
            rgb_image = np.asarray(images[index])

            enhanced = apply_clahe(rgb_image)

            if enhanced.shape != (height, width):
                raise ValueError(
                    f"Unexpected AFE output shape at image {index}."
                )

            if not np.isfinite(enhanced).all():
                raise ValueError(
                    f"Invalid AFE values at image {index}."
                )

            enhanced_images[index, :, :, 0] = enhanced

        enhanced_images.flush()
        del enhanced_images

        # Replace the previous output only after processing succeeds.
        temp_path.replace(output_path)

    except Exception:
        del enhanced_images

        if temp_path.exists():
            temp_path.unlink()

        raise

    metadata = {
        "dataset_id": dataset_id,
        "method": "AFE CLAHE grayscale enhancement",
        "image_count": image_count,
        "input_shape": list(images.shape),
        "output_shape": [image_count, height, width, 1],
        "dtype": "float32",
        "normalization": "0-1",
        "status": "processed",
    }

    temp_metadata_path = metadata_path.with_suffix(".tmp")

    with open(temp_metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    temp_metadata_path.replace(metadata_path)

    return metadata



DCNN_MODEL_FILENAME = "coin_dcnn.keras"
DCNN_HISTORY_FILENAME = "training_history.json"
DCNN_TRAIN_INDICES_FILENAME = "train_indices.npy"
DCNN_VAL_INDICES_FILENAME = "validation_indices.npy"


class CoinImageSequence(Sequence):
    """Load image batches from disk-backed arrays."""

    def __init__(
        self,
        images,
        labels,
        indices,
        batch_size=16,
        shuffle=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.images = images
        self.labels = labels
        self.indices = np.asarray(indices)
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.indices))

        batch_indices = self.indices[start:end]

        X = np.asarray(
            self.images[batch_indices],
            dtype=np.float32,
        )

        y = np.asarray(
            self.labels[batch_indices],
            dtype=np.int32,
        )

        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def build_coin_cnn(input_shape, num_classes):
    """Build the proposed AFE-enhanced CNN classifier."""

    model = models.Sequential(
        [
            layers.Input(shape=input_shape),

            layers.Conv2D(
                32, (3, 3), activation="relu", padding="same"
            ),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(
                64, (3, 3), activation="relu", padding="same"
            ),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(
                128, (3, 3), activation="relu", padding="same"
            ),
            layers.MaxPooling2D((2, 2)),

            layers.GlobalAveragePooling2D(),

            layers.Dense(128, activation="relu"),
            layers.Dropout(0.4),

            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_coin_dcnn(dataset_id, force_retrain=False):
    """Train the CNN using AFE-enhanced images."""

    processed_dir = Path(PROCESSED_DIR) / dataset_id
    dcnn_dir = processed_dir / DCNN_DIRNAME

    images_path = dcnn_dir / DCNN_IMAGES_FILENAME
    labels_path = processed_dir / "labels.npy"

    split_dir = processed_dir / "split"
    train_path = split_dir / "train_indices.npy"
    test_path = split_dir / "test_indices.npy"

    model_path = dcnn_dir / DCNN_MODEL_FILENAME
    history_path = dcnn_dir / DCNN_HISTORY_FILENAME
    metadata_path = dcnn_dir / "dcnn_model_metadata.json"

    required_files = [
        images_path,
        labels_path,
        train_path,
        test_path,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required CNN input file missing: {path}"
            )

    images = np.load(images_path, mmap_mode="r")
    labels = np.load(labels_path, mmap_mode="r")

    train_indices = np.load(train_path)
    test_indices = np.load(test_path)

    if images.ndim != 4 or images.shape[-1] != 1:
        raise ValueError("CNN images must have shape (N, H, W, 1).")

    if len(images) != len(labels):
        raise ValueError("CNN image and label counts do not match.")

    if len(train_indices) == 0 or len(test_indices) == 0:
        raise ValueError("Training or testing split is empty.")

    if (
        np.any(train_indices < 0)
        or np.any(train_indices >= len(images))
        or np.any(test_indices < 0)
        or np.any(test_indices >= len(images))
    ):
        raise ValueError("Split contains invalid sample indices.")

    if np.intersect1d(train_indices, test_indices).size:
        raise ValueError("Training and testing indices overlap.")

    # Reuse a compatible saved model.
    if (
        not force_retrain
        and model_path.exists()
        and metadata_path.exists()
    ):
        try:
            with open(metadata_path, "r", encoding="utf-8") as file:
                old_metadata = json.load(file)

            if (
                old_metadata.get("dataset_id") == dataset_id
                and old_metadata.get("training_samples")
                == len(train_indices)
                and old_metadata.get("input_shape")
                == list(images.shape[1:])
                and old_metadata.get("model_file")
                == model_path.name
            ):
                tf.keras.models.load_model(model_path)

                return {
                    **old_metadata,
                    "status": "reused",
                }

        except Exception:
            pass

    # Map labels to consecutive class indices.
    classes = np.unique(labels)

    if len(classes) < 2:
        raise ValueError("CNN requires at least two classes.")

    if not np.isfinite(np.asarray(labels)).all():
        raise ValueError("Labels contain invalid values.")

    encoded_labels = np.searchsorted(
        classes,
        np.asarray(labels),
    ).astype(np.int32)

    training_labels = encoded_labels[train_indices]

    # Create validation data ONLY from the training split.
    train_sub_indices, val_indices = train_test_split(
        train_indices,
        test_size=0.20,
        random_state=42,
        stratify=training_labels,
    )

    # Save exact indices for reproducibility.
    np.save(
        dcnn_dir / DCNN_TRAIN_INDICES_FILENAME,
        train_sub_indices,
    )

    np.save(
        dcnn_dir / DCNN_VAL_INDICES_FILENAME,
        val_indices,
    )

    train_sequence = CoinImageSequence(
        images,
        encoded_labels,
        train_sub_indices,
        batch_size=16,
        shuffle=True,
    )

    validation_sequence = CoinImageSequence(
        images,
        encoded_labels,
        val_indices,
        batch_size=16,
        shuffle=False,
    )

    # Balance class contributions during training.
    class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(classes)),
    y=encoded_labels[train_sub_indices],
)

    class_weights = {
        index: float(weight)
        for index, weight in enumerate(class_weights_array)
    }

    model = build_coin_cnn(
        input_shape=images.shape[1:],
        num_classes=len(classes),
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=0.00001,
        ),
    ]

    history = model.fit(
        train_sequence,
        validation_data=validation_sequence,
        epochs=25,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    # Save the model after restoring the best validation weights.
    model.save(model_path)

    history_data = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }

    with open(history_path, "w", encoding="utf-8") as file:
        json.dump(history_data, file, indent=4)

    metadata = {
        "dataset_id": dataset_id,
        "model_name": "AFE-Enhanced Deep Convolutional Neural Network",
        "model_file": model_path.name,
        "input_shape": list(images.shape[1:]),
        "num_classes": len(classes),
        "class_labels": [
            value.item() if isinstance(value, np.generic) else value
            for value in classes
        ],
        "total_samples": len(images),
        "training_samples": len(train_sub_indices),
        "validation_samples": len(val_indices),
        "testing_samples": len(test_indices),
        "batch_size": 16,
        "maximum_epochs": 25,
        "completed_epochs": len(history.epoch),
        "optimizer": "Adam",
        "loss": "sparse_categorical_crossentropy",
        "class_weights": class_weights,
        "best_validation_accuracy": float(
            max(history.history["val_accuracy"])
        ),
        "best_validation_loss": float(
            min(history.history["val_loss"])
        ),
        "status": "trained",
    }

    temp_metadata_path = metadata_path.with_suffix(".tmp")

    with open(temp_metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    temp_metadata_path.replace(metadata_path)

    return metadata



def save_dcnn_training_plots(dataset_id):
    """
    Generate and save DCNN training/validation
    accuracy and loss graphs.
    """

    dcnn_dir = Path(PROCESSED_DIR) / dataset_id / DCNN_DIRNAME

    history_path = dcnn_dir / DCNN_HISTORY_FILENAME

    if not history_path.exists():
        raise FileNotFoundError(
            f"Training history not found: {history_path}"
        )

    with open(history_path, "r", encoding="utf-8") as file:
        history = json.load(file)

    required_keys = [
        "accuracy",
        "val_accuracy",
        "loss",
        "val_loss",
    ]

    for key in required_keys:
        if key not in history:
            raise ValueError(
                f"Missing training history field: {key}"
            )

    # Accuracy graph
    epochs = range(1, len(history["accuracy"]) + 1)

    plt.figure(figsize=(9, 5))

    plt.plot(
        epochs,
        history["accuracy"],
        label="Training Accuracy",
    )

    plt.plot(
        epochs,
        history["val_accuracy"],
        label="Validation Accuracy",
    )

    plt.title("DCNN Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    accuracy_path = dcnn_dir / "accuracy_plot.png"

    plt.savefig(accuracy_path, dpi=150)
    plt.close()

    # Loss graph
    plt.figure(figsize=(9, 5))

    plt.plot(
        epochs,
        history["loss"],
        label="Training Loss",
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss",
    )

    plt.title("DCNN Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    loss_path = dcnn_dir / "loss_plot.png"

    plt.savefig(loss_path, dpi=150)
    plt.close()

    return {
        "accuracy_plot": str(accuracy_path),
        "loss_plot": str(loss_path),
        "status": "saved",
    }

