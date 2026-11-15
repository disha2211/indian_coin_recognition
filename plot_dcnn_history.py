
import json
from pathlib import Path

import matplotlib.pyplot as plt

from app.services.state_service import load_project_state
from app.config import PROCESSED_DIR


state = load_project_state()
dataset = state.get("dataset")

if not dataset:
    raise RuntimeError("No saved dataset found.")

dataset_id = dataset["dataset_id"]

dcnn_dir = Path(PROCESSED_DIR) / dataset_id / "dcnn"

history_path = dcnn_dir / "training_history.json"

if not history_path.exists():
    raise FileNotFoundError("CNN training history not found.")

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
        raise ValueError(f"Missing history metric: {key}")

epochs = range(1, len(history["accuracy"]) + 1)

best_epoch = max(
    range(len(history["val_accuracy"])),
    key=lambda index: history["val_accuracy"][index],
) + 1

print("\n--- CNN Training Summary ---")
print("Completed epochs:", len(history["accuracy"]))
print("Best validation epoch:", best_epoch)
print(
    "Best validation accuracy:",
    round(max(history["val_accuracy"]) * 100, 2),
    "%",
)
print(
    "Training accuracy at best epoch:",
    round(history["accuracy"][best_epoch - 1] * 100, 2),
    "%",
)
print(
    "Validation loss at best epoch:",
    round(history["val_loss"][best_epoch - 1], 4),
)
print(
    "Training loss at best epoch:",
    round(history["loss"][best_epoch - 1], 4),
)

# Accuracy graph
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

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("DCNN Training and Validation Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()

accuracy_path = dcnn_dir / "accuracy_graph.png"
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

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("DCNN Training and Validation Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()

loss_path = dcnn_dir / "loss_graph.png"
plt.savefig(loss_path, dpi=150)
plt.close()

print("\nGraphs saved:")
print(accuracy_path)
print(loss_path)
