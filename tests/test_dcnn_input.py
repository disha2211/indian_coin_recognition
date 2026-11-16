
import numpy as np

from app.services.state_service import load_project_state
from app.services.dcnn_service import prepare_dcnn_images
from app.config import PROCESSED_DIR

from pathlib import Path


state = load_project_state()
dataset = state.get("dataset")

if not dataset:
    raise RuntimeError("No saved dataset found.")

dataset_id = dataset["dataset_id"]

print("Preparing AFE-enhanced CNN images...")

result = prepare_dcnn_images(dataset_id)

print("\n--- DCNN Input Preparation Results ---")

for key, value in result.items():
    print(f"{key}: {value}")

output_path = (
    Path(PROCESSED_DIR)
    / dataset_id
    / "dcnn"
    / "afe_images.npy"
)

enhanced = np.load(output_path, mmap_mode="r")

print("\n--- Verification ---")
print("Shape:", enhanced.shape)
print("Data type:", enhanced.dtype)
print("Minimum:", float(enhanced.min()))
print("Maximum:", float(enhanced.max()))
print("All finite:", bool(np.isfinite(enhanced).all()))
