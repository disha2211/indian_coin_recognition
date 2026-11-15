
import inspect
import numpy as np

from app.services.state_service import load_project_state
from app.services.afe_service import apply_clahe
from app.config import PROCESSED_DIR

from pathlib import Path


state = load_project_state()
dataset = state.get("dataset")

if not dataset:
    raise RuntimeError("No saved dataset found.")

dataset_id = dataset["dataset_id"]

images_path = (
    Path(PROCESSED_DIR)
    / dataset_id
    / "images.npy"
)

images = np.load(images_path, mmap_mode="r")

print("\n--- CNN Input Verification ---")
print("Image array shape:", images.shape)
print("Image data type:", images.dtype)
print("Minimum pixel value:", float(images.min()))
print("Maximum pixel value:", float(images.max()))

print("\n--- Existing AFE CLAHE Function ---")
print(inspect.signature(apply_clahe))
print(inspect.getsource(apply_clahe))

print("\n--- One Sample Image ---")
sample = np.asarray(images[0])

print("Sample shape:", sample.shape)
print("Sample dtype:", sample.dtype)
print("Sample minimum:", float(sample.min()))
print("Sample maximum:", float(sample.max()))
