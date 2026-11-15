import json
import numpy as np

from app.config import STATE_FILE, PROCESSED_DIR

# Load persistent project state
state = json.loads(STATE_FILE.read_text(encoding="utf-8"))

dataset = state["dataset"]
afe = state["afe"]

dataset_id = dataset["dataset_id"]
afe_dir = PROCESSED_DIR / dataset_id / "afe"

# Load saved arrays using memory mapping
features = np.load(
    afe_dir / "features.npy",
    mmap_mode="r",
)

afe_labels = np.load(
    afe_dir / "labels.npy",
    mmap_mode="r",
)

original_labels = np.load(
    PROCESSED_DIR / dataset_id / "labels.npy",
    mmap_mode="r",
)

# Validate saved configuration
config = json.loads(
    (afe_dir / "afe_config.json").read_text(encoding="utf-8")
)

print("Feature shape:", features.shape)
print("Feature dtype:", features.dtype)
print("Labels shape:", afe_labels.shape)
print("Labels match:", np.array_equal(afe_labels, original_labels))
print("All features finite:", np.isfinite(features).all())
print("AFE configuration:", config["method"])
print("AFE feature count:", config["total_features"])