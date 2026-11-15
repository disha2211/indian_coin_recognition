import numpy as np

from app.config import PROCESSED_DIR
from app.services.afe_service import extract_afe_features


# Change this to your actual dataset ID.
DATASET_ID = "6caa6b1b872242ee9fccdfe65835d7e9"

images_path = (
    PROCESSED_DIR
    / DATASET_ID
    / "images.npy"
)

X = np.load(
    images_path,
    mmap_mode="r",
)

print("Processed dataset shape:", X.shape)

# Take the first actual preprocessed image.
image = X[0]

print("Single image shape:", image.shape)
print("Single image dtype:", image.dtype)
print(
    "Single image range:",
    float(image.min()),
    float(image.max()),
)

features = extract_afe_features(image)

print("AFE feature shape:", features.shape)
print("AFE feature dtype:", features.dtype)
print("AFE features:")
print(features)