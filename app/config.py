import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
EXTRACTED_DIR = DATA_DIR / "extracted"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
ARTIFACTS_DIR = DATA_DIR / "artifacts"

ALLOWED_DATASET_EXTENSIONS = {"zip"}
ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}
IMAGE_SIZE = (128, 128)
TEST_SIZE = 0.20
RANDOM_STATE = 42

PREPROCESS_DTYPE = "float32"
NORMALIZATION_SCALE = 255.0
PREPROCESS_BATCH_SIZE = 32

PREPROCESS_METADATA_FILENAME = "preprocessing_metadata.json"
PROCESSED_IMAGES_FILENAME = "images.npy"
PROCESSED_LABELS_FILENAME = "labels.npy"
CLASS_MAPPING_FILENAME = "class_mapping.json"

STATE_FILENAME = "project_state.json"
STATE_FILE = ARTIFACTS_DIR / STATE_FILENAME

# Override these through environment variables when deploying on a machine
# with more/less disk capacity. These are byte limits, not RAM allocations.
MAX_CONTENT_LENGTH = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024**3)))  # 10 GiB
MAX_EXTRACTED_BYTES = int(os.getenv("MAX_EXTRACTED_BYTES", str(50 * 1024**3)))  # 50 GiB
MAX_ARCHIVE_FILES = int(os.getenv("MAX_ARCHIVE_FILES", "1000000"))
UPLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MiB

for directory in [UPLOAD_DIR, EXTRACTED_DIR, PROCESSED_DIR, MODELS_DIR, ARTIFACTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
