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
MAX_CONTENT_LENGTH = 512 * 1024 * 1024  # 512 MB

for directory in [UPLOAD_DIR, EXTRACTED_DIR, PROCESSED_DIR, MODELS_DIR, ARTIFACTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
