
from pathlib import Path

from app.services.prediction_service import predict_coin


DATASET_ID = "6caa6b1b872242ee9fccdfe65835d7e9"

# Change this to an actual image from your dataset.
IMAGE_PATH = Path(
    r"data\extracted\6caa6b1b872242ee9fccdfe65835d7e9\indian_coins_dataset\rupee_2\rupee2_001.JPG"
)

if __name__ == "__main__":

    result = predict_coin(
        image_path=IMAGE_PATH,
        dataset_id=DATASET_ID,
        model_key="random_forest",
    )

    print("\n========== COIN PREDICTION ==========")

    print("Model:", result["model_name"])
    print("Predicted coin:", result["predicted_category"])
    print("Confidence:", result["confidence_percentage"], "%")
