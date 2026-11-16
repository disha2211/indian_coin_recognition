
from app.services.state_service import load_project_state
from app.services.dcnn_service import train_coin_dcnn


state = load_project_state()
dataset = state.get("dataset")

if not dataset:
    raise RuntimeError("No saved dataset found.")

dataset_id = dataset["dataset_id"]

print("Starting DCNN training...")

result = train_coin_dcnn(dataset_id)

print("\n--- DCNN Training Results ---")

for key, value in result.items():
    print(f"{key}: {value}")
