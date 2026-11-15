
from app.services.state_service import load_project_state
from app.services.dcnn_service import save_dcnn_training_plots


state = load_project_state()
dataset = state.get("dataset")

if not dataset:
    raise RuntimeError("No saved dataset found.")

result = save_dcnn_training_plots(
    dataset["dataset_id"]
)

print("\n--- DCNN Graph Results ---")

for key, value in result.items():
    print(f"{key}: {value}")
