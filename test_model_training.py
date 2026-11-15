
from app.services.model_training_service import train_classical_model
from app.services.state_service import load_project_state


state = load_project_state()

dataset = state.get("dataset")

if not dataset:
    raise RuntimeError("No saved dataset found.")

dataset_id = dataset.get("dataset_id")

if not dataset_id:
    raise RuntimeError("Dataset ID not found in project state.")

for model_key in [
    "naive_bayes",
    "decision_tree",
    "random_forest",
]:
    print(f"\nTraining: {model_key}")

    result = train_classical_model(
        dataset_id=dataset_id,
        model_key=model_key
    )

    print(result)
