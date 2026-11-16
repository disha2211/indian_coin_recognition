
from app.services.state_service import load_project_state
from app.services.evaluation_service import evaluate_all_models


state = load_project_state()

dataset = state.get("dataset")

if not dataset:
    raise RuntimeError("No saved dataset found.")

print("Starting evaluation of all four models...")

result = evaluate_all_models(
    dataset_id=dataset["dataset_id"],
    dataset_metadata=dataset,
)

print("\n========== INDIVIDUAL MODEL RESULTS ==========")

for model_key, metrics in result["individual_results"].items():

    print(f"\nModel: {metrics['model_name']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro Precision: {metrics['precision_macro']:.4f}")
    print(f"Macro Recall: {metrics['recall_macro']:.4f}")
    print(f"Macro F1-score: {metrics['f1_macro']:.4f}")

print("\n========== MODEL COMPARISON ==========")

for model in result["comparison"]["models"]:
    print(
        f"{model['model_name']}: "
        f"Accuracy={model['accuracy']:.4f}, "
        f"Macro F1={model['f1_macro']:.4f}"
    )

print("\n========== BEST MODEL ==========")

best = result["comparison"]["best_model"]

print("Best model:", best["model_name"])
print("Macro F1-score:", best["f1_macro"])

print("\nSaved evaluation directory:")
print(result["evaluation_directory"])
