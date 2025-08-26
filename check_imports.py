import os
import importlib
import traceback
import sys

# Optionally add current directory to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

MODULES_TO_TEST = [
    "data.loaders.abide_loader",
    "models.dsam",
    "models.dsam.dsamcomponents",
    "models.dsam.dsamcomponents.models",
    "models.domain_adaptation",
    "models.bnt",
    "models.braingnn",
    "training.trainers",
    "training.trainers.loso_trainer",
    "evaluation.metrics",
    "evaluation.metrics.classification",
    "evaluation.visualisation",
    "evaluation.visualisation.roc_plot",
    "evaluation.visualisation.tsne_umap",
    "utils",
    "utils.logging_utils",
    # "run_loso",
]

def test_import(module_path):
    try:
        importlib.import_module(module_path)
        print(f"[OK] {module_path}")
    except Exception:
        print(f"[FAIL] {module_path}")
        traceback.print_exc()

if __name__ == "__main__":
    print("[INFO] Checking all key imports...\n")
    for module in MODULES_TO_TEST:
        test_import(module)
    print("\n[COMPLETE] ✅")