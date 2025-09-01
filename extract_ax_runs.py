import re
import glob
import ast
import pandas as pd
from pathlib import Path

# Point to your SLURM log folder
log_files = glob.glob("/Volumes/shared/ai4h/Shared/acq21js/hpc_log/ax_loso_*.out")

# Manual mapping from SLURM_ARRAY_TASK_ID to model and GRL
model_map = {
    0: ("dsam", False),
    1: ("dsam", True),
    2: ("bnt", False),
    3: ("bnt", True),
    4: ("braingnn", False),
    5: ("braingnn", True),
}

results = []

for file in log_files:
    filename = Path(file).stem  # e.g. ax_loso_123456_0
    print(f"Processing {filename}...")
    try:
        slurm_id = int(filename.split("_")[-1])
        print(f"SLURM ID: {slurm_id}")
        model_name, use_grl = model_map[slurm_id]
        print(f"Model: {model_name}, Use GRL: {use_grl}")
    except:
        print(f"Could not parse model/GRL from {filename}")
        continue

    with open(file, 'r') as f:
        content = f.read()
        # print(f"Content of {filename}:")
        # print(content)

    params_match = re.search(r"\[AX\] Best Parameters: (.+)", content)
    val_auc_match = re.search(r"\[AX\] Best val_auc: (.+)", content)
    trial_metrics_match = re.search(
        r"Trial results → AUC: ([\d.]+), F1: ([\d.]+), BalAcc: ([\d.]+), Acc: ([\d.]+)", content
    )

    if params_match and val_auc_match and trial_metrics_match:
        try:
            params = ast.literal_eval(params_match.group(1).strip())
            val_auc_dict = ast.literal_eval(val_auc_match.group(1).strip())
            val_auc = val_auc_dict[0]["val_auc"]  # safely extract val_auc value

            auc, f1, balacc, acc = map(float, trial_metrics_match.groups())
            results.append({
                "slurm_log": filename,
                "model": model_name,
                "use_grl": use_grl,
                "learning_rate": params["learning_rate"],
                "dropout": params["dropout"],
                "grl_lambda": params["grl_lambda"],
                "seed": params["seed"],
                "val_auc": val_auc,
                "test_auc": auc,
                "test_f1": f1,
                "test_bal_acc": balacc,
                "test_acc": acc
            })
        except Exception as e:
            print(f"Error parsing metrics in {filename}: {e}")
    else:
        print(f"Could not find all required metrics in {filename}")

df = pd.DataFrame(results)
df.to_csv("/Volumes/shared/ai4h/Shared/acq21js/code/abide/GRL_GraphNN/ax_logs/recovered_ax_results.csv", index=False)
print("✅ Saved recovered results to: ax_logs/recovered_ax_results.csv")