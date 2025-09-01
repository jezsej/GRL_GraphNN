
import os
import json
import pandas as pd

results = []
for file in os.listdir("ax_logs"):
    if file.endswith(".json"):
        path = os.path.join("ax_logs", file)
        with open(path, "r") as f:
            data = json.load(f)
        model_variant = file.replace(".json", "")
        row = {"variant": model_variant}
        row.update(data["best_parameters"])
        row["val_auc"] = data["val_auc"][0]
        results.append(row)

df = pd.DataFrame(results)
df.to_csv("ax_logs/summary.csv", index=False)
print("[INFO] Summary saved to ax_logs/summary.csv")
