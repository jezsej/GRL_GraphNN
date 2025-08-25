import os
import subprocess
from datetime import datetime


def run(config_path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("runs", f"loso_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    cmd = [
        "python", "main.py",
        "--config", config_path
    ]

    print(f"[INFO] Running LOSO experiment with config: {config_path}")
    print(f"[INFO] Logs saved to: {log_dir}")

    with open(os.path.join(log_dir, "stdout.txt"), "w") as out:
        subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT)


if __name__ == "__main__":
    CONFIG_PATH = "config/full_config.yaml"
    run(CONFIG_PATH)
