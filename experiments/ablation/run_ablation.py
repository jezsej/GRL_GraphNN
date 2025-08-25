import os
import copy
import yaml
import subprocess


def run_ablation(base_config_path, ablations):
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    for key, options in ablations.items():
        for value in options:
            config = copy.deepcopy(base_config)
            section, param = key.split('.')
            config[section][param] = value

            tag = f"{param}_{value}"
            config['logging']['run_name'] = f"ablation_{tag}"
            save_path = f"config/tmp_ablation_{tag}.yaml"

            with open(save_path, 'w') as fout:
                yaml.safe_dump(config, fout)

            print(f"\n[INFO] Running ablation for {key}={value}")
            subprocess.run(["python", "main.py", "--config", save_path])


if __name__ == "__main__":
    BASE_CONFIG = "config/full_config.yaml"
    ABLATIONS = {
        "model.pooling": ["attention", "mean"],
        "training.lr": [0.001, 0.0001],
        "model.hidden_dim": [32, 64, 128],
        "model.dropout": [0.0, 0.3, 0.5],
        "training.weight_decay": [0.0, 0.0001, 0.001],
        "domain_adaptation.grl_lambda": [0.1, 0.5, 1.0],
        "domain_adaptation.domain_loss_weight": [0.5, 1.0, 2.0]
    }
    run_ablation(BASE_CONFIG, ABLATIONS)
