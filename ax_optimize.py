
import os
os.environ["WANDB_MODE"] = "disabled"
import wandb
import sys
import json
import torch
import random
import argparse
import numpy as np
from ax.service.managed_loop import optimize
from omegaconf import OmegaConf
from training.tuning.loso_util_ax import run_loso
from hydra import initialize, compose

class DummyWandbTable:
    def __init__(self, *args, **kwargs): pass

wandb.init(project="ax-dummy", mode="disabled")
wandb.log = lambda *args, **kwargs: None
wandb.watch = lambda *args, **kwargs: None
wandb.define_metric = lambda *args, **kwargs: None
wandb.Table = DummyWandbTable

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="Model name: dsam, bnt, bgnn")
parser.add_argument("--use_grl", required=True, type=str, help="Use GRL: true or false")
# parser.add_argument("--site", default=None, help="(Optional) site to restrict LOSO to")
parser.add_argument("--run_id", default=None)
parser.add_argument("--log_dir", default="ax_logs")
parser.add_argument("--base_config", default="config/config.yaml")
parser.add_argument("--trials", type=int, default=10)
args = parser.parse_args()

model_name = args.model
use_grl = args.use_grl.lower() == "true"
# site = args.site


def evaluate(params):
    print(f"Evaluating: {params}")
    with initialize(config_path="config", version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[f"models={model_name}"]
        )
        cfg.models.alias = model_name
        cfg.domain_adaptation.use_grl = use_grl
        cfg.optimizer.lr = params['learning_rate']
        cfg.models.dropout = params['dropout']
        cfg.domain_adaptation.grl_lambda = params['grl_lambda']
        cfg.training.epochs = 30
        # cfg.models.patience = 10
        # cfg.wandb.mode = "disabled"
        cfg.training.seed = int(params['seed'])
        # if site is not None and site != "":
        #     cfg.dataset.site = site

    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    random.seed(cfg.training.seed)

    sites = ["NYU", "PITT", "USM", "YALE", "UM_1"]

    results = run_loso(cfg, fold=sites)

    mean_val_auc = np.mean([site["val_auc"] for site in results])
    mean_val_acc = np.mean([site["val_acc"] for site in results])
    mean_test_auc = np.mean([site["test_auc"] for site in results])
    mean_test_f1 = np.mean([site["test_f1"] for site in results])
    mean_test_bal_acc = np.mean([site["test_bal_acc"] for site in results])
    mean_test_acc = np.mean([site["test_acc"] for site in results])

    print(f"Trial results → AUC: {mean_val_auc:.4f}, F1: {mean_test_f1:.4f}, BalAcc: {mean_test_bal_acc:.4f}, Acc: {mean_test_acc:.4f}")

    return {"val_auc": (mean_val_auc, 0.0), "val_acc": (mean_val_acc, 0.0), "test_auc": (mean_test_auc, 0.0), "test_f1": (mean_test_f1, 0.0), "test_bal_acc": (mean_test_bal_acc, 0.0), "test_acc": (mean_test_acc, 0.0)}

if __name__ == "__main__":
    best_parameters, values, experiment, model = optimize(
        parameters=[
            {"name": "learning_rate", "type": "range", "bounds": [1e-5, 1e-2], "log_scale": True},
            {"name": "dropout", "type": "range", "bounds": [0.0, 0.6]},
            {"name": "grl_lambda", "type": "range", "bounds": [0.0, 1.0]},
            {"name": "seed", "type": "choice", "values": [42, 1337], "sort_values": True}
        ],
        evaluation_function=evaluate,
        total_trials=10,
        objective_name="val_auc",
        minimize=False,
        )
    
    os.makedirs("/app/ax_logs", exist_ok=True)
    outfile = f"/app/ax_logs/{model_name}_{'grl' if use_grl else 'base'}.json"
    print(f"\n[AX] Best Parameters: {best_parameters}")
    print(f"[AX] Best val_auc: {values}")
    
    with open(outfile, "w") as f:
        json.dump({
            "best_parameters": best_parameters,
            "val_auc": values,
            "model": model_name,
            "use_grl": use_grl,
        }, f, indent=4)