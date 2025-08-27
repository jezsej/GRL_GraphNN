import os
import torch
from ax.service.managed_loop import optimize
from omegaconf import OmegaConf
import run_loso  # your main entry point


BASE_CONFIG_PATH = "configs/training/loso.yaml"

def evaluate(params):
    """
    Objective function for Ax.
    Must return a dict: {"metric_name": float, ...}
    """
    print(f"\n[Ax] Evaluating params: {params}")

    # Load base config
    cfg = OmegaConf.load(BASE_CONFIG_PATH)

    # Inject Ax hyperparameters
    cfg.optim.lr = params["learning_rate"]
    cfg.model.dropout = params["dropout"]
    cfg.domain_adaptation.grl_lambda = params["grl_lambda"]
    cfg.domain_adaptation.use_grl = True  # ensure GRL is active

    # Optional: disable wandb for sweeps
    cfg.wandb.mode = "disabled"

    # Run LOSO experiment (returns dict of per-site metrics)
    site_results = run_loso(cfg)

    # Compute macro average AUC across all sites
    macro_auc = sum(site_results[site]["auc"] for site in site_results) / len(site_results)

    print(f"[Ax] Macro AUC: {macro_auc:.4f}")
    return {"macro_auc": (macro_auc, 0.0)}  # return metric + SEM

if __name__ == "__main__":
    best_parameters, values, experiment, model = optimize(
        parameters=[
            {
                "name": "learning_rate",
                "type": "range",
                "bounds": [1e-5, 1e-2],
                "log_scale": True,
            },
            {
                "name": "dropout",
                "type": "range",
                "bounds": [0.0, 0.6],
            },
            {
                "name": "grl_lambda",
                "type": "range",
                "bounds": [0.0, 1.0],
            }
        ],
        objective_name="macro_auc",
        evaluation_function=evaluate,
        total_trials=20,
        minimize=False,
    )

    print("\n[Ax] Best config:")
    print(best_parameters)
    print("\n[Ax] Best macro AUC:")
    print(values)