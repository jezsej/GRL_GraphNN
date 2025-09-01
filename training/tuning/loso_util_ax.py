import numpy as np
import random
import torch
import wandb
import os
from omegaconf import OmegaConf
from data.loaders.abide_loader import get_abide_dataloaders
from models.model_factory import model_factory
from training.trainers.loso_trainer import LOSOTrainer


def run_loso(cfg, fold=None):
    """Stripped-down version of run_loso() without Hydra or WandB sweep overrides."""
    assert os.environ.get("WANDB_API_KEY") is not None, "WANDB_API_KEY not set in environment!"

    print(OmegaConf.to_yaml(cfg))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if cfg.domain_adaptation.use_grl:
        cfg.logging.run_name = f"{cfg.models.alias}-grl_lambda-{cfg.domain_adaptation.grl_lambda}"
    else:
        cfg.logging.run_name = f"{cfg.models.alias}-baseline"

    site_graphs, site_names = get_abide_dataloaders(cfg)
    # if hasattr(cfg.dataset, "site") and cfg.dataset.site:
    #     if cfg.dataset.site not in site_names:
    #         raise ValueError(f"[ERROR] Requested site '{cfg.dataset.site}' not found in available sites: {site_names}")
    #     if fold is None:
    #         print(f"[INFO] Running LOSO for single site: {cfg.dataset.site}")
    #         site_graphs = {cfg.dataset.site: site_graphs[cfg.dataset.site]}
    #         site_names = [cfg.dataset.site]

    model = model_factory(cfg, site_graphs).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.optimizer.lr_scheduler.base_lr,
        weight_decay=cfg.optimizer.weight_decay
    )

    trainer = LOSOTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        config=cfg,
        site_names=site_names if fold is None else fold
    )

    print(f"Running LOSO training for model: {cfg.models.alias}, GRL={cfg.domain_adaptation.use_grl}")
    return trainer.train(site_graphs)