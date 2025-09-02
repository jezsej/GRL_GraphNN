import numpy as np
import random
import torch
import hydra
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
import wandb
import os
from omegaconf import DictConfig, OmegaConf
from data.loaders.abide_loader import get_abide_dataloaders
from models.model_factory import model_factory
from training.trainers.loso_trainer import LOSOTrainer


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    # Initialise WandB
    # wandb.login(anonymous="must", key=os.environ.get("WANDB_API_KEY"), verify=True)

    assert os.environ.get("WANDB_API_KEY") is not None, "WANDB_API_KEY not set in environment!"
    # -------------------------------
    # Manual patch for wandb sweep params
    # -------------------------------

    project_name = cfg.logging.project if cfg.local_host else "domainadaptation"

    wandb.init(project=project_name, entity=cfg.logging.entity)


    if "models" in wandb.config:
        model_name = wandb.config["models"]
        

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        with initialize(config_path="config/models", job_name="sweep_model_override"):
            model_cfg = compose(config_name=f"{model_name}.yaml")


        cfg.models = model_cfg

        # cfg.models.name = {"bgnn":"BrainGNN", "dsam":"SpatioTemporalModel",
        #                    "bnt":"BrainNetworkTransformer"}.get(model_name)
        cfg.models.alias = model_name
    
    if "dataset.site" in wandb.config:
        cfg.dataset.site = wandb.config["dataset.site"]

    if "domain_adaptation.use_grl" in wandb.config:
        cfg.domain_adaptation.use_grl = wandb.config["domain_adaptation.use_grl"]

    if "training.lr" in wandb.config:
        cfg.optimizer.lr_scheduler.base_lr = wandb.config["optimizer.lr_scheduler.base_lr"]

    if "training.seed" in wandb.config:
        cfg.training.seed = wandb.config["training.seed"]

    if "models.dropout" in wandb.config:
        cfg.models.dropout = wandb.config["models.dropout"]

    
    wandb.config.update(OmegaConf.to_container(cfg, resolve=True), allow_val_change=True)
    print(OmegaConf.to_yaml(cfg))


    seed = cfg.training.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print(f"Training model {cfg.models.name} with GRL={cfg.domain_adaptation.use_grl}")
    print(f"Learning rate: {cfg.training.lr} | Dropout: {cfg.models.dropout} | Seed: {cfg.training.seed}")


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dynamic run name update
    if cfg.domain_adaptation.use_grl:
        cfg.logging.run_name = f"{cfg.models.alias}-grl_lambda-{cfg.domain_adaptation.grl_lambda}"
    else:
        cfg.logging.run_name = f"{cfg.models.alias}-baseline"

    # setup_wandb(cfg)

    site_graphs, site_names = get_abide_dataloaders(cfg)
    print(f"Loaded {len(site_names)} sites: {site_names}")
    # print(f"site_graphs: {site_graphs}")
    if hasattr(cfg.dataset, "site") and cfg.dataset.site:
        if cfg.dataset.site not in site_names:
            raise ValueError(f"[ERROR] Requested site '{cfg.dataset.site}' not found in available sites: {site_names}")
        print(f"[INFO] Running LOSO for single site: {cfg.dataset.site}")
        site_graphs = {cfg.dataset.site: site_graphs[cfg.dataset.site]}
        site_names = [cfg.dataset.site]

    model = model_factory(cfg, site_graphs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.lr_scheduler.base_lr, weight_decay=cfg.optimizer.weight_decay)
    print(f"{model.__class__.__name__} initialised and moved to {device}")

    trainer = LOSOTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        config=cfg,
        site_names=site_names
    )

    trainer.train(site_graphs)



if __name__ == '__main__':
    main()