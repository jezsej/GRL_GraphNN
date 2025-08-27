import os
import torch
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from data.loaders.abide_loader import get_abide_dataloaders
from models.model_factory import model_factory
from training.trainers.loso_trainer import LOSOTrainer
from utils.logging_utils import setup_wandb


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):

    assert os.environ.get("WANDB_API_KEY") is not None, "WANDB_API_KEY not set in environment!"
    wandb.init(project=cfg.logging.project, entity=cfg.logging.entity)
    print(OmegaConf.to_yaml(cfg))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Run name setup
    if cfg.domain_adaptation.use_grl:
        cfg.logging.run_name = f"{cfg.models.alias}-grl_lambda-{cfg.domain_adaptation.grl_lambda}"
    else:
        cfg.logging.run_name = f"{cfg.models.alias}-baseline"

    setup_wandb(cfg)

    # Load data
    site_graphs, site_names = get_abide_dataloaders(cfg)
    print(f"Loaded {len(site_names)} sites: {site_names}")

    # Build model + trainer
    model = model_factory(cfg, site_graphs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.lr)
    print(f"{model.__class__.__name__} initialised and moved to {device}")

    trainer = LOSOTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        config=cfg,
        site_names=site_names
    )

    # LOSO logic — single site vs full LOSO
    if hasattr(cfg.loso, "site_name") and cfg.loso.site_name:
        site_name = cfg.loso.site_name
        assert site_name in site_graphs, f"Site {site_name} not found in site_graphs!"
        print(f"[RUN MODE] Single-site LOSO on: {site_name}")
        trainer.train_loso_one_site(site_name, site_graphs)
    else:
        print("[RUN MODE] Full LOSO on all sites")
        trainer.train_loso(site_graphs)


if __name__ == '__main__':
    main()