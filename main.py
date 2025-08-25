import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from data.loaders.abide_loader import get_abide_dataloaders
from models import model_factory
from training.trainers.loso_trainer import LOSOTrainer
from utils.logging_utils import setup_wandb


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dynamic run name update
    if cfg.domain_adaptation.use_grl:
        cfg.logging.run_name = f"{cfg.models.name}_grl_lambda{cfg.domain_adaptation.grl_lambda}"
    else:
        cfg.logging.run_name = f"{cfg.models.name}_baseline"

    setup_wandb(cfg)

    site_graphs, site_names = get_abide_dataloaders(cfg)
    print(f"Loaded {len(site_names)} sites: {site_names}")
    # print(f"site_graphs: {site_graphs}")

    model = model_factory(cfg, site_graphs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.lr)
    print(f"{model.__class__.__name__} initialized and moved to {device}")

    trainer = LOSOTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        config=cfg,
        site_names=site_names
    )

    trainer.train_loso(site_graphs)


if __name__ == '__main__':
    main()