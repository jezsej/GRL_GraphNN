import torch
import hydra
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
    
    project_name = cfg.logging.project if cfg.local_host else "domainadaptation"
    
    wandb.init(project=project_name, entity=cfg.logging.entity)
    print(OmegaConf.to_yaml(cfg))
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

    trainer.train_loso(site_graphs)



if __name__ == '__main__':
    main()