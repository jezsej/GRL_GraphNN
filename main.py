# Entry point for the full codebase setup
# File: autism_gnn_domain_adaptation/main.py

import argparse
import torch
from config import load_config
from data.loaders.abide_loader import get_abide_dataloaders
from models import build_model
from training.trainers.loso_trainer import LOSOTrainer
from utils.logging_utils import setup_wandb


def main():
    parser = argparse.ArgumentParser(description='ABIDE Domain-Adapted GNNs')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    setup_wandb(config)

    dataloaders, site_names = get_abide_dataloaders(config)
    model = build_model(config['model']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])

    trainer = LOSOTrainer(model=model,
                          optimizer=optimizer,
                          device=device,
                          config=config,
                          site_names=site_names)

    trainer.train_loso(dataloaders)


if __name__ == '__main__':
    main()