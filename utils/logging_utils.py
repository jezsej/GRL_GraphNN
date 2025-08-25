import wandb


def setup_wandb(config):
    if not config.get('logging', {}).get('use_wandb', False):
        return

    wandb.init(
        project=config['logging'].get('project', 'autism-gnn'),
        name=config['logging'].get('run_name', 'default-run'),
        config=config
    )


def log_metrics_to_wandb(metrics, step=None, prefix=""):
    if wandb.run is None:
        return

    wandb.log({f"{prefix}{k}": v for k, v in metrics.items()}, step=step)
