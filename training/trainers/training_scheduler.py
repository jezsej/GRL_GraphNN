import torch

def get_scheduler(optimizer, config, total_steps):
    mode = config.mode.lower()

    if mode == "step":
        # convert milestone fractions to epochs
        milestones = [int(m * total_steps) for m in config.milestones]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=config.decay_factor
        )

    elif mode == "poly":
        class PolyLR(torch.optim.lr_scheduler._LRScheduler):
            def __init__(self, optimizer, total_iters, power=1.0, last_epoch=-1):
                self.total_iters = total_iters
                self.power = power
                super().__init__(optimizer, last_epoch)

            def get_lr(self):
                return [
                    base_lr * (1 - self.last_epoch / self.total_iters) ** self.power
                    for base_lr in self.base_lrs
                ]

        scheduler = PolyLR(
            optimizer,
            total_iters=total_steps,
            power=config.poly_power
        )

    elif mode == "cos":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps
        )

    else:
        raise ValueError(f"Unknown scheduler mode: {mode}")

    return scheduler