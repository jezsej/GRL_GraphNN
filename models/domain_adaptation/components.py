import torch
import torch.nn as nn
import numpy as np


class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x, alpha):
    return GradientReversalLayer.apply(x, alpha)



class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_domains, dropout=0.5, use_bn=True):
        super().__init__()

        def block(in_features, out_features):
            layers = [nn.Linear(in_features, out_features)]
            if use_bn:
                layers.append(nn.BatchNorm1d(out_features))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout))
            return nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            block(input_dim, hidden_dim),
            block(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, num_domains)
        )

    def forward(self, x, alpha=1.0):
        x = grad_reverse(x, alpha)
        return self.classifier(x)

class AdversarialLoss(nn.Module):
    def __init__(self, loss_type='cross_entropy', weight=1.0):
        super().__init__()
        self.weight = weight
        if loss_type == 'cross_entropy':
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def forward(self, domain_preds, domain_labels):
        return self.weight * self.loss_fn(domain_preds, domain_labels)



class GRLScheduler:
    def __init__(
        self,
        total_steps: int,
        schedule: str = "dann",         # "dann", "cosine", "linear"
        gamma: float = 10.0,
        warmup_steps: int = 0,
        ramp_steps: int = 1000,
        max_lambda: float = 1.0
    ):
        """
        GRL lambda scheduler with warm-up and ramp-up.
        
        Args:
            total_steps (int): Total number of training steps.
            schedule (str): Type of ramp schedule: "dann", "cosine", or "linear".
            gamma (float): Used only for "dann" schedule.
            warmup_steps (int): Number of steps to keep GRL at 0.
            ramp_steps (int): Number of steps over which lambda ramps to max_lambda.
            max_lambda (float): The maximum value GRL lambda can reach.
        """
        self.total_steps = total_steps
        self.schedule = schedule
        self.gamma = gamma
        self.warmup_steps = warmup_steps
        self.ramp_steps = ramp_steps
        self.max_lambda = max_lambda

    def get_lambda(self, step: int) -> float:
        if step < self.warmup_steps:
            return 0.0

        p = (step - self.warmup_steps) / max(self.ramp_steps, 1)
        p = min(p, 1.0)  # cap at 1.0

        if self.schedule == "dann":
            lambd = 2. / (1. + np.exp(-self.gamma * p)) - 1
        elif self.schedule == "cosine":
            lambd = 0.5 * (1 + np.cos(np.pi * (1 - p)))
        elif self.schedule == "linear":
            lambd = p
        else:
            raise ValueError(f"Unsupported GRL schedule: {self.schedule}")

        return min(lambd * self.max_lambda, self.max_lambda)