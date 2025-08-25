import torch
import torch.nn as nn


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
    def __init__(self, input_dim, hidden_dim, num_domains):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_domains)
        )

    def forward(self, x, alpha=1.0):
        x = grad_reverse(x, alpha)
        return self.net(x)


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
