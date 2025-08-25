import torch
import torch.nn as nn


class BaseGraphModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, **kwargs):
        super(BaseGraphModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

    def forward(self, data, return_attention=False):
        """
        Args:
            data (torch_geometric.data.Data): Input graph data
            return_attention (bool): Whether to return attention weights
        Returns:
            torch.Tensor: logits
            Optional[dict]: attention weights
        """
        raise NotImplementedError("Must be implemented in subclass")

    def extract_features(self, data):
        """
        Extract intermediate representations (e.g., before classification layer)
        Args:
            data (torch_geometric.data.Data)
        Returns:
            torch.Tensor: Feature representation
        """
        raise NotImplementedError("Must be implemented in subclass")