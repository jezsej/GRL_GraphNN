""" 
Brain Network Transformer (BNT) model. 
Implementation adapted from
https://github.com/Wayfear/BrainNetworkTransformer

"""


import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer
from .ptdec.dec import DEC
from typing import List
from .bntcomponents.transformer_encoder import InterpretableTransformerEncoder
from omegaconf import DictConfig
from .base import BaseModel




class TransPoolingEncoder(nn.Module):
    """
    Transformer encoder with Pooling mechanism.
    Input size: (batch_size, input_node_num, input_feature_size)
    Output size: (batch_size, output_node_num, input_feature_size)
    """

    def __init__(self, input_feature_size, input_node_num, hidden_size, output_node_num, pooling=True, orthogonal=True, freeze_center=False, project_assignment=True):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transformer = InterpretableTransformerEncoder(
            d_model=input_feature_size,
            nhead=4,
            dim_feedforward=hidden_size,
            batch_first=True,
            device=self.device
        )
       

        self.pooling = pooling
        if pooling:
            encoder_hidden_size = 32
            self.encoder = nn.Sequential(
                nn.Linear(input_feature_size *
                          input_node_num, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size,
                          input_feature_size * input_node_num),
            )
            self.dec = DEC(cluster_number=output_node_num, hidden_dimension=input_feature_size, encoder=self.encoder,
                           orthogonal=orthogonal, freeze_center=freeze_center, project_assignment=project_assignment)

    def is_pooling_enabled(self):
        return self.pooling

    def forward(self, x):
        if x is None:
            raise ValueError("Input x to TransPoolingEncoder is None!")
        assert x is not None, "Transformer input x is None during validation!"
        assert isinstance(x, torch.Tensor), f"Expected tensor, got {type(x)}"
        assert x.dim() == 3, f"Expected 3D tensor (batch_size, num_nodes, feature_dim), got {x.shape}"
        assert x.dtype == torch.float32, f"Expected float32, got {x.dtype}"
        assert x.device == next(self.transformer.parameters()).device, \
            f"x device {x.device} doesn't match model device {next(self.transformer.parameters()).device}"

        print(f"[DEBUG] Transformer input shape: {x.shape} | dtype: {x.dtype} | device: {x.device}")

        # Defensive print for supported device types
        for name, param in self.transformer.named_parameters():
            if param is None:
                print(f"[DEBUG] Transformer param '{name}' is None!")
            else:
                print(f"[DEBUG] Transformer param '{name}' device: {param.device}")
        x = self.transformer(x)
        if self.pooling:
            x, assignment = self.dec(x)
            return x, assignment
        return x, None

    def get_attention_weights(self):
        return self.transformer.get_attention_weights()

    def loss(self, assignment):
        return self.dec.loss(assignment)


class BrainNetworkTransformer(BaseModel):

    def __init__(self, config: DictConfig):

        super().__init__()

        self.attention_list = nn.ModuleList()
        forward_dim = config.dataset.num_nodes

        self.pos_encoding = config.models.pos_encoding
        if self.pos_encoding == 'identity':
            self.node_identity = nn.Parameter(torch.zeros(
                config.dataset.num_nodes, config.models.pos_embed_dim), requires_grad=True)
            forward_dim = config.dataset.num_nodes + config.models.pos_embed_dim
            nn.init.kaiming_normal_(self.node_identity)

        sizes = config.models.sizes
        sizes[0] = config.dataset.num_nodes
        in_sizes = [config.dataset.num_nodes] + sizes[:-1]
        do_pooling = config.models.pooling
        self.do_pooling = do_pooling
        for index, size in enumerate(sizes):
            self.attention_list.append(
                TransPoolingEncoder(input_feature_size=forward_dim,
                                    input_node_num=in_sizes[index],
                                    hidden_size=1024,
                                    output_node_num=size,
                                    pooling=do_pooling[index],
                                    orthogonal=config.models.orthogonal,
                                    freeze_center=config.models.freeze_center,
                                    project_assignment=config.models.project_assignment))

        self.dim_reduction = nn.Sequential(
            nn.Linear(forward_dim, 8),
            nn.LeakyReLU()
        )

        print(f"[DEBUG] Expected input to fc: {8 * sizes[-1]}")

        self.fc = nn.Sequential(
            nn.Linear(8 * sizes[-1], 256),
            nn.LeakyReLU(),
            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 2)
        )

    def forward(self,
                time_seires: torch.tensor,
                node_feature: torch.tensor):
        print(f"[DEBUG] BNT forward called with time_series shape: {time_seires.shape}, node_feature shape: {node_feature.shape}")
        bz, _, _, = node_feature.shape # (batch_size, num_nodes, feature_dim)

        if self.pos_encoding == 'identity':
            pos_emb = self.node_identity.expand(bz, *self.node_identity.shape)
            node_feature = torch.cat([node_feature, pos_emb], dim=-1)

        assignments = []

        for atten in self.attention_list:
            print(f"[DEBUG] Input to TransPoolingEncoder: {node_feature.shape}")
            node_feature, assignment = atten(node_feature)
            assignments.append(assignment)

        node_feature = self.dim_reduction(node_feature)

        node_feature = node_feature.reshape((bz, -1))

        return self.fc(node_feature)
    
    def extract_features(self, time_series, node_feature):
        bz = node_feature.shape[0]
        
        if self.pos_encoding == 'identity':
            pos_emb = self.node_identity.expand(bz, *self.node_identity.shape)
            node_feature = torch.cat([node_feature, pos_emb], dim=-1)

        for atten in self.attention_list:
            node_feature, _ = atten(node_feature)

        node_feature = self.dim_reduction(node_feature)
        node_feature = node_feature.reshape((bz, -1))
        print(f"[DEBUG] Extracted features shape: {node_feature.shape}")
        return node_feature  

    def get_attention_weights(self):
        return [atten.get_attention_weights() for atten in self.attention_list]

    def get_cluster_centers(self) -> torch.Tensor:
        """
        Get the cluster centers, as computed by the encoder.

        :return: [number of clusters, hidden dimension] Tensor of dtype float
        """
        return self.dec.get_cluster_centers()

    def loss(self, assignments):
        """
        Compute KL loss for the given assignments. Note that not all encoders contain a pooling layer.
        Inputs: assignments: [batch size, number of clusters]
        Output: KL loss
        """
        decs = list(
            filter(lambda x: x.is_pooling_enabled(), self.attention_list))
        assignments = list(filter(lambda x: x is not None, assignments))
        loss_all = None

        for index, assignment in enumerate(assignments):
            if loss_all is None:
                loss_all = decs[index].loss(assignment)
            else:
                loss_all += decs[index].loss(assignment)
        return loss_all