import torch
import torch.nn as nn
from models.base import BaseGraphModel
from models.dsam.dsam_original import SpatioTemporalModel

class DSAMModel(BaseGraphModel):
    def __init__(self, input_dim, spatial_dim, temporal_dim, num_classes, cfg, dataloader=None):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes

        # Create native DSAM model from original repo logic
        self.model = SpatioTemporalModel(cfg, dataloader)

        # Final classifier head if needed
        self.classifier = nn.Linear(cfg.model.n_clustered_communities, num_classes)

    def forward(self, data):
        # Expecting: time_series, edge_index_tensor, edge_attr_tensor, node_feature, pseudo_torch, batch_tensor
        logits, _, _, _, _ = self.model(data, 
                                        data.time_series, 
                                        data.edge_index, 
                                        data.edge_attr,
                                        data.x, 
                                        data.pseudo,
                                        data.batch)
        return self.classifier(logits)

    def extract_features(self, data):
        with torch.no_grad():
            features, _, _, _, _ = self.model(data, 
                                              data.time_series, 
                                              data.edge_index, 
                                              data.edge_attr,
                                              data.x, 
                                              data.pseudo,
                                              data.batch)
        return features