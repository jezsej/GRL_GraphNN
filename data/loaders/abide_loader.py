import os
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from data.utils.abide_data_processor import ABIDEDataProcessor


def get_abide_dataloaders(config):
    """
    Load ABIDE data and return dataloaders for each site for LOSO-CV.
    Returns:
        dict: Mapping of site_name -> (train_loader, test_loader)
        list: Ordered list of site names
    """
    processor = ABIDEDataProcessor(
        data_dir=config['dataset']['root'],
        atlas=config['dataset']['atlas'],
        connectivity=config['dataset']['connectivity']
    )

    site_data = {}
    site_names = processor.get_all_sites()

    for site in site_names:
        graphs, labels = processor.load_site_data(site)
        pyg_graphs = []
        for g, y in zip(graphs, labels):
            x = torch.tensor(g['node_features'], dtype=torch.float)
            edge_index = torch.tensor(g['edge_index'], dtype=torch.long)
            edge_attr = torch.tensor(g['edge_attr'], dtype=torch.float) if 'edge_attr' in g else None
            y = torch.tensor([y], dtype=torch.long)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            data.site = site
            pyg_graphs.append(data)
        site_data[site] = pyg_graphs

    return site_data, site_names

