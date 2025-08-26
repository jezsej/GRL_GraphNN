import os
import numpy as np
import torch
from torch_geometric.data import Data
from typing import Dict, List
from collections import defaultdict


def load_abide_graph_data(data_dir: str, use_1d_timeseries: bool = True,
                          edge_threshold: float = 0.25,
                          include_timeseries: bool = True,
                          return_site_mapping: bool = True) -> Dict[str, List[Data]]:
    """
    Loads ABIDE data into PyTorch Geometric Data objects.
    Supports .1D or .h5 inputs and attaches time series for DSAM.

    Args:
        data_dir (str): Path to the root ABIDE dataset directory
        use_1d_timeseries (bool): Whether to use .1D files or .h5 blobs
        edge_threshold (float): Percentile threshold for connectivity edge sparsity
        include_timeseries (bool): Whether to attach raw time series to `data` objects (for DSAM)
        return_site_mapping (bool): Whether to return a dict mapping site to list of Data objects

    Returns:
        Dict[site_name -> List[Data]]
    """
    import pandas as pd
    from nilearn.connectome import ConnectivityMeasure

    sitewise_data = defaultdict(list)

    phenotypic_path = os.path.join(data_dir, 'phenotypic.csv')
    phenotypic_df = pd.read_csv(phenotypic_path)

    if use_1d_timeseries:
        ts_dir = os.path.join(data_dir, 'timeseries/CC200')  # You can parameterize atlas
        for _, row in phenotypic_df.iterrows():
            sub_id = row['SUB_ID']
            site = row['SITE_ID']
            label = int(row['DX_GROUP']) - 1
            ts_path = os.path.join(ts_dir, f'{sub_id}.1D')
            if not os.path.exists(ts_path):
                continue
            try:
                ts = np.loadtxt(ts_path)
            except:
                continue

            ts = (ts - ts.mean(0)) / (ts.std(0) + 1e-5)
            ts = ts.T

            conn_measure = ConnectivityMeasure(kind='correlation')
            corr_matrix = conn_measure.fit_transform([ts])[0]

            threshold = np.percentile(np.abs(corr_matrix), edge_threshold * 100)
            adj = np.where(np.abs(corr_matrix) >= threshold, 1, 0)
            edge_index = np.array(np.nonzero(np.triu(adj, 1)))
            edge_attr = corr_matrix[edge_index[0], edge_index[1]].reshape(-1, 1)

            data = Data(
                x=torch.eye(ts.shape[0], dtype=torch.float32),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
                y=torch.tensor([label], dtype=torch.long),
            )
            data.site = site

            if include_timeseries:
                data.time_series = torch.tensor(ts, dtype=torch.float32)

            sitewise_data[site].append(data)

    else:
        # To be implemented: h5 support path if needed in future
        raise NotImplementedError("Only .1D time series supported in this version")

    return dict(sitewise_data)
