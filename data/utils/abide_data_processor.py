import os
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
from nilearn.connectome import ConnectivityMeasure


class ABIDEDataProcessor:
    def __init__(self, data_dir, atlas='CC200', connectivity='correlation'):
        self.data_dir = data_dir
        self.atlas = atlas
        self.connectivity = connectivity
        self.subject_info = pd.read_csv(os.path.join(data_dir, 'phenotypic.csv'))
        self.time_series_dir = os.path.join(data_dir, f'timeseries/{atlas}')
        self.graph_cache = {}

    def get_all_sites(self):
        return sorted(self.subject_info['SITE_ID'].unique())

    def load_site_data(self, site):
        site_df = self.subject_info[self.subject_info['SITE_ID'] == site]
        graphs = []
        labels = []

        for _, row in site_df.iterrows():
            sub_id = row['SUB_ID']
            label = int(row['DX_GROUP'])  # 1 = ASD, 2 = TC
            ts_path = os.path.join(self.time_series_dir, f'{sub_id}.npy')
            if not os.path.exists(ts_path):
                continue
            ts = np.load(ts_path)
            ts = StandardScaler().fit_transform(ts.T).T
            graph = self.construct_graphs(ts)
            graphs.append(graph)
            labels.append(label - 1)  # remap to 0 = TC, 1 = ASD

        return graphs, labels

    def construct_graphs(self, time_series):
        if self.connectivity == 'correlation':
            corr = np.corrcoef(time_series)
        elif self.connectivity == 'partial_correlation':
            lw = LedoitWolf()
            prec = lw.fit(time_series.T).precision_
            d = np.sqrt(np.diag(prec))
            corr = -prec / np.outer(d, d)
            np.fill_diagonal(corr, 1.0)
        else:
            raise ValueError(f"Unsupported connectivity: {self.connectivity}")

        threshold = np.percentile(np.abs(corr), 75)
        adj = np.where(np.abs(corr) >= threshold, 1, 0)
        edge_index = np.array(np.nonzero(np.triu(adj, 1)))
        edge_attr = corr[edge_index[0], edge_index[1]]

        return {
            'node_features': np.eye(corr.shape[0]),
            'edge_index': edge_index,
            'edge_attr': edge_attr[:, None],
        }
