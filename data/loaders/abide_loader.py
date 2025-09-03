import os
import torch
import wandb
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from typing import Dict, List
from scipy.io import loadmat

def get_abide_dataloaders(config) -> (Dict[str, List[Data]], List[str]):
    """
    Loads ABIDE data from .1D + matrix files and returns:
    - Dict of site name -> list of Data objects
    - List of site names
    Also sets config.dataset.time_points based on T_min.
    """
    pheno = pd.read_csv(config.dataset.phenotypic_file)

    time_series_root = config.dataset.root
    atlas = config.dataset.atlas.lower()  
    conn_type = config.dataset.connectivity.lower() 
    threshold_percentile = config.dataset.edge_threshold
    dynamic = config.dataset.dynamic
    print(f"[INFO] Loading ABIDE data from: {time_series_root}")
    site_data = {}
    site_names = sorted(pheno["SITE_ID"].dropna().unique())
    print(f"[INFO] Found {len(site_names)} sites: {site_names}")
    # -------------------------------------
    # STEP 1: Scan all time series for T_min
    # -------------------------------------
    time_lengths = []
    for site in site_names:
        sub_df = pheno[pheno["SITE_ID"] == site]
        for _, row in sub_df.iterrows():
            sub_id = str(int(row["SUB_ID"]))
            subject_folder = os.path.join(time_series_root, sub_id)
            print(f"[DEBUG] Checking subject folder: {subject_folder}")
            if not os.path.exists(subject_folder): continue

            ts_files = [f for f in os.listdir(subject_folder) if f.endswith(".1D")]
            if not ts_files: continue
            ts_path = os.path.join(subject_folder, ts_files[0])
            try:
                ts = np.loadtxt(ts_path)
                time_lengths.append(ts.shape[0])  # T = num rows
            except Exception as e:
                print(f"[WARN] Failed to read {ts_path}: {e}")
                continue

    if len(time_lengths) == 0:
        raise RuntimeError("No valid .1D files found for computing T_min")

    T_min = config.dataset.time_points  # Use predefined time points
    T_mean = int(np.mean(time_lengths))
    T_median = int(np.median(time_lengths))
    T_max = max(time_lengths)

    print(f"[INFO] Global T stats for ABIDE .1D files:")
    print(f" - Min: {T_min}, Mean: {T_mean}, Median: {T_median}, Max: {T_max}")

    # -------------------------------------
    # STEP 2: Build site graphs
    # -------------------------------------
    for site in site_names:
        data_list = []
        sub_df = pheno[pheno["SITE_ID"] == site]

        for _, row in sub_df.iterrows():
            sub_id = str(int(row["SUB_ID"]))

            subject_folder = os.path.join(time_series_root, sub_id)
            print(f"[DEBUG] Processing subject folder: {subject_folder}")
            if not os.path.exists(subject_folder):
                continue

            # --- Time series ---
            ts_files = [f for f in os.listdir(subject_folder) if f.endswith(".1D")]
            if not ts_files:
                print(f"[WARN] Subject {sub_id} has no .1D files")
                continue
            ts_path = os.path.join(subject_folder, ts_files[0])
            try:
                ts = np.loadtxt(ts_path).T  # Shape: (N, T)
            except Exception:
                print(f"[WARN] Failed to read {ts_path}")
                continue
            if ts.shape[0] != config.dataset.num_nodes:
                continue
            if ts.shape[1] < T_min:
                continue

            ts_tensor = torch.tensor(ts[:, :T_min], dtype=torch.float)

            # --- Correlation matrix ---
            txt_path = os.path.join(subject_folder, f"{sub_id}_{atlas}_{conn_type}_matrix.txt")
            mat_path = os.path.join(subject_folder, f"{sub_id}_{atlas}_{conn_type}.mat")

            if os.path.exists(txt_path):
                try:
                    mat = np.loadtxt(txt_path)
                except Exception as e:
                    print(f"[WARN] Failed to load TXT matrix {txt_path}: {e}")
                    continue
            elif os.path.exists(mat_path):
                try:
                    mat_dict = loadmat(mat_path)
                    mat_key = [k for k in mat_dict.keys() if not k.startswith("__")][0]
                    mat = mat_dict[mat_key]
                except Exception as e:
                    print(f"[WARN] Failed to load MAT matrix {mat_path}: {e}")
                    continue
            else:
                print(f"[WARN] No matrix file found for subject {sub_id}")
                continue

            if mat.shape[0] != config.dataset.num_nodes:
                print(f"[WARN] Matrix shape mismatch for subject {sub_id}: expected {config.dataset.num_nodes}, got {mat.shape[0]}")
                continue

            # --- Graph Construction ---
            edge_mask = np.triu(np.ones_like(mat), k=1).astype(bool)
            values = np.abs(mat[edge_mask])
            threshold = np.percentile(values, threshold_percentile)
            mask = (np.abs(mat) >= threshold) & edge_mask
            edge_index = np.array(np.where(mask))
            edge_attr = mat[edge_index[0], edge_index[1]]
            
            if config.models.name == "BrainNetworkTransformer":
                node_features = mat.astype(np.float32)
            else:
                node_features = np.eye(mat.shape[0], dtype=np.float32)

            # --- Labels & Metadata ---
            y = int(row["DX_GROUP"]) - 1
            sex = int(row["SEX"]) - 1
            age = float(row["AGE_AT_SCAN"])
            ados = row.get("ADOS_TOTAL", -1.0)

            data = Data(
                x=torch.tensor(node_features, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                edge_attr=torch.tensor(edge_attr, dtype=torch.float).unsqueeze(1),
                y=torch.tensor([y], dtype=torch.long),
            )

            data.site = site
            data.age = torch.tensor([age], dtype=torch.float)
            data.sex = torch.tensor([sex], dtype=torch.long)
            data.ados = torch.tensor([ados], dtype=torch.float)
            data.subject_id = int(row["SUB_ID"])

            if dynamic:
                data.time_series = ts_tensor
                identity = torch.eye(config.dataset.num_nodes)
                data.pseudo = identity[data.edge_index[0]]

            data_list.append(data)

     
        if not data_list:
            print(f"[WARN] Skipping site {site} — no valid data")
            continue

        site_data[site] = data_list
        print(f"[INFO] Site {site}: {len(data_list)} subjects")

    valid_site_data = {k: v for k, v in site_data.items() if len(v) > 0}
    valid_site_names = list(valid_site_data.keys())

    # ----------------------------
    # Print & Export Dataset Stats
    # ----------------------------
    print("\n[SUMMARY] Dataset Statistics:")
    site_stats = []

    total_asd = 0
    total_td = 0
    total_subjects = 0

    for site, subjects in site_data.items():
        asd = sum([1 for d in subjects if d.y.item() == 0])
        td = sum([1 for d in subjects if d.y.item() == 1])
        total = len(subjects)

        site_stats.append({
            "Site": site,
            "ASD": asd,
            "TD": td,
            "Total": total
        })

        total_asd += asd
        total_td += td
        total_subjects += total
        print(f" - {site:10s} | ASD: {asd:3d} | TD: {td:3d} | Total: {total:3d}")

    # Add overall total
    site_stats.append({
        "Site": "TOTAL",
        "ASD": total_asd,
        "TD": total_td,
        "Total": total_subjects
    })

    print(f"\n[TOTAL] Subjects: {total_subjects}, ASD: {total_asd}, TD: {total_td}\n")

    # Save to CSV
    stats_df = pd.DataFrame(site_stats)
    stats_path = os.path.join(config.log_path, "stats/abide_site_stats.csv")
    os.makedirs(os.path.dirname(stats_path), exist_ok=True) # Ensure log directory exists
    stats_df.to_csv(stats_path, index=False) # Save site stats to CSV
    print(f"[SAVED] Site statistics saved to: {stats_path}")
    # wandb.log({"abide_site_stats": wandb.Table(dataframe=stats_df)})
    return valid_site_data, valid_site_names