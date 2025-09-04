import os
import torch
from torch_geometric.loader import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from omegaconf import OmegaConf
from data.loaders.abide_loader import get_abide_dataloaders
import hydra
from omegaconf import DictConfig

# ------------------------------
# CONFIG
# ------------------------------
CHECKPOINT_DIR = "/Users/jessessempijja/Documents/Adulting/Education/MSc/Dissertation/Domain Adaptation/checkpoints/best_model"
OUTPUT_DIR = "embeddings"
MODEL_NAME = "bnt"  # or "dsam", "bnt", "braingnn"
BATCH_SIZE = 16

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ------------------------------
# MODEL LOADER
# ------------------------------
def load_model_checkpoint(ckpt_path, cfg, dataloader=None):
    if cfg.models.name == "BrainNetworkTransformer":
        from models.bnt.bnt import BrainNetworkTransformer
        model = BrainNetworkTransformer(cfg)
    elif cfg.models.name == "BrainGNN":
        from models.braingnn.braingnn import Network as BrainGNN
        model = BrainGNN(cfg, dataloader)
    elif cfg.models.name == "SpatioTemporalModel":
        from models.dsam.dsam_original import SpatioTemporalModel
        model = SpatioTemporalModel(cfg, dataloader)
    else:
        raise ValueError(f"Unknown model type: {cfg.models.name}")

    # Step 2: Load state_dict
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model.to(device)

# ------------------------------
# EMBEDDING EXTRACTION
# ------------------------------
def extract_embeddings(model, dataloader, device, domain_id):
    model.to(device)
    model.eval()

    all_embeddings, all_labels, all_domains = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Embeddings"):
            batch = batch.to(device)

            # Match training input
            node_feature = batch.x.view(batch.num_graphs, 200, -1)  # N = 200 nodes
            time_series = batch.time_series

            feats = model.extract_features(time_series, node_feature)

            all_embeddings.append(feats.cpu())
            all_labels.append(batch.y.cpu())
            all_domains.append(torch.full_like(batch.y, fill_value=domain_id))

    return torch.cat(all_embeddings), torch.cat(all_labels), torch.cat(all_domains)
# ------------------------------
# t-SNE VISUALISATION
# ------------------------------
def plot_tsne(embeddings, domains, title="t-SNE by Domain"):
    tsne = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto")
    reduced = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=domains, palette="tab20", s=60, alpha=0.8)
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

# ------------------------------
# MAIN DRIVER
# ------------------------------
@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(OmegaConf.to_yaml(cfg))
    site_data, site_names = get_abide_dataloaders(cfg)
    all_emb, all_lbl, all_dom = [], [], []

    for site in site_names:
        domain_id = site_names.index(site)
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"Baseline-{MODEL_NAME}_{site}.pt")
        if not os.path.exists(ckpt_path):
            print(f"[SKIP] Missing checkpoint for {site}")
            continue
        print(f"[LOAD] Loading checkpoint for {site} from {ckpt_path}")
        dataloader = DataLoader(site_data[site], batch_size=BATCH_SIZE, shuffle=False)
        model = load_model_checkpoint(ckpt_path, cfg, dataloader)
        
        emb, lbl, dom = extract_embeddings(model, dataloader, device, domain_id)
        all_emb.append(emb)
        all_lbl.append(lbl)
        all_dom.append(dom)

        torch.save((emb, lbl, dom), os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_{site}.pt"))

    if len(all_emb) == 0:
        print("[ERROR] No embeddings found.")
        return

    all_emb = torch.cat(all_emb).cpu().numpy()
    all_dom = torch.cat(all_dom).cpu().numpy()
    plot_tsne(all_emb, all_dom, title=f"{MODEL_NAME.upper()} t-SNE by Site")

if __name__ == "__main__":
    main()