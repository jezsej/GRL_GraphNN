import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    f1_score, balanced_accuracy_score
)
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import StratifiedShuffleSplit
from models.domain_adaptation.components import DomainDiscriminator, AdversarialLoss
from evaluation.visualisation.roc_plot import plot_roc, plot_macro_micro_roc
from evaluation.visualisation.tsne_umap import plot_embeddings
from tqdm import tqdm
import wandb
import copy

class LOSOTrainerParallel:
    def __init__(self, model, optimizer, device, config, site_names):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.site_names = site_names

        self.use_grl = config.domain_adaptation.use_grl
        self.grl_lambda = config.domain_adaptation.grl_lambda
        self.domain_weight = config.domain_adaptation.domain_loss_weight

        if self.use_grl:
            self.domain_discriminator = DomainDiscriminator(
                input_dim=config.models.hidden_dim,
                hidden_dim=64,
                num_domains=len(site_names)
            ).to(device)
            self.domain_loss_fn = AdversarialLoss(weight=self.domain_weight)

        self.best_model_path = os.path.join("checkpoints", "best_model")
        os.makedirs(self.best_model_path, exist_ok=True)

    def model_forward(self, batch):
        if self.config.models.name == "SpatioTemporalModel":
            num_nodes = self.config.dataset.num_nodes
            batch_size = batch.num_graphs
            device = batch.x.device

            identity = torch.eye(num_nodes, device=device)
            pseudo = identity.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, num_nodes)
            batch.pseudo = pseudo

            return self.model(
                batch,
                batch.time_series,
                batch.edge_index,
                batch.edge_attr,
                batch.x,
                batch.pseudo,
                batch.batch
            )
        else:
            return self.model(batch)

    def stratified_split(self, site_graphs, val_split):
        labels = [g.y.item() for g in site_graphs]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=42)
        indices = list(range(len(site_graphs)))
        train_idx, val_idx = next(sss.split(indices, labels))
        train_graphs = [site_graphs[i] for i in train_idx]
        val_graphs = [site_graphs[i] for i in val_idx]
        return train_graphs, val_graphs

    def log_distribution(self, name, graphs):
        labels = [g.y.item() for g in graphs]
        total = len(labels)
        asd = labels.count(1)
        td = labels.count(0)
        print(f"[{name}] Total: {total}, ASD: {asd}, TD: {td}")
        return asd, td

    def train_loso(self, site_data):
        print(f"Training on site: {site_data}...")
        results = {}
        all_y_true = []
        all_y_score = []
        metrics_summary = []

        for held_out_site in self.site_names:
            site_result = self.train_loso_one_site(site_data, held_out_site)
            results[held_out_site] = site_result["accuracy"]
            all_y_true.append(site_result["y_true"])
            all_y_score.append(site_result["y_score"])
            metrics_summary.append({k: site_result[k] for k in [
                "site", "accuracy", "auc", "sensitivity", "specificity", "f1_score", "balanced_accuracy"
            ]})

        os.makedirs("figures", exist_ok=True)
        macro_roc_path = "figures/macro_roc.png"
        plot_macro_micro_roc(all_y_true, all_y_score, save_path=macro_roc_path)

        os.makedirs("result/stats", exist_ok=True)
        df = pd.DataFrame(metrics_summary)
        df.to_csv("result/stats/metrics_summary.csv", index=False)
        print("\n[LOSO] All-site Results:")
        print(df.to_string(index=False))

        mean_metrics = df.mean(numeric_only=True)
        wandb.log({
            "overall/mean_test_acc": mean_metrics["accuracy"],
            "overall/mean_test_auc": mean_metrics["auc"],
            "overall/mean_test_sens": mean_metrics["sensitivity"],
            "overall/mean_test_spec": mean_metrics["specificity"],
            "overall/mean_test_f1": mean_metrics["f1_score"],
            "overall/mean_test_bal_acc": mean_metrics["balanced_accuracy"],
        })
        return results

    def train_loso_one_site(self, site_data, held_out_site):
        print(f"\n[LOSO] Held-out site: {held_out_site}")
        run = wandb.init(
            project=self.config.logging.project,
            entity=self.config.logging.entity,
            name=self.config.logging.run_name + "-site-" + held_out_site,
            config=OmegaConf.to_container(self.config, resolve=True),
            reinit=True
        )

        train_graphs, val_graphs, test_graphs = [], [], site_data[held_out_site]
        domain_labels = []

        for i, site in enumerate(self.site_names):
            if site == held_out_site:
                continue
            site_graphs = site_data[site]
            train_split, val_split = self.stratified_split(site_graphs, self.config.dataset.val_split)
            train_graphs.extend(train_split)
            val_graphs.extend(val_split)
            domain_labels.extend([i] * len(train_split))

        train_loader = DataLoader(train_graphs, batch_size=self.config.dataset.batch_size, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=self.config.dataset.batch_size, shuffle=False)
        test_loader = DataLoader(test_graphs, batch_size=self.config.dataset.batch_size, shuffle=False)
        domain_labels = torch.tensor(domain_labels).to(self.device)

        asd_train, td_train = self.log_distribution(f"Train (excluding {held_out_site})", train_graphs)
        asd_val, td_val = self.log_distribution(f"Val (excluding {held_out_site})", val_graphs)
        asd_test, td_test = self.log_distribution(f"Test ({held_out_site})", test_graphs)
        wandb.log({
            f"{held_out_site}/train_asd": asd_train,
            f"{held_out_site}/train_td": td_train,
            f"{held_out_site}/val_asd": asd_val,
            f"{held_out_site}/val_td": td_val,
            f"{held_out_site}/test_asd": asd_test,
            f"{held_out_site}/test_td": td_test
        })

        best_auc = 0
        patience = self.config.models.patience
        min_delta = self.config.models.min_delta
        stop_counter = 0
        best_model = None

        for epoch in tqdm(range(self.config.training.epochs), desc=f"[Training] {held_out_site}"):
            train_loss = self.train_epoch(train_loader, domain_labels)
            val_auc, val_loss, val_acc = self.validate(val_loader)

            wandb.log({
                f"{held_out_site}/val_auc": val_auc,
                f"{held_out_site}/val_loss": val_loss,
                f"{held_out_site}/val_acc": val_acc,
                f"{held_out_site}/train_loss": train_loss,
                "epoch": epoch
            })

            if val_auc > best_auc + min_delta:
                best_auc = val_auc
                best_model = copy.deepcopy(self.model.state_dict())
                stop_counter = 0
            else:
                stop_counter += 1
                if stop_counter >= patience:
                    print(f"[Early Stopping] Epoch {epoch} for site {held_out_site}")
                    break

            print(f"Epoch {epoch}: | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f} | Best Val AUC: {best_auc:.4f} | Patience: {stop_counter}/{patience} | LR: {self.optimizer.param_groups[0]['lr']:.6f} | GRL Lambda: {self.grl_lambda if self.use_grl else 'N/A'}")

        if best_model is not None:
            torch.save(best_model, os.path.join(self.best_model_path, f"best_model_{held_out_site}.pt"))
            self.model.load_state_dict(best_model)

        acc, auc, y_true, y_score, sensitivity, specificity, f1, bal_acc = self.evaluate(test_loader, held_out_site, return_scores=True)

        wandb.log({
            f"{held_out_site}/test_acc": acc,
            f"{held_out_site}/test_auc": auc,
            f"{held_out_site}/test_sens": sensitivity,
            f"{held_out_site}/test_spec": specificity,
            f"{held_out_site}/test_f1": f1,
            f"{held_out_site}/test_bal_acc": bal_acc,
        })

        print(f"[LOSO] Site {held_out_site} - Test Acc: {acc:.4f}, AUC: {auc:.4f}, Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, F1: {f1:.4f}, Bal. Acc: {bal_acc:.4f}")

        run.finish()
        return {
            "site": held_out_site,
            "accuracy": acc,
            "auc": auc,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "f1_score": f1,
            "balanced_accuracy": bal_acc,
            "y_true": y_true,
            "y_score": y_score
        }
    
    def train_epoch(self, train_loader, domain_labels):
        print("Training epoch...")
        self.model.train()
        if self.use_grl:
            self.domain_discriminator.train()

        total_loss = 0
        for batch in train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            out = self.model_forward(batch)
            logits = out[0] if isinstance(out, tuple) else out
            loss_cls = F.cross_entropy(logits, batch.y)

            if self.use_grl:
                features = None
                if self.config.models.name == "SpatioTemporalModel":
                    num_nodes = self.config.dataset.num_nodes
                    batch_size = batch.num_graphs
                    device = batch.x.device
                    identity = torch.eye(num_nodes, device=device)
                    pseudo = identity.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, num_nodes)
                    batch.pseudo = pseudo

                    features = self.model.extract_features(
                        batch,
                        batch.time_series,
                        batch.edge_index,
                        batch.edge_attr,
                        batch.x,
                        batch.pseudo,
                        batch.batch
                    )
                else:
                    features = self.model.extract_features(batch)
                domain_preds = self.domain_discriminator(features, alpha=self.grl_lambda)
                loss_domain = self.domain_loss_fn(domain_preds, domain_labels[batch.ptr[:-1]])
                loss = loss_cls + loss_domain
            else:
                loss = loss_cls

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        print(f"Training Loss: {total_loss / len(train_loader):.4f} | LR: {self.optimizer.param_groups[0]['lr']:.6f}")
        return total_loss / len(train_loader)

    def validate(self, loader):
        print("Validating...")
        self.model.eval()
        all_preds, all_labels, all_probs, all_logits = [], [], [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                out = self.model_forward(batch)
                logits = out[0] if isinstance(out, tuple) else out
                probs = F.softmax(logits, dim=1)[:, 1]
                preds = torch.argmax(logits, dim=1)

                all_preds.append(preds.cpu())
                all_labels.append(batch.y.cpu())
                all_probs.append(probs.cpu())
                all_logits.append(logits.cpu())

        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_labels).numpy()
        y_score = torch.cat(all_probs).numpy()
        loss = F.cross_entropy(torch.cat(all_logits), torch.cat(all_labels)).item()
        auc = roc_auc_score(y_true, y_score)
        acc = accuracy_score(y_true, y_pred)

        print(f"AUC: {auc:.4f} | Loss: {loss:.4f} | Acc: {acc:.4f} | LR: {self.optimizer.param_groups[0]['lr']:.6f}")

        return auc, loss, acc

    def evaluate(self, test_loader, site_name, return_scores=False):
        print(f"Evaluating on test set of site: {site_name}...")
        self.model.eval()
        all_preds, all_labels, all_probs, all_features, all_logits = [], [], [], [], []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                out = self.model_forward(batch)
                logits = out[0] if isinstance(out, tuple) else out
                probs = F.softmax(logits, dim=1)[:, 1]
                preds = torch.argmax(logits, dim=1)

                if self.use_grl:
                    if self.config.models.name == "SpatioTemporalModel":
                        num_nodes = self.config.dataset.num_nodes
                        batch_size = batch.num_graphs
                        device = batch.x.device
                        identity = torch.eye(num_nodes, device=device)
                        pseudo = identity.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, num_nodes)
                        batch.pseudo = pseudo

                        features = self.model.extract_features(
                            batch,
                            batch.time_series,
                            batch.edge_index,
                            batch.edge_attr,
                            batch.x,
                            batch.pseudo,
                            batch.batch
                        )
                    else:
                        features = self.model.extract_features(batch)

                    all_features.append(features.cpu())

                all_preds.append(preds.cpu())
                all_labels.append(batch.y.cpu())
                all_probs.append(probs.cpu())
                all_logits.append(logits.cpu())

        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_labels).numpy()
        y_score = torch.cat(all_probs).numpy()
        X_feat = torch.cat(all_features).numpy() if self.use_grl else None

        os.makedirs("figures", exist_ok=True)
        roc_path = f"figures/roc_{site_name}.png"
        tsne_path = f"figures/tsne_{site_name}.png"
        umap_path = f"figures/umap_{site_name}.png"

        auc = plot_roc(y_true, y_score, title=f"ROC - {site_name}", save_path=roc_path)

        if self.use_grl:
            plot_embeddings(X_feat, y_true, method='tsne', title=site_name, save_path=tsne_path)
            plot_embeddings(X_feat, y_true, method='umap', title=site_name, save_path=umap_path)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn + 1e-6)
        specificity = tn / (tn + fp + 1e-6)
        f1 = f1_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)

        if return_scores:
            return accuracy_score(y_true, y_pred), auc, y_true, y_score, sensitivity, specificity, f1, bal_acc
        return accuracy_score(y_true, y_pred)