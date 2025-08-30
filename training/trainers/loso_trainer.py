import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    f1_score, balanced_accuracy_score, recall_score
)
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import StratifiedShuffleSplit
from models.domain_adaptation.components import DomainDiscriminator, AdversarialLoss
# from evaluation.visualisation.roc_plot import plot_roc, plot_macro_micro_roc
# from evaluation.visualisation.tsne_umap import plot_embeddings
from tqdm import tqdm
import wandb
import copy
from models.domain_adaptation.components import GRLScheduler
from utils.logging_utils import wandb_log


class LOSOTrainer:
    def __init__(self, model, optimizer, device, config, site_names):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.site_names = site_names

        self.use_grl = self.config.domain_adaptation.use_grl
        self.grl_lambda = self.config.domain_adaptation.grl_lambda
        self.domain_weight = self.config.domain_adaptation.domain_loss_weight
        self.global_step = 0
        self.grl_scheduler = None
        self.mode = f"GRL-{self.config.models.alias}" if self.use_grl else f"Baseline-{self.config.models.alias}"

        if self.use_grl:
            self.domain_discriminator = DomainDiscriminator(
                input_dim=self.config.models.feature_dim,
                hidden_dim=self.config.domain_adaptation.hidden_dim,
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
        elif self.config.models.name == "BrainGNN":
            num_nodes = self.config.dataset.num_nodes
            batch_size = batch.num_graphs
            device = batch.x.device

            identity = torch.eye(num_nodes, device=device)
            pseudo = identity.expand(batch_size, -1, -1).reshape(-1, num_nodes)
            
            batch.pseudo = pseudo
            return self.model(batch.x, batch.edge_index, batch.batch, batch.edge_attr, batch.pseudo)
        elif self.config.models.name == "BrainNetworkTransformer":
            time_series = batch.time_series
            node_feature = batch.x
            
            assert node_feature is not None, "batch.x is None"
            assert node_feature.dim() == 2, f"Expected 2D node features (N*F), got {node_feature.shape}"
            assert node_feature.shape[0] == batch.num_graphs * self.config.dataset.num_nodes, \
                f"Expected shape[0] = {batch.num_graphs * self.config.dataset.num_nodes}, got {node_feature.shape[0]}"

            node_feature = node_feature.view(batch.num_graphs, self.config.dataset.num_nodes, -1)
            print(f"[DEBUG] node_feature reshaped: {node_feature.shape}")


            return self.model(time_series, node_feature)
        else:
            return self.model(batch)
        
    def extract_features(self, batch):
        if self.config.models.name == "SpatioTemporalModel":
            num_nodes = self.config.dataset.num_nodes
            batch_size = batch.num_graphs
            device = batch.x.device

            identity = torch.eye(num_nodes, device=device)
            pseudo = identity.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, num_nodes)
            batch.pseudo = pseudo

            return self.model.extract_features(batch)
        
        elif self.config.models.name == "BrainGNN":
            num_nodes = self.config.dataset.num_nodes
            batch_size = batch.num_graphs
            device = batch.x.device

            identity = torch.eye(num_nodes, device=device)
            pseudo = identity.expand(batch_size, -1, -1).reshape(-1, num_nodes)

            batch.pseudo = pseudo
            print(f"[DEBUG] batch.pseudo shape: {batch.pseudo.shape}")
            return self.model.extract_features(batch)
        elif self.config.models.name == "BrainNetworkTransformer":
            node_feature = batch.x.view(batch.num_graphs, self.config.dataset.num_nodes, -1)
            time_series = batch.time_series
            return self.model.extract_features(time_series, node_feature)
        else:
            raise NotImplementedError("Feature extraction not implemented for this model.")
    
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

        #scale threads to match SLURM allocation
        num_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
        torch.set_num_threads(num_threads)
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        print(f"[INFO] Using {num_threads // 8} CPU threads for training")

        results = {}
        all_y_true = []
        all_y_score = []
        metrics_summary = []
        run = None

        for held_out_site in self.site_names:
            print(f"\n[LOSO] Held-out site: {held_out_site}")
            run = wandb.init(
                project=self.config.logging.project,
                entity=self.config.logging.entity,
                name=self.config.logging.run_name+"-site-"+held_out_site,
                config=OmegaConf.to_container(self.config, resolve=True),
                reinit=True
            )

            train_graphs, val_graphs, test_graphs = [], [], site_data[held_out_site]
            # domain_labels = []

            domain_counter = 0
            for site in self.site_names:
                if site == held_out_site:
                    continue
                site_graphs = site_data[site]

                for g in site_graphs:
                    g.domain = torch.tensor([domain_counter], dtype=torch.long).to(self.device) # assign domain label to each graph

                train_split, val_split = self.stratified_split(site_graphs, self.config.dataset.val_split)
                train_graphs.extend(train_split)
                val_graphs.extend(val_split)
                # domain_labels.extend([domain_counter] * len(train_split)) # assign domain labels

                domain_counter += 1

            # multiple workers for data loading to improve CPU throughput
            print(f"[DEBUG] Domain labels in training graphs: {set([g.domain.item() for g in train_graphs])}")
            num_workers = min(4, num_threads // 8)  # limit to avoid too many threads
            train_loader = DataLoader(train_graphs, batch_size=self.config.dataset.batch_size, shuffle=True, num_workers=num_workers)
            val_loader = DataLoader(val_graphs, batch_size=self.config.dataset.batch_size, shuffle=False, num_workers=num_workers)
            held_out_idx = self.site_names.index(held_out_site)
            for g in test_graphs:
                g.domain = torch.tensor([held_out_idx], dtype=torch.long).to(self.device)

            test_loader = DataLoader(test_graphs, batch_size=self.config.dataset.batch_size, shuffle=False, num_workers=num_workers)

            # domain_labels = torch.tensor(domain_labels).to(self.device)

            if self.config.domain_adaptation.use_grl:
                total_steps = self.config.training.epochs * len(train_loader)
                warm_up_steps = self.config.domain_adaptation.grl_warmup_epochs * len(train_loader)
                ramp_steps = self.config.domain_adaptation.grl_ramp_epochs * len(train_loader)
                
                self.grl_scheduler = GRLScheduler(
                    total_steps=total_steps,
                    schedule=self.config.domain_adaptation.grl_schedule,
                    gamma=self.config.domain_adaptation.grl_gamma,
                    warmup_steps=warm_up_steps,
                    ramp_steps=ramp_steps,
                    max_lambda=self.config.domain_adaptation.grl_max_lambda
                )
                print(f"[INFO] GRL Scheduler: {self.grl_scheduler.schedule} with max lambda {self.grl_scheduler.max_lambda}, total steps {total_steps}, warmup {warm_up_steps}, ramp {ramp_steps}")

            asd_train, td_train = self.log_distribution(f"Train (excluding {held_out_site})", train_graphs)
            asd_val, td_val = self.log_distribution(f"Val (excluding {held_out_site})", val_graphs)
            asd_test, td_test = self.log_distribution(f"Test ({held_out_site})", test_graphs)
            wandb_log({
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
                train_loss = self.train_epoch(train_loader, held_out_site, epoch)

                val_auc, val_loss, val_acc = self.validate(val_loader, held_out_site)
                wandb_log({
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
                torch.save(best_model, os.path.join(self.best_model_path, f"{self.mode}_{held_out_site}.pt"))
                self.model.load_state_dict(best_model)

            acc, auc, y_true, y_score, sensitivity, specificity, f1, bal_acc = self.evaluate(test_loader, held_out_site, return_scores=True)
            results[held_out_site] = acc
            all_y_true.append(y_true)
            all_y_score.append(y_score)
            metrics_summary.append({
                "site": held_out_site,
                "accuracy": acc,
                "auc": auc,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "f1_score": f1,
                "balanced_accuracy": bal_acc
            })

            wandb_log({
                f"{held_out_site}/test_acc": acc,
                f"{held_out_site}/test_auc": auc,
                f"{held_out_site}/test_sens": sensitivity,
                f"{held_out_site}/test_spec": specificity,
                f"{held_out_site}/test_f1": f1,
                f"{held_out_site}/test_bal_acc": bal_acc,
            })

            print(f"[LOSO] Site {held_out_site} - Test Acc: {acc:.4f}, AUC: {auc:.4f}, Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, F1: {f1:.4f}, Bal. Acc: {bal_acc:.4f}")

        os.makedirs("figures", exist_ok=True)
        # macro_roc_path = "figures/macro_roc.png"
        # plot_macro_micro_roc(all_y_true, all_y_score, save_path=macro_roc_path)

        os.makedirs("result/stats", exist_ok=True)
        df = pd.DataFrame(metrics_summary)
        df.to_csv(f"result/stats/{self.mode}_metrics_summary.csv", index=False)
        print("\n[LOSO] All-site Results:")
        print(df.to_string(index=False))

        # Log overall metrics
        mean_metrics = df.mean(numeric_only=True)
        wandb_log({
            "overall/mean_test_acc": mean_metrics["accuracy"],
            "overall/mean_test_auc": mean_metrics["auc"],
            "overall/mean_test_sens": mean_metrics["sensitivity"],
            "overall/mean_test_spec": mean_metrics["specificity"],
            "overall/mean_test_f1": mean_metrics["f1_score"],
            "overall/mean_test_bal_acc": mean_metrics["balanced_accuracy"],
        })
        if run is not None:
            run.finish()
        return results

    def train_epoch(self, train_loader, held_out_site, epoch=None):
        print("Training epoch...")
        self.model.train()
        if self.use_grl:
            self.domain_discriminator.train()

        total_loss = 0
        correct_domains = 0
        total_domains = 0
        for batch in train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            out = self.model_forward(batch)
            logits = out[0] if isinstance(out, tuple) else out
            loss_cls = F.cross_entropy(logits, batch.y)

            if self.use_grl:
                features = self.extract_features(batch)

                print("Extracted features shape:", features.shape)

                if self.config.models.name == "SpatioTemporalModel":
                    # convert node features to graph features by averaging over nodes
                    features = global_mean_pool(features, batch.batch)
                self.grl_lambda = self.grl_scheduler.get_lambda(self.global_step)

                print(f"[DEBUG] GRL Lambda at step {self.global_step}: {self.grl_lambda} of {self.grl_scheduler.total_steps}")

                print(f"[DEBUG] features input to fc: {features.shape}")
                domain_preds = self.domain_discriminator(features, alpha=self.grl_lambda)
                domain_targets = batch.domain.view(-1).to(self.device)  # shape [B] # get domain labels from graphs in the batch
                print(f"[DEBUG] Domain preds shape: {domain_preds.shape}, Domain targets shape: {domain_targets.shape}")
                loss_domain = self.domain_loss_fn(domain_preds, domain_targets)
                loss = loss_cls + loss_domain

                pred_domains = torch.argmax(domain_preds, dim=1)
                correct_domains += (pred_domains == domain_targets).sum().item()
                total_domains += domain_targets.size(0)
                self.global_step += 1
            else:
                loss = loss_cls

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        
        if self.use_grl:
            domain_acc = correct_domains / total_domains
            wandb_log({f"{held_out_site}/train_domain_acc": domain_acc,
                       "epoch": epoch})
            print(f"Domain Discriminator Accuracy: {domain_acc:.4f}")
        print(f"Training Loss: {total_loss / len(train_loader):.4f} | LR: {self.optimizer.param_groups[0]['lr']:.6f}")
        return total_loss / len(train_loader)

    def validate(self, loader, held_out_site=None):
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

        # per-domain collection
        per_site_metrics = {s: {"y_true": [], "y_pred": [], "y_score": []} for s in self.site_names}

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                out = self.model_forward(batch)
                logits = out[0] if isinstance(out, tuple) else out
                probs = F.softmax(logits, dim=1)[:, 1]
                preds = torch.argmax(logits, dim=1)

                if self.use_grl:
                    features = self.extract_features(batch)
                    all_features.append(features.cpu())

                all_preds.append(preds.cpu())
                all_labels.append(batch.y.cpu())
                all_probs.append(probs.cpu())
                all_logits.append(logits.cpu())

                # aggregate per-site
                domains = batch.domain.cpu().numpy()
                for i, domain_idx in enumerate(domains):
                    site = self.site_names[domain_idx]
                    per_site_metrics[site]["y_true"].append(batch.y[i].item())
                    per_site_metrics[site]["y_pred"].append(preds[i].item())
                    per_site_metrics[site]["y_score"].append(probs[i].item())

        # overall stats
        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_labels).numpy()
        y_score = torch.cat(all_probs).numpy()
        X_feat = torch.cat(all_features).numpy() if self.use_grl else None

        print(f"[DEBUG] X_feat shape: {X_feat.shape if self.use_grl else None}")
        print(f"[DEBUG] y_true shape: {y_true.shape}")

        os.makedirs("figures", exist_ok=True)
        # roc_path = f"figures/roc_{site_name}.png"
        # tsne_path = f"figures/tsne_{site_name}.png"
        # umap_path = f"figures/umap_{site_name}.png"

        auc = roc_auc_score(y_true, y_score)

        if self.use_grl:
            num_graphs = y_true.shape[0]
            nodes_per_graph = X_feat.shape[0] // num_graphs

            X_feat = X_feat.reshape(num_graphs, nodes_per_graph, -1).mean(axis=1)
            print(f"[DEBUG] Averaged node features to graph features: {X_feat.shape}")

            # plot_embeddings(X_feat, y_true, method='tsne', title=site_name, save_path=tsne_path)
            # plot_embeddings(X_feat, y_true, method='umap', title=site_name, save_path=umap_path)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn + 1e-6)
        specificity = tn / (tn + fp + 1e-6)
        f1 = f1_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)

        # Per-domain metric logging
        for domain, values in per_site_metrics.items():
            if not values["y_true"]:
                continue
            y_t = np.array(values["y_true"])
            y_p = np.array(values["y_pred"])
            y_s = np.array(values["y_score"])

            site_auc = roc_auc_score(y_t, y_s)
            site_acc = accuracy_score(y_t, y_p)
            site_f1 = f1_score(y_t, y_p)
            site_bal_acc = balanced_accuracy_score(y_t, y_p)
            site_sens = recall_score(y_t, y_p, pos_label=1)
            site_spec = recall_score(y_t, y_p, pos_label=0)

            wandb_log({
                f"{site_name}/domain_{domain}_auc": site_auc,
                f"{site_name}/domain_{domain}_acc": site_acc,
                f"{site_name}/domain_{domain}_f1": site_f1,
                f"{site_name}/domain_{domain}_bal_acc": site_bal_acc,
                f"{site_name}/domain_{domain}_sens": site_sens,
                f"{site_name}/domain_{domain}_spec": site_spec
            })

        print(f"Test AUC: {auc:.4f} | Acc: {accuracy_score(y_true, y_pred):.4f} | Sensitivity: {sensitivity:.4f} | Specificity: {specificity:.4f} | F1: {f1:.4f} | Bal. Acc: {bal_acc:.4f}")

        # Per-domain metric logging
        domain_metric_rows = []
        for domain, values in per_site_metrics.items():
            if not values["y_true"]:
                continue
            y_t = np.array(values["y_true"])
            y_p = np.array(values["y_pred"])
            y_s = np.array(values["y_score"])

            site_auc = roc_auc_score(y_t, y_s)
            site_acc = accuracy_score(y_t, y_p)
            site_f1 = f1_score(y_t, y_p)
            site_bal_acc = balanced_accuracy_score(y_t, y_p)
            site_sens = recall_score(y_t, y_p, pos_label=1)
            site_spec = recall_score(y_t, y_p, pos_label=0)

            wandb_log({
                f"{site_name}/domain_{domain}_auc": site_auc,
                f"{site_name}/domain_{domain}_acc": site_acc,
                f"{site_name}/domain_{domain}_f1": site_f1,
                f"{site_name}/domain_{domain}_bal_acc": site_bal_acc,
                f"{site_name}/domain_{domain}_sens": site_sens,
                f"{site_name}/domain_{domain}_spec": site_spec
            })

            domain_metric_rows.append({
                "Held-Out Site": site_name,
                "Evaluated Domain": domain,
                "AUC": site_auc,
                "Accuracy": site_acc,
                "F1": site_f1,
                "Balanced Accuracy": site_bal_acc,
                "Sensitivity": site_sens,
                "Specificity": site_spec
            })

        if domain_metric_rows:
            domain_df = pd.DataFrame(domain_metric_rows)
            
            # Append to global summary CSV
            global_csv_path = f"result/stats/{self.mode}_per_domain_summary.csv"
            if os.path.exists(global_csv_path):
                existing_df = pd.read_csv(global_csv_path)
                domain_df = pd.concat([existing_df, domain_df], ignore_index=True)
            domain_df.to_csv(global_csv_path, index=False)
            print("\n[Per-domain metrics Summary]")
            print(domain_df.to_string(index=False))
            print(f"[SAVED] Per-domain metrics written to {global_csv_path}")

        if return_scores:
            return accuracy_score(y_true, y_pred), auc, y_true, y_score, sensitivity, specificity, f1, bal_acc
        return accuracy_score(y_true, y_pred)


    # def train_loso_one_site(self, site_data, held_out_site):
    #     print(f"\n[LOSO] Held-out site: {held_out_site}")
    #     run = wandb.init(
    #         project=self.config.logging.project,
    #         entity=self.config.logging.entity,
    #         name=self.config.logging.run_name + "-site-" + held_out_site,
    #         config=OmegaConf.to_container(self.config, resolve=True),
    #         reinit=True
    #     )

    #     #scale threads to match SLURM allocation
    #     num_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
    #     torch.set_num_threads(num_threads)
    #     os.environ["OMP_NUM_THREADS"] = str(num_threads)
    #     print(f"[INFO] Using {num_threads} CPU threads for training")

    #     train_graphs, val_graphs, test_graphs = [], [], site_data[held_out_site]
    #     domain_labels = []

    #     for i, site in enumerate(self.site_names):
    #         if site == held_out_site:
    #             continue
    #         site_graphs = site_data[site]
    #         train_split, val_split = self.stratified_split(site_graphs, self.config.dataset.val_split)
    #         train_graphs.extend(train_split)
    #         val_graphs.extend(val_split)
    #         domain_labels.extend([i] * len(train_split))

    #     num_workers = min(8, num_threads)
    #     train_loader = DataLoader(train_graphs, batch_size=self.config.dataset.batch_size, shuffle=True, num_workers=num_workers)
    #     val_loader = DataLoader(val_graphs, batch_size=self.config.dataset.batch_size, shuffle=False, num_workers=num_workers)
    #     test_loader = DataLoader(test_graphs, batch_size=self.config.dataset.batch_size, shuffle=False, num_workers=num_workers)

    #     domain_labels = torch.tensor(domain_labels).to(self.device)

    #     asd_train, td_train = self.log_distribution(f"Train (excluding {held_out_site})", train_graphs)
    #     asd_val, td_val = self.log_distribution(f"Val (excluding {held_out_site})", val_graphs)
    #     asd_test, td_test = self.log_distribution(f"Test ({held_out_site})", test_graphs)
    #     wandb_log({
    #         f"{held_out_site}/train_asd": asd_train,
    #         f"{held_out_site}/train_td": td_train,
    #         f"{held_out_site}/val_asd": asd_val,
    #         f"{held_out_site}/val_td": td_val,
    #         f"{held_out_site}/test_asd": asd_test,
    #         f"{held_out_site}/test_td": td_test
    #     })

    #     best_auc = 0
    #     patience = self.config.models.patience
    #     min_delta = self.config.models.min_delta
    #     stop_counter = 0
    #     best_model = None

    #     for epoch in tqdm(range(self.config.training.epochs), desc=f"[Training] {held_out_site}"):
    #         train_loss = self.train_epoch(train_loader, domain_labels)
    #         val_auc, val_loss, val_acc = self.validate(val_loader)

    #         wandb_log({
    #             f"{held_out_site}/val_auc": val_auc,
    #             f"{held_out_site}/val_loss": val_loss,
    #             f"{held_out_site}/val_acc": val_acc,
    #             f"{held_out_site}/train_loss": train_loss,
    #             "epoch": epoch
    #         })

    #         if val_auc > best_auc + min_delta:
    #             best_auc = val_auc
    #             best_model = copy.deepcopy(self.model.state_dict())
    #             stop_counter = 0
    #         else:
    #             stop_counter += 1
    #             if stop_counter >= patience:
    #                 print(f"[Early Stopping] Epoch {epoch} for site {held_out_site}")
    #                 break

    #         print(f"Epoch {epoch}: | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f} | Best Val AUC: {best_auc:.4f} | Patience: {stop_counter}/{patience} | LR: {self.optimizer.param_groups[0]['lr']:.6f} | GRL Lambda: {self.grl_lambda if self.use_grl else 'N/A'}")

    #     if best_model is not None:
    #         torch.save(best_model, os.path.join(self.best_model_path, f"best_model_{held_out_site}.pt"))
    #         self.model.load_state_dict(best_model)

    #     acc, auc, y_true, y_score, sensitivity, specificity, f1, bal_acc = self.evaluate(test_loader, held_out_site, return_scores=True)

    #     wandb_log({
    #         f"{held_out_site}/test_acc": acc,
    #         f"{held_out_site}/test_auc": auc,
    #         f"{held_out_site}/test_sens": sensitivity,
    #         f"{held_out_site}/test_spec": specificity,
    #         f"{held_out_site}/test_f1": f1,
    #         f"{held_out_site}/test_bal_acc": bal_acc,
    #     })

    #     print(f"[LOSO] Site {held_out_site} - Test Acc: {acc:.4f}, AUC: {auc:.4f}, Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, F1: {f1:.4f}, Bal. Acc: {bal_acc:.4f}")

    #     run.finish()
    #     return {
    #         "site": held_out_site,
    #         "accuracy": acc,
    #         "auc": auc,
    #         "sensitivity": sensitivity,
    #         "specificity": specificity,
    #         "f1_score": f1,
    #         "balanced_accuracy": bal_acc,
    #         "y_true": y_true,
    #         "y_score": y_score
    #     }