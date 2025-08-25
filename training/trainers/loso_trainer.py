import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score
from models.domain_adaptation.components import DomainDiscriminator, AdversarialLoss


class LOSOTrainer:
    def __init__(self, model, optimizer, device, config, site_names):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.site_names = site_names
        self.domain_discriminator = DomainDiscriminator(
            input_dim=config['model']['hidden_dim'],
            hidden_dim=64,
            num_domains=len(site_names)
        ).to(device)
        self.domain_loss_fn = AdversarialLoss()

    def train_loso(self, site_data):
        results = {}
        for held_out_site in self.site_names:
            print(f"\n[LOSO] Held-out site: {held_out_site}")

            # Construct training and test sets
            train_graphs = []
            test_graphs = site_data[held_out_site]
            domain_labels = []

            for i, site in enumerate(self.site_names):
                if site == held_out_site:
                    continue
                train_graphs.extend(site_data[site])
                domain_labels.extend([i] * len(site_data[site]))

            train_loader = DataLoader(train_graphs, batch_size=self.config['training']['batch_size'], shuffle=True)
            test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)
            domain_labels = torch.tensor(domain_labels).to(self.device)

            for epoch in range(self.config['training']['epochs']):
                self.train_epoch(train_loader, domain_labels)

            acc = self.evaluate(test_loader)
            results[held_out_site] = acc

        print("\n[LOSO] All-site Results:")
        for site, acc in results.items():
            print(f"{site}: {acc:.4f}")
        return results

    def train_epoch(self, train_loader, domain_labels):
        self.model.train()
        self.domain_discriminator.train()

        for i, batch in enumerate(train_loader):
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            # Forward pass
            logits = self.model(batch)
            loss_cls = F.cross_entropy(logits, batch.y)

            # Extract domain-invariant features
            features = self.model.extract_features(batch)
            domain_preds = self.domain_discriminator(features)
            loss_domain = self.domain_loss_fn(domain_preds, domain_labels[batch.ptr[:-1]])

            loss = loss_cls + loss_domain
            loss.backward()
            self.optimizer.step()

    def evaluate(self, test_loader):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                logits = self.model(batch)
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(batch.y.cpu())

        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_labels).numpy()
        return accuracy_score(y_true, y_pred)
