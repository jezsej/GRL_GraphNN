import torch.nn.functional as F
from training.trainers.loso_trainer import LOSOTrainer
from sklearn.metrics import roc_auc_score

class BrainGNNTrainer(LOSOTrainer):
    def model_forward(self, batch):
        return self.model(batch.x, batch.edge_index, batch.batch, batch.edge_attr, batch.pseudo)
    
    def train_epoch(self, train_loader, held_out_site, epoch=None):
        self.model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            logits, _, _, _, _ = self.model_forward(batch)
            loss = F.cross_entropy(logits, batch.y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"[BrainGNN] Epoch {epoch} | Train Loss: {avg_loss:.4f}")
        return avg_loss