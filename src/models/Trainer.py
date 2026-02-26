import torch


class NNTrainer:
    def __init__(self, model: torch.nn.Module, lr: float = 1e-3):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()

    def train_step(self, X_batch, y_batch):
        self.model.train()
        self.optimizer.zero_grad()
        y_pred = self.model(X_batch)
        loss = self.criterion(y_pred.squeeze(), y_batch)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def train(self, train_loader, val_loader=None, epochs=10):
        for epoch in range(epochs):
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                train_loss += self.train_step(X_batch, y_batch)
            avg_train_loss = train_loss / len(train_loader)
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}')
            if val_loader is not None:
                val_loss = 0.0
                with torch.no_grad():
                    for X_val, y_val in val_loader:
                        y_val_pred = self.model(X_val)
                        val_loss += self.criterion(y_val_pred.squeeze(), y_val).item()
                avg_val_loss = val_loss / len(val_loader)
                print(f'Validation Loss: {avg_val_loss:.4f}')
    
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            return self.model(X).squeeze().cpu().numpy()