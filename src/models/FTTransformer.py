"""
Feature Tokenizer + Transformer (FT-Transformer) for FPL points prediction.

Each numerical feature is projected independently to a d_token-dimensional
embedding (feature tokenization). A learnable CLS token is prepended and the
full sequence is processed by a standard pre-norm Transformer encoder.
The regression head reads the CLS output.

This architecture explicitly models pairwise feature interactions via
self-attention, making it especially effective when features have complex
non-linear relationships — as is the case with FPL performance metrics.

Reference:
  Gorishniy et al. (2021) "Revisiting Deep Learning Models for Tabular Data"
  NeurIPS 2021.  https://arxiv.org/abs/2106.11959
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ─────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────

class FTTransformer(nn.Module):
    """
    Feature Tokenizer + Transformer for tabular regression.

    Architecture:
      1. Feature tokenization  — each scalar feature x_i is projected as
                                 x_i * W_i + b_i  →  (batch, n_features, d_token)
      2. CLS token prepended   — (batch, n_features + 1, d_token)
      3. Transformer encoder   — n_layers of pre-norm MHA + FFN blocks
      4. Regression head       — LayerNorm → Linear on the CLS position
    """
    def __init__(self, n_features: int, d_token: int = 64,
                 n_heads: int = 8, n_layers: int = 3,
                 ffn_factor: float = 4 / 3, dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features

        # Per-feature linear tokenizer: weight shape (n_features, d_token)
        self.feat_W = nn.Parameter(torch.empty(n_features, d_token))
        self.feat_b = nn.Parameter(torch.zeros(n_features, d_token))
        nn.init.kaiming_uniform_(self.feat_W, a=0)

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))

        # Pre-norm Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=max(4, int(d_token * ffn_factor)),
            dropout=dropout,
            batch_first=True,
            norm_first=True,    # pre-LayerNorm (more stable)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Regression head on CLS token output
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features)
        # Feature tokenization → (batch, n_features, d_token)
        tokens = x.unsqueeze(-1) * self.feat_W + self.feat_b

        # Prepend CLS token → (batch, n_features + 1, d_token)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        # Transformer encoder
        out = self.transformer(tokens)          # (batch, seq_len, d_token)

        # Regression from CLS position (index 0)
        return self.head(out[:, 0, :]).squeeze(-1)


# ─────────────────────────────────────────────────────────────
# SKLEARN-COMPATIBLE TRAINER
# ─────────────────────────────────────────────────────────────

class FTTransformerTrainer:
    """
    Sklearn-compatible wrapper for FTTransformer.

    Uses AdamW (recommended for transformers) + ReduceLROnPlateau scheduler
    + Huber loss (robust to the heavy-tailed distribution of FPL points).

    Usage:
        trainer = FTTransformerTrainer(n_features=29)
        trainer.fit(X_train_np, y_train_np)
        preds = trainer.predict(X_test_np)
    """
    def __init__(self, n_features: int, d_token: int = 64,
                 n_heads: int = 8, n_layers: int = 3,
                 ffn_factor: float = 4 / 3, dropout: float = 0.1,
                 lr: float = 1e-3, epochs: int = 100,
                 batch_size: int = 512, patience: int = 10,
                 device: str = None):
        self.hparams = dict(n_features=n_features, d_token=d_token,
                            n_heads=n_heads, n_layers=n_layers,
                            ffn_factor=ffn_factor, dropout=dropout)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.train_losses: list = []
        self.val_losses: list = []

    def fit(self, X: np.ndarray, y: np.ndarray, val_split: float = 0.1,
            sample_weight: np.ndarray = None):
        """
        Args:
            sample_weight: Optional per-sample weights (e.g. hauler_weights(y)).
                           Normalised internally so mean weight = 1.
                           Validation loss is always unweighted.
        """
        self.model = FTTransformer(**self.hparams).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5)

        n_val = int(len(X) * val_split)
        idx = np.random.permutation(len(X))
        X_val, y_val = X[idx[:n_val]], y[idx[:n_val]]
        X_tr,  y_tr  = X[idx[n_val:]], y[idx[n_val:]]

        sw_tr = None
        if sample_weight is not None:
            sw = np.array(sample_weight, dtype='float32')
            sw /= sw.mean()
            sw_tr = sw[idx[n_val:]]

        tr_loader  = self._loader(X_tr, y_tr, sw_tr, shuffle=True)
        val_loader = self._loader(X_val, y_val, None, shuffle=False)

        crit_per_sample = nn.HuberLoss(delta=1.5, reduction='none')
        crit_val        = nn.HuberLoss(delta=1.5)

        best_val, wait, best_state = float('inf'), 0, None

        for epoch in range(self.epochs):
            self.model.train()
            tr_loss = 0.0
            for batch in tr_loader:
                Xb, yb = batch[0], batch[1]
                wb = batch[2] if len(batch) == 3 else None
                tr_loss += self._step(Xb, yb, optimizer, crit_per_sample, wb)
            tr_loss /= len(X_tr)

            val_loss = self._eval(val_loader, crit_val)
            scheduler.step(val_loss)
            self.train_losses.append(tr_loss)
            self.val_losses.append(val_loss)

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d}/{self.epochs}  "
                      f"train={tr_loss:.4f}  val={val_loss:.4f}")

            if val_loss < best_val - 1e-4:
                best_val, wait = val_loss, 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    print(f"  Early stop at epoch {epoch + 1}")
                    break

        if best_state:
            self.model.load_state_dict(best_state)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        loader = self._loader(X, np.zeros(len(X)), shuffle=False)
        preds = []
        with torch.no_grad():
            for Xb, _ in loader:
                preds.append(self.model(Xb.to(self.device)).cpu().numpy())
        return np.concatenate(preds)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _loader(self, X, y, weights=None, shuffle=True):
        tensors = [torch.tensor(X, dtype=torch.float32),
                   torch.tensor(y, dtype=torch.float32)]
        if weights is not None:
            tensors.append(torch.tensor(weights, dtype=torch.float32))
        return DataLoader(TensorDataset(*tensors), batch_size=self.batch_size,
                          shuffle=shuffle, drop_last=False)

    def _step(self, Xb, yb, opt, crit, wb=None):
        Xb, yb = Xb.to(self.device), yb.to(self.device)
        opt.zero_grad()
        losses = crit(self.model(Xb), yb)  # per-sample when reduction='none'
        loss = (losses * wb.to(self.device)).mean() if wb is not None else losses.mean()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        opt.step()
        return loss.item() * len(Xb)

    def _eval(self, loader, crit):
        self.model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for batch in loader:
                Xb, yb = batch[0].to(self.device), batch[1].to(self.device)
                total += crit(self.model(Xb), yb).item() * len(Xb)
                n += len(Xb)
        return total / n
