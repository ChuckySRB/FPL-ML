"""
HTMT_GT — Adapted for FPL Tabular Data
========================================
Original concept: Hierarchical Temporal Multi-Task Graph Transformer
Source: Gemini architecture suggestion, adapted for our flat tabular feature vectors.

Removed from original:
  - TacticalGraphNetwork (GATv2Conv) — needs teammate/opponent graph connectivity
  - Sequential TemporalFusionEncoder — our features are already pre-aggregated rolling stats
  - LLM-derived expected minutes — not available in our data

Kept and adapted:
  - GatedResidualNetwork (GRN) — core non-linear processing block with skip connections
  - VariableSelectionNetwork (VSN) — learns per-sample feature attention weights
  - MultiTaskDecoder — predicts sub-components (goals, assists, CS) as regularizers
  - UncertaintyWeightedLoss — balances task losses automatically

Input:  flat feature vector (batch_size, n_features)        — our 29 Tier 2 features
Output: total_points prediction (main) + optional sub-targets (auxiliary)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ─────────────────────────────────────────────────────────────
# BUILDING BLOCKS
# ─────────────────────────────────────────────────────────────

class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network — provides flexible non-linear processing.
    Learns to skip complex transformations when not needed via a gating mechanism.

    Bug fixed from original: GLU gate now correctly splits the projection
    rather than applying sigmoid(x)*x on the unsplit tensor.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Linear(input_dim, output_dim, bias=False)
        self.layer_norm = nn.LayerNorm(output_dim)
        # GLU gate: projects to output_dim and produces a gate in [0,1]
        self.gate = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Skip connection (residual)
        residual = self.skip(x)
        # Non-linear branch
        h = F.elu(self.fc1(x))
        h = self.dropout(h)
        value = self.fc2(h)
        gate = torch.sigmoid(self.gate(h))   # Fixed: proper GLU gating
        out = gate * value
        return self.layer_norm(out + residual)


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network — learns which input features matter most
    for each individual sample (instance-wise soft attention over features).

    Adapted from the original for flat tabular input (batch, n_features).
    Each feature is first embedded to feature_dim via a linear layer,
    then a shared GRN computes attention weights across all features.
    """
    def __init__(self, n_features: int, feature_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        self.feature_dim = feature_dim

        # Project each scalar feature to a feature_dim embedding
        self.feature_embeddings = nn.ModuleList([
            nn.Linear(1, feature_dim) for _ in range(n_features)
        ])

        # GRN over all concatenated embeddings → soft weights per feature
        self.selection_grn = GatedResidualNetwork(
            input_dim=n_features * feature_dim,
            hidden_dim=hidden_dim,
            output_dim=n_features,
            dropout=dropout,
        )

        # Per-feature GRNs for individual processing
        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(feature_dim, hidden_dim, hidden_dim, dropout)
            for _ in range(n_features)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, n_features) — flat feature vector
        Returns:
            (batch_size, hidden_dim) — attention-weighted feature representation
        """
        batch_size = x.shape[0]

        # Embed each feature: (batch, n_features, feature_dim)
        embedded = torch.stack(
            [self.feature_embeddings[i](x[:, i:i+1]) for i in range(self.n_features)],
            dim=1
        )  # (batch, n_features, feature_dim)

        # Compute attention weights from flattened embeddings
        flat = embedded.view(batch_size, -1)                         # (batch, n_features * feature_dim)
        weights = F.softmax(self.selection_grn(flat), dim=-1)        # (batch, n_features)
        weights = weights.unsqueeze(-1)                               # (batch, n_features, 1)

        # Process each feature through its own GRN
        processed = torch.stack(
            [self.feature_grns[i](embedded[:, i, :]) for i in range(self.n_features)],
            dim=1
        )  # (batch, n_features, hidden_dim)

        # Weighted sum across features
        out = (weights * processed).sum(dim=1)   # (batch, hidden_dim)
        return out


# ─────────────────────────────────────────────────────────────
# MAIN MODELS
# ─────────────────────────────────────────────────────────────

class FPLNet(nn.Module):
    """
    Single-task model — predicts total_points directly.

    Architecture:
      1. Variable Selection Network  — soft attention over 29 input features
      2. Two GRN layers              — deep non-linear processing with residuals
      3. Linear output head          — predict total_points
    """
    def __init__(self, n_features: int, feature_dim: int = 8,
                 hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.vsn = VariableSelectionNetwork(n_features, feature_dim, hidden_dim, dropout)
        self.grn1 = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.grn2 = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.vsn(x)
        h = self.grn1(h)
        h = self.grn2(h)
        return self.head(h).squeeze(-1)


class FPLMultiTaskNet(nn.Module):
    """
    Multi-task model — predicts total_points (main) plus sub-components
    (goals, assists, clean_sheets) as auxiliary tasks that regularise the shared
    representation, matching the HTMT_GT spirit.

    Architecture:
      1. Variable Selection Network  — shared input processing
      2. Two GRN layers              — shared representation
      3. Separate output heads:
           - total_points  (main)    Huber loss
           - goals_lambda            Poisson NLL (softplus output)
           - assists_lambda          Poisson NLL (softplus output)
           - cs_prob                 BCE loss   (sigmoid output)
    """
    def __init__(self, n_features: int, feature_dim: int = 8,
                 hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.vsn  = VariableSelectionNetwork(n_features, feature_dim, hidden_dim, dropout)
        self.grn1 = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.grn2 = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)

        # Task-specific heads
        self.head_pts     = nn.Linear(hidden_dim, 1)   # total_points (continuous)
        self.head_goals   = nn.Linear(hidden_dim, 1)   # goals lambda (Poisson)
        self.head_assists = nn.Linear(hidden_dim, 1)   # assists lambda (Poisson)
        self.head_cs      = nn.Linear(hidden_dim, 1)   # clean sheet probability

    def forward(self, x: torch.Tensor):
        h = self.vsn(x)
        h = self.grn1(h)
        h = self.grn2(h)
        pts     = self.head_pts(h).squeeze(-1)
        goals   = F.softplus(self.head_goals(h)).squeeze(-1)    # must be > 0 (Poisson rate)
        assists = F.softplus(self.head_assists(h)).squeeze(-1)
        cs_prob = torch.sigmoid(self.head_cs(h)).squeeze(-1)    # probability in [0, 1]
        return pts, goals, assists, cs_prob


class UncertaintyWeightedLoss(nn.Module):
    """
    Automatically balances multiple task losses using learnable uncertainty weights
    (Kendall et al., 2018). Prevents manual weight tuning.
    """
    def __init__(self, num_tasks: int = 4):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses: torch.Tensor) -> torch.Tensor:
        precision = torch.exp(-self.log_vars)
        return (precision * losses + self.log_vars).sum()


# ─────────────────────────────────────────────────────────────
# SKLEARN-COMPATIBLE TRAINING WRAPPERS
# ─────────────────────────────────────────────────────────────

class FPLNetTrainer:
    """
    Sklearn-compatible wrapper around FPLNet (single-task).
    Usage:
        trainer = FPLNetTrainer(n_features=29)
        trainer.fit(X_train_np, y_train_np)
        preds = trainer.predict(X_test_np)
    """
    def __init__(self, n_features: int, feature_dim: int = 8,
                 hidden_dim: int = 128, dropout: float = 0.2,
                 lr: float = 1e-3, epochs: int = 50, batch_size: int = 512,
                 patience: int = 10, device: str = None):
        self.hparams = dict(n_features=n_features, feature_dim=feature_dim,
                            hidden_dim=hidden_dim, dropout=dropout)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.train_losses = []
        self.val_losses = []

    def fit(self, X: np.ndarray, y: np.ndarray, val_split: float = 0.1):
        self.model = FPLNet(**self.hparams).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        criterion = nn.HuberLoss(delta=1.5)

        # Split validation set
        n_val = int(len(X) * val_split)
        idx = np.random.permutation(len(X))
        X_val, y_val = X[idx[:n_val]], y[idx[:n_val]]
        X_tr,  y_tr  = X[idx[n_val:]], y[idx[n_val:]]

        tr_loader = self._make_loader(X_tr, y_tr, shuffle=True)
        val_loader = self._make_loader(X_val, y_val, shuffle=False)

        best_val, patience_cnt = float('inf'), 0
        best_state = None

        for epoch in range(self.epochs):
            # Train
            self.model.train()
            tr_loss = 0.0
            for Xb, yb in tr_loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                pred = self.model(Xb)
                loss = criterion(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                tr_loss += loss.item() * len(Xb)
            tr_loss /= len(X_tr)

            # Validate
            val_loss = self._eval_loss(val_loader, criterion)
            scheduler.step(val_loss)
            self.train_losses.append(tr_loss)
            self.val_losses.append(val_loss)

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d}/{self.epochs}  "
                      f"train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}")

            # Early stopping
            if val_loss < best_val - 1e-4:
                best_val = val_loss
                patience_cnt = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_cnt += 1
                if patience_cnt >= self.patience:
                    print(f"  Early stop at epoch {epoch+1}")
                    break

        if best_state:
            self.model.load_state_dict(best_state)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        loader = self._make_loader(X, np.zeros(len(X)), shuffle=False)
        preds = []
        with torch.no_grad():
            for Xb, _ in loader:
                preds.append(self.model(Xb.to(self.device)).cpu().numpy())
        return np.concatenate(preds)

    def _make_loader(self, X, y, shuffle):
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        return DataLoader(TensorDataset(Xt, yt), batch_size=self.batch_size,
                          shuffle=shuffle, drop_last=False)

    def _eval_loss(self, loader, criterion):
        self.model.eval()
        total = 0.0
        with torch.no_grad():
            for Xb, yb in loader:
                pred = self.model(Xb.to(self.device))
                total += criterion(pred, yb.to(self.device)).item() * len(Xb)
        return total / sum(len(b[0]) for b in loader)


class FPLMultiTaskTrainer:
    """
    Sklearn-compatible wrapper around FPLMultiTaskNet.

    Requires auxiliary targets: goals_scored, assists, clean_sheets
    (all available in train_full / test_full DataFrames).

    Main prediction is still total_points. The auxiliary heads are only
    used during training to regularise the shared representation.
    """
    def __init__(self, n_features: int, feature_dim: int = 8,
                 hidden_dim: int = 128, dropout: float = 0.2,
                 lr: float = 1e-3, epochs: int = 50, batch_size: int = 512,
                 patience: int = 10, device: str = None):
        self.hparams = dict(n_features=n_features, feature_dim=feature_dim,
                            hidden_dim=hidden_dim, dropout=dropout)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.loss_fn = None
        self.train_losses = []
        self.val_losses = []

    def fit(self, X: np.ndarray, y: np.ndarray,
            goals: np.ndarray, assists: np.ndarray, cs: np.ndarray,
            val_split: float = 0.1):
        """
        Args:
            X:       Feature matrix (n_samples, n_features)
            y:       total_points targets
            goals:   goals_scored targets (for Poisson head)
            assists: assists targets (for Poisson head)
            cs:      clean_sheets targets 0/1 (for BCE head)
        """
        self.model   = FPLMultiTaskNet(**self.hparams).to(self.device)
        self.loss_fn = UncertaintyWeightedLoss(num_tasks=4).to(self.device)
        optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.loss_fn.parameters()),
            lr=self.lr, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        huber = nn.HuberLoss(delta=1.5)
        poisson = nn.PoissonNLLLoss(log_input=False, full=True)
        bce = nn.BCELoss()

        # Build dataset arrays
        aux = np.stack([goals, assists, cs], axis=1)

        # Validation split
        n_val = int(len(X) * val_split)
        idx = np.random.permutation(len(X))
        X_val, y_val, aux_val = X[idx[:n_val]], y[idx[:n_val]], aux[idx[:n_val]]
        X_tr,  y_tr,  aux_tr  = X[idx[n_val:]], y[idx[n_val:]], aux[idx[n_val:]]

        tr_loader  = self._make_loader(X_tr, y_tr, aux_tr, shuffle=True)
        val_loader = self._make_loader(X_val, y_val, aux_val, shuffle=False)

        best_val, patience_cnt, best_state = float('inf'), 0, None

        for epoch in range(self.epochs):
            self.model.train()
            self.loss_fn.train()
            tr_loss = 0.0
            for Xb, yb, auxb in tr_loader:
                Xb  = Xb.to(self.device)
                yb  = yb.to(self.device)
                g_b = auxb[:, 0].to(self.device)
                a_b = auxb[:, 1].to(self.device)
                c_b = auxb[:, 2].to(self.device)

                optimizer.zero_grad()
                pts, goals_p, assists_p, cs_p = self.model(Xb)

                losses = torch.stack([
                    huber(pts, yb),
                    poisson(goals_p, g_b),
                    poisson(assists_p, a_b),
                    bce(cs_p, c_b),
                ])
                loss = self.loss_fn(losses)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                tr_loss += loss.item() * len(Xb)
            tr_loss /= len(X_tr)

            val_loss = self._eval_loss(val_loader, huber, poisson, bce)
            scheduler.step(val_loss)
            self.train_losses.append(tr_loss)
            self.val_losses.append(val_loss)

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d}/{self.epochs}  "
                      f"train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}")

            if val_loss < best_val - 1e-4:
                best_val = val_loss
                patience_cnt = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_cnt += 1
                if patience_cnt >= self.patience:
                    print(f"  Early stop at epoch {epoch+1}")
                    break

        if best_state:
            self.model.load_state_dict(best_state)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns total_points predictions only (main task head)."""
        self.model.eval()
        dummy_aux = np.zeros((len(X), 3))
        loader = self._make_loader(X, np.zeros(len(X)), dummy_aux, shuffle=False)
        preds = []
        with torch.no_grad():
            for Xb, _, _ in loader:
                pts, *_ = self.model(Xb.to(self.device))
                preds.append(pts.cpu().numpy())
        return np.concatenate(preds)

    def _make_loader(self, X, y, aux, shuffle):
        Xt   = torch.tensor(X,   dtype=torch.float32)
        yt   = torch.tensor(y,   dtype=torch.float32)
        auxt = torch.tensor(aux, dtype=torch.float32)
        return DataLoader(TensorDataset(Xt, yt, auxt), batch_size=self.batch_size,
                          shuffle=shuffle, drop_last=False)

    def _eval_loss(self, loader, huber, poisson, bce):
        self.model.eval()
        self.loss_fn.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for Xb, yb, auxb in loader:
                Xb = Xb.to(self.device)
                pts, g, a, c = self.model(Xb)
                losses = torch.stack([
                    huber(pts, yb.to(self.device)),
                    poisson(g, auxb[:, 0].to(self.device)),
                    poisson(a, auxb[:, 1].to(self.device)),
                    bce(c, auxb[:, 2].to(self.device)),
                ])
                total += self.loss_fn(losses).item() * len(Xb)
                n += len(Xb)
        return total / n
