"""
PyTorch MLP podium predictor.
Includes learnable embeddings for driver, team, and circuit.
Outputs P(P1), P(P2), P(P3) simultaneously via sigmoid.
"""
# STATUS: FROZEN — 2026
# The MLP underperforms XGBoost on this dataset size (~5k rows).
# Categorical embeddings add no predictive value over the existing
# rolling-form and circuit-history features at this scale.
# Do not enable for inference. Revisit if training set exceeds ~15k rows
# (approximately 7+ hybrid-era seasons with full grid).
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from loguru import logger

ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Dataset helpers
# --------------------------------------------------------------------------- #

class RaceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        X_num: np.ndarray,
        driver_idx: np.ndarray,
        team_idx: np.ndarray,
        circuit_idx: np.ndarray,
        y: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
    ):
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.driver_idx = torch.tensor(driver_idx, dtype=torch.long)
        self.team_idx = torch.tensor(team_idx, dtype=torch.long)
        self.circuit_idx = torch.tensor(circuit_idx, dtype=torch.long)

        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None
        if y is not None:
            if sample_weight is None:
                self.sample_weight = torch.ones(len(X_num), dtype=torch.float32)
            else:
                sw = np.asarray(sample_weight, dtype=float)
                if len(sw) != len(X_num):
                    raise ValueError(
                        f"sample_weight length mismatch: got {len(sw)} expected {len(X_num)}"
                    )
                self.sample_weight = torch.tensor(sw, dtype=torch.float32)
        else:
            self.sample_weight = None

    def __len__(self) -> int:
        return len(self.X_num)

    def __getitem__(self, idx: int):
        if self.y is not None:
            return (
                self.X_num[idx],
                self.driver_idx[idx],
                self.team_idx[idx],
                self.circuit_idx[idx],
                self.y[idx],
                self.sample_weight[idx],
            )
        return (
            self.X_num[idx],
            self.driver_idx[idx],
            self.team_idx[idx],
            self.circuit_idx[idx],
        )


# --------------------------------------------------------------------------- #
# Model architecture
# --------------------------------------------------------------------------- #

class PodiumMLP(nn.Module):
    """
    MLP with categorical embeddings for driver, team, circuit.
    Input: numeric features + embedding lookups.
    Output: 3 logits (P1, P2, P3).
    """

    def __init__(
        self,
        num_numeric_features: int,
        n_drivers: int,
        n_teams: int,
        n_circuits: int,
        driver_emb_dim: int = 8,
        team_emb_dim: int = 4,
        circuit_emb_dim: int = 6,
        hidden_dims: list[int] = [384, 192, 96],
        dropout: float = 0.25,
    ):
        super().__init__()

        self.driver_emb = nn.Embedding(n_drivers + 1, driver_emb_dim, padding_idx=0)
        self.team_emb = nn.Embedding(n_teams + 1, team_emb_dim, padding_idx=0)
        self.circuit_emb = nn.Embedding(n_circuits + 1, circuit_emb_dim, padding_idx=0)

        input_dim = num_numeric_features + driver_emb_dim + team_emb_dim + circuit_emb_dim

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 3))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        x_num: torch.Tensor,
        driver_idx: torch.Tensor,
        team_idx: torch.Tensor,
        circuit_idx: torch.Tensor,
    ) -> torch.Tensor:
        d_emb = self.driver_emb(driver_idx)
        t_emb = self.team_emb(team_idx)
        c_emb = self.circuit_emb(circuit_idx)
        x = torch.cat([x_num, d_emb, t_emb, c_emb], dim=1)
        return self.net(x)


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #

class NNPodiumPredictor:
    """
    Wraps PodiumMLP with training, evaluation, and serialization logic.
    """

    def __init__(
        self,
        num_numeric_features: int,
        n_drivers: int,
        n_teams: int,
        n_circuits: int,
        hidden_dims: list[int] = [384, 192, 96],
        dropout: float = 0.25,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        batch_size: int = 128,
        max_epochs: int = 300,
        patience: int = 35,
        min_epochs: int = 60,
        min_delta: float = 1e-4,
        lr_scheduler_patience: int = 8,
        driver_emb_dim: int = 16,
        team_emb_dim: int = 8,
        circuit_emb_dim: int = 12,
    ):
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.min_epochs = min_epochs
        self.min_delta = min_delta

        self.model = PodiumMLP(
            num_numeric_features=num_numeric_features,
            n_drivers=n_drivers,
            n_teams=n_teams,
            n_circuits=n_circuits,
            hidden_dims=hidden_dims,
            dropout=dropout,
            driver_emb_dim=driver_emb_dim,
            team_emb_dim=team_emb_dim,
            circuit_emb_dim=circuit_emb_dim,
        ).to(DEVICE)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            patience=lr_scheduler_patience,
            factor=0.5,
            threshold=self.min_delta,
            min_lr=1e-5,
        )
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")

        self._mean: Optional[torch.Tensor] = None
        self._std: Optional[torch.Tensor] = None

        self.n_drivers = n_drivers
        self.n_teams = n_teams
        self.n_circuits = n_circuits
        self.num_numeric_features = num_numeric_features

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(
            f"NN architecture: hidden_dims={hidden_dims}, dropout={dropout}, lr={lr}, "
            f"params={trainable_params:,}/{total_params:,} trainable/total"
        )

    def _normalize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self._mean = X.mean(axis=0)
            self._std = X.std(axis=0) + 1e-8
        return (X - self._mean) / self._std

    def _make_dataset(
        self,
        X_num: np.ndarray,
        driver_idx: np.ndarray,
        team_idx: np.ndarray,
        circuit_idx: np.ndarray,
        y: Optional[np.ndarray],
        sample_weight: Optional[np.ndarray] = None,
    ) -> RaceDataset:
        return RaceDataset(X_num, driver_idx, team_idx, circuit_idx, y, sample_weight=sample_weight)

    def fit(
        self,
        X_num: np.ndarray,
        driver_idx: np.ndarray,
        team_idx: np.ndarray,
        circuit_idx: np.ndarray,
        y: np.ndarray,
        X_val_num: Optional[np.ndarray] = None,
        driver_val: Optional[np.ndarray] = None,
        team_val: Optional[np.ndarray] = None,
        circuit_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "NNPodiumPredictor":

        if sample_weight is not None and len(sample_weight) != len(X_num):
            raise ValueError(
                f"sample_weight length mismatch: got {len(sample_weight)} expected {len(X_num)}"
            )

        X_num = self._normalize(X_num, fit=True)

        pos_weights = []
        for i in range(y.shape[1]):
            pos_rate = float(y[:, i].mean())
            w = (1.0 - pos_rate) / pos_rate if pos_rate > 0 else 1.0
            pos_weights.append(w)
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weights, dtype=torch.float32).to(DEVICE),
            reduction="none",
        )
        logger.info(f"NN pos_weight: {[f'{w:.1f}x' for w in pos_weights]}")

        train_ds = self._make_dataset(
            X_num,
            driver_idx,
            team_idx,
            circuit_idx,
            y,
            sample_weight=sample_weight,
        )
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        has_val = X_val_num is not None
        if has_val:
            X_val_num = self._normalize(X_val_num, fit=False)
            val_ds = self._make_dataset(
                X_val_num,
                driver_val,
                team_val,
                circuit_val,
                y_val,
                sample_weight=None,
            )
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0
        best_state = None

        for epoch in range(self.max_epochs):
            self.model.train()
            train_loss = 0.0
            for batch in train_loader:
                x_num, d_idx, t_idx, c_idx, targets, weights = [b.to(DEVICE) for b in batch]

                self.optimizer.zero_grad()
                logits = self.model(x_num, d_idx, t_idx, c_idx)
                loss_matrix = self.criterion(logits, targets)
                sample_loss = loss_matrix.mean(dim=1)
                weight_sum = torch.clamp(weights.sum(), min=1e-8)
                loss = (sample_loss * weights).sum() / weight_sum

                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * len(x_num)
            train_loss /= len(train_ds)

            if has_val:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        x_num, d_idx, t_idx, c_idx, targets, _weights = [b.to(DEVICE) for b in batch]
                        logits = self.model(x_num, d_idx, t_idx, c_idx)
                        loss = self.criterion(logits, targets).mean()
                        val_loss += loss.item() * len(x_num)
                val_loss /= len(val_ds)

                self.scheduler.step(val_loss)
                current_lr = float(self.optimizer.param_groups[0]["lr"])

                if (epoch + 1) % 10 == 0 or epoch < 3:
                    logger.info(
                        f"Epoch {epoch + 1}/{self.max_epochs} | "
                        f"train={train_loss:.4f} | val={val_loss:.4f} | lr={current_lr:.2e}"
                    )

                improved = val_loss < (best_val_loss - self.min_delta)
                if improved:
                    best_val_loss = val_loss
                    best_epoch = epoch + 1
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                elif (epoch + 1) >= self.min_epochs:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.info(
                            f"Early stopping at epoch {epoch + 1} "
                            f"(best_epoch={best_epoch}, best_val={best_val_loss:.4f})"
                        )
                        break
            else:
                if (epoch + 1) % 10 == 0 or epoch < 3:
                    logger.info(f"Epoch {epoch + 1}/{self.max_epochs} | train={train_loss:.4f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)
            logger.info(
                f"Restored best model (best_epoch={best_epoch}, val_loss={best_val_loss:.4f})"
            )

        return self

    def predict_proba(
        self,
        X_num: np.ndarray,
        driver_idx: np.ndarray,
        team_idx: np.ndarray,
        circuit_idx: np.ndarray,
    ) -> np.ndarray:
        """Returns shape (N, 3) array of probabilities [P1, P2, P3]."""
        X_num = self._normalize(X_num, fit=False)
        ds = self._make_dataset(X_num, driver_idx, team_idx, circuit_idx, y=None)
        loader = DataLoader(ds, batch_size=256, shuffle=False)

        self.model.eval()
        all_probs = []
        with torch.no_grad():
            for batch in loader:
                x_num, d_idx, t_idx, c_idx = [b.to(DEVICE) for b in batch]
                logits = self.model(x_num, d_idx, t_idx, c_idx)
                probs = torch.sigmoid(logits)
                all_probs.append(probs.cpu().numpy())

        return np.vstack(all_probs)

    def save(self, name: str = "nn_podium") -> None:
        path = MODELS_DIR / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Saved NN model -> {path}")

    @classmethod
    def load(cls, name: str = "nn_podium") -> "NNPodiumPredictor":
        path = MODELS_DIR / f"{name}.pkl"
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"Loaded NN model <- {path}")
        return obj
