"""
Neural Policy Trainer for DislocationDynamicsNet.
Optimizes portfolio weights directly for Net Sharpe Ratio with 5 bps transaction costs.
"""
from dataclasses import dataclass
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from cmbr.models.dynamics_net import DislocationDynamicsNet
from cmbr.models.loss import DifferentiableNetSharpeLoss
from cmbr.models.dataset import ResidualSequenceDataset

@dataclass
class TrainingHistory:
    epochs: list
    train_net_sharpe: list
    train_gross_sharpe: list
    train_turnover: list
    val_net_sharpe: list

class NeuralPolicyTrainer:
    def __init__(
        self,
        n_assets: int = 423,
        hidden_dim: int = 64,
        lookback: int = 15,
        transaction_cost_bps: float = 5.0,
        max_gross_leverage: float = 2.5,
        turnover_penalty_weight: float = 0.02,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.lookback = lookback
        self.max_gross_leverage = max_gross_leverage
        self.model = DislocationDynamicsNet(n_assets=n_assets, hidden_dim=hidden_dim).to(self.device)
        self.criterion = DifferentiableNetSharpeLoss(
            transaction_cost_bps=transaction_cost_bps,
            turnover_penalty_weight=turnover_penalty_weight
        )
        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=5)

    def train_epoch(
        self,
        X_train: torch.Tensor,
        Y_train: torch.Tensor,
        batch_size: int = 252
    ) -> dict:
        """
        Trains over rolling trajectory blocks to compute empirical Net Sharpe with realistic turnover.
        """
        self.model.train()
        T = len(X_train)

        indices = list(range(0, T - batch_size + 1, batch_size // 2))
        np.random.shuffle(indices)

        epoch_telemetry = []

        for start_idx in indices:
            end_idx = start_idx + batch_size
            seq_batch = X_train[start_idx:end_idx].to(self.device) # [batch_size, lookback, n_assets]
            ret_batch = Y_train[start_idx:end_idx].to(self.device) # [batch_size, n_assets]

            self.optimizer.zero_grad()

            raw_weights = self.model(seq_batch)
            weights = self.model.project_dollar_neutral(raw_weights, max_gross_leverage=self.max_gross_leverage)

            loss, tel = self.criterion(weights, ret_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            epoch_telemetry.append(tel)

        # Average metrics
        avg_metrics = {
            k: float(np.mean([t[k] for t in epoch_telemetry]))
            for k in epoch_telemetry[0]
        }
        return avg_metrics

    @torch.no_grad()
    def evaluate(self, X: torch.Tensor, Y: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Evaluates full trajectory out-of-sample.
        """
        self.model.eval()
        X_dev = X.to(self.device)
        Y_dev = Y.to(self.device)

        # Process in chunks if needed to conserve memory
        raw_w = self.model(X_dev)
        weights = self.model.project_dollar_neutral(raw_w, max_gross_leverage=self.max_gross_leverage)

        loss, tel = self.criterion(weights, Y_dev)
        return weights.cpu(), tel

    def fit(
        self,
        residuals: pd.DataFrame,
        returns: pd.DataFrame,
        train_end_date: str = "2020-12-31",
        num_epochs: int = 50,
        batch_size: int = 252,
        verbose: bool = True
    ) -> tuple[TrainingHistory, dict]:
        dataset = ResidualSequenceDataset(residuals, returns, lookback=self.lookback)
        tensors = dataset.get_full_tensors()

        dates = pd.to_datetime(tensors.dates)
        split_date = pd.to_datetime(train_end_date)

        train_mask = dates <= split_date
        val_mask = dates > split_date

        X_train, Y_train = tensors.residual_sequences[train_mask], tensors.forward_returns[train_mask]
        X_val, Y_val = tensors.residual_sequences[val_mask], tensors.forward_returns[val_mask]

        if verbose:
            print(f"[Neural Policy Trainer] Dataset Split:")
            print(f"  Training samples:   {len(X_train)} days ({dates[train_mask][0].date()} to {dates[train_mask][-1].date()})")
            print(f"  Validation samples: {len(X_val)} days ({dates[val_mask][0].date()} to {dates[val_mask][-1].date()})")
            print(f"  Direct Objective:   Maximize Net Sharpe after 5 bps transaction friction")

        history = TrainingHistory([], [], [], [], [])
        best_val_sharpe = -999.0
        best_state = None

        t0 = time.time()
        for epoch in range(1, num_epochs + 1):
            train_tel = self.train_epoch(X_train, Y_train, batch_size=batch_size)
            _, val_tel = self.evaluate(X_val, Y_val)

            self.scheduler.step(val_tel["loss"])

            history.epochs.append(epoch)
            history.train_net_sharpe.append(train_tel["net_sharpe"])
            history.train_gross_sharpe.append(train_tel["gross_sharpe"])
            history.train_turnover.append(train_tel["avg_turnover"])
            history.val_net_sharpe.append(val_tel["net_sharpe"])

            if val_tel["net_sharpe"] > best_val_sharpe:
                best_val_sharpe = val_tel["net_sharpe"]
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == num_epochs):
                print(
                    f"Epoch {epoch:2d}/{num_epochs:2d} | "
                    f"Train Net Sharpe: {train_tel['net_sharpe']:5.2f} (Gross: {train_tel['gross_sharpe']:5.2f}, Turn: {train_tel['avg_turnover']*100:4.1f}%) | "
                    f"Val Net Sharpe: {val_tel['net_sharpe']:5.2f} (Gross: {val_tel['gross_sharpe']:5.2f}) | "
                    f"Elapsed: {time.time()-t0:.1f}s"
                )

        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Final Out-Of-Sample Evaluation on Validation
        opt_weights, final_val_tel = self.evaluate(X_val, Y_val)
        val_dates = dates[val_mask]
        weights_df = pd.DataFrame(opt_weights.numpy(), index=val_dates, columns=returns.columns)

        return history, {
            "val_telemetry": final_val_tel,
            "val_weights": weights_df,
            "val_dates": val_dates
        }
