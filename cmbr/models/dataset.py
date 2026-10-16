"""
Sequence dataset builder for residual inputs and forward returns.
Transforms orthogonal residuals e_t into sliding sequence windows [T, Lookback, 423] for GRU training.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

@dataclass
class SequentialTensors:
    residual_sequences: torch.Tensor # [N_samples, Lookback, 423]
    forward_returns: torch.Tensor    # [N_samples, 423]
    dates: list

class ResidualSequenceDataset(Dataset):
    def __init__(
        self,
        residuals: pd.DataFrame,
        returns: pd.DataFrame,
        lookback: int = 15
    ):
        """
        Args:
            residuals: Residual dislocations e_t (T x n_assets)
            returns: Asset returns r_t (T x n_assets)
            lookback: Number of historical lag days to ingest into GRU
        """
        self.lookback = lookback
        
        # Align timestamps
        aligned_res, aligned_ret = residuals.align(returns, join="inner")
        
        res_vals = aligned_res.values.astype(np.float32)
        ret_vals = aligned_ret.values.astype(np.float32)
        dates = aligned_res.index

        T_total = len(aligned_res)
        if T_total <= lookback + 1:
            raise ValueError(f"Insufficient samples ({T_total}) for lookback {lookback}")

        X_list = []
        Y_list = []
        sample_dates = []

        # At day t: features are residuals from [t - lookback + 1 : t + 1]
        # Target is next-day forward return: r_{t+1}
        for t in range(lookback - 1, T_total - 1):
            seq = res_vals[t - lookback + 1 : t + 1] # [lookback, n_assets]
            fwd_r = ret_vals[t + 1]                  # [n_assets]
            
            X_list.append(seq)
            Y_list.append(fwd_r)
            sample_dates.append(dates[t + 1])

        self.X = torch.from_numpy(np.stack(X_list, axis=0)) # [N_samples, lookback, n_assets]
        self.Y = torch.from_numpy(np.stack(Y_list, axis=0)) # [N_samples, n_assets]
        self.dates = sample_dates

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]

    def get_full_tensors(self) -> SequentialTensors:
        return SequentialTensors(
            residual_sequences=self.X,
            forward_returns=self.Y,
            dates=self.dates
        )
