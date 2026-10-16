"""
Vectorized residual dislocation signal generator and state machine.
Computes the rolling cumulative Z-score of orthogonal residuals and generates
asymmetric long/short/exit/stop-loss trading signals based on elastic manifold dislocation.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from cmbr.config import CMBRConfig

@dataclass
class SignalMetrics:
    total_entries_long: int
    total_entries_short: int
    total_exits_normal: int
    total_exits_stoploss: int
    avg_holding_period_days: float

class DislocationSignalGenerator:
    def __init__(self, config: CMBRConfig = None):
        self.config = config or CMBRConfig()

    def compute_zscores(self, residuals: pd.DataFrame) -> pd.DataFrame:
        """
        Computes rolling cumulative Z-score s_{i,t} of orthogonal residuals:
        x_{i,t} = cumsum(e_{i,t})
        s_{i,t} = (x_{i,t} - SMA(x_{i,t}, W)) / Std(x_{i,t}, W)
        """
        W = self.config.cum_residual_window
        # Cumulative sum of residual idiosyncratic returns
        cum_residuals = residuals.cumsum()

        # Rolling mean and standard deviation
        rolling_mean = cum_residuals.rolling(window=W, min_periods=max(W // 2, 10)).mean()
        rolling_std = cum_residuals.rolling(window=W, min_periods=max(W // 2, 10)).std()

        # Z-score of dislocation from the local trend
        zscores = (cum_residuals - rolling_mean) / (rolling_std + 1e-8)
        return zscores

    def generate_positions(self, zscores: pd.DataFrame) -> tuple[pd.DataFrame, SignalMetrics]:
        """
        Runs the state machine across time for all assets:
        - Long Entry: s_{i,t} < long_entry_z (-2.0)
        - Short Entry: s_{i,t} > short_entry_z (+2.0)
        - Exit Mean Reversion: cross back through exit_z (0.0)
        - Stop Loss Regime Break: |s_{i,t}| > stop_loss_z (4.0)

        Returns:
            positions_df: DataFrame of {-1, 0, +1} target position states
            metrics: Signal generation telemetry
        """
        long_thresh = self.config.long_entry_z
        short_thresh = self.config.short_entry_z
        exit_thresh = self.config.exit_z
        stop_thresh = self.config.stop_loss_z

        T, N = zscores.shape
        z_vals = zscores.values
        positions = np.zeros((T, N), dtype=np.int8)

        n_long_entries = 0
        n_short_entries = 0
        n_exits_norm = 0
        n_exits_stop = 0
        holding_periods = []

        # Current trade entry day for each asset
        entry_day = np.full(N, -1, dtype=int)

        for t in range(1, T):
            # Carry over previous position
            prev_p = positions[t - 1]
            curr_z = z_vals[t]
            curr_p = np.copy(prev_p)

            for i in range(N):
                z = curr_z[i]
                if np.isnan(z):
                    curr_p[i] = 0
                    continue

                p = prev_p[i]

                if p == 0:
                    # Flat -> Look for entry within safe zone (|z| < stop_thresh)
                    if z <= long_thresh and z > -stop_thresh:
                        curr_p[i] = 1
                        n_long_entries += 1
                        entry_day[i] = t
                    elif z >= short_thresh and z < stop_thresh:
                        curr_p[i] = -1
                        n_short_entries += 1
                        entry_day[i] = t
                elif p == 1:
                    # Long position
                    if z >= exit_thresh:
                        # Mean-reverted back to manifold
                        curr_p[i] = 0
                        n_exits_norm += 1
                        if entry_day[i] >= 0:
                            holding_periods.append(t - entry_day[i])
                            entry_day[i] = -1
                    elif z <= -stop_thresh:
                        # Severe regime break / structural breakdown
                        curr_p[i] = 0
                        n_exits_stop += 1
                        if entry_day[i] >= 0:
                            holding_periods.append(t - entry_day[i])
                            entry_day[i] = -1
                elif p == -1:
                    # Short position
                    if z <= exit_thresh:
                        # Mean-reverted back to manifold
                        curr_p[i] = 0
                        n_exits_norm += 1
                        if entry_day[i] >= 0:
                            holding_periods.append(t - entry_day[i])
                            entry_day[i] = -1
                    elif z >= stop_thresh:
                        # Severe upside regime break (e.g. buyout)
                        curr_p[i] = 0
                        n_exits_stop += 1
                        if entry_day[i] >= 0:
                            holding_periods.append(t - entry_day[i])
                            entry_day[i] = -1

            positions[t] = curr_p

        pos_df = pd.DataFrame(positions, index=zscores.index, columns=zscores.columns)
        avg_holding = float(np.mean(holding_periods)) if holding_periods else 0.0

        metrics = SignalMetrics(
            total_entries_long=n_long_entries,
            total_entries_short=n_short_entries,
            total_exits_normal=n_exits_norm,
            total_exits_stoploss=n_exits_stop,
            avg_holding_period_days=round(avg_holding, 1)
        )
        return pos_df, metrics
