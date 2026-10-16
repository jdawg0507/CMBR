"""
Dollar-neutral portfolio allocation and execution engine with 5 bps transaction costs.
Enforces max gross leverage, strict dollar neutrality, and turnover cost tracking.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from cmbr.config import CMBRConfig

@dataclass
class BacktestResult:
    daily_returns_gross: pd.Series
    daily_returns_net: pd.Series
    cumulative_returns_gross: pd.Series
    cumulative_returns_net: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    transaction_costs: pd.Series
    long_count: pd.Series
    short_count: pd.Series

class PortfolioAllocator:
    def __init__(self, config: CMBRConfig = None):
        self.config = config or CMBRConfig()

    def allocate_weights(self, positions: pd.DataFrame) -> pd.DataFrame:
        """
        Translates discrete position states {-1, 0, 1} into dollar-neutral portfolio weights
        such that:
        sum(w_long) = +0.5 * max_leverage
        sum(w_short) = -0.5 * max_leverage
        Net exposure = 0.0, Gross leverage = 1.0 * max_leverage
        """
        max_lev = self.config.max_gross_leverage
        half_lev = max_lev / 2.0

        pos_vals = positions.values
        T, N = pos_vals.shape
        weights = np.zeros((T, N), dtype=np.float64)

        for t in range(T):
            p = pos_vals[t]
            long_idx = np.where(p > 0)[0]
            short_idx = np.where(p < 0)[0]
            n_long = len(long_idx)
            n_short = len(short_idx)

            if n_long > 0 and n_short > 0:
                weights[t, long_idx] = half_lev / n_long
                weights[t, short_idx] = -half_lev / n_short
            elif n_long > 0 and n_short == 0:
                # If only longs exist, scale to half_lev and balance with uniform short market hedge
                weights[t, long_idx] = half_lev / n_long
                weights[t] -= (half_lev / N) # equal-weight market short hedge
            elif n_short > 0 and n_long == 0:
                # If only shorts exist, scale to half_lev and balance with uniform long market hedge
                weights[t, short_idx] = -half_lev / n_short
                weights[t] += (half_lev / N) # equal-weight market long hedge

        return pd.DataFrame(weights, index=positions.index, columns=positions.columns)

    def simulate(self, weights: pd.DataFrame, asset_returns: pd.DataFrame) -> BacktestResult:
        """
        Simulates portfolio execution day-by-day including drift and transaction costs (5 bps).
        """
        fee_rate = self.config.transaction_cost_bps / 10000.0  # e.g. 5 bps = 0.0005

        aligned_weights, aligned_returns = weights.align(asset_returns, join="inner")
        dates = aligned_weights.index
        T, N = aligned_weights.shape

        w_vals = aligned_weights.values
        r_vals = aligned_returns.values

        ret_gross = np.zeros(T, dtype=np.float64)
        ret_net = np.zeros(T, dtype=np.float64)
        turnover = np.zeros(T, dtype=np.float64)
        costs = np.zeros(T, dtype=np.float64)

        # Track holdings across days
        w_prev_end = np.zeros(N, dtype=np.float64)

        for t in range(T):
            w_curr = w_vals[t]
            # Turnover is the rebalancing difference from previous end-of-day drifted weights
            delta_w = np.sum(np.abs(w_curr - w_prev_end))
            turnover[t] = delta_w
            cost = delta_w * fee_rate
            costs[t] = cost

            # Daily return: dot product of weights with asset returns
            # Using simple return equivalent r_simple = exp(r_log) - 1
            r_simple = np.expm1(r_vals[t])
            rg = float(np.dot(w_curr, r_simple))
            rn = rg - cost

            ret_gross[t] = rg
            ret_net[t] = rn

            # Weights drift due to asset returns: w_{end} = w * (1 + r) / (1 + r_port)
            w_drifted = w_curr * (1.0 + r_simple)
            # Normalize to avoid numerical drift blowup
            w_prev_end = w_drifted / (1.0 + rg) if abs(1.0 + rg) > 1e-6 else w_curr

        gross_series = pd.Series(ret_gross, index=dates)
        net_series = pd.Series(ret_net, index=dates)
        turnover_series = pd.Series(turnover, index=dates)
        costs_series = pd.Series(costs, index=dates)

        cum_gross = (1.0 + gross_series).cumprod() - 1.0
        cum_net = (1.0 + net_series).cumprod() - 1.0

        long_counts = (aligned_weights > 1e-5).sum(axis=1)
        short_counts = (aligned_weights < -1e-5).sum(axis=1)

        return BacktestResult(
            daily_returns_gross=gross_series,
            daily_returns_net=net_series,
            cumulative_returns_gross=cum_gross,
            cumulative_returns_net=cum_net,
            weights=aligned_weights,
            turnover=turnover_series,
            transaction_costs=costs_series,
            long_count=long_counts,
            short_count=short_counts
        )
