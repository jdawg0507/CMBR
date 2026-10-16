"""
Rolling Backtest Engine for the CMBR Invariant Trading Engine.
Orchestrates rolling RMT spectral re-estimation, orthogonal projection,
dislocation signal generation, and dollar-neutral portfolio execution.
"""
from dataclasses import dataclass
import time
import numpy as np
import pandas as pd
from cmbr.config import CMBRConfig
from cmbr.engine.rmt import RMTEngine
from cmbr.engine.projector import ManifoldProjector
from cmbr.signals.dislocation import DislocationSignalGenerator, SignalMetrics
from cmbr.backtest.portfolio import PortfolioAllocator, BacktestResult
from cmbr.backtest.metrics import PerformanceEvaluator, PerformanceSummary

@dataclass
class ManifoldTelemetry:
    dates: list
    signal_modes_count: list
    market_mode_lambda: list
    market_variance_pct: list
    subspace_stability: list

class RollingBacktestEngine:
    def __init__(self, config: CMBRConfig = None):
        self.config = config or CMBRConfig()
        self.rmt_engine = RMTEngine(method=self.config.noise_variance_estimator)
        self.signal_gen = DislocationSignalGenerator(self.config)
        self.portfolio_alloc = PortfolioAllocator(self.config)
        self.evaluator = PerformanceEvaluator(self.config.annual_trading_days)

    def run(self, returns: pd.DataFrame, verbose: bool = True) -> tuple[BacktestResult, PerformanceSummary, SignalMetrics, ManifoldTelemetry]:
        """
        Executes rolling out-of-sample backtest over the full historical period.
        """
        t0 = time.time()
        T_total, N = returns.shape
        T_window = self.config.rolling_window_T
        freq = self.config.re_estimation_freq

        if T_total <= T_window:
            raise ValueError(f"Insufficient history ({T_total} days) for rolling window T={T_window}")

        dates = returns.index
        out_dates = dates[T_window:]
        T_out = len(out_dates)

        if verbose:
            print(f"[CMBR Engine] Initializing rolling simulation...")
            print(f"  Universe: {N} S&P 500 constituents")
            print(f"  Historical sample: {T_total} trading days ({dates[0].date()} to {dates[-1].date()})")
            print(f"  Out-of-sample test: {T_out} trading days ({out_dates[0].date()} to {out_dates[-1].date()})")
            print(f"  Lookback window T: {T_window} days | Re-estimation frequency: every {freq} days")

        residuals = np.zeros((T_out, N), dtype=np.float64)
        projected = np.zeros((T_out, N), dtype=np.float64)

        telemetry = ManifoldTelemetry([], [], [], [], [])
        prev_V = None
        current_projector = None

        # Rolling loop
        for idx, t in enumerate(range(T_window, T_total)):
            curr_date = dates[t]

            # Re-estimate CMBR eigenspace every `freq` days
            if idx % freq == 0 or current_projector is None:
                train_data = returns.iloc[t - T_window:t].values
                # Standardize returns across rolling window
                means = np.mean(train_data, axis=0)
                stds = np.std(train_data, axis=0) + 1e-8
                Z = (train_data - means) / stds

                # Empirical correlation matrix C = 1/T * Z^T Z
                C = (Z.T @ Z) / T_window

                # RMT Marchenko-Pastur Decomposition
                rmt_res = self.rmt_engine.decompose(C, T_window)
                current_projector = ManifoldProjector(rmt_res.signal_eigenvectors)

                # Track manifold telemetry
                k = rmt_res.num_signal_modes
                l1 = float(rmt_res.eigenvalues[0])
                var_pct = float((l1 / N) * 100.0)

                stab = 1.0
                if prev_V is not None:
                    stab = ManifoldProjector.subspace_stability(prev_V, rmt_res.signal_eigenvectors)
                prev_V = rmt_res.signal_eigenvectors

                telemetry.dates.append(curr_date)
                telemetry.signal_modes_count.append(k)
                telemetry.market_mode_lambda.append(l1)
                telemetry.market_variance_pct.append(var_pct)
                telemetry.subspace_stability.append(stab)

            # Daily orthogonal projection of out-of-sample return r_t
            r_t = returns.iloc[t].values
            r_hat, e_t = current_projector.project(r_t)

            residuals[idx] = e_t
            projected[idx] = r_hat

            if verbose and (idx + 1) % 500 == 0:
                elapsed = time.time() - t0
                pct = ((idx + 1) / T_out) * 100.0
                print(f"  Progress: {idx+1}/{T_out} days ({pct:.1f}%) | Subspace stability: {stab:.3f} | Elapsed: {elapsed:.1f}s")

        res_df = pd.DataFrame(residuals, index=out_dates, columns=returns.columns)
        oos_returns = returns.loc[out_dates]

        if verbose:
            print(f"[CMBR Engine] Manifold projection completed in {time.time()-t0:.2f}s. Generating signals...")

        # Compute cumulative residual Z-scores and discrete positions
        zscores_df = self.signal_gen.compute_zscores(res_df)
        positions_df, signal_metrics = self.signal_gen.generate_positions(zscores_df)

        if verbose:
            print(f"[CMBR Engine] Allocating dollar-neutral portfolio and executing simulation...")

        # Dollar-neutral allocation and backtest simulation
        # Out-of-sample execution: signals generated at day t close trade on day t+1
        weights_raw = self.portfolio_alloc.allocate_weights(positions_df)
        weights_df = weights_raw.shift(1).fillna(0.0)
        backtest_res = self.portfolio_alloc.simulate(weights_df, oos_returns)

        # Performance evaluation and metrics
        summary = self.evaluator.evaluate(
            daily_returns_gross=backtest_res.daily_returns_gross,
            daily_returns_net=backtest_res.daily_returns_net,
            turnover=backtest_res.turnover,
            costs=backtest_res.transaction_costs,
            zscores=zscores_df,
            residuals=res_df
        )

        if verbose:
            print(f"[CMBR Engine] Backtest complete!")
            print(f"  Annualized Net Return: {summary.annualized_return_net:.2f}%")
            print(f"  Annualized Volatility: {summary.annualized_volatility:.2f}%")
            print(f"  Sharpe Ratio (Net):    {summary.sharpe_ratio_net:.2f}")
            print(f"  Max Drawdown:          {summary.max_drawdown_pct:.2f}%")
            print(f"  Information Coeff 1D:  {summary.ic_1d_mean:.4f} (t-stat: {summary.ic_1d_tstat:.2f})")

        return backtest_res, summary, signal_metrics, telemetry
