"""
Neural CMBR Runner: Trains DislocationDynamicsNet to maximize Net Sharpe after 5 bps friction
and benchmarks its out-of-sample performance against the rule-based baseline.
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cmbr.config import CMBRConfig
from cmbr.data.loader import DataLoader
from cmbr.engine.rmt import RMTEngine
from cmbr.engine.projector import ManifoldProjector
from cmbr.signals.dislocation import DislocationSignalGenerator
from cmbr.backtest.portfolio import PortfolioAllocator
from cmbr.backtest.metrics import PerformanceEvaluator
from cmbr.models.trainer import NeuralPolicyTrainer

def compute_all_residuals(returns: pd.DataFrame, config: CMBRConfig, cache_path: Path) -> pd.DataFrame:
    """Computes or loads cached rolling RMT orthogonal residuals."""
    if cache_path.exists():
        print(f"[CMBR Pipeline] Loading cached residuals from {cache_path}...")
        return pd.read_parquet(cache_path)

    print("[CMBR Pipeline] Computing rolling RMT residuals...")
    T_total, N = returns.shape
    T_window = config.rolling_window_T
    freq = config.re_estimation_freq

    rmt = RMTEngine(method=config.noise_variance_estimator)
    residuals = []
    out_dates = returns.index[T_window:]

    current_projector = None
    for idx, t in enumerate(range(T_window, T_total)):
        if idx % freq == 0 or current_projector is None:
            train_data = returns.iloc[t - T_window:t].values
            Z = (train_data - train_data.mean(axis=0)) / (train_data.std(axis=0) + 1e-8)
            C = (Z.T @ Z) / T_window
            res = rmt.decompose(C, T_window)
            current_projector = ManifoldProjector(res.signal_eigenvectors)

        r_t = returns.iloc[t].values
        _, e_t = current_projector.project(r_t)
        residuals.append(e_t)

    res_df = pd.DataFrame(residuals, index=out_dates, columns=returns.columns)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_parquet(cache_path)
    print(f"[CMBR Pipeline] Cached {len(res_df)} residual days to {cache_path}")
    return res_df

def plot_neural_diagnostics(
    history,
    neural_res,
    baseline_res,
    benchmark_returns,
    neural_summary,
    baseline_summary,
    output_dir: Path
):
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 11))

    # 1. Out-of-Sample Cumulative Returns Comparison
    aligned_bm = benchmark_returns.reindex(neural_res.daily_returns_net.index).mean(axis=1)
    cum_bm = (1.0 + aligned_bm).cumprod() - 1.0

    ax1.plot(neural_res.cumulative_returns_net.index, neural_res.cumulative_returns_net * 100.0,
             label=f"Neural Policy Net (Sharpe: {neural_summary.sharpe_ratio_net:.2f})", color="#2563eb", lw=2.2)
    ax1.plot(baseline_res.cumulative_returns_net.index, baseline_res.cumulative_returns_net * 100.0,
             label=f"Rule Baseline Net (Sharpe: {baseline_summary.sharpe_ratio_net:.2f})", color="#f59e0b", lw=1.8, ls="--")
    ax1.plot(cum_bm.index, cum_bm * 100.0,
             label="S&P 500 Market Baseline", color="#94a3b8", lw=1.4, ls=":")

    ax1.set_title("Out-of-Sample Performance (2021 - 2026 Test Period)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Net Cumulative Return (%)", fontsize=11)
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(True, alpha=0.3)

    # 2. Drawdowns Comparison
    cum_net_nn = (1.0 + neural_res.daily_returns_net).cumprod()
    dd_nn = (cum_net_nn - cum_net_nn.cummax()) / cum_net_nn.cummax() * 100.0

    cum_net_base = (1.0 + baseline_res.daily_returns_net).cumprod()
    dd_base = (cum_net_base - cum_net_base.cummax()) / cum_net_base.cummax() * 100.0

    ax2.plot(dd_nn.index, dd_nn.values, label=f"Neural Policy DD (Max: {neural_summary.max_drawdown_pct:.1f}%)", color="#2563eb", lw=1.8)
    ax2.plot(dd_base.index, dd_base.values, label=f"Rule Baseline DD (Max: {baseline_summary.max_drawdown_pct:.1f}%)", color="#f59e0b", lw=1.5, ls="--")
    ax2.set_title("Drawdown Profile (2021 - 2026)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Drawdown (%)", fontsize=11)
    ax2.legend(loc="lower left", frameon=True)
    ax2.grid(True, alpha=0.3)

    # 3. Training Dynamics: Net Sharpe Learning Curve
    ax3.plot(history.epochs, history.train_net_sharpe, label="Train Net Sharpe", color="#10b981", lw=2)
    ax3.plot(history.epochs, history.val_net_sharpe, label="Validation Net Sharpe", color="#6366f1", lw=2, ls="--")
    ax3.axhline(0, color="#94a3b8", ls=":", lw=1)
    ax3.set_title("Optimization Objective: Net Sharpe after 5 bps Friction", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Epoch", fontsize=11)
    ax3.set_ylabel("Annualized Net Sharpe Ratio", fontsize=11)
    ax3.legend(loc="upper left", frameon=True)
    ax3.grid(True, alpha=0.3)

    # 4. Churn Suppression: Daily Turnover Comparison
    ax4.plot(neural_res.turnover.index, neural_res.turnover.rolling(20).mean() * 100.0,
             label=f"Neural Policy Turnover (Avg: {neural_summary.avg_daily_turnover_pct:.1f}%)", color="#2563eb", lw=1.8)
    ax4.plot(baseline_res.turnover.index, baseline_res.turnover.rolling(20).mean() * 100.0,
             label=f"Rule Baseline Turnover (Avg: {baseline_summary.avg_daily_turnover_pct:.1f}%)", color="#ef4444", lw=1.5, ls="--")
    ax4.set_title("Turnover & Churn Suppression (20-day Rolling)", fontsize=13, fontweight="bold")
    ax4.set_xlabel("Date", fontsize=11)
    ax4.set_ylabel("Daily Turnover (%)", fontsize=11)
    ax4.legend(loc="upper right", frameon=True)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = output_dir / "neural_vs_baseline_comparison.png"
    plt.savefig(chart_path, dpi=200)
    plt.close()
    print(f"[Neural CMBR] Saved comparison chart to {chart_path}")

def main():
    print("=" * 75)
    print("  DislocationDynamicsNet: Second-Order Neural Portfolio Policy")
    print("=" * 75)

    config = CMBRConfig()
    loader = DataLoader(config)
    prices, returns, metadata = loader.load_aligned_data()

    residuals_cache = config.cache_dir / "rmt_residuals.parquet"
    residuals = compute_all_residuals(returns, config, residuals_cache)

    n_assets = returns.shape[1]
    lookback = 15
    train_end = "2020-12-31"

    print(f"\n[Step 1/3] Initializing Neural Policy Trainer...")
    trainer = NeuralPolicyTrainer(
        n_assets=n_assets,
        hidden_dim=64,
        lookback=lookback,
        transaction_cost_bps=config.transaction_cost_bps,
        max_gross_leverage=config.max_gross_leverage,
        lr=1e-3,
        weight_decay=1e-3
    )

    print(f"\n[Step 2/3] Training DislocationDynamicsNet directly on Net Sharpe...")
    history, train_out = trainer.fit(
        residuals=residuals,
        returns=returns,
        train_end_date=train_end,
        num_epochs=30,
        batch_size=252,
        verbose=True
    )

    val_weights = train_out["val_weights"]
    val_dates = train_out["val_dates"]

    print(f"\n[Step 3/3] Benchmarking Neural Policy vs Rule Baseline on Test Period (2021-2026)...")
    allocator = PortfolioAllocator(config)
    evaluator = PerformanceEvaluator(config.annual_trading_days)

    # 1. Neural Policy Simulation
    # Lag weights by 1 day: signals at day t close execute into day t+1
    neural_weights_lagged = val_weights.shift(1).fillna(0.0)
    test_returns = returns.loc[val_dates]
    neural_sim = allocator.simulate(neural_weights_lagged, test_returns)
    neural_summary = evaluator.evaluate(
        neural_sim.daily_returns_gross,
        neural_sim.daily_returns_net,
        neural_sim.turnover,
        neural_sim.transaction_costs
    )

    # 2. Rule Baseline Simulation on the same test period
    sig_gen = DislocationSignalGenerator(config)
    res_test = residuals.loc[val_dates]
    zscores_test = sig_gen.compute_zscores(res_test)
    pos_test, _ = sig_gen.generate_positions(zscores_test)
    base_weights_raw = allocator.allocate_weights(pos_test)
    base_weights_lagged = base_weights_raw.shift(1).fillna(0.0)
    base_sim = allocator.simulate(base_weights_lagged, test_returns)
    base_summary = evaluator.evaluate(
        base_sim.daily_returns_gross,
        base_sim.daily_returns_net,
        base_sim.turnover,
        base_sim.transaction_costs
    )

    # Plot Diagnostics
    plot_neural_diagnostics(
        history=history,
        neural_res=neural_sim,
        baseline_res=base_sim,
        benchmark_returns=returns,
        neural_summary=neural_summary,
        baseline_summary=base_summary,
        output_dir=config.output_dir
    )

    print("\n" + "=" * 75)
    print("  Out-of-Sample Performance Comparison (2021-01-04 to 2026-10-15)")
    print("=" * 75)
    print(f"{'Metric':<30} | {'Rule Baseline':<18} | {'Neural Dynamics Net':<18}")
    print("-" * 75)
    print(f"{'Annualized Gross Return':<30} | {base_summary.annualized_return_gross:>15.2f}% | {neural_summary.annualized_return_gross:>15.2f}%")
    print(f"{'Annualized Net Return':<30} | {base_summary.annualized_return_net:>15.2f}% | {neural_summary.annualized_return_net:>15.2f}%")
    print(f"{'Annualized Volatility':<30} | {base_summary.annualized_volatility:>15.2f}% | {neural_summary.annualized_volatility:>15.2f}%")
    print(f"{'Gross Sharpe Ratio':<30} | {base_summary.sharpe_ratio_gross:>15.2f}  | {neural_summary.sharpe_ratio_gross:>15.2f} ")
    print(f"{'Net Sharpe Ratio':<30} | {base_summary.sharpe_ratio_net:>15.2f}  | {neural_summary.sharpe_ratio_net:>15.2f} ")
    print(f"{'Max Drawdown':<30} | {base_summary.max_drawdown_pct:>15.2f}% | {neural_summary.max_drawdown_pct:>15.2f}%")
    print(f"{'Daily Win Rate':<30} | {base_summary.win_rate_pct:>15.2f}% | {neural_summary.win_rate_pct:>15.2f}%")
    print(f"{'Average Daily Turnover':<30} | {base_summary.avg_daily_turnover_pct:>15.2f}% | {neural_summary.avg_daily_turnover_pct:>15.2f}%")
    print(f"{'Total 5 bps Costs Paid':<30} | {base_summary.total_transaction_costs_pct:>15.2f}% | {neural_summary.total_transaction_costs_pct:>15.2f}%")
    print("=" * 75)

if __name__ == "__main__":
    main()
