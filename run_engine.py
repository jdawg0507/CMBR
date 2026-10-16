"""
Main execution script for the CMBR Invariant Trading Engine.
Runs the complete 16-year out-of-sample backtest, generates performance tables,
and exports publication-quality diagnostic charts.
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from cmbr.config import CMBRConfig
from cmbr.data.loader import DataLoader
from cmbr.engine.rmt import RMTEngine
from cmbr.backtest.engine import RollingBacktestEngine

def plot_performance_suite(
    backtest_res,
    summary,
    telemetry,
    market_returns,
    meta,
    output_dir: Path
):
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Performance and Drawdown Chart
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True, gridspec_kw={'height_ratios': [3, 1.5, 1.5]})

    # Benchmark: S&P 500 equal-weighted or cap-weighted market return
    aligned_mkt = market_returns.reindex(backtest_res.daily_returns_net.index).mean(axis=1)
    cum_mkt = (1.0 + aligned_mkt).cumprod() - 1.0

    ax1.plot(backtest_res.cumulative_returns_gross.index, backtest_res.cumulative_returns_gross * 100.0,
             label=f"CMBR Strategy Gross (Sharpe: {summary.sharpe_ratio_gross:.2f})", color="#2563eb", lw=2)
    ax1.plot(backtest_res.cumulative_returns_net.index, backtest_res.cumulative_returns_net * 100.0,
             label=f"CMBR Strategy Net (-5 bps) (Sharpe: {summary.sharpe_ratio_net:.2f})", color="#059669", lw=2.2)
    ax1.plot(cum_mkt.index, cum_mkt * 100.0,
             label="S&P 500 Market Baseline", color="#64748b", lw=1.5, ls="--", alpha=0.8)

    ax1.set_title("Eigenspace Stress Decomposition & CMBR Invariant Trading Engine: Out-of-Sample Performance", fontsize=15, fontweight="bold", pad=12)
    ax1.set_ylabel("Cumulative Return (%)", fontsize=12)
    ax1.legend(loc="upper left", frameon=True, fontsize=11)
    ax1.grid(True, alpha=0.3)

    # 2. Drawdown Chart
    cum_net = (1.0 + backtest_res.daily_returns_net).cumprod()
    dd = (cum_net - cum_net.cummax()) / cum_net.cummax() * 100.0
    ax2.fill_between(dd.index, dd.values, 0, color="#dc2626", alpha=0.35, label=f"Max Drawdown ({summary.max_drawdown_pct:.1f}%)")
    ax2.plot(dd.index, dd.values, color="#b91c1c", lw=1.2)
    ax2.set_ylabel("Drawdown (%)", fontsize=12)
    ax2.set_ylim(min(dd.min() * 1.2, -5), 2)
    ax2.legend(loc="lower left", frameon=True, fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 3. Manifold Subspace Stability & Active Holdings
    ax3_twin = ax3.twinx()
    dates_tel = pd.to_datetime(telemetry.dates)
    ax3.plot(dates_tel, telemetry.subspace_stability, color="#7c3aed", lw=1.5, label="Manifold Subspace Stability (Grassmannian Overlap)")
    ax3_twin.plot(backtest_res.long_count.index, backtest_res.long_count, color="#10b981", lw=1, alpha=0.6, label="Active Longs")
    ax3_twin.plot(backtest_res.short_count.index, backtest_res.short_count, color="#ef4444", lw=1, alpha=0.6, label="Active Shorts")

    ax3.set_ylabel("Subspace Stability", color="#7c3aed", fontsize=11)
    ax3_twin.set_ylabel("Active Positions Count", fontsize=11)
    ax3.set_xlabel("Date", fontsize=12)
    ax3.set_ylim(0.0, 1.05)
    ax3.legend(loc="lower left", frameon=True, fontsize=10)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    chart1_path = output_dir / "cmbr_performance_and_drawdown.png"
    plt.savefig(chart1_path, dpi=200)
    plt.close()
    print(f"[CMBR Engine] Saved performance chart: {chart1_path}")

    # 2. Spectral Noise Filtering vs Marchenko-Pastur Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    T = 252
    sample_returns = market_returns.iloc[-T:].values
    Z = (sample_returns - sample_returns.mean(axis=0)) / (sample_returns.std(axis=0) + 1e-8)
    C = (Z.T @ Z) / T
    rmt = RMTEngine()
    res = rmt.decompose(C, T)

    # Plot empirical eigenvalue histogram vs MP bound
    ax.hist(res.eigenvalues[1:], bins=60, density=True, alpha=0.6, color="#3b82f6", label="Empirical Eigenvalue Bulk (λ_2 to λ_N)")
    ax.axvline(res.lambda_max, color="#ef4444", lw=2, ls="--", label=f"Marchenko-Pastur Noise Ceiling (λ_max = {res.lambda_max:.2f})")
    ax.axvline(res.eigenvalues[0], color="#10b981", lw=2.5, ls="-.", label=f"Market Mode v1 (λ_1 = {res.eigenvalues[0]:.1f})")

    for k in range(1, res.num_signal_modes):
        val = res.eigenvalues[k]
        if val > res.lambda_max:
            ax.axvline(val, color="#f59e0b", lw=1.5, alpha=0.8, ls=":", label="CMBR Sector/Rotation Modes" if k==1 else None)

    ax.set_title("Random Matrix Theory: Empirical Spectrum vs Marchenko-Pastur Noise Boundary", fontsize=13, fontweight="bold")
    ax.set_xlabel("Eigenvalue (λ)", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.set_xlim(0, max(res.lambda_max * 2.5, 15))
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)

    chart2_path = output_dir / "marchenko_pastur_spectrum.png"
    plt.savefig(chart2_path, dpi=200)
    plt.close()
    print(f"[CMBR Engine] Saved spectrum chart: {chart2_path}")

def main():
    print("=" * 70)
    print("  Eigenspace Stress Decomposition & CMBR Invariant Trading Engine")
    print("=" * 70)

    config = CMBRConfig()
    loader = DataLoader(config)

    print("\n[Step 1/4] Loading and aligning S&P 500 dataset...")
    prices, returns, metadata = loader.load_aligned_data()
    print(f"  Loaded {returns.shape[1]} stocks across {len(returns)} trading days.")

    print("\n[Step 2/4] Executing Rolling CMBR Out-of-Sample Backtest...")
    engine = RollingBacktestEngine(config)
    backtest_res, summary, sig_metrics, telemetry = engine.run(returns, verbose=True)

    print("\n[Step 3/4] Exporting Results and Visual Diagnostics...")
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Save summary to CSV and text
    summary_dict = summary.__dict__
    summary_df = pd.DataFrame([summary_dict])
    summary_df.to_csv(config.output_dir / "performance_summary.csv", index=False)

    # Save signal metrics
    sig_df = pd.DataFrame([sig_metrics.__dict__])
    sig_df.to_csv(config.output_dir / "signal_metrics.csv", index=False)

    # Save daily net equity curve
    backtest_res.daily_returns_net.to_frame("daily_return_net").to_csv(config.output_dir / "daily_returns_net.csv")

    plot_performance_suite(
        backtest_res=backtest_res,
        summary=summary,
        telemetry=telemetry,
        market_returns=returns,
        meta=metadata,
        output_dir=config.output_dir
    )

    print("\n[Step 4/4] Summary of Key Findings:")
    print("-" * 50)
    print(f"  Period:                 {summary.start_date} to {summary.end_date} ({summary.total_days} days)")
    print(f"  Annualized Net Return:  {summary.annualized_return_net}% (Gross: {summary.annualized_return_gross}%)")
    print(f"  Annualized Volatility:  {summary.annualized_volatility}%")
    print(f"  Sharpe Ratio (Net):     {summary.sharpe_ratio_net}")
    print(f"  Sortino Ratio (Net):    {summary.sortino_ratio_net}")
    print(f"  Max Drawdown:           {summary.max_drawdown_pct}%")
    print(f"  Calmar Ratio:           {summary.calmar_ratio_net}")
    print(f"  Win Rate:               {summary.win_rate_pct}%")
    print(f"  Profit Factor:          {summary.profit_factor}")
    print(f"  Average Daily Turnover: {summary.avg_daily_turnover_pct}%")
    print(f"  Information Coeff (1D): {summary.ic_1d_mean} (t-stat: {summary.ic_1d_tstat})")
    print(f"  Information Coeff (5D): {summary.ic_5d_mean} (t-stat: {summary.ic_5d_tstat})")
    print("-" * 50)
    print(f"  Trade Signals: Long Entries={sig_metrics.total_entries_long}, Short Entries={sig_metrics.total_entries_short}")
    print(f"  Trade Exits:   Normal Reversions={sig_metrics.total_exits_normal}, Stop-Losses={sig_metrics.total_exits_stoploss}")
    print(f"  Average Holding Period: {sig_metrics.avg_holding_period_days} days")
    print("=" * 70)

if __name__ == "__main__":
    main()
