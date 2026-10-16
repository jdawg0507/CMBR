"""
Configuration settings for the CMBR Invariant Trading Engine.
"""
from pathlib import Path
from dataclasses import dataclass

@dataclass
class CMBRConfig:
    # Data Paths
    data_dir: Path = Path(r"C:\Users\jdude\Downloads\S&PArchive")
    price_dir: Path = Path(r"C:\Users\jdude\Downloads\S&PArchive\price_data")
    metadata_file: Path = Path(r"C:\Users\jdude\Downloads\S&PArchive\01_company_info.csv")
    cache_dir: Path = Path(r"c:\Users\jdude\.gemini\CMBR\cache")
    output_dir: Path = Path(r"c:\Users\jdude\.gemini\CMBR\output")

    # Manifold / RMT Estimation Parameters
    rolling_window_T: int = 252           # Lookback window for correlation matrix
    re_estimation_freq: int = 5           # Re-estimate eigenvectors every N trading days
    min_history_pct: float = 0.95         # Filter constituents with >95% valid history
    noise_variance_estimator: str = "peeling" # "peeling" or "analytical"

    # Residual Signal Parameters
    cum_residual_window: int = 15         # Rolling window for calculating residual mean/std (optimal mean-reversion horizon)
    long_entry_z: float = -2.5            # Tightened dislocation z-score threshold for Long (eliminates churn)
    short_entry_z: float = 2.5            # Tightened dislocation z-score threshold for Short (eliminates churn)
    exit_z: float = 0.0                   # Mean-reversion exit threshold
    stop_loss_z: float = 4.0              # Regime-break structural stop-loss threshold

    # Execution & Portfolio Parameters
    max_gross_leverage: float = 2.5       # 2.5x synthetic portfolio gross leverage (+1.25x Long, -1.25x Short)
    dollar_neutral: bool = True           # Enforce Longs = Shorts in notional value
    transaction_cost_bps: float = 5.0     # 5 bps (0.0005) per side
    annual_trading_days: int = 252        # Standard trading days per year
