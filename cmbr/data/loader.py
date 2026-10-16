"""
Data loader and alignment pipeline for S&P 500 historical prices and metadata.
Caches aligned return matrices to Parquet for sub-second subsequent loads.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from cmbr.config import CMBRConfig

class DataLoader:
    def __init__(self, config: CMBRConfig = None):
        self.config = config or CMBRConfig()
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        self.prices_cache = self.config.cache_dir / "prices.parquet"
        self.returns_cache = self.config.cache_dir / "log_returns.parquet"
        self.metadata_cache = self.config.cache_dir / "metadata.parquet"

    def _load_single_ticker(self, filepath: Path) -> pd.Series:
        """Reads a single ticker CSV and returns a Series of Close prices indexed by Date."""
        try:
            df = pd.read_csv(filepath, usecols=["Date", "Close"])
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.dropna(subset=["Date", "Close"]).drop_duplicates(subset=["Date"])
            df = df.sort_values("Date")
            symbol = filepath.stem
            return df.set_index("Date")["Close"].rename(symbol)
        except Exception:
            return None

    def load_metadata(self) -> pd.DataFrame:
        """Loads and cleans company metadata."""
        if self.metadata_cache.exists():
            return pd.read_parquet(self.metadata_cache)

        if not self.config.metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.config.metadata_file}")

        meta = pd.read_csv(self.config.metadata_file, low_memory=False)
        cols_to_keep = [
            col for col in ["symbol", "shortName", "sector", "industry", "marketCap", "beta"]
            if col in meta.columns
        ]
        meta = meta[cols_to_keep].dropna(subset=["symbol"]).drop_duplicates(subset=["symbol"])
        meta = meta.set_index("symbol")
        meta.to_parquet(self.metadata_cache)
        return meta

    def load_aligned_data(self, force_reload: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads aligned adjusted close prices, calculates log returns, and applies
        universe filtering based on minimum history percentage.

        Returns:
            (prices_df, log_returns_df, metadata_df)
        """
        if not force_reload and self.prices_cache.exists() and self.returns_cache.exists() and self.metadata_cache.exists():
            prices = pd.read_parquet(self.prices_cache)
            log_returns = pd.read_parquet(self.returns_cache)
            metadata = pd.read_parquet(self.metadata_cache)
            return prices, log_returns, metadata

        metadata = self.load_metadata()

        csv_files = list(self.config.price_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV price files found in {self.config.price_dir}")

        series_list = []
        with ThreadPoolExecutor(max_workers=16) as executor:
            for s in executor.map(self._load_single_ticker, csv_files):
                if s is not None and len(s) > 0:
                    series_list.append(s)

        prices = pd.concat(series_list, axis=1, join="outer").sort_index()

        # Universe filtering: keep tickers with >= min_history_pct valid data
        total_trading_days = len(prices)
        valid_ratio = prices.notna().sum(axis=0) / total_trading_days
        eligible_tickers = valid_ratio[valid_ratio >= self.config.min_history_pct].index.tolist()

        prices = prices[eligible_tickers]
        # Forward fill up to 5 consecutive days for minor trading halts, then backward fill remaining
        prices = prices.ffill(limit=5).bfill()

        # Log returns r_{i,t} = ln(P_{i,t} / P_{i,t-1})
        log_returns = np.log(prices / prices.shift(1)).iloc[1:]

        # Drop any remaining NaN rows or columns
        log_returns = log_returns.dropna(axis=0, how="all")
        valid_cols = log_returns.columns[log_returns.notna().all(axis=0)]
        log_returns = log_returns[valid_cols]
        prices = prices.loc[log_returns.index, valid_cols]

        # Cache datasets
        prices.to_parquet(self.prices_cache)
        log_returns.to_parquet(self.returns_cache)
        metadata = metadata.reindex(valid_cols)
        metadata.to_parquet(self.metadata_cache)

        return prices, log_returns, metadata
