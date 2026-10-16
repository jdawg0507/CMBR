"""
Performance analytics, drawdown analysis, and Information Coefficient (IC) calculation.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

@dataclass
class PerformanceSummary:
    total_days: int
    start_date: str
    end_date: str
    annualized_return_gross: float
    annualized_return_net: float
    annualized_volatility: float
    sharpe_ratio_gross: float
    sharpe_ratio_net: float
    sortino_ratio_net: float
    max_drawdown_pct: float
    calmar_ratio_net: float
    win_rate_pct: float
    profit_factor: float
    avg_daily_turnover_pct: float
    total_transaction_costs_pct: float
    ic_1d_mean: float
    ic_1d_tstat: float
    ic_5d_mean: float
    ic_5d_tstat: float

class PerformanceEvaluator:
    def __init__(self, trading_days_per_year: int = 252):
        self.N_ANNUAL = trading_days_per_year

    def evaluate(
        self,
        daily_returns_gross: pd.Series,
        daily_returns_net: pd.Series,
        turnover: pd.Series,
        costs: pd.Series,
        zscores: pd.DataFrame = None,
        residuals: pd.DataFrame = None
    ) -> PerformanceSummary:
        T = len(daily_returns_net)
        if T == 0:
            raise ValueError("Empty returns series provided")

        # Annualized Returns
        total_ret_gross = (1.0 + daily_returns_gross).prod() - 1.0
        total_ret_net = (1.0 + daily_returns_net).prod() - 1.0
        ann_ret_gross = (1.0 + total_ret_gross) ** (self.N_ANNUAL / T) - 1.0
        ann_ret_net = (1.0 + total_ret_net) ** (self.N_ANNUAL / T) - 1.0

        # Volatility
        daily_vol = daily_returns_net.std()
        ann_vol = daily_vol * np.sqrt(self.N_ANNUAL)

        # Sharpe Ratios (assuming zero risk-free rate for long/short market neutral)
        sharpe_gross = (daily_returns_gross.mean() / (daily_returns_gross.std() + 1e-8)) * np.sqrt(self.N_ANNUAL)
        sharpe_net = (daily_returns_net.mean() / (daily_vol + 1e-8)) * np.sqrt(self.N_ANNUAL)

        # Sortino Ratio (downside deviation)
        downside = daily_returns_net[daily_returns_net < 0]
        downside_std = downside.std() * np.sqrt(self.N_ANNUAL) if len(downside) > 0 else 1e-8
        sortino_net = ann_ret_net / (downside_std + 1e-8)

        # Drawdown
        cum_ret = (1.0 + daily_returns_net).cumprod()
        running_max = cum_ret.cummax()
        drawdowns = (cum_ret - running_max) / running_max
        max_dd = float(drawdowns.min())
        max_dd_pct = abs(max_dd) * 100.0

        # Calmar Ratio
        calmar = ann_ret_net / (abs(max_dd) + 1e-8)

        # Win Rate & Profit Factor
        pos_days = daily_returns_net[daily_returns_net > 0]
        neg_days = daily_returns_net[daily_returns_net < 0]
        win_rate = (len(pos_days) / T) * 100.0
        profit_factor = float(pos_days.sum() / (abs(neg_days.sum()) + 1e-8))

        # Turnover and Costs
        avg_turnover = float(turnover.mean()) * 100.0
        total_costs_pct = float(costs.sum()) * 100.0

        # Information Coefficient (IC) of residual mean-reversion
        # Predictor: -zscore (since negative z implies positive forward return)
        ic_1d_mean, ic_1d_tstat = 0.0, 0.0
        ic_5d_mean, ic_5d_tstat = 0.0, 0.0

        if zscores is not None and residuals is not None:
            aligned_z, aligned_res = zscores.align(residuals, join="inner")
            # 1-day forward residual return
            fwd_res_1d = aligned_res.shift(-1)
            # 5-day forward cumulative residual return
            fwd_res_5d = aligned_res.rolling(5).sum().shift(-5)

            # Signal is -z (reversion expected)
            sig = -aligned_z

            ic_1d_list = []
            ic_5d_list = []

            for d in range(len(aligned_z) - 5):
                s_row = sig.iloc[d].dropna()
                f1_row = fwd_res_1d.iloc[d].dropna()
                f5_row = fwd_res_5d.iloc[d].dropna()

                common1 = s_row.index.intersection(f1_row.index)
                if len(common1) > 20:
                    r1, _ = spearmanr(s_row.loc[common1], f1_row.loc[common1])
                    if not np.isnan(r1):
                        ic_1d_list.append(r1)

                common5 = s_row.index.intersection(f5_row.index)
                if len(common5) > 20:
                    r5, _ = spearmanr(s_row.loc[common5], f5_row.loc[common5])
                    if not np.isnan(r5):
                        ic_5d_list.append(r5)

            if ic_1d_list:
                ic_1d_mean = float(np.mean(ic_1d_list))
                ic_1d_tstat = float(ic_1d_mean / (np.std(ic_1d_list) / np.sqrt(len(ic_1d_list)) + 1e-8))
            if ic_5d_list:
                ic_5d_mean = float(np.mean(ic_5d_list))
                ic_5d_tstat = float(ic_5d_mean / (np.std(ic_5d_list) / np.sqrt(len(ic_5d_list)) + 1e-8))

        return PerformanceSummary(
            total_days=T,
            start_date=str(daily_returns_net.index[0].date()),
            end_date=str(daily_returns_net.index[-1].date()),
            annualized_return_gross=round(ann_ret_gross * 100.0, 2),
            annualized_return_net=round(ann_ret_net * 100.0, 2),
            annualized_volatility=round(ann_vol * 100.0, 2),
            sharpe_ratio_gross=round(sharpe_gross, 2),
            sharpe_ratio_net=round(sharpe_net, 2),
            sortino_ratio_net=round(sortino_net, 2),
            max_drawdown_pct=round(max_dd_pct, 2),
            calmar_ratio_net=round(calmar, 2),
            win_rate_pct=round(win_rate, 2),
            profit_factor=round(profit_factor, 2),
            avg_daily_turnover_pct=round(avg_turnover, 2),
            total_transaction_costs_pct=round(total_costs_pct, 2),
            ic_1d_mean=round(ic_1d_mean, 4),
            ic_1d_tstat=round(ic_1d_tstat, 2),
            ic_5d_mean=round(ic_5d_mean, 4),
            ic_5d_tstat=round(ic_5d_tstat, 2)
        )
