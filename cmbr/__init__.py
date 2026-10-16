"""
CMBR: Eigenspace Stress Decomposition & CMBR Invariant Trading Engine.
"""
from cmbr.config import CMBRConfig
from cmbr.data.loader import DataLoader
from cmbr.engine.rmt import RMTEngine, RMTResult
from cmbr.engine.projector import ManifoldProjector
from cmbr.signals.dislocation import DislocationSignalGenerator, SignalMetrics
from cmbr.backtest.portfolio import PortfolioAllocator, BacktestResult
from cmbr.backtest.metrics import PerformanceEvaluator, PerformanceSummary
from cmbr.backtest.engine import RollingBacktestEngine, ManifoldTelemetry

__all__ = [
    "CMBRConfig",
    "DataLoader",
    "RMTEngine",
    "RMTResult",
    "ManifoldProjector",
    "DislocationSignalGenerator",
    "SignalMetrics",
    "PortfolioAllocator",
    "BacktestResult",
    "PerformanceEvaluator",
    "PerformanceSummary",
    "RollingBacktestEngine",
    "ManifoldTelemetry"
]
