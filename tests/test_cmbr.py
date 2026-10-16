"""
Unit tests for the CMBR Invariant Trading Engine.
Tests Marchenko-Pastur bounds, projection orthogonality, and signal state transitions.
"""
import numpy as np
import pandas as pd
import pytest
from cmbr.config import CMBRConfig
from cmbr.engine.rmt import RMTEngine
from cmbr.engine.projector import ManifoldProjector
from cmbr.signals.dislocation import DislocationSignalGenerator
from cmbr.backtest.portfolio import PortfolioAllocator

def test_marchenko_pastur_bounds():
    rmt = RMTEngine()
    T = 252
    N = 423
    l_min, l_max = rmt.marchenko_pastur_bounds(T, N, sigma2=1.0)

    # For q = 423/252 = 1.6785, sqrt(q) = 1.2956
    # lambda_max = (1 + 1.2956)^2 = 5.269
    assert l_max > 5.0 and l_max < 5.5
    assert l_min >= 0.0

def test_projection_orthogonality():
    # Construct random orthogonal vectors
    N, k = 50, 4
    np.random.seed(42)
    A = np.random.randn(N, k)
    Q, _ = np.linalg.qr(A)

    projector = ManifoldProjector(Q)
    r = np.random.randn(N)
    r_hat, e = projector.project(r)

    # Reconstruction: r_hat + e == r
    np.testing.assert_allclose(r_hat + e, r, atol=1e-10)

    # Orthogonality: e should be orthogonal to all columns in Q (Q.T @ e == 0)
    residuals_on_subspace = Q.T @ e
    np.testing.assert_allclose(residuals_on_subspace, np.zeros(k), atol=1e-10)

def test_dollar_neutral_allocation():
    allocator = PortfolioAllocator(CMBRConfig(max_gross_leverage=1.0))
    # 5 assets: 2 long, 2 short, 1 flat
    positions = pd.DataFrame([[1, 1, -1, -1, 0]], columns=["A", "B", "C", "D", "E"])
    weights = allocator.allocate_weights(positions)

    # Dollar neutrality: sum of weights == 0
    assert abs(weights.iloc[0].sum()) < 1e-10
    # Gross leverage: sum of abs weights == 1.0
    assert abs(weights.iloc[0].abs().sum() - 1.0) < 1e-10

def test_signal_state_transitions():
    sig_gen = DislocationSignalGenerator(CMBRConfig(long_entry_z=-2.0, exit_z=0.0, stop_loss_z=4.0))

    # Trace a single asset through Long entry, hold, normal exit, and stop loss
    z_series = pd.DataFrame({
        "TEST": [0.0, -1.5, -2.2, -1.0, 0.2, 0.0, -2.5, -4.2, -1.0]
    })
    positions, metrics = sig_gen.generate_positions(z_series)
    pos = positions["TEST"].tolist()

    # t=0: flat (0)
    # t=1: z=-1.5 (0)
    # t=2: z=-2.2 (enters Long: +1)
    # t=3: z=-1.0 (holds Long: +1)
    # t=4: z=+0.2 (crosses 0: exits to 0)
    # t=5: z=0.0 (0)
    # t=6: z=-2.5 (enters Long: +1)
    # t=7: z=-4.2 (extreme dislocation: stop loss exits to 0)
    # t=8: z=-1.0 (remains flat: 0)
    assert pos == [0, 0, 1, 1, 0, 0, 1, 0, 0]
    assert metrics.total_entries_long == 2
    assert metrics.total_exits_normal == 1
    assert metrics.total_exits_stoploss == 1

def test_dislocation_dynamics_net_forward_and_projection():
    import torch
    from cmbr.models.dynamics_net import DislocationDynamicsNet
    from cmbr.models.loss import DifferentiableNetSharpeLoss

    n_assets = 423
    hidden_dim = 64
    lookback = 10
    batch_size = 8

    model = DislocationDynamicsNet(n_assets=n_assets, hidden_dim=hidden_dim)
    x = torch.randn(batch_size, lookback, n_assets)
    raw_w = model(x)

    # Check output shape and tanh range [-1, 1]
    assert raw_w.shape == (batch_size, n_assets)
    assert (raw_w >= -1.0).all() and (raw_w <= 1.0).all()

    # Check dollar-neutral projection
    weights = model.project_dollar_neutral(raw_w, max_gross_leverage=1.0)
    assert weights.shape == (batch_size, n_assets)
    
    # Sum of weights across asset dimension should be zero
    sums = weights.sum(dim=-1)
    torch.testing.assert_close(sums, torch.zeros(batch_size), atol=1e-5, rtol=1e-5)

    # Gross leverage should be <= 1.0
    l1 = torch.sum(torch.abs(weights), dim=-1)
    torch.testing.assert_close(l1, torch.ones(batch_size), atol=1e-5, rtol=1e-5)

def test_differentiable_net_sharpe_loss_backprop():
    import torch
    from cmbr.models.dynamics_net import DislocationDynamicsNet
    from cmbr.models.loss import DifferentiableNetSharpeLoss

    n_assets = 20
    model = DislocationDynamicsNet(n_assets=n_assets, hidden_dim=16)
    criterion = DifferentiableNetSharpeLoss(transaction_cost_bps=5.0)

    T = 30
    lookback = 5
    x = torch.randn(T, lookback, n_assets)
    fwd_ret = torch.randn(T, n_assets) * 0.02 + 0.001

    raw_w = model(x)
    weights = model.project_dollar_neutral(raw_w)

    loss, tel = criterion(weights, fwd_ret)
    assert torch.is_tensor(loss)
    assert not torch.isnan(loss)
    assert "net_sharpe" in tel

    # Test backward pass
    loss.backward()
    for param in model.parameters():
        assert param.grad is not None
        assert not torch.isnan(param.grad).any()

