"""
Differentiable Net Sharpe Ratio Loss with 5 bps transaction friction.
Directly optimizes portfolio weights via backpropagation, penalizing high turnover and low-conviction churn.
"""
import torch
import torch.nn as nn

class DifferentiableNetSharpeLoss(nn.Module):
    def __init__(
        self,
        transaction_cost_bps: float = 5.0,
        annual_factor: float = 252.0,
        turnover_penalty_weight: float = 0.0,
        eps: float = 1e-8
    ):
        """
        Args:
            transaction_cost_bps: Friction per side in basis points (default 5.0 bps = 0.0005)
            annual_factor: Trading days per year (default 252)
            turnover_penalty_weight: Optional auxiliary L1 turnover regularization
        """
        super().__init__()
        self.fee_rate = transaction_cost_bps / 10000.0
        self.annual_sqrt = float(annual_factor ** 0.5)
        self.turnover_penalty_weight = turnover_penalty_weight
        self.eps = eps

    def forward(
        self,
        weights: torch.Tensor,
        forward_returns: torch.Tensor,
        initial_weights: torch.Tensor = None
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            weights: [Batch_T, n_assets] sequence of dollar-neutral portfolio weights
            forward_returns: [Batch_T, n_assets] next-step asset returns
            initial_weights: [n_assets] optional portfolio state at t=0
        Returns:
            loss: Scalar tensor (-Annualized Net Sharpe)
            telemetry: Dict of gross return, net return, turnover, and Sharpe values
        """
        T, N = weights.shape

        # Daily gross returns: dot product along asset dimension
        gross_returns = torch.sum(weights * forward_returns, dim=-1) # [T]

        # Calculate daily turnover: sum(|w_t - w_{t-1}|)
        if initial_weights is None:
            w_prev = torch.cat([weights[:1], weights[:-1]], dim=0)
        else:
            w_prev = torch.cat([initial_weights.unsqueeze(0), weights[:-1]], dim=0)

        daily_turnover = torch.sum(torch.abs(weights - w_prev), dim=-1) # [T]
        daily_costs = daily_turnover * self.fee_rate # [T]

        # Daily net returns after friction
        net_returns = gross_returns - daily_costs # [T]

        # Compute empirical mean and standard deviation over the trajectory
        mean_net = torch.mean(net_returns)
        std_net = torch.std(net_returns) + self.eps

        # Annualized Net Sharpe Ratio
        net_sharpe = self.annual_sqrt * (mean_net / std_net)

        # Loss is negative Net Sharpe (plus optional auxiliary turnover penalty)
        loss = -net_sharpe
        if self.turnover_penalty_weight > 0:
            loss = loss + self.turnover_penalty_weight * torch.mean(daily_turnover)

        mean_gross = torch.mean(gross_returns)
        std_gross = torch.std(gross_returns) + self.eps
        gross_sharpe = self.annual_sqrt * (mean_gross / std_gross)

        telemetry = {
            "loss": float(loss.detach().item()),
            "net_sharpe": float(net_sharpe.detach().item()),
            "gross_sharpe": float(gross_sharpe.detach().item()),
            "ann_net_return": float((mean_net * 252.0).detach().item()),
            "ann_gross_return": float((mean_gross * 252.0).detach().item()),
            "avg_turnover": float(torch.mean(daily_turnover).detach().item())
        }
        return loss, telemetry
