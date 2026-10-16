"""
Manifold orthogonal projection and residual dislocation calculator.
Projects daily return vectors onto the structural CMBR eigenspace to isolate idiosyncratic deviations.
"""
import numpy as np

class ManifoldProjector:
    def __init__(self, V_signal: np.ndarray):
        """
        Args:
            V_signal: Matrix of orthonormal signal eigenvectors (N x k)
        """
        self.V_signal = V_signal
        # Orthogonal projection operator P = V @ V.T (k x k inner formulation is faster)
        # For return vector r: r_hat = V @ (V.T @ r)
        self.k = V_signal.shape[1] if V_signal.ndim > 1 else 1

    def project(self, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Projects return vector(s) r onto the CMBR manifold.

        Args:
            r: Return array. Either 1D vector of shape (N,) or 2D matrix of shape (T_batch, N).

        Returns:
            (r_hat, e): Projected systematic return, and idiosyncratic residual dislocation
        """
        if r.ndim == 1:
            # r_hat = V @ (V.T @ r)
            coord = self.V_signal.T @ r          # (k,)
            r_hat = self.V_signal @ coord         # (N,)
            e = r - r_hat                         # (N,)
            return r_hat, e
        elif r.ndim == 2:
            # Batch matrix formulation: R_hat = R @ V @ V.T
            coord = r @ self.V_signal             # (T_batch, k)
            r_hat = coord @ self.V_signal.T       # (T_batch, N)
            e = r - r_hat                         # (T_batch, N)
            return r_hat, e
        else:
            raise ValueError(f"Expected 1D or 2D returns array, got shape {r.shape}")

    @staticmethod
    def subspace_stability(V_prev: np.ndarray, V_curr: np.ndarray) -> float:
        """
        Measures the Grassmannian subspace overlap between successive eigenvector bases:
        overlap = Tr(V_prev^T @ V_curr @ V_curr^T @ V_prev) / min(k1, k2)
        Yields 1.0 for identical invariant subspaces and 0.0 for orthogonal subspaces.
        """
        k = min(V_prev.shape[1], V_curr.shape[1])
        if k == 0:
            return 0.0
        M = V_prev.T @ V_curr
        overlap = np.sum(M ** 2) / k
        return float(np.clip(overlap, 0.0, 1.0))
