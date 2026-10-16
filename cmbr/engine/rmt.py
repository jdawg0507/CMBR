"""
Random Matrix Theory (RMT) and Marchenko-Pastur spectral filtering for correlation matrices.
Separates structural signal eigenspace (the CMBR modes) from the bulk noise spectrum.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize_scalar

@dataclass
class RMTResult:
    eigenvalues: np.ndarray          # Sorted descending
    eigenvectors: np.ndarray         # Corresponding eigenvectors as columns
    lambda_max: float                # Marchenko-Pastur upper bound
    lambda_min: float                # Marchenko-Pastur lower bound
    sigma2_noise: float              # Estimated noise variance
    signal_mask: np.ndarray          # Boolean mask for signal eigenvalues
    signal_eigenvalues: np.ndarray   # Eigenvalues > lambda_max
    signal_eigenvectors: np.ndarray  # Eigenvectors for signal modes (N x k)
    noise_eigenvalues: np.ndarray    # Eigenvalues in the noise bulk
    num_signal_modes: int            # k (typically 3 to 7)

class RMTEngine:
    def __init__(self, method: str = "analytical", min_modes: int = 3, max_modes: int = 15):
        """
        Args:
            method: 'analytical' (standard MP bound with sigma2=1.0 - sum(top)/N) or 'peeling'
            min_modes: minimum number of signal modes to retain (at least market mode)
            max_modes: maximum number of signal modes to retain
        """
        self.method = method
        self.min_modes = min_modes
        self.max_modes = max_modes

    def marchenko_pastur_bounds(self, T: int, N: int, sigma2: float = 1.0) -> tuple[float, float]:
        """
        Calculates theoretical Marchenko-Pastur bounds:
        lambda_max = sigma2 * (1 + sqrt(N/T))^2
        lambda_min = sigma2 * (1 - sqrt(N/T))^2
        """
        q = N / T
        lambda_max = sigma2 * (1.0 + np.sqrt(q)) ** 2
        lambda_min = sigma2 * (1.0 - np.sqrt(q)) ** 2
        return lambda_min, lambda_max

    def decompose(self, C: np.ndarray, T: int) -> RMTResult:
        """
        Performs spectral decomposition on empirical correlation matrix C (N x N)
        and isolates the CMBR signal subspace using Marchenko-Pastur theory.

        Args:
            C: Empirical correlation matrix (N x N)
            T: Number of observations in the rolling window

        Returns:
            RMTResult with signal/noise separation
        """
        N = C.shape[0]
        # Hermitian / symmetric eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(C)

        # Sort in descending order
        idx = np.argsort(eigvals)[::-1]
        eigvals = np.real(eigvals[idx])
        eigvecs = np.real(eigvecs[:, idx])

        # Numerical floor
        eigvals = np.maximum(eigvals, 1e-10)

        # Ensure market mode has predominantly positive loadings
        if np.sum(eigvecs[:, 0]) < 0:
            eigvecs[:, 0] = -eigvecs[:, 0]

        # Standard null hypothesis: sigma2 = 1.0 - lambda_1 / N (peeling the dominant market mode)
        # or iterative peeling for residual noise variance
        if self.method == "peeling":
            sigma2 = 1.0
            for _ in range(10):
                l_min, l_max = self.marchenko_pastur_bounds(T, N, sigma2)
                sig_count = int(np.sum(eigvals > l_max))
                sig_count = max(self.min_modes, min(sig_count, self.max_modes))
                new_sigma2 = 1.0 - (np.sum(eigvals[:sig_count]) / N)
                if abs(new_sigma2 - sigma2) < 1e-4:
                    break
                sigma2 = max(new_sigma2, 0.01)
        else:
            # Analytical: base Marchenko-Pastur boundary under null of independent returns
            # Market mode variance is peeled to prevent masking factor/sector modes
            sigma2 = max(1.0 - (eigvals[0] / N), 0.05)
            l_min, l_max = self.marchenko_pastur_bounds(T, N, sigma2)

        l_min, l_max = self.marchenko_pastur_bounds(T, N, sigma2)
        signal_mask = eigvals > l_max

        # Ensure bounded range between min_modes and max_modes
        k = int(np.sum(signal_mask))
        if k < self.min_modes:
            signal_mask[:self.min_modes] = True
        elif k > self.max_modes:
            signal_mask[self.max_modes:] = False

        num_signal = int(np.sum(signal_mask))

        return RMTResult(
            eigenvalues=eigvals,
            eigenvectors=eigvecs,
            lambda_max=l_max,
            lambda_min=l_min,
            sigma2_noise=sigma2,
            signal_mask=signal_mask,
            signal_eigenvalues=eigvals[signal_mask],
            signal_eigenvectors=eigvecs[:, signal_mask],
            noise_eigenvalues=eigvals[~signal_mask],
            num_signal_modes=num_signal
        )

    def denoise_correlation_matrix(self, rmt_res: RMTResult) -> np.ndarray:
        """
        Reconstructs a clean, filtered correlation matrix by replacing the noise bulk
        with a constant trace-preserving floor.
        """
        N = len(rmt_res.eigenvalues)
        k = rmt_res.num_signal_modes

        # Average noise eigenvalue to preserve trace
        lambda_noise_mean = np.mean(rmt_res.noise_eigenvalues) if k < N else 0.0

        clean_eigvals = np.copy(rmt_res.eigenvalues)
        clean_eigvals[k:] = lambda_noise_mean

        # Reconstruct: C_clean = V @ diag(lambda_clean) @ V.T
        V = rmt_res.eigenvectors
        C_clean = V @ np.diag(clean_eigvals) @ V.T

        # Rescale diagonals to 1.0
        d = np.sqrt(np.diag(C_clean))
        C_clean = C_clean / np.outer(d, d)
        np.fill_diagonal(C_clean, 1.0)
        return C_clean
