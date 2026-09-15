"""
Core Signature Distance computation.

This module provides the SignatureDistance class, a PyTorch-accelerated
implementation of Signature Distance (SD) and its local variant, Neighborhood
Signature Distance (NSD).
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Optional, Literal


class SignatureDistance:
    """Compute Signature Distance (SD) and Energy Distance (ED) between point clouds.

    This isolated implementation preserves the estimator used for the
    normalization/component illustration.

    Parameters
    ----------
    k : int, optional
        Number of nearest neighbors for signature construction. Only used when
        mode="signature". Default: all points (k=0 means use full distance row).
    mode : {"default", "signature", "simplified"}
        - "default":     Standard energy distance (scalar pairwise distances).
        - "signature":   Full signature comparison via sorted distance vectors.
        - "simplified":  Simplified signature (mean difference). Requires equal-
                             sized sets.
    p : float, optional
        The p-norm to use for pairwise distances (default: 2, Euclidean).
    device : str, optional
        PyTorch device, e.g. "cpu", "cuda", "mps" (default: "cpu").
    batch_size : int or None, optional
        If set, compute pairwise distances in batches to limit GPU memory. None
        means compute the full matrix at once.

    Examples
    --------
    >>> import numpy as np
    >>> from sigdist import SignatureDistance
    >>> sd = SignatureDistance(mode="signature")
    >>> sd.fit(np.random.randn(200, 10))
    >>> score = sd.score(np.random.randn(200, 10))
    >>> print(f"SD = {score:.4f}")
    """

    def __init__(
        self,
        k: int = 0,
        mode: Literal["default", "signature", "simplified"] = "default",
        p: float = 2.0,
        device: str = "cpu",
        batch_size: Optional[int] = None,
        normalize: bool = False,
    ):
        self.k = k
        self.mode = mode
        self.p = p
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.normalize = normalize

        # Internal state set by fit()
        self._X: Optional[torch.Tensor] = None
        self._XX: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "SignatureDistance":
        """Fit the reference distribution.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Reference point cloud.

        Returns
        -------
        self
        """
        self._X = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(self.device)
        self._XX = self._pairwise_distances(self._X, self._X)
        return self

    def score(self, Y: np.ndarray) -> float:
        """Compute Signature Distance between fitted X and new set Y.

        Parameters
        ----------
        Y : ndarray of shape (m_samples, n_features)

        Returns
        -------
        float
            The Signature Distance SD(X, Y). If mode="default" this is
            equivalent to energy distance.
        """
        self._check_is_fitted()
        Y_t = torch.from_numpy(np.asarray(Y, dtype=np.float32)).to(self.device)

        if self.mode == "default":
            XX, XY, YY = self._compute_distance_matrices(Y_t)
            return self._energy_distance(XX, XY, YY).item()
        else:
            # Signature distance needs the raw distance matrices
            XY = self._pairwise_distances(self._X, Y_t)
            YY = self._pairwise_distances(Y_t, Y_t)
            return self._signature_distance(self._XX, XY, YY).item()

    def energy_distance(self, Y: np.ndarray) -> float:
        """Compute Energy Distance between fitted X and new set Y.

        Always uses scalar pairwise distances, regardless of mode.

        Parameters
        ----------
        Y : ndarray of shape (m_samples, n_features)

        Returns
        -------
        float
        """
        self._check_is_fitted()
        Y_t = torch.from_numpy(np.asarray(Y, dtype=np.float32)).to(self.device)
        XY = self._pairwise_distances(self._X, Y_t)
        YY = self._pairwise_distances(Y_t, Y_t)
        return self._energy_distance(self._XX, XY, YY).item()

    def pvalue(
        self,
        Y: np.ndarray,
        n_permutations: int = 200,
        metric: Literal["sd", "ed"] = "sd",
    ) -> tuple[np.ndarray, float, float]:
        """Permutation test for distributional difference.

        Parameters
        ----------
        Y : ndarray of shape (m_samples, n_features)
        n_permutations : int
            Number of random permutations for the null distribution.
        metric : {"sd", "ed"}
            Which distance to use for the test.

        Returns
        -------
        null_distribution : ndarray of shape (n_permutations,)
        observed : float
            The observed distance.
        p_value : float
            Fraction of null distances >= observed distance.
        """
        self._check_is_fitted()
        Y_t = torch.from_numpy(np.asarray(Y, dtype=np.float32)).to(self.device)

        dist_fn = self._signature_distance if metric == "sd" else self._energy_distance

        # Pool X and Y
        n = self._X.shape[0]
        m = Y_t.shape[0]
        Z = torch.cat([self._X, Y_t], dim=0)

        # Compute full pairwise distance matrix on Z (raw distances)
        dist_matrix = self._pairwise_distances(Z, Z)

        # Observed distance
        XX_obs = dist_matrix[:n, :n]
        XY_obs = dist_matrix[:n, n:]
        YY_obs = dist_matrix[n:, n:]
        observed = dist_fn(XX_obs, XY_obs, YY_obs)

        # Null distribution via permutation
        null_dist = torch.empty(n_permutations, device=self.device)
        for i in range(n_permutations):
            perm = torch.randperm(n + m, device=self.device)
            p1, p2 = perm[:n], perm[n:]
            XX_p = dist_matrix[p1][:, p1]
            XY_p = dist_matrix[p1][:, p2]
            YY_p = dist_matrix[p2][:, p2]
            null_dist[i] = dist_fn(XX_p, XY_p, YY_p)

        p_val = (torch.sum(null_dist >= observed) + 1).float() / (n_permutations + 1)

        return null_dist.cpu().numpy(), observed.item(), p_val.item()

    def bootstrap_ci(
        self,
        Y: np.ndarray,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        metric: Literal["sd", "ed"] = "sd",
    ) -> tuple[np.ndarray, float, float, float]:
        """Bootstrap confidence interval for the observed distance.

        Complements the permutation test (which evaluates the conditional
        randomisation null for this specific sample) with a measure of
        population-level stability.  Resampling with replacement simulates
        drawing new cohorts from the broader population, so a wide CI
        honestly reflects high sampling uncertainty.

        Parameters
        ----------
        Y : ndarray of shape (m_samples, n_features)
        n_bootstrap : int
            Number of bootstrap resamples.
        confidence : float
            Confidence level for the interval (default: 0.95 → 95% CI).
        metric : {"sd", "ed"}
            Which distance to use.

        Returns
        -------
        boot_distribution : ndarray of shape (n_bootstrap,)
            Bootstrap distance values.
        observed : float
            The observed distance SD(X, Y).
        ci_lower : float
            Lower bound of the confidence interval.
        ci_upper : float
            Upper bound of the confidence interval.
        """
        self._check_is_fitted()
        Y_t = torch.from_numpy(np.asarray(Y, dtype=np.float32)).to(self.device)

        dist_fn = self._signature_distance if metric == "sd" else self._energy_distance

        # Pool all points for efficient index-based resampling
        n = self._X.shape[0]
        m = Y_t.shape[0]
        Z = torch.cat([self._X, Y_t], dim=0)

        # Precompute full pairwise distance matrix once
        dist_matrix = self._pairwise_distances(Z, Z)

        # Observed distance
        XX_obs = dist_matrix[:n, :n]
        XY_obs = dist_matrix[:n, n:]
        YY_obs = dist_matrix[n:, n:]
        observed = dist_fn(XX_obs, XY_obs, YY_obs)

        # Bootstrap: resample each group WITH replacement
        alpha = 1.0 - confidence
        boot_dist = torch.empty(n_bootstrap, device=self.device)
        idx_x = torch.arange(n, device=self.device)
        idx_y = torch.arange(n, n + m, device=self.device)

        for i in range(n_bootstrap):
            bx = idx_x[torch.randint(n, (n,), device=self.device)]
            by = idx_y[torch.randint(m, (m,), device=self.device)]
            XX_b = dist_matrix[bx][:, bx]
            XY_b = dist_matrix[bx][:, by]
            YY_b = dist_matrix[by][:, by]
            boot_dist[i] = dist_fn(XX_b, XY_b, YY_b)

        boot_np = boot_dist.cpu().numpy()
        ci_lo = float(np.percentile(boot_np, 100 * alpha / 2))
        ci_hi = float(np.percentile(boot_np, 100 * (1 - alpha / 2)))

        return boot_np, observed.item(), ci_lo, ci_hi


    def nsd_scores(self, Y: np.ndarray, k: Optional[int] = None) -> np.ndarray:
        """Compute per-point Neighborhood Signature Distance scores.

        For each point in Y, computes how anomalous its local signature is
        compared to the reference distribution X. Higher scores indicate
        greater deviation from the reference geometry.

        Parameters
        ----------
        Y : ndarray of shape (m_samples, n_features)
        k : int, optional
            Number of neighbors for NSD. Defaults to self.k or 20.

        Returns
        -------
        scores : ndarray of shape (m_samples,)
            Per-point NSD anomaly scores.
        """
        self._check_is_fitted()
        k = k or self.k or 20

        Y_t = torch.from_numpy(np.asarray(Y, dtype=np.float32)).to(self.device)

        # For each point in Y, find k nearest neighbors in X
        XY_dists = self._pairwise_distances(self._X, Y_t)  # (n, m)

        scores = torch.empty(Y_t.shape[0], device=self.device)
        for j in range(Y_t.shape[0]):
            # Get distances from all X points to this Y point
            dists_to_y = XY_dists[:, j]  # (n,)

            # k nearest neighbors in X
            knn_idx = torch.topk(dists_to_y, k, largest=False).indices

            # Signature of Y[j]: distances to its k neighbors
            sig_y = dists_to_y[knn_idx].sort().values  # (k,)

            # Reference signatures: for each neighbor, its distances to
            # the other k-1 neighbors in this set
            neighbor_dists = self._XX[knn_idx][:, knn_idx]  # (k, k)
            ref_sigs = neighbor_dists.sort(dim=1).values  # (k, k)

            # Average Wasserstein-1 distance between sig_y and each ref signature
            # Align lengths if needed (sig_y has k elements, ref_sigs has k elements)
            scores[j] = torch.mean(torch.abs(sig_y.unsqueeze(0) - ref_sigs))

        return scores.cpu().numpy()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_is_fitted(self):
        if self._X is None:
            raise RuntimeError("Must call fit() before using the model.")

    def _pairwise_distances(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distance matrix, optionally in batches."""
        if self.batch_size is None:
            return torch.cdist(X, Y, p=self.p)

        n = X.shape[0]
        dist_matrix = torch.empty(n, Y.shape[0], device=self.device)
        for i in range(0, n, self.batch_size):
            end = min(i + self.batch_size, n)
            dist_matrix[i:end] = torch.cdist(X[i:end], Y, p=self.p)
        return dist_matrix

    @staticmethod
    def _energy_distance(
        XX: torch.Tensor, XY: torch.Tensor, YY: torch.Tensor
    ) -> torch.Tensor:
        """Energy distance from pairwise distance matrices."""
        val = 2 * torch.mean(XY) - torch.mean(XX) - torch.mean(YY)
        return torch.sqrt(torch.clamp(val, min=0.0))

    def _signature_distance(
        self, XX: torch.Tensor, XY: torch.Tensor, YY: torch.Tensor
    ) -> torch.Tensor:
        """Signature Distance: symmetrized comparison of sorted distance distributions.

        This illustration uses the square root of the averaged radial
        discrepancies. No general triangle-inequality guarantee is assumed.

        When self.normalize is True, uses an alternative per-point
        normalization: sqrt(W1 / mean_distance) per point, then averages.
        """
        n, m = XX.shape[0], YY.shape[0]

        # Sort each row (= distance distribution for each point)
        sXX = torch.sort(XX, dim=1).values
        sYY = torch.sort(YY, dim=1).values
        sXY = torch.sort(XY, dim=1).values
        sYX = torch.sort(XY.T, dim=1).values

        # Align via quantile interpolation when set sizes differ
        aXX = self._align_distributions(sXX, m)
        aYY = self._align_distributions(sYY, n)

        # Per-point W1 (mean absolute difference of sorted vectors)
        w1_fwd = torch.abs(sXY - aXX).mean(dim=1)   # (n,)
        w1_bwd = torch.abs(sYX - aYY).mean(dim=1)   # (m,)

        if self.normalize:
            # Per-point normalized variant: sqrt(W1 / mean_distance)
            mean_dist_x = 0.5 * (XX.mean(dim=1) + XY.mean(dim=1))
            mean_dist_y = 0.5 * (YY.mean(dim=1) + XY.T.mean(dim=1))
            w1_fwd = torch.sqrt(w1_fwd / mean_dist_x.clamp(min=1e-9))
            w1_bwd = torch.sqrt(w1_bwd / mean_dist_y.clamp(min=1e-9))
            return 0.5 * (w1_fwd.mean() + w1_bwd.mean())

        # Default: sqrt of symmetrized mean radial W1
        raw = 0.5 * (w1_fwd.mean() + w1_bwd.mean())
        return torch.sqrt(raw)

    def _align_distributions(
        self, sorted_dists: torch.Tensor, target_len: int
    ) -> torch.Tensor:
        """Align sorted distance vectors to a target length via quantile mapping."""
        if sorted_dists.shape[1] == target_len:
            return sorted_dists
        q = torch.linspace(0, 1, target_len, device=self.device)
        return torch.quantile(sorted_dists, q, dim=1, interpolation="midpoint").T

    def _signature_dist_matrix(
        self, S1: torch.Tensor, S2: torch.Tensor
    ) -> torch.Tensor:
        """Pairwise signature distance matrix between two sets of distance rows.

        Used by the permutation test to transform the full pooled distance matrix
        into signature space before computing null distributions.
        """
        if self.mode == "simplified":
            # Mean difference (cheapest)
            return torch.abs(
                torch.mean(S1, dim=1, keepdim=True)
                - torch.mean(S2, dim=1, keepdim=True).T
            )

        # Full sorted Wasserstein-1 comparison
        S1_sorted = torch.sort(S1, dim=1).values
        S2_sorted = torch.sort(S2, dim=1).values

        k1, k2 = S1.shape[1], S2.shape[1]
        if k1 == k2:
            s1_exp = S1_sorted.unsqueeze(1)
            s2_exp = S2_sorted.unsqueeze(0)
        elif k1 > k2:
            s2_aligned = self._align_distributions(S2_sorted, k1)
            s1_exp = S1_sorted.unsqueeze(1)
            s2_exp = s2_aligned.unsqueeze(0)
        else:
            s1_aligned = self._align_distributions(S1_sorted, k2)
            s1_exp = s1_aligned.unsqueeze(1)
            s2_exp = S2_sorted.unsqueeze(0)

        return torch.mean(torch.abs(s1_exp - s2_exp), dim=2)

    def _compute_distance_matrices(
        self, Y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute XX, XY, YY:  optionally transforming to signature space."""
        XY = self._pairwise_distances(self._X, Y)
        YY = self._pairwise_distances(Y, Y)

        if self.mode in ("signature", "simplified"):
            if self.mode == "simplified" and self._XX.shape[0] != YY.shape[0]:
                raise ValueError(
                    f"Simplified mode requires equal-sized sets. "
                    f"Got {self._XX.shape[0]} and {YY.shape[0]}."
                )
            return (
                self._signature_dist_matrix(self._XX, self._XX),
                self._signature_dist_matrix(self._XX, YY),
                self._signature_dist_matrix(YY, YY),
            )

        return self._XX, XY, YY
