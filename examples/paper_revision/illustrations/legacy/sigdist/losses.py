"""
sigdist/losses.py:  Canonical loss functions for Signature Distance generative training.

All losses take tensors X (generated/fake) and Y (real/reference) and return
a scalar loss suitable for backprop through X.

Losses:
    sd       :  full symmetric Signature Distance           (row, symmetric)
    sdiv     :  SDiv: asymmetric recall-only SD             (row, asymmetric)
    cd       :  Column Distance: symmetric column-dual of SD (col, symmetric)
    csd      :  Coupled SD = sqrt(SD^2 + CD^2)              (row+col, symmetric)
    gsd      :  Grounded SD = sqrt(SD^2 + GDiv)             (row+NN, symmetric)
    nn_cross :  NN cross-profile matching (ablation component)
    nsd      :  Normalised SD: scale-invariant variant
    ediv     :  Energy Distance Divergence (Cramer GAN objective)

Helpers:
    _profiles(X) :  sorted within-set distance profiles
    _cross(D, n) :  sorted cross-distance profiles, truncated to width n

All implementations are MPS-safe (no float('inf'), no F.interpolate).
"""

from __future__ import annotations
import torch
import torch.nn.functional as F

__all__ = [
    "sd", "sdiv", "cd", "csd",
    "gsd", "nn_cross", "nsd", "ediv",
]


# ── helpers ────────────────────────────────────────────────────────────────────

def _profiles(X: torch.Tensor) -> torch.Tensor:
    """
    Sorted within-set L2 distance profiles, self-distance included.

    Including the zero self-distance is consistent with the continuous
    formulation where the signature of z counts all mass within radius r
    of z, including the atom at z itself.  This ensures SD(X, X) = 0.

    Args:
        X: (N, d) tensor.

    Returns:
        (N, N) tensor:  each row is the sorted distances from that point
        to every point in X (including itself, which gives a 0).
    """
    D    = torch.cdist(X, X)                         # (N, N)
    return torch.sort(D, dim=1).values               # (N, N)


def _cross(D: torch.Tensor, n: int) -> torch.Tensor:
    """
    Sorted cross-distance profiles, truncated to width n.

    Args:
        D: (M, K) cross-distance matrix (one distribution per row).
        n: target width:  truncate or pad to this size.

    Returns:
        (M, n) tensor of sorted cross-distances.
    """
    D_sorted = torch.sort(D, dim=1).values  # (M, K)
    if D_sorted.shape[1] >= n:
        return D_sorted[:, :n]
    return F.pad(D_sorted, (0, n - D_sorted.shape[1]))


# ── loss functions ─────────────────────────────────────────────────────────────

def sd(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Full symmetric Signature Distance loss.

    Minimising encourages X to match Y in terms of both:
      - Precision: each generated point has real neighbours with matching
        inter-point distances (X side).
      - Recall:    each real point has generated neighbours matching its
        local density (Y side).

    Gradient flows through both X and Y terms.

    Args:
        X: (N, d) generated samples.
        Y: (M, d) reference samples.

    Returns:
        Scalar tensor.
    """
    N, M  = X.shape[0], Y.shape[0]
    D     = torch.cdist(X, Y)                     # (N, M)
    n     = min(N, M) - 1
    SXX   = _profiles(X)                          # (N, N)
    SYY   = _profiles(Y)                          # (M, M)
    SXY   = _cross(D,     n)                      # (N, n)
    SYX   = _cross(D.t(), n)                      # (M, n)
    return 0.5 * (F.l1_loss(SXX[:, :n], SXY) + F.l1_loss(SYY[:, :n], SYX))


def sdiv(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Asymmetric Signature Divergence:  recall direction only.

    Pulls each real point's cross-profile toward its within-real profile.
    Gradient flows through X only (Y's signature is detached).

    This is the recommended default for pure generative training:
    it drives X to cover the support of Y without penalising over-coverage.

    Args:
        X: (N, d) generated samples.
        Y: (M, d) reference samples.

    Returns:
        Scalar tensor.
    """
    M   = Y.shape[0]
    n   = M - 1
    D   = torch.cdist(X, Y)                       # (N, M)
    SYY = _profiles(Y).detach()[:, :n]            # (M, n)
    SYX = _cross(D.t(), n)                        # (M, n)
    return F.l1_loss(SYY, SYX)


def cd(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Symmetric Column Distance:  column-dual of SD.

    CD = 0.5 * (CDiv(X || Y) + CDiv(Y || X))

    SD constrains per-point neighbourhood topology (row-wise);  CD constrains
    the population-level density at each radial scale (column-wise). Together
    they cover orthogonal projections of the same cross-distance matrix.

    Gradient flows through both X and Y.

    Args:
        X: (N, d) generated samples.
        Y: (M, d) reference samples.

    Returns:
        Scalar tensor.
    """
    N, M  = X.shape[0], Y.shape[0]
    D     = torch.cdist(X, Y)                     # (N, M)
    n     = min(N, M) - 1
    SXX   = _profiles(X)[:, :n]                   # (N, n)
    SYY   = _profiles(Y)[:, :n]                   # (M, n)
    SXY   = _cross(D,     n)                      # (N, n)
    SYX   = _cross(D.t(), n)                      # (M, n)
    cXY   = F.l1_loss(torch.sort(SXX, dim=0).values, torch.sort(SXY, dim=0).values)
    cYX   = F.l1_loss(torch.sort(SYY, dim=0).values, torch.sort(SYX, dim=0).values)
    return 0.5 * (cXY + cYX)


def csd(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Coupled Signature Distance (CSD) = sqrt(SD^2 + CD^2).

    The symmetric joint objective combining row and column signals
    from the signature matrix via their Pythagorean (L2 product-space)
    combination.

    Enforces:
      - Row constraint (SD):  each point's local neighbourhood topology matches.
      - Column constraint (CD): the population density at each radial scale matches.

    Gradient flows through both X and Y.

    Args:
        X: (N, d) generated samples.
        Y: (M, d) reference samples.

    Returns:
        Scalar tensor:  sqrt(SD^2 + CD^2).
    """
    sd_sq = sd(X, Y)   # SD^2 (raw W1 average over rows)
    cd_sq = cd(X, Y)   # CD^2 (raw W1 average over columns)
    return torch.sqrt(sd_sq + cd_sq)


def gsd(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Grounded Signature Distance (GSD) = sqrt(GSD^2).

    GSD^2 = SD^2 + 1/2 GDiv(X || Y) + 1/2 GDiv(Y || X)

    Full symmetric SD plus nearest-neighbour-grounded coupling in both
    directions. The NN cross term anchors each point's cross-profile to
    its nearest counterpart's self-profile.

    NN targets are detached to provide stable spatial anchors.
    Gradient flows through SXX, SXY, SYX (via X).

    Args:
        X: (N, d) generated samples.
        Y: (M, d) reference samples.
    Returns:
        Scalar tensor:  sqrt(SD^2 + 1/2 GDiv_fwd + 1/2 GDiv_bwd).
    """
    N, M = X.shape[0], Y.shape[0]
    n    = min(N, M) - 1
    D    = torch.cdist(X, Y)
    SXX  = _profiles(X)[:, :n]
    SYY  = _profiles(Y)[:, :n]
    SXY  = _cross(D, n)
    SYX  = _cross(D.t(), n)
    # Full symmetric SD^2
    sd_sq = 0.5 * (F.l1_loss(SXX, SXY) + F.l1_loss(SYY, SYX))
    # Symmetric NN grounding: 1/2 GDiv(X||Y) + 1/2 GDiv(Y||X)
    nn_fwd = D.argmin(dim=1)                            # (N,) x_i → nearest y
    nn_bwd = D.argmin(dim=0)                            # (M,) y_j → nearest x
    gdiv_fwd = F.l1_loss(SXY, SYY[nn_fwd].detach())    # GDiv(X || Y)
    gdiv_bwd = F.l1_loss(SYX, SXX[nn_bwd].detach())    # GDiv(Y || X)
    return torch.sqrt(sd_sq + 0.5 * (gdiv_fwd + gdiv_bwd))


def nn_cross(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    NN cross-profile matching only (ablation component).

    For each generated x_i, compare its cross-profile SXY[i,:] to the
    within-Y profile of its nearest real neighbour SYY[nn_i,:].
    No SDiv recall, no SD:  isolated NN coupling signal.

    WARNING: Without a base recall/coverage term, this WILL mode-collapse
    on most tasks.  Use only for ablation studies.

    Args:
        X: (N, d) generated samples.
        Y: (M, d) reference samples.
    Returns:
        Scalar tensor.
    """
    M      = Y.shape[0]
    n      = M - 1
    D      = torch.cdist(X, Y)
    SYY    = _profiles(Y).detach()[:, :n]
    SXY    = _cross(D, n)
    nn_idx = D.argmin(dim=1)
    return F.l1_loss(SXY, SYY[nn_idx])


def nsd(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalised Signature Distance:  scale-invariant SD.

    Args:
        X   : (N, d) generated samples.
        Y   : (M, d) reference samples.
        eps : Minimum scale to avoid division by zero.

    Returns:
        Scalar tensor (dimensionless).
    """
    with torch.no_grad():
        scale = torch.cdist(X, Y).median().clamp(min=eps)
    return sd(X / scale, Y / scale)


def ediv(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Energy Distance Divergence:  ED as a differentiable generative training loss.

    Minimising encourages X to cover Y via two complementary terms:
      - Cross-attraction:  pulls each generated point toward all real points.
      - Self-repulsion:    pushes generated points apart, preventing collapse.

    Formally:
        L = 2 * mean(D_XY) - mean_off_diag(D_XX)

    The Y-Y term of the full ED is constant w.r.t. the generator and omitted.
    This is the Cramer GAN / Energy Score objective.

    Args:
        X: (N, d) generated samples.
        Y: (M, d) reference samples.

    Returns:
        Scalar tensor.
    """
    N    = X.shape[0]
    D_XY = torch.cdist(X, Y)
    D_XX = torch.cdist(X, X)
    mask = torch.eye(N, dtype=torch.bool, device=X.device)
    repulsion = D_XX.masked_fill(mask, 0.).sum() / max(N * (N - 1), 1)
    return 2. * D_XY.mean() - repulsion
