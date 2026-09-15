"""Exact W1 integrals between equally weighted empirical radial laws."""
import torch


def row_wasserstein(a, b):
    """Compare corresponding rows, allowing different numbers of atoms.

    Each row represents a uniform empirical law. Integrating over the union
    of both quantile grids preserves every atom's mass; interpolation does not.
    """
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[0]:
        raise ValueError("Expected two matrices with the same number of rows.")
    n, m = a.shape[1], b.shape[1]
    if not n or not m:
        raise ValueError("Radial distributions must be nonempty.")
    a = a.sort(dim=1).values
    b = b.sort(dim=1).values
    if n == m:
        return (a - b).abs().mean(dim=1)
    grid = torch.unique(torch.cat([
        torch.arange(n + 1, device=a.device, dtype=a.dtype) / n,
        torch.arange(m + 1, device=a.device, dtype=a.dtype) / m,
    ]), sorted=True)
    widths = grid[1:] - grid[:-1]
    mid = (grid[1:] + grid[:-1]) / 2
    ia = (mid * n).long().clamp(max=n - 1)
    ib = (mid * m).long().clamp(max=m - 1)
    return ((a[:, ia] - b[:, ib]).abs() * widths).sum(dim=1)
