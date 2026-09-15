"""Differentiable radial-distribution objectives, using the summed convention.

sd returns the raw statistic S; SignatureDistance.score returns sqrt(S).
Inputs are generated X and reference Y. Every empirical atom is retained.
"""
from __future__ import annotations
import torch
from ._quantiles import row_wasserstein

__all__ = ["sd", "sdiv", "cd", "csd", "gsd", "nn_cross", "nsd", "ediv", "energy"]


def _distance(x, y):
    return torch.cdist(x, y, compute_mode="donot_use_mm_for_euclid_dist")


def _profiles(x):
    return _distance(x, x).sort(dim=1).values


def _cross(d, n=None):
    """Sorted cross profile; n is accepted only when no truncation is needed."""
    if n is not None and n != d.shape[1]:
        raise ValueError("Full empirical profiles cannot be rank-truncated; use exact quantile integration.")
    return d.sort(dim=1).values


def sd(x, y):
    """Summed raw signature statistic S=SDiv(X||Y)+SDiv(Y||X)."""
    xy = _distance(x, y)
    return row_wasserstein(_distance(x,x),xy).mean() + row_wasserstein(_distance(y,y),xy.T).mean()


def sdiv(x, y):
    """Raw real-anchored recall direction SDiv(Y||X); Y is a fixed target."""
    return row_wasserstein(_distance(y,y).detach(),_distance(y,x)).mean()


def _column_cost(a, b):
    # Each radial quantile u defines a distribution across anchor rows.
    a=a.sort(dim=1).values;b=b.sort(dim=1).values
    n,m=a.shape[1],b.shape[1]
    if n==m:
        return row_wasserstein(a.T,b.T).mean()
    grid=torch.unique(torch.cat([torch.arange(n+1,device=a.device,dtype=a.dtype)/n,
                                  torch.arange(m+1,device=a.device,dtype=a.dtype)/m]),sorted=True)
    mid=(grid[:-1]+grid[1:])/2
    ia=(mid*n).long().clamp(max=n-1);ib=(mid*m).long().clamp(max=m-1)
    return (row_wasserstein(a[:,ia].T,b[:,ib].T)*torch.diff(grid)).sum()


def cd(x, y):
    """Summed column discrepancy C, integrated over exact radial quantiles."""
    xy=_distance(x,y)
    return _column_cost(_distance(x,x),xy)+_column_cost(_distance(y,y),xy.T)


def csd(x, y):
    """sqrt(S+C); an objective, with no asserted general metric guarantee."""
    xy=_distance(x,y);xx=_distance(x,x);yy=_distance(y,y)
    raw=row_wasserstein(xx,xy).mean()+row_wasserstein(yy,xy.T).mean()
    col=_column_cost(xx,xy)+_column_cost(yy,xy.T)
    return (raw+col).clamp_min(1e-12).sqrt()


def gsd(x, y):
    """sqrt(S+GDiv(X||Y)+GDiv(Y||X)); nearest-profile targets are detached."""
    xy=_distance(x,y);xx=_distance(x,x);yy=_distance(y,y)
    raw=row_wasserstein(xx,xy).mean()+row_wasserstein(yy,xy.T).mean()
    ix=xy.argmin(dim=1);iy=xy.argmin(dim=0)
    g=row_wasserstein(xy,yy[ix].detach()).mean()+row_wasserstein(xy.T,xx[iy].detach()).mean()
    return (raw+g).clamp_min(1e-12).sqrt()


def nn_cross(x, y):
    """Generated-anchored nearest-profile component, for component analysis."""
    xy=_distance(x,y);yy=_distance(y,y).detach()
    return row_wasserstein(xy,yy[xy.argmin(dim=1)]).mean()


def nsd(x, y, eps=1e-8):
    """Scale-normalized raw signature objective; normalization is detached."""
    with torch.no_grad():scale=_distance(x,y).median().clamp_min(eps)
    return sd(x/scale,y/scale)


def energy(x, y):
    """Full unrooted energy expression E=2C-A-B, including self-atoms."""
    return 2*_distance(x,y).mean()-_distance(x,x).mean()-_distance(y,y).mean()


def ediv(x, y):
    """Raw energy training objective 2C-A; target-target constant is omitted."""
    return 2*_distance(x,y).mean()-_distance(x,x).mean()
