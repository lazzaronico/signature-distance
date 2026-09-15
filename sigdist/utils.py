"""
Utility functions for data loading, clustering, and visualization.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


# ======================================================================
# Data loading
# ======================================================================

def load_tcga(
    path: Optional[str] = None,
    n_features: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load the TCGA RNA-seq dataset.

    Parameters
    ----------
    path : str, optional
        Path to tcga_rnaseq.csv. If None, looks in the default location
        relative to this package.
    n_features : int, optional
        If set, select the top-N most variable features (genes) to reduce
        dimensionality before experiments.
    random_state : int
        Random seed for reproducibility of feature selection.

    Returns
    -------
    pd.DataFrame
        Samples × features.
    """
    if path is None:
        path = os.environ.get("SIGDIST_TCGA_RNASEQ")
        if not path:
            raise ValueError("Supply path=... or set SIGDIST_TCGA_RNASEQ to the input CSV.")

    df = pd.read_csv(path, index_col=0)

    if n_features is not None and n_features < df.shape[1]:
        # Select top-N most variable genes
        variances = df.var(axis=0)
        top_genes = variances.nlargest(n_features).index
        df = df[top_genes]

    return df


def train_test_split_by_index(
    df: pd.DataFrame, test_frac: float = 0.2, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Split a DataFrame into train/test numpy arrays.

    Parameters
    ----------
    df : pd.DataFrame
    test_frac : float
    random_state : int

    Returns
    -------
    X_train, X_test : ndarray
    """
    rng = np.random.RandomState(random_state)
    n = len(df)
    indices = rng.permutation(n)
    split = int(n * (1 - test_frac))
    train_idx = indices[:split]
    test_idx = indices[split:]
    return df.values[train_idx], df.values[test_idx]


# ======================================================================
# Clustering
# ======================================================================

def leiden_clustering(
    X: np.ndarray,
    n_neighbors: int = 10,
    resolution: float = 1.0,
) -> np.ndarray:
    """Perform Leiden clustering on a point cloud.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    n_neighbors : int
        Number of neighbors for the KNN graph.
    resolution : float
        Resolution parameter for the Leiden algorithm.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        Cluster label for each sample.
    """
    from sklearn.neighbors import kneighbors_graph
    import igraph as ig
    import leidenalg

    knn_graph = kneighbors_graph(
        X, n_neighbors=n_neighbors, mode="connectivity", include_self=False
    )
    sources, targets = knn_graph.nonzero()
    G = ig.Graph(edges=list(zip(sources, targets)), directed=False)
    partition = leidenalg.find_partition(
        G,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
    )
    return np.array(partition.membership)


# ======================================================================
# Visualization helpers
# ======================================================================

def plot_signatures_2d(X, Y, point_indices=None, k=None, ax=None):
    """Plot 2D point clouds and visualize per-point distance signatures.

    Parameters
    ----------
    X, Y : ndarray of shape (n, 2) and (m, 2)
    point_indices : list of int, optional
        Indices in X to highlight. Defaults to first 3.
    k : int, optional
        Number of neighbors to show connections to.
    ax : matplotlib Axes, optional
    """
    import matplotlib.pyplot as plt

    if point_indices is None:
        point_indices = list(range(min(3, len(X))))

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(X[:, 0], X[:, 1], alpha=0.5, label="X", zorder=2)
    ax.scatter(Y[:, 0], Y[:, 1], alpha=0.5, label="Y", zorder=2)

    for idx in point_indices:
        ref = X[idx]
        # Draw lines to Y points
        for y in Y:
            ax.plot(
                [ref[0], y[0]], [ref[1], y[1]],
                "grey", linestyle="--", linewidth=0.5, alpha=0.3, zorder=1,
            )
        ax.scatter([ref[0]], [ref[1]], s=100, zorder=3, edgecolors="red", facecolors="none", linewidths=2)

    ax.legend()
    ax.set_aspect("equal")
    return ax
