#!/usr/bin/env python3
"""
Per-point loss landscapes: ED vs SD penalty surfaces for two topologies
(two-cluster interpolation and ring mode collapse).

Output: figures/fig03_loss_landscape.png
"""

import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# -- Style -----------------------------------------------------------------
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "savefig.bbox": "tight",
})

SEED = 42
FIGDIR = os.environ.get("SIGDIST_FIGURES", "outputs/component-illustrations")



def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def compute_landscapes(Y_true, grid_points, device="cpu"):
    """Compute per-point ED and SD penalty landscapes.

    For each grid point g, computes:
      - ED penalty: mean distance from g to all points in Y_true
      - SD penalty: W1 between g's cross-signature and the typical
        intra-signature of Y_true points (actual SD per-point computation)

    Returns:
        ed_landscape: (n_grid,) ED penalty for each grid point
        sd_landscape: (n_grid,) SD penalty for each grid point
    """
    Y = torch.from_numpy(Y_true.astype(np.float32)).to(device)
    G = torch.from_numpy(grid_points.astype(np.float32)).to(device)

    # Intra-distances of Y (for reference signatures)
    D_YY = torch.cdist(Y, Y)
    n_y = Y.shape[0]
    # Mask self-distance
    D_YY_masked = D_YY + torch.eye(n_y, device=D_YY.device) * 1e9
    S_YY = torch.sort(D_YY_masked, dim=1).values[:, :n_y - 1]  # (n_y, n_y-1)

    # Mean intra-signature (the "expected" structural profile)
    mean_intra_signature = S_YY.mean(dim=0)  # (n_y-1,)

    # Cross-distances from grid to Y
    D_GY = torch.cdist(G, Y)  # (n_grid, n_y)

    # ED landscape: mean distance from each grid point to Y
    ed_landscape = D_GY.mean(dim=1)  # (n_grid,)

    # SD landscape: W1 between each grid point's cross-signature and
    # the mean intra-signature
    S_GY = torch.sort(D_GY, dim=1).values  # (n_grid, n_y)

    # Align to same length as intra-signature (n_y - 1)
    target_len = n_y - 1
    if S_GY.shape[1] != target_len:
        src_len = S_GY.shape[1]
        indices = torch.linspace(0, src_len - 1, target_len, device=G.device)
        floor_idx = indices.long().clamp(max=src_len - 2)
        frac = indices - floor_idx.float()
        S_GY_aligned = S_GY[:, floor_idx] * (1 - frac) + \
                        S_GY[:, floor_idx + 1] * frac
    else:
        S_GY_aligned = S_GY

    # Per-point W1 against the mean intra-signature
    sd_landscape = torch.abs(S_GY_aligned - mean_intra_signature).mean(dim=1)

    return ed_landscape.cpu().numpy(), sd_landscape.cpu().numpy()


def fig03_loss_landscape():
    """Loss landscape comparison: ED vs SD for two topologies."""
    np.random.seed(SEED)

    # -- Topology A: Two Distinct Clusters --
    n_per = 150
    cluster_A = np.random.randn(n_per, 2) * 0.6 + np.array([-5, 0])
    cluster_B = np.random.randn(n_per, 2) * 0.6 + np.array([5, 0])
    Y_clusters = np.vstack([cluster_A, cluster_B])

    # -- Topology B: Continuous Ring (e.g., cell cycle) --
    theta = np.linspace(0, 2 * np.pi, 300, endpoint=False)
    Y_ring = np.column_stack([
        5 * np.cos(theta) + np.random.randn(300) * 0.3,
        5 * np.sin(theta) + np.random.randn(300) * 0.3
    ])

    # -- Grids --
    res = 100
    xx_c, yy_c = np.meshgrid(np.linspace(-10, 10, res),
                               np.linspace(-6, 6, res))
    grid_c = np.c_[xx_c.ravel(), yy_c.ravel()]

    xx_r, yy_r = np.meshgrid(np.linspace(-8, 8, res),
                               np.linspace(-8, 8, res))
    grid_r = np.c_[xx_r.ravel(), yy_r.ravel()]

    print("Computing cluster landscapes...")
    ed_c, sd_c = compute_landscapes(Y_clusters, grid_c)
    print("Computing ring landscapes...")
    ed_r, sd_r = compute_landscapes(Y_ring, grid_r)

    # -- Visualization --
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=150)
    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    cmap = "magma_r"

    def plot_landscape(ax, xx, yy, Z, Y, panel_label):
        contour = ax.contourf(xx, yy, Z.reshape(xx.shape),
                              levels=30, cmap=cmap, alpha=0.85)
        ax.scatter(Y[:, 0], Y[:, 1], c="white", edgecolor="black",
                   s=12, alpha=0.8, linewidths=0.4)
        ax.text(0.02, 0.98, panel_label, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top")
        ax.set_xticks([]); ax.set_yticks([])
        cbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Generative penalty", rotation=270, labelpad=15,
                       fontsize=9)

    # Row 1: Two Clusters
    plot_landscape(axes[0, 0], xx_c, yy_c, ed_c, Y_clusters, "A")
    plot_landscape(axes[0, 1], xx_c, yy_c, sd_c, Y_clusters, "B")

    # Row 2: Ring Topology
    plot_landscape(axes[1, 0], xx_r, yy_r, ed_r, Y_ring, "C")
    plot_landscape(axes[1, 1], xx_r, yy_r, sd_r, Y_ring, "D")

    fig.tight_layout()

    ensure_dir(FIGDIR)
    fig.savefig(f"{FIGDIR}/fig03_loss_landscape.png")
    print(f"Saved: {FIGDIR}/fig03_loss_landscape.png")
    plt.close(fig)


if __name__ == "__main__":
    fig03_loss_landscape()
    print("Done -- Figure 3 (loss landscape) generated.")
