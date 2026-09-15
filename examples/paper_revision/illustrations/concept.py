from pathlib import Path
#!/usr/bin/env python3
"""
Signature concept illustration with directional self-bootstrap references.

Output:
  figures/fig01_signature_concept.png
  figures/fig02_sd_vs_ed.png
  figures/fig_perpoint.png
  figures/fig04_density_outliers.png
"""

import sys


import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist
import seaborn as sns



# -- Style -----------------------------------------------------------------
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

SEED = 42
FIGDIR = "src/figures"

# Consistent palette across fig01 and fig01b
COLOR_X = "#D4A017"   # gold
COLOR_Y = "#2A9D8F"   # teal
COLOR_DENSITY = "#16a085"  # teal for SDiv/SD null panels (matching fig08b)
COLOR_VLINE   = "#888888"  # grey observed-value line



def ensure_dir(path):
    import os
    os.makedirs(path, exist_ok=True)


# ======================================================================
# Figure 1: Signature Concept
# ======================================================================

def _plot_null_panel(ax, null_vals, observed, p_val, color, title_ann, xlabel):
    """Shared helper: histogram of a null distribution + observed vline."""
    sns.histplot(null_vals, ax=ax, color=color, alpha=0.35,
                 stat="density", kde=True, linewidth=0)
    ax.axvline(observed, color="#222222", linewidth=2.0, linestyle="--", zorder=5)
    # ensure enough right margin so the vline and annotation are never clipped
    xmin, xmax = ax.get_xlim()
    margin = (observed - xmin) * 0.25
    ax.set_xlim(xmin, max(xmax, observed + margin))
    ax.text(0.04, 0.95,
            f"obs = {observed:.3f}\np = {p_val:.3f}",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=8, family="monospace",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))
    ax.text(0.97, 0.95, title_ann,
            transform=ax.transAxes, va="top", ha="right",
            fontsize=8, style="italic",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
    ax.set_xlabel(xlabel, fontsize=8)
    ax.tick_params(labelsize=8)


def _compute_sdiv(A, B):
    """
    Compute SDiv(A || B) for equal empirical sample sizes.
    Takes mean(abs(sort(dAA) - sort(dAB))) where rows are sorted.
    """
    dAA = np.sort(cdist(A, A), axis=1)
    dAB = np.sort(cdist(A, B), axis=1)
    return np.mean(np.abs(dAA - dAB))


def _bootstrap_self_sdiv(Z, n_boot, rng):
    """
    Bootstrap null for self-distance using the directed SDiv statistic.
    Resamples Z twice with replacement to get Z1 and Z2, then computes SDiv(Z1, Z2).
    """
    n = len(Z)
    null_vals = np.empty(n_boot)
    for i in range(n_boot):
        Z1 = Z[rng.choice(n, n, replace=True)]
        Z2 = Z[rng.choice(n, n, replace=True)]
        null_vals[i] = _compute_sdiv(Z1, Z2)
    return null_vals


def fig01_signature_concept():
    """
    2-row x 3-column figure matching the exact aesthetics and dataset of the
    original cytobench PED notebook.
    """
    rng = np.random.RandomState(SEED)

    # Original notebook generation (n=30)
    n_points = 30
    X = np.array([rng.normal(5, 2, n_points), rng.normal(5, 0.5, n_points)]).T
    Y = np.array([rng.normal(5, 0.2, n_points), rng.normal(5, 0.2, n_points)]).T

    # The notebook iterated over points. X[0] is typically an edge point,
    # which makes the connection lines sweep across the space clearly,
    # instead of the centroid which sits inside Y.
    ref = X[0]

    # Distance vectors from reference point
    d_X = np.linalg.norm(X - ref, axis=1)
    d_Y = np.linalg.norm(Y - ref, axis=1)
    wd  = wasserstein_distance(np.sort(d_X), np.sort(d_Y))

    # ---------- null distributions ------------------------------------
    pooled=np.vstack([X,Y]); rng_perm=np.random.default_rng(742)
    def score(a,b):
        f=_compute_sdiv(a,b); r=_compute_sdiv(b,a)
        return f,r,np.sqrt(f+r)
    observed=np.asarray(score(X,Y));obs_xy,obs_yx,obs_sd=observed
    null=np.asarray([score(*(lambda q:(pooled[q[:30]],pooled[q[30:]]))(rng_perm.permutation(60))) for _ in range(999)])
    null_combined=null[:,2];p_combined=float((1+(null_combined>=obs_sd).sum())/1000)
    rng_boot=np.random.default_rng(743)
    boot=np.asarray([score(X[rng_boot.integers(30,size=30)],Y[rng_boot.integers(30,size=30)]) for _ in range(500)])
    ci_lower,ci_upper=np.percentile(boot,[2.5,97.5],axis=0)
    ci_lo,ci_hi=ci_lower[2],ci_upper[2]
    # Original directional construction: two independent empirical bootstrap draws
    # from each reference, using the original RNG state after drawing X and Y.
    null_X=_bootstrap_self_sdiv(X,500,rng)
    null_Y=_bootstrap_self_sdiv(Y,500,rng)
    p_X=(1+(null_X>=obs_xy).sum())/501
    p_Y=(1+(null_Y>=obs_yx).sum())/501
    np.savez_compressed(Path(FIGDIR)/'fig1_self_reference.npz',X=X,Y=Y,
      observed=observed,null_X=null_X,null_Y=null_Y,null_combined=null_combined,
      tail_fraction=[p_X,p_Y],p_sd=p_combined,bootstrap=boot,ci=[ci_lower,ci_upper])

    # ---------- layout ------------------------------------------------
    fig = plt.figure(figsize=(18, 12))
    gs  = fig.add_gridspec(2, 3, hspace=0.15, wspace=0.15)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)]

    def _scatter_base(ax, draw_lines_to):
        """Draw both clouds + grey dashed connection lines from ref."""
        for pt in draw_lines_to:
            ax.plot([ref[0], pt[0]], [ref[1], pt[1]],
                    '#bbbbbb', linestyle='--', linewidth=0.8, alpha=0.5, zorder=2)

        ax.scatter(X[:, 0], X[:, 1], alpha=0.6, c=COLOR_X, label='X', zorder=3)
        ax.scatter(Y[:, 0], Y[:, 1], alpha=0.6, c=COLOR_Y, label='Y', zorder=3)

        ax.tick_params(labelsize=10)

    # --- Row 0, Col A: ref -> X (intra) ---
    ax = axes[0][0]
    _scatter_base(ax, draw_lines_to=X)
    ax.legend(fontsize=12, loc="upper right")

    # --- Row 0, Col B: ref -> Y (cross) ---
    ax = axes[0][1]
    _scatter_base(ax, draw_lines_to=Y)
    ax.legend(fontsize=12, loc="upper right")

    # Shared limits for scatter plots based on all data
    all_pts = np.concatenate([X, Y])
    xlim = [all_pts[:, 0].min() - 0.5, all_pts[:, 0].max() + 0.5]
    ylim = [all_pts[:, 1].min() - 0.5, all_pts[:, 1].max() + 0.5]
    for ax in [axes[0][0], axes[0][1]]:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    # --- Row 0, Col C: overlaid histograms ---
    ax = axes[0][2]
    bins = np.histogram_bin_edges(np.concatenate([d_X, d_Y]), bins=n_points)

    ax.hist(d_X, bins=bins, alpha=0.6, color=COLOR_X, label=r"$d(x_0, X)$")
    ax.hist(d_Y, bins=bins, alpha=0.6, color=COLOR_Y, label=r"$d(x_0, Y)$")
    ax.legend(fontsize=12, loc="upper right")
    ax.tick_params(labelsize=10)

    def _plot_null(ax, null_vals, label, p_val, obs_val, obs_label, color):
        q=0 if color==COLOR_X else 1
        lo,hi=ci_lower[q],ci_upper[q]
        ax.axvspan(lo,hi,alpha=0.15,color=COLOR_VLINE,zorder=1,label="95% CI")
        sns.histplot(null_vals, ax=ax, stat='density', kde=True,
                     label=label, color=color, alpha=0.45)
        ax.axvline(x=obs_val, color=COLOR_VLINE, linewidth=2.0,
                   linestyle='--', zorder=5)

        # Ensure enough right margin
        xmin, xmax = ax.get_xlim()
        right_edge=max(obs_val,hi)
        if right_edge > xmax - (xmax - xmin) * 0.05:
            ax.set_xlim(xmin, right_edge + (right_edge - xmin) * 0.1)

        # Annotation box instead of title
        ax.text(0.04, 0.95,
                f"obs = {obs_val:.3f}\ntail = {p_val:.3f}\n95% CI [{lo:.3f}, {hi:.3f}]",
                transform=ax.transAxes, va="top", ha="left",
                fontsize=8, family="monospace",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))
        ax.legend(fontsize=10, loc="upper right")
        ax.tick_params(labelsize=10)
        ax.set_xlabel('Value', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)

    # --- Row 1, Col A: X null (gold:  formed from X) ---
    _plot_null(axes[1][0], null_X, "SDiv(X,X') self", p_X, obs_xy, "SDiv(X,Y)", color=COLOR_X)

    # --- Row 1, Col B: Y null (teal:  formed from Y) ---
    _plot_null(axes[1][1], null_Y, "SDiv(Y,Y') self", p_Y, obs_yx, "SDiv(Y,X)", color=COLOR_Y)

    # --- Row 1, Col C: Combined SD null + bootstrap CI (red:  SD color) ---
    COLOR_SD = "#e85d4a"
    ax = axes[1][2]
    sns.histplot(null_combined, ax=ax, stat='density', kde=True,
                 label="SD(X,Y) null", color=COLOR_SD, alpha=0.45)
    # Bootstrap CI shaded region
    ax.axvspan(ci_lo, ci_hi, alpha=0.15, color=COLOR_VLINE, zorder=1,
               label=f"95% CI")
    # Observed vline
    ax.axvline(x=obs_sd, color=COLOR_VLINE, linewidth=2.0,
               linestyle='--', zorder=5)

    # Ensure enough right margin
    xmin, xmax = ax.get_xlim()
    right_edge = max(obs_sd, ci_hi)
    if right_edge > xmax - (xmax - xmin) * 0.05:
        ax.set_xlim(xmin, right_edge + (right_edge - xmin) * 0.1)

    ax.text(0.04, 0.95,
            f"obs = {obs_sd:.3f}\np = {p_combined:.3f}\n"
            f"95% CI [{ci_lo:.3f}, {ci_hi:.3f}]",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=8, family="monospace",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))
    ax.legend(fontsize=10, loc="upper right")
    ax.tick_params(labelsize=10)
    ax.set_xlabel('Value', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)

    ensure_dir(FIGDIR)
    fig.savefig(f"{FIGDIR}/fig01_signature_concept.png", bbox_inches='tight')
    fig.savefig(f"{FIGDIR}/fig01_signature_concept.pdf", bbox_inches='tight')
    print(f"Saved: {FIGDIR}/fig01_signature_concept.png")
    plt.close(fig)



if __name__ == "__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    FIGDIR=str(a.out);a.out.mkdir(parents=True,exist_ok=True);fig01_signature_concept()
