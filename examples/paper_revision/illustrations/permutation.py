#!/usr/bin/env python3
"""
Null distribution comparison: ED / SD / CD / CSD across three scenarios
(same data, location shift, density change).

Layout: 5 rows (scatter, ED null, SD null, CD null, CSD null) x 3 cols.

Output: figures/fig01b_null_comparison.png
"""

import sys, os


import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 150, "font.size": 10,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.1})

SEED   = 42
FIGDIR = "src/figures"
DEVICE = "cpu"   # permutation loop:  cpu is fine


METRIC_COLORS = {
    "ED":  "#4a90d9",
    "SD":  "#e85d4a",
    "CD":  "#e74c3c",
    "CSD": "#9b59b6",
}

def _null_panel(ax, null_vals, obs, p, metric, col):
    color = METRIC_COLORS[metric]
    sns.histplot(null_vals, ax=ax, color=color, alpha=0.35,
                 stat="density", kde=True, linewidth=0, label="Null")
    ax.axvline(obs, color="#888888", linewidth=2.0, linestyle="--",
               label=f"Obs (p={p:.3f})")
    ax.legend(fontsize=7, framealpha=0.7)
    ax.set_xlabel(f"{metric} score", fontsize=8)
    ax.tick_params(labelsize=8)
    if col == 0:
        ax.set_ylabel(f"{metric}: Density", fontsize=9, fontweight="bold", color=color)
    # Expand xlim if observed is near/beyond edge
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, max(xmax, obs + (obs - xmin) * 0.15))


# ── Main figure ────────────────────────────────────────────────────────────────

def main():

    scenario_keys=['null','shift_original','scale_original']
    data=[np.load(W/f'n50_{k}_000.npz') for k in scenario_keys]
    scenarios=[(label,d['x'],d['y']) for label,d in zip(['Same distribution','Location shift','Density change'],data)]
    metrics = ["ED", "SD", "CSD"]

    nrows = 1 + len(metrics)   # scatter + 3 metrics
    ncols = len(scenarios)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5*ncols, 4.0 + 2.5*len(metrics)),
                             gridspec_kw={"height_ratios": [2.5] + [1]*len(metrics)})

    col_titles = [s[0] for s in scenarios]

    for col, (sc_name, X, Y) in enumerate(scenarios):
        print(f"\n── Scenario: {sc_name} ──", flush=True)

        # Row 0: scatter
        ax = axes[0, col]
        ax.scatter(X[:, 0], X[:, 1], alpha=0.45, s=12, c="#D4A017", label="X")
        ax.scatter(Y[:, 0], Y[:, 1], alpha=0.45, s=12, c="#2A9D8F", label="Y")
        ax.set_aspect("equal")
        lim = max(np.abs(np.vstack([X, Y])).max() * 1.25, 3.5)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_title(sc_name, fontsize=11, fontweight="bold")
        ax.tick_params(labelsize=8)
        if col == 0:
            ax.legend(fontsize=8); ax.set_ylabel("Scatter", fontsize=9)

        # Metric rows
        for row, mname in enumerate(metrics, start=1):
            q={'ED':0,'SD':1,'CSD':2}[mname]
            null_v=data[col]['null'][:,q];obs_v=data[col]['observed'][q];p_val=data[col]['p'][q]
            if q in [0,1]:null_v=np.sqrt(np.maximum(null_v,0));obs_v=np.sqrt(max(obs_v,0))
            print(f"    obs={obs_v:.4f}  p={p_val:.3f}", flush=True)
            _null_panel(axes[row, col], null_v, obs_v, p_val, mname, col)

    # Ensure all density panels start at ylim=0
    for row in range(1, nrows):
        for col in range(ncols):
            axes[row, col].set_ylim(bottom=0)

    fig.tight_layout(h_pad=0.3)
    os.makedirs(FIGDIR, exist_ok=True)
    out = f"{FIGDIR}/fig01b_null_comparison.png"
    fig.savefig(out); fig.savefig(Path(out).with_suffix(".pdf")); plt.close(fig)
    print(f"\nSaved: {out}", flush=True)



if __name__ == '__main__':
    import argparse
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('--statistics',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    W=a.statistics;FIGDIR=str(a.out);a.out.mkdir(parents=True,exist_ok=True);main()
