#!/usr/bin/env python3
"""
Normalisation and component ablation on the ring generation task.

Shows BatchNorm failure with SD loss, and extended losses (SDiv, NSD, NN-Cross)
not included in the main generative loss comparison.

Output: figures/fig08b_norm_comparison.png
"""

import sys, os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sigdist import SignatureDistance
from sigdist.losses import (
    sd as sd_loss, sdiv as sdiv_loss, nsd as nsd_loss,
    nn_cross as nn_cross_loss,
)
from sigdist.generators import FICGenerator, FICGeneratorBN

sns.set_style("white")
plt.rcParams.update({"figure.dpi": 150, "font.size": 11, "savefig.bbox": "tight",
                     "axes.facecolor": "white", "figure.facecolor": "white"})

SEED       = 7
FIGDIR     = os.environ.get("SIGDIST_FIGURES", "outputs/component-illustrations")
CACHEDIR   = os.environ.get("SIGDIST_CACHE", "outputs/component-cache")
DEVICE     = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 256
NUM_EPOCHS = 10_000
LR         = 1e-3
HIDDEN     = 128
LATENT_DIM = 2
N_LAYERS   = 5

os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(CACHEDIR, exist_ok=True)
torch.manual_seed(SEED); np.random.seed(SEED)
print(f"Device: {DEVICE}", flush=True)


# ── Data ──────────────────────────────────────────────────────────────────────

def make_circles_data(n_per_class=2000, seed=SEED):
    rng    = np.random.RandomState(seed)
    theta0 = rng.uniform(0, 2 * np.pi, n_per_class)
    theta1 = rng.uniform(0, 2 * np.pi, n_per_class)
    x0 = 2.0 * np.cos(theta0) - 1.0 + rng.normal(0, 0.08, n_per_class)
    y0 = 2.0 * np.sin(theta0)        + rng.normal(0, 0.08, n_per_class)
    x1 = 3.5 * np.cos(theta1) + 2.5 + rng.normal(0, 0.08, n_per_class)
    y1 = 3.5 * np.sin(theta1)       + rng.normal(0, 0.08, n_per_class)
    X  = np.vstack([np.column_stack([x0, y0]),
                    np.column_stack([x1, y1])]).astype(np.float32)
    labels = np.array([0]*n_per_class + [1]*n_per_class, dtype=np.int32)
    return X, labels


# ── Training ──────────────────────────────────────────────────────────────────

def sample_real(X_norm, labels, cl, bs):
    pool = X_norm[labels == cl]
    idx  = np.random.choice(len(pool), bs, replace=True)
    return torch.tensor(pool[idx]).to(DEVICE)

def sample_gen(G, n, cl):
    z   = torch.randn(n, LATENT_DIM, device=DEVICE)
    c   = torch.full((n, 1), float(cl), device=DEVICE)
    with torch.no_grad():
        return G(z, c).cpu().numpy()

def train(G, X_norm, labels, loss_fn, label):
    G.to(DEVICE)
    opt = torch.optim.Adam(G.parameters(), lr=LR)
    for ep in range(1, NUM_EPOCHS + 1):
        G.train()
        cl  = ep % 2
        Y_r = sample_real(X_norm, labels, cl, BATCH_SIZE)
        z   = torch.randn(BATCH_SIZE, LATENT_DIM, device=DEVICE)
        c   = torch.full((BATCH_SIZE, 1), float(cl), device=DEVICE)
        X_g = G(z, c)
        loss = loss_fn(X_g, Y_r)
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 2500 == 0 or ep == 1:
            print(f"  [{label}] ep {ep:5d}", flush=True)
    G.eval()
    return G


# ── Config ────────────────────────────────────────────────────────────────────
# Each entry: (panel_key, title, color, GenClass, loss_fn)

CONFIGS = [
    ("sd_nonorm", "SD",              "#e74c3c", FICGenerator,   sd_loss),
    ("sd_bn",     "SD (BatchNorm)",   "#b03a2e", FICGeneratorBN, sd_loss),
    ("sdiv",      "SDiv",             "#16a085", FICGenerator,   sdiv_loss),
    ("nsd",       "NSD",              "#1a5276", FICGenerator,   nsd_loss),
    ("nn_cross",  "NN-Cross",         "#34495e", FICGenerator,   nn_cross_loss),
]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    X_raw, labels = make_circles_data(2000)
    mu  = X_raw.mean(0); sig = X_raw.std(0) + 1e-8
    X_norm = (X_raw - mu) / sig

    sd_calc = SignatureDistance(mode="signature", device=DEVICE)
    ed_calc = SignatureDistance(mode="default",   device=DEVICE)

    gen_cache, metrics = {}, {}
    for key, title, color, GenClass, loss_fn in CONFIGS:
        run_key = f"{key}_{BATCH_SIZE}_{NUM_EPOCHS}_{SEED}"
        gen_cache[key] = {}
        metrics[key] = {}

        cached = all(
            os.path.exists(f"{CACHEDIR}/{run_key}_cl{cl}.npy")
            for cl in [0, 1])
        if cached:
            print(f"  [{key}] loading from cache", flush=True)
            for cl in [0, 1]:
                gen_cache[key][cl] = np.load(f"{CACHEDIR}/{run_key}_cl{cl}.npy")
        else:
            print(f"\n{'='*50}\n  {title}\n{'='*50}", flush=True)
            torch.manual_seed(SEED)
            G = GenClass(noise_dim=LATENT_DIM, cond_dim=1,
                         hidden=HIDDEN, n_layers=N_LAYERS)
            train(G, X_norm, labels, loss_fn, title)
            for cl in [0, 1]:
                gen_norm = sample_gen(G, 2000, cl)
                gen_raw  = gen_norm * sig + mu
                gen_cache[key][cl] = gen_raw
                np.save(f"{CACHEDIR}/{run_key}_cl{cl}.npy", gen_raw)

        for cl in [0, 1]:
            gen_raw  = gen_cache[key][cl]
            real_raw = X_raw[labels == cl]
            sd_calc.fit(real_raw); sd_v = sd_calc.score(gen_raw)
            ed_calc.fit(real_raw); ed_v = ed_calc.score(gen_raw)
            metrics[key][cl] = {"sd": sd_v, "ed": ed_v}
            print(f"  {key} cl={cl}: SD={sd_v:.4f}  ED={ed_v:.4f}", flush=True)

    # -- Figure: 2 rows (class 0 / class 1) x N cols --
    ncols = len(CONFIGS)
    fig, axes = plt.subplots(2, ncols, figsize=(ncols * 4.2, 8))

    for row, cl in enumerate([0, 1]):
        for col, (key, title, color, _, _) in enumerate(CONFIGS):
            ax  = axes[row, col]
            m   = metrics[key][cl]
            gen_raw = gen_cache[key][cl]
            ax.scatter(X_raw[:, 0], X_raw[:, 1], c="#cccccc", s=3, alpha=0.20,
                       linewidths=0, zorder=1)
            ax.scatter(gen_raw[:, 0], gen_raw[:, 1], c=color, s=10, alpha=0.55,
                       linewidths=0, zorder=2)
            ax.text(0.04, 0.97, f"SD={m['sd']:.3f}\nED={m['ed']:.3f}",
                    transform=ax.transAxes, va="top", fontsize=9,
                    fontfamily="monospace",
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
            if row == 0:
                ax.set_title(title, fontsize=13, fontweight="bold", color=color)
            if col == 0:
                ax.set_ylabel(f"Class {cl}", fontsize=11, fontweight="bold")
            ax.set_xlim(-8, 8); ax.set_ylim(-5, 5)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = f"{FIGDIR}/fig08b_norm_comparison.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"\nSaved: {out}", flush=True)


if __name__ == "__main__":
    main()
    print("Done.", flush=True)
