"""Original manuscript layouts, parameterized by experiment outputs."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from examples.paper_revision.train_toys import circles

T_RANGE=(0.,2*np.pi)
PALETTE=dict(zip(['MSE','ED','SD','CSD','GSD','MMD','SW1'],['#444444','#4a90d9','#e85d4a','#9b59b6','#e67e22','#2A9D8F','#b58a28']))

def style():
    plt.rcdefaults();sns.set_style('whitegrid')
    plt.rcParams.update({'figure.dpi':150,'font.size':11,'savefig.bbox':'tight','savefig.pad_inches':.1})

def named(m):return r'Sliced $W_1$' if m=='SW1' else m

def rings(work,out,methods,filename):
    style();FIGDIR=str(out);out.mkdir(parents=True,exist_ok=True)
    raw,labels=circles();X_raw=raw;mu=raw.mean(0);scale=raw.std(0)
    CONFIGS=[(named(m),PALETTE[m],None) for m in methods];gen_cache={};metrics={}
    for method in methods:
        z=np.load(work/f'circles_{method.lower()}_seed42/samples.npz');gen=z['generated']*scale+mu;truth=z['truth']*scale+mu
        gen_cache[named(method)]={};metrics[named(method)]={}
        for cl in [0,1]:
            g=gen[2000*cl:2000*(cl+1)];t=truth[2000*cl:2000*(cl+1)]
            xx=cdist(g,g);yy=cdist(t,t);xy=cdist(g,t)
            e=max(0,2*xy.mean()-xx.mean()-yy.mean());raws=np.abs(np.sort(xx,1)-np.sort(xy,1)).mean()+np.abs(np.sort(yy,1)-np.sort(xy.T,1)).mean()
            gen_cache[named(method)][cl]=g;metrics[named(method)][cl]={'ed':np.sqrt(e),'sd':np.sqrt(raws)}
    # -- Figure: 2 rows x N cols (paper subset) --
    ncols = len(CONFIGS)
    fig, axes = plt.subplots(2, ncols, figsize=(ncols * 4.2, 8))
    for row, cl in enumerate([0, 1]):
        for col, (name, color, _) in enumerate(CONFIGS):
            ax  = axes[row, col]
            m   = metrics[name][cl]
            gen_raw = gen_cache[name][cl]
            ax.scatter(X_raw[:, 0], X_raw[:, 1], c="#cccccc", s=3, alpha=0.20,
                       linewidths=0, zorder=1)
            ax.scatter(gen_raw[:, 0], gen_raw[:, 1], c=color, s=10, alpha=0.55,
                       linewidths=0, zorder=2)
            ax.text(0.04, 0.97,
                    f"SD = {m['sd']:.3f}\nED = {m['ed']:.3f}",
                    transform=ax.transAxes, va="top", fontsize=9,
                    fontfamily="monospace",
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
            if row == 0:
                ax.set_title(name, fontsize=13, fontweight="bold", color=color)
            if col == 0:
                ax.set_ylabel(f"Class {cl}", fontsize=11, fontweight="bold")
            ax.set_xlim(-8, 8); ax.set_ylim(-5, 5)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.tight_layout()

    out = str(Path(FIGDIR)/filename)
    fig.savefig(out, bbox_inches="tight");fig.savefig(Path(out).with_suffix(".pdf"),bbox_inches="tight");plt.close(fig)
    print(f"\nSaved: {out}", flush=True)



def plot_panel(ax, truth, pred, key, title):
    ax.scatter(truth[:, 0], truth[:, 1],
               c="#aaaaaa", s=12, alpha=0.50, zorder=1, linewidths=0, rasterized=True)
    ax.scatter(pred[:, 0], pred[:, 1],
               c=PALETTE[key], s=12, alpha=0.55, zorder=2, linewidths=0, rasterized=True)
    ax.set_xlim(T_RANGE[0] - 0.1, T_RANGE[1] + 0.1)
    ax.set_ylim(-1.6, 1.6)
    ax.set_xticks([0, np.pi, 2 * np.pi])
    ax.set_xticklabels(["0", "π", "2π"])
    ax.set_yticks([-1, 0, 1])
    ax.set_title(title, fontsize=11, fontweight="bold", color=PALETTE[key])
    ax.set_xlabel("t")

def sine(work,out,methods,filename):
    global PALETTE
    style();sns.set_style('white');plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False})
    PALETTE.update({k.lower():v for k,v in list(PALETTE.items())})
    FIGDIR=str(out);out.mkdir(parents=True,exist_ok=True)
    configs=[(m.lower(),named(m),None,None) for m in methods]
    results={m.lower():np.load(work/f'sine_{m.lower()}_seed42/samples.npz')['generated'][:2000] for m in methods}
    truth=np.load(work/'sine_mse_seed42/samples.npz')['truth'][:2000]
    # -- Figure: 1 row x N cols --
    fig, axes = plt.subplots(1, len(configs), figsize=(4.5 * len(configs), 4.5), sharey=True)

    for ax, (key, label, _, _) in zip(axes, configs):
        plot_panel(ax, truth, results[key], key, label)

    fig.tight_layout()

    out = str(Path(FIGDIR)/filename)
    fig.savefig(out, bbox_inches="tight");fig.savefig(Path(out).with_suffix(".pdf"),bbox_inches="tight")
    print(f"\nSaved: {out}", flush=True)
    plt.close(fig)


def expansion(work,out):
    style();FIGDIR=str(out);out.mkdir(parents=True,exist_ok=True);SEED=42
    COL_SD='#e85d4a';COL_ED='#4a90d9';COL_CSD='#9b59b6';GUIDANCES=['ed','sd','csd'];boot_res={}
    for m in GUIDANCES:
        boot_res[m]=[]
        for b in range(5):
            z=np.load(work/'expansion'/f'{m}_bootstrap{b}.npz');tr=z['trace'];st=int(z['selected_step'])
            boot_res[m].append({'results':dict(step=np.arange(len(tr)),ed_to_B=tr[:,2],sd_to_B=tr[:,3],ed_to_C=tr[:,4],sd_to_C=tr[:,5]),'best_ed':st,'best_sd':st,'snapshots':{st:z['selected']},'initial':z['initial']})
    z=np.load(work/'geometry/geometry.npz');X_c=z['x'][z['labels']==int(z['largest'])]
    # -- Figure: 2x3 --
    sns.set_style("whitegrid")
    plt.rcParams.update({"figure.dpi": 150, "font.size": 11,
                          "savefig.bbox": "tight"})

    pca = PCA(n_components=2, random_state=SEED)
    pca.fit(X_c)
    X_c_2d = pca.transform(X_c)
    p5, p95 = np.percentile(X_c_2d, [2, 98], axis=0)
    margin  = (p95 - p5) * 0.15
    xlim = (p5[0]-margin[0], p95[0]+margin[0])
    ylim = (p5[1]-margin[1], p95[1]+margin[1])

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    col_cfg = {
        "sd":  dict(key_B="sd_to_B", key_C="sd_to_C",
                    key_stop="best_sd", metric="SD",  color=COL_SD),
        "ed":  dict(key_B="ed_to_B", key_C="ed_to_C",
                    key_stop="best_ed", metric="ED",  color=COL_ED),
        # CSD guidance evaluated with SD metric (canonical evaluator)
        "csd": dict(key_B="sd_to_B", key_C="sd_to_C",
                    key_stop="best_sd", metric="SD",  color=COL_CSD),
    }



    best_runs = {}
    for g, cfg in col_cfg.items():
        runs   = boot_res[g]
        scores = [r["results"][cfg["key_B"]][r[cfg["key_stop"]]] for r in runs]
        b_best = int(np.argmin(scores))
        best_runs[g] = (b_best, runs[b_best])
        print(f"  {g.upper()} best: b={b_best}  "
              f"{cfg['metric']}(B)={scores[b_best]:.4f}  "
              f"stop={runs[b_best][cfg['key_stop']]}", flush=True)

    all_display = np.concatenate([X_c_2d]+[pca.transform(r['snapshots'][r[col_cfg[g]['key_stop']]]) for g,(_,r) in best_runs.items()])
    lo,hi=all_display.min(0),all_display.max(0);margin=(hi-lo)*.05
    xlim=(lo[0]-margin[0],hi[0]+margin[0]);ylim=(lo[1]-margin[1],hi[1]+margin[1])

    # Row 0: traces
    steps = boot_res["sd"][0]["results"]["step"]
    for col, g in enumerate(GUIDANCES):
        cfg = col_cfg[g]; b_best, r_best = best_runs[g]; ax = axes[0, col]
        for b, r in enumerate(boot_res[g]):
            if b == b_best: continue
            ax.plot(steps, r["results"][cfg["key_B"]], "-",
                    color=cfg["color"], lw=0.8, alpha=0.15)
            ax.plot(steps, r["results"][cfg["key_C"]], "--",
                    color=cfg["color"], lw=0.8, alpha=0.10)
        ax.plot(steps, r_best["results"][cfg["key_B"]], "-",
                color=cfg["color"], lw=2.0,
                label=f"{cfg['metric']}(pop, B)")
        ax.plot(steps, r_best["results"][cfg["key_C"]], "--",
                color=cfg["color"], lw=2.0, alpha=0.7,
                label=f"{cfg['metric']}(pop, C)")
        vline_step = r_best[cfg["key_stop"]]
        ax.axvline(vline_step, color="black", ls="--", lw=1.5, alpha=0.75,
                   label=f"stop={vline_step} (b={b_best})")
        ax.set_xlabel("Step")
        label_g = {"sd": "SD", "ed": "ED", "csd": "CSD"}[g]
        ax.set_ylabel(f"{'Signature' if g in ('sd','csd') else 'Energy'} Distance")
        ax.legend(fontsize=8, title=f"{label_g}-guided", title_fontsize=9)

    # Row 1: PCA
    for col, g in enumerate(GUIDANCES):
        cfg = col_cfg[g]; b_best, r_best = best_runs[g]; ax = axes[1, col]
        vline_step = r_best[cfg["key_stop"]]
        pop_b      = r_best["snapshots"][vline_step]
        pop_clean  = pop_b
        n_out      = 0
        score_B    = r_best["results"][cfg["key_B"]][vline_step]
        score_C    = r_best["results"][cfg["key_C"]][vline_step]
        print(f"  {g.upper()} PCA: b={b_best} step={vline_step}  "
              f"B={score_B:.4f}  C={score_C:.4f}  "
              f"pts={len(pop_clean)}  out={n_out}", flush=True)
        label_g = {"sd": "SD", "ed": "ED", "csd": "CSD"}[g]
        unique_cents = np.unique(pca.transform(r_best['initial']),axis=0)
        pop_2d = pca.transform(pop_clean)
        ax.scatter(X_c_2d[:, 0], X_c_2d[:, 1],
                   c="grey", s=4, alpha=0.15, linewidths=0, zorder=1,
                   label="Original")
        ax.scatter(pop_2d[:, 0], pop_2d[:, 1],
                   c=cfg["color"], s=10, alpha=0.45, linewidths=0, zorder=2,
                   label=f"Expanded ({len(pop_clean)} pts)")
        ax.scatter(unique_cents[:, 0], unique_cents[:, 1],
                   c="black", s=20, alpha=0.8,
                   linewidths=0.5, edgecolors="white", zorder=5,
                   label="Initial centroids")
        ax.text(0.03, 0.03,
                f"{cfg['metric']}(B)={score_B:.3f}\n{cfg['metric']}(C)={score_C:.3f}",
                transform=ax.transAxes, va="bottom", fontsize=8, fontfamily="monospace",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.legend(fontsize=7, title=f"{label_g}-guided", title_fontsize=9)

    fig.tight_layout()
    out_path = f"{FIGDIR}/fig07c_langevin_csd.png"
    fig.savefig(out_path);fig.savefig(Path(out_path).with_suffix(".pdf"));plt.close(fig)
    print(f"\nSaved: {out_path}", flush=True)
