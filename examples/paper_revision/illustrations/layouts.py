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


def expansion(work,out,show=0):
    """Traces of all five bootstraps and the displayed bootstrap at its validation-selected step."""
    style();out.mkdir(parents=True,exist_ok=True)
    sns.set_style("whitegrid");plt.rcParams.update({"figure.dpi":150,"font.size":11,"savefig.bbox":"tight"})
    arms=[("none","No guidance","#4d4d4d"),("ed","ED-guided","#4a90d9"),("sd","SD-guided","#e85d4a"),("csd","CSD-guided","#9b59b6")]
    z=np.load(work/"geometry/geometry.npz");cluster=z["x"][z["labels"]==int(z["largest"])]
    pca=PCA(n_components=2,random_state=42).fit(cluster);cluster_2d=pca.transform(cluster)
    fig,axes=plt.subplots(3,4,figsize=(22,15))
    for col,(key,title,color) in enumerate(arms):
        for b in range(5):
            run=np.load(work/"expansion"/f"{key}_bootstrap{b}.npz");trace=run["trace"];steps=np.arange(len(trace));main=b==show
            # trace columns: (d_E, SD, CSD) to the bootstrap seed, to validation B and to held-out C
            for row,(to_b,to_c,label) in enumerate([(3,6,"ED"),(4,7,"SD")]):
                ax=axes[row,col]
                ax.plot(steps,trace[:,to_b],"-",color=color,lw=2. if main else .8,alpha=1 if main else .15,label=f"{label}(pop, B)" if main else None)
                ax.plot(steps,trace[:,to_c],"--",color=color,lw=2. if main else .8,alpha=.7 if main else .10,label=f"{label}(pop, C)" if main else None)
            if main:shown=run;stop=int(run["selected_step"]);shown_trace=trace
        b_to_c=[float(np.sqrt(max(0.,v))) for v in _raw_reference(shown["b"],shown["c"])]
        for row,(label,limits) in enumerate([("Energy Distance",(.5,2.5)),("Signature Distance",(1.,5.))]):
            ax=axes[row,col];ax.axhline(b_to_c[row],color="black",ls=":",lw=1.2,alpha=.75,label="real–real (B, C)")
            ax.axvline(stop,color="black",ls="--",lw=1.5,alpha=.75,label=f"stop={stop} (b={show})")
            ax.set_xlabel("Step");ax.set_ylabel(label);ax.set_ylim(limits);ax.legend(fontsize=8,title=title,title_fontsize=9,loc="upper left")
        ax=axes[2,col];population=shown["selected"]
        ax.scatter(*cluster_2d.T,c="grey",s=4,alpha=.15,linewidths=0,zorder=1,label="Original")
        ax.scatter(*pca.transform(population).T,c=color,s=10,alpha=.45,linewidths=0,zorder=2,label=f"Expanded, step {stop} ({len(population)} pts)")
        ax.scatter(*np.unique(pca.transform(shown["initial"]),axis=0).T,c="black",s=20,alpha=.8,linewidths=.5,edgecolors="white",zorder=5,label="Initial centroids")
        ax.text(.03,.03,f"ED(C)={shown_trace[stop,6]:.3f}\nSD(C)={shown_trace[stop,7]:.3f}",transform=ax.transAxes,va="bottom",fontsize=8,fontfamily="monospace",bbox=dict(facecolor="white",alpha=.85,edgecolor="none"))
        ax.set_xlabel("PC1");ax.set_ylabel("PC2");ax.set_xlim(-45,45);ax.set_ylim(-45,45);ax.legend(fontsize=7,title=title,title_fontsize=9,loc="upper left")
    fig.tight_layout();path=out/"fig07c_langevin_csd.png"
    fig.savefig(path);fig.savefig(path.with_suffix(".pdf"));plt.close(fig);print(f"Saved: {path}",flush=True)


def _raw_reference(b,c):
    """Raw energy and signature statistics between the validation and held-out sets."""
    import torch
    from sigdist import losses
    x,y=torch.from_numpy(np.asarray(b)),torch.from_numpy(np.asarray(c))
    return float(losses.energy(x,y)),float(losses.sd(x,y))
