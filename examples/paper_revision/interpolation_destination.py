"""Main Figure 4 and Supporting Figure S5: interpolation to the destination sample."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from examples.paper_revision.statistical_tests import score,NAMES
from examples.paper_revision.common import save_json,digest


def compute(rnaseq,audit,out):
    out.mkdir(parents=True,exist_ok=True)
    genes=pd.read_csv(audit/'rnaseq_top256_genes.csv').gene.tolist()
    first=pd.read_csv(rnaseq,nrows=0).columns[0]
    frame=pd.read_csv(rnaseq,usecols=[first]+genes,index_col=0)[genes]
    x=frame.to_numpy();rng=np.random.RandomState(42);ix=rng.permutation(len(x))
    A=x[ix[:400]];B=x[ix[400:800]]
    row,col=linear_sum_assignment(cdist(A,B));am=A[row];bm=B[col]
    pool=np.r_[A,B];D=cdist(pool,pool);bw=float(np.median(D[D>0]))
    directions=np.random.default_rng(711).normal(size=(len(genes),128));directions/=np.linalg.norm(directions,axis=0)
    alpha=np.linspace(0,1,11);values=[]
    for t in alpha:
        z=np.r_[(1-t)*am+t*bm,B];dist=cdist(z,z);kernel=np.exp(-dist**2/(2*bw**2))
        values.append(score(dist,kernel,z@directions,np.arange(400),np.arange(400,800)))
    values=np.asarray(values);rooted=values.copy();rooted[:,:2]=np.sqrt(np.maximum(rooted[:,:2],0))
    np.savez_compressed(out/'destination.npz',matched_A=am,matched_B=bm,alpha=alpha,values=values,rooted=rooted)
    save_json(out/'configuration.json',{'input_sha256':digest(rnaseq),'n_per_group':400,'selection':'Original specimen order and RandomState(42); top-variance features from the full input.','projection_seed':711,'projections':128,'bandwidth':bw,'normalization':'Each curve divided by its own starting value; E and S rooted.'})
    return out/'destination.npz'


def render(data,out):
    out.mkdir(parents=True,exist_ok=True);z=np.load(data);curves=z['rooted']/z['rooted'][0]
    if not np.allclose(curves[0],1) or not np.allclose(curves[-1],0,atol=1e-6):raise ValueError('Interpolation endpoint mismatch')
    sns.set_style('whitegrid');plt.rcParams.update({'figure.dpi':150,'font.size':11,'savefig.bbox':'tight','savefig.pad_inches':.1})
    am=z['matched_A'];bm=z['matched_B'];pca=PCA(n_components=2,svd_solver='full').fit(np.r_[am,bm]);pca.components_[0]*=-1
    fig,axes=plt.subplots(1,2,figsize=(14,5),gridspec_kw={'width_ratios':[1,1.2]})
    for pts,label,color in [(am,'Cluster A (Start)','#D4A017'),(bm,'Cluster B (Target)','#2A9D8F')]:axes[0].scatter(*pca.transform(pts).T,color=color,alpha=.6,s=15,label=label,zorder=2)
    axes[0].set_xlabel('Principal Component 1');axes[0].set_ylabel('Principal Component 2');axes[0].legend(fontsize=9)
    labels=['Energy Distance','Signature Distance','Coupled SD (CSD)','Gaussian MMD$^2$',r'Sliced $W_1$'];colors=['#4a90d9','#e85d4a','#9b59b6','#2A9D8F','#b58a28'];markers=['o','s','^','D','v']
    for i in range(3):axes[1].plot(z['alpha'],curves[:,i],markers[i]+'-',color=colors[i],lw=2,markersize=6,label=labels[i])
    axes[1].axhline(1,color='#888888',ls='--',alpha=.7,label='Starting distance');axes[1].set_xlabel(r'Interpolation $\alpha$  (0 = cluster A, 1 = cluster B)');axes[1].set_ylabel('Distance / starting value');axes[1].legend(fontsize=9);fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'fig06_interpolation.{ext}')
    plt.close(fig);fig,ax=plt.subplots(figsize=(9,5))
    for i in range(5):ax.plot(z['alpha'],curves[:,i],markers[i]+'-',color=colors[i],lw=2,markersize=6,label=labels[i])
    ax.axhline(1,color='grey',ls='--',lw=1,alpha=.7);ax.set_xlabel(r'Interpolation $\alpha$  (0 = cluster A, 1 = cluster B)');ax.set_ylabel('Score / starting value');ax.legend(fontsize=9);ax.set_ylim(bottom=-.03);fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'figS08_interpolation_combined.{ext}')
    plt.close(fig)
    pd.DataFrame({'alpha':z['alpha'],**{n:curves[:,i] for i,n in enumerate(NAMES)}}).to_csv(out/'destination_normalized.csv',index=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--rnaseq',type=Path);p.add_argument('--audit',type=Path);p.add_argument('--data',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.data is None and (a.rnaseq is None or a.audit is None):p.error('provide --data, or both --rnaseq and --audit')
    render(a.data or compute(a.rnaseq,a.audit,a.out),a.out)
