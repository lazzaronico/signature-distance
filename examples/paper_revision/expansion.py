#!/usr/bin/env python3
"""Five-bootstrap centroid expansion with validation-only stopping and held-out checks."""
import argparse,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from sklearn.cluster import KMeans
from sigdist import losses
from examples.paper_revision.common import save_json,digest,independent_metrics


def metrics(p,q):
    with torch.no_grad():return np.array([float(losses.energy(p,q).clamp_min(0).sqrt()),float(losses.sd(p,q).clamp_min(0).sqrt())])


def run(a):
    torch.set_num_threads(1);a.out.mkdir(parents=True,exist_ok=True)
    data=np.load(a.geometry/'geometry.npz');pool=data['x'][data['labels']==int(data['largest'])];n=len(pool)//3;ix=np.random.default_rng(42).permutation(len(pool));A,B,C=[pool[ix[i*n:(i+1)*n]].astype('float32') for i in range(3)]
    save_json(a.out/'configuration.json',{'bootstrap_replicates':5,'steps':500,'eta':.1,'noise':.1,'batch':256,'n_per_partition':n,'pool_size':len(pool),'centroids':max(1,int(n*.1)),
      'seed':42,'covariance':'Bootstrap covariance + 1e-6 I; Cholesky divided by sqrt(trace/d)',
      'guidance':['ED','SD','CSD'],'losses':'E, S, CSD (no compensating multipliers)','stopping':'argmin [(D(pop,Aboot)-D(pop,B))^2+(D(pop,Aboot)-D(Aboot,B))^2]; D=d_E for ED guidance, SD for SD/CSD guidance.',
      'display':'Minimum validation-distance bootstrap at its criterion-selected step, per method; all five retained; no test-based selection or point removal.',
      'limits':'Descriptive noise-injected optimization, not a proof of sampling from a calibrated target density. Global feature/cluster selection precedes the partition.',
      'source':digest(__file__),'geometry_source':digest(a.geometry/'geometry.npz')})
    for b in range(5):
        rng=np.random.default_rng(5200+b);Ak=A[rng.integers(n,size=n)];cov=np.cov(Ak.T)+1e-6*np.eye(A.shape[1]);sh=np.linalg.cholesky(cov)/np.sqrt(np.trace(cov)/A.shape[1]);km=KMeans(n_clusters=max(1,int(n*.1)),n_init=10,random_state=42).fit(Ak)
        init=np.repeat(km.cluster_centers_,np.bincount(km.labels_,minlength=km.n_clusters),axis=0);init=init[rng.permutation(n)].astype('float32');refs=[torch.from_numpy(x) for x in [Ak,B,C]];ab=metrics(refs[0],refs[1])
        for name in ['ED','SD','CSD']:
            out=a.out/f'{name.lower()}_bootstrap{b}.npz'
            if out.exists():continue
            noise_rng=np.random.default_rng(6200+b);pop=init.copy();trace=[];best=float('inf');beststep=0;bestpop=pop.copy();start=time.monotonic()
            for step in range(500):
                pt=torch.from_numpy(pop);ds=np.stack([metrics(pt,r) for r in refs]);criterion=(ds[0]-ds[1])**2+(ds[0]-ab)**2;j=0 if name=='ED' else 1
                if criterion[j]<best:best=float(criterion[j]);beststep=step;bestpop=pop.copy()
                trace.append(np.r_[ds.ravel(),criterion]);grad=np.empty_like(pop)
                for i in range(0,n,256):
                    v=pt[i:i+256].clone().requires_grad_(True)
                    value=losses.energy(v,refs[0]) if name=='ED' else losses.sd(v,refs[0]) if name=='SD' else losses.csd(v,refs[0])
                    grad[i:i+256]=torch.autograd.grad(value,v)[0].numpy()
                pop=np.maximum(0,pop-.1*grad+.1*noise_rng.normal(size=pop.shape)@sh.T).astype('float32')
                if not np.isfinite(pop).all():raise RuntimeError('Nonfinite expansion')
                if step%100==0:save_json(a.out/'progress.json',{'bootstrap':b,'method':name,'step':step,'seconds':time.monotonic()-start});print(b,name,step,flush=True)
            independently=independent_metrics(C,bestpop);control=independent_metrics(C,B)
            np.savez_compressed(out,trace=trace,selected_step=beststep,selected=bestpop,initial=init,a=Ak,b=B,c=C)
            save_json(out.with_suffix('.json'),{'bootstrap':b,'method':name,'selected_step':beststep,'criterion':best,'independent':independently,'real_real':control,'seconds':time.monotonic()-start})
    save_json(a.out/'progress.json',{'stage':'complete'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--geometry',type=Path,required=True);p.add_argument('--out',type=Path,required=True);run(p.parse_args())
