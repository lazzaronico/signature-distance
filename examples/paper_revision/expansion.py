#!/usr/bin/env python3
"""Langevin population expansion with validation-selected stopping.

The largest cluster is split into a seed set A, a validation set B and a
held-out set C. A bootstrap resample A_k of A guides the update

    x <- clip(x - eta * n * grad_x D(pop, A_k) + noise, 0),

with D the raw energy statistic E, the raw signature statistic S or S + C
(the squared CSD), evaluated on the complete population of n points; `none`
omits the guidance term. The factor n gives each point the gradient of its own
term in the population average. Each chain stops at the step, among multiples
of --every, with the smallest distance to B in the measure of its guidance
(SD for the unguided chain). C is used for evaluation only.
"""
import argparse,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from sklearn.cluster import KMeans
from sigdist import losses
from examples.paper_revision.common import save_json,digest,independent_metrics

# guidance objective and index of the selection measure among (d_E, SD, CSD)
GUIDANCE={'none':(None,1),'ED':(losses.energy,0),'SD':(losses.sd,1),'CSD':(lambda x,y:losses.csd(x,y)**2,2)}
COLUMNS=[f'{m}({r})' for r in ['A_k','B','C'] for m in ['d_E','SD','CSD']]


def distances(p,q):
    with torch.no_grad():
        return np.array([float(losses.energy(p,q).clamp_min(0).sqrt()),float(losses.sd(p,q).clamp_min(0).sqrt()),float(losses.csd(p,q))])


def chain(name,init,refs,shape,noise_rng,steps,every,eta=.1,noise=.1):
    fn,measure=GUIDANCE[name];n=len(init);pop=init.copy();trace=[];independent=[];snapshots={}
    C=refs[2].numpy()
    for step in range(steps+1):
        pt=torch.from_numpy(pop);trace.append(np.stack([distances(pt,r) for r in refs]).ravel())
        if step%every==0:independent.append({'step':step,**independent_metrics(C,pop)});snapshots[step]=pop.copy()
        if step==steps:break
        if fn is None:grad=np.zeros_like(pop)
        else:
            v=pt.clone().requires_grad_(True);grad=n*torch.autograd.grad(fn(v,refs[0]),v)[0].numpy()
        pop=np.maximum(0,pop-eta*grad+noise*noise_rng.normal(size=pop.shape)@shape.T).astype('float32')
        if not np.isfinite(pop).all():raise RuntimeError('Nonfinite expansion')
    trace=np.array(trace);grid=np.arange(0,steps+1,every)
    selected=int(grid[trace[grid,3+measure].argmin()])
    return trace,independent,selected,snapshots[selected]


def run(a):
    torch.set_num_threads(a.threads);a.out.mkdir(parents=True,exist_ok=True)
    data=np.load(a.geometry/'geometry.npz');pool=data['x'][data['labels']==int(data['largest'])];n=len(pool)//3
    ix=np.random.default_rng(42).permutation(len(pool));A,B,C=[pool[ix[i*n:(i+1)*n]].astype('float32') for i in range(3)]
    save_json(a.out/'configuration.json',{'bootstrap_replicates':5,'steps':a.steps,'eta':.1,'noise':.1,'n_per_partition':n,'pool_size':len(pool),'centroids':max(1,int(n*.1)),
      'seed':42,'covariance':'Bootstrap covariance + 1e-6 I; Cholesky divided by sqrt(trace/d)',
      'guidance':{'none':'no guidance term','ED':'eta * n * grad E','SD':'eta * n * grad S','CSD':'eta * n * grad (S + C)'},
      'stopping':f'Smallest distance to validation B among multiples of {a.every} steps; d_E for ED, SD for SD and none, CSD for CSD guidance.',
      'display':'Bootstrap 0 at its selected step; all five retained; held-out C never selects.',
      'limits':'Descriptive noise-injected optimization at one step size and noise level. Global feature/cluster selection precedes the partition.',
      'source':digest(__file__),'geometry_source':digest(a.geometry/'geometry.npz')})
    for b in a.bootstraps:
        rng=np.random.default_rng(5200+b);Ak=A[rng.integers(n,size=n)];cov=np.cov(Ak.T)+1e-6*np.eye(A.shape[1])
        shape=np.linalg.cholesky(cov)/np.sqrt(np.trace(cov)/A.shape[1]);km=KMeans(n_clusters=max(1,int(n*.1)),n_init=10,random_state=42).fit(Ak)
        init=np.repeat(km.cluster_centers_,np.bincount(km.labels_,minlength=km.n_clusters),axis=0);init=init[rng.permutation(n)].astype('float32')
        refs=[torch.from_numpy(x) for x in [Ak,B,C]]
        for name in a.guidance:
            out=a.out/f'{name.lower()}_bootstrap{b}.npz'
            if out.exists():continue
            start=time.monotonic()
            # Chains of a bootstrap share the initial population and the noise draws.
            trace,independent,selected,population=chain(name,init,refs,shape,np.random.default_rng(6200+b),a.steps,a.every)
            np.savez_compressed(out,trace=trace,selected_step=selected,selected=population,initial=init,a=Ak,b=B,c=C)
            save_json(out.with_suffix('.json'),{'bootstrap':b,'method':name,'selected_step':selected,'trace_columns':COLUMNS,
              'validation_distances':dict(zip(['d_E','SD','CSD'],trace[selected,3:6].tolist())),'test_distances':dict(zip(['d_E','SD','CSD'],trace[selected,6:9].tolist())),
              'independent':next(i for i in independent if i['step']==selected),'independent_trace':independent,'real_real':independent_metrics(C,B),
              'real_real_distances':dict(zip(['d_E','SD','CSD'],distances(refs[1],refs[2]).tolist())),'seconds':time.monotonic()-start})
            print(b,name,'selected step',selected,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--geometry',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--guidance',nargs='+',choices=list(GUIDANCE),default=list(GUIDANCE));p.add_argument('--bootstraps',nargs='+',type=int,default=list(range(5)))
    p.add_argument('--steps',type=int,default=1000);p.add_argument('--every',type=int,default=25);p.add_argument('--threads',type=int,default=1);run(p.parse_args())
