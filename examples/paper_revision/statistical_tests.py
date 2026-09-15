#!/usr/bin/env python3
"""Prespecified independent-draw statistical comparisons for the revised paper."""
import os,sys,argparse,json,time,concurrent.futures
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scipy.spatial.distance import cdist
from examples.paper_revision.common import save_json,digest

NAMES=['E','S','CSD','MMD2','sliced_W1']
SCENARIOS={'null':(0.,1.),'shift_original':(2.,1.),'scale_original':(0.,.7),'shift_subtle':(.25,1.),'scale_subtle':(0.,.9)}


def score(D,K,P,ix,iy):
    xx=D[np.ix_(ix,ix)];yy=D[np.ix_(iy,iy)];xy=D[np.ix_(ix,iy)]
    a=np.sort(xx,axis=1);b=np.sort(xy,axis=1);c=np.sort(yy,axis=1);d=np.sort(xy.T,axis=1)
    s=np.abs(a-b).mean()+np.abs(c-d).mean()
    col=np.abs(np.sort(a,axis=0)-np.sort(b,axis=0)).mean()+np.abs(np.sort(c,axis=0)-np.sort(d,axis=0)).mean()
    e=2*xy.mean()-xx.mean()-yy.mean()
    mmd=K[np.ix_(ix,ix)].mean()+K[np.ix_(iy,iy)].mean()-2*K[np.ix_(ix,iy)].mean()
    sw=np.abs(np.sort(P[ix],axis=0)-np.sort(P[iy],axis=0)).mean()
    return np.array([e,s,np.sqrt(s+col),mmd,sw])


def compare(x,y,permutations,seed):
    if len(x)!=len(y):raise ValueError('This comparison design uses equal sample counts.')
    z=np.vstack([x,y]);n=len(x);D=cdist(z,z);positive=D[D>0];bw=float(np.median(positive)) if len(positive) else 1.
    K=np.exp(-D**2/(2*bw*bw));prng=np.random.default_rng(711);directions=prng.normal(size=(x.shape[1],128));directions/=np.linalg.norm(directions,axis=0)
    P=z@directions;obs=score(D,K,P,np.arange(n),np.arange(n,2*n));rng=np.random.default_rng(seed);null=np.empty((permutations,len(NAMES)))
    for i in range(permutations):
        perm=rng.permutation(2*n);null[i]=score(D,K,P,perm[:n],perm[n:])
    exc=(null>=obs).sum(axis=0);p=(exc+1)/(permutations+1);order=np.argsort(p);adj=np.empty_like(p);adj[order]=np.minimum(1,np.maximum.accumulate(p[order]*np.arange(len(NAMES),0,-1)))
    return {'observed':obs,'null':null,'exceedances':exc,'p':p,'holm':adj,'bandwidth':bw}


def run_job(args):
    n,scenario,rep,B,out=args;seed=193000+n*1000+list(SCENARIOS).index(scenario)*100+rep;target=Path(out)/f'n{n}_{scenario}_{rep:03d}.npz'
    if target.exists():return str(target)
    rng=np.random.default_rng(seed);x=rng.normal(size=(n,2));shift,scale=SCENARIOS[scenario];y=rng.normal(size=(n,2))*scale;y[:,0]+=shift
    start=time.monotonic();result=compare(x,y,B,seed+900000);np.savez_compressed(target,x=x,y=y,seed=seed,seconds=time.monotonic()-start,**result)
    return str(target)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--replicates',type=int,default=50);p.add_argument('--permutations',type=int,default=999);p.add_argument('--workers',type=int,default=2)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    config={'samples':[50,200],'scenarios':SCENARIOS,'replicates':args.replicates,'permutations':args.permutations,'metrics':NAMES,'source_sha256':digest(__file__),
            'alpha':.05,'multiplicity':'Power uses each individual test at alpha=.05; per-example simultaneous method comparisons also have Holm values.',
            'limits':'Prespecified independent pairs, common permutations, no outcome-based parameter selection. Wilson intervals quantify finite replication uncertainty.'}
    if (args.out/'config.json').exists() and json.loads((args.out/'config.json').read_text())!=json.loads(json.dumps(config)):raise ValueError('Configuration changed')
    save_json(args.out/'config.json',config)
    jobs=[(n,s,rep,args.permutations,args.out) for n in [50,200] for s in SCENARIOS for rep in range(args.replicates)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,_ in enumerate(pool.map(run_job,jobs)):
            if i%10==0:save_json(args.out/'progress.json',{'complete':i+1,'total':len(jobs)})
    save_json(args.out/'progress.json',{'complete':len(jobs),'total':len(jobs)})
