#!/usr/bin/env python3
"""Three-seed repeats of the paper's circle and heteroscedastic sine tasks."""
import argparse,sys,time,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from scipy.spatial.distance import pdist
from examples.paper_revision.common import TOY_METHODS,SCALES,SineMLP,objective,independent_metrics,digest,save_json
from sigdist.generators import FICGenerator


def circles(n=2000,seed=7):
    r=np.random.RandomState(seed);a=r.uniform(0,2*np.pi,n);b=r.uniform(0,2*np.pi,n)
    x0=2*np.cos(a)-1+r.normal(0,.08,n);y0=2*np.sin(a)+r.normal(0,.08,n)
    x1=3.5*np.cos(b)+2.5+r.normal(0,.08,n);y1=3.5*np.sin(b)+r.normal(0,.08,n)
    return np.r_[np.c_[x0,y0],np.c_[x1,y1]].astype('float32'),np.repeat([0,1],n)


def sine(n=2000,seed=42):
    r=np.random.default_rng(seed);t=r.uniform(0,2*np.pi,n).astype('float32');y=np.sin(t)+np.where(t<np.pi,.06,.28)*r.normal(size=n)
    return np.c_[t,y].astype('float32')


def train(a):
    torch.set_num_threads(a.threads);torch.manual_seed(a.seed);rng=np.random.default_rng(a.seed+100);out=a.out/f'{a.task}_{a.method.lower()}_seed{a.seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():return
    if a.task=='circles':
        raw,labels=circles();mu=raw.mean(0);sd=raw.std(0);real=(raw-mu)/sd;model=FICGenerator(noise_dim=2);lr=.001;steps=a.steps or 10000
    else:
        real=sine();model=SineMLP();lr=3e-4;steps=a.steps or 7000
    bw=float(np.median(pdist(real[np.random.default_rng(7741).choice(len(real),min(1024,len(real)),replace=False)])))
    config={'task':a.task,'method':a.method,'seed':a.seed,'steps':steps,'batch':256,'lr':lr,'bandwidth':bw,'loss_multiplier':SCALES[a.method],
      'data_seed':7 if a.task=='circles' else 42,'torch':str(torch.__version__),'sources':{p.name:digest(p) for p in [Path(__file__),Path(__file__).with_name('common.py'),Path(__file__).resolve().parents[2]/'sigdist/losses.py',Path(__file__).resolve().parents[2]/'sigdist/generators.py']}}
    if (out/'config.json').exists() and json.loads((out/'config.json').read_text())!=config:raise ValueError('Changed resume configuration')
    save_json(out/'config.json',config);opt=torch.optim.Adam(model.parameters(),lr=lr);begin=time.monotonic();start=0
    if (out/'training.pt').exists():
        c=torch.load(out/'training.pt',weights_only=False);model.load_state_dict(c['model']);opt.load_state_dict(c['opt']);rng.bit_generator.state=c['rng'];torch.set_rng_state(c['torch_rng']);start=c['step']
    # Sine checkpoints only occur at epoch boundaries: seven full batches per epoch.
    for step in range(start+1,steps+1):
        if a.task=='circles':
            cl=step%2;pool=real[labels==cl];target=torch.from_numpy(pool[rng.integers(len(pool),size=256)]);pred=model(torch.randn(256,2),torch.full((256,1),float(cl)))
        else:
            if (step-1)%7==0:order=rng.permutation(len(real));inputs=torch.stack([torch.from_numpy(real[:,0]),torch.randn(len(real))],1)
            ids=order[((step-1)%7)*256:((step-1)%7+1)*256];target=torch.from_numpy(real[ids]);pred=model(inputs[ids])
        loss=objective(pred,target,a.method,bw)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite objective')
        opt.zero_grad();loss.backward();opt.step()
        interval=500 if a.task=='circles' else 700
        if step%interval==0 or step==steps:
            torch.save({'model':model.state_dict(),'opt':opt.state_dict(),'rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),'step':step},out/'training.pt')
            save_json(out/'progress.json',{'step':step,'steps':steps,'seconds_this_process':time.monotonic()-begin});print(a.task,a.method,a.seed,step,flush=True)
    model.eval();torch.manual_seed(a.seed+10000)
    with torch.no_grad():
        if a.task=='circles':
            generated=np.concatenate([model(torch.randn(2000,2),torch.full((2000,1),float(cl))).numpy() for cl in [0,1]])
            truth,tl=circles(seed=991);truth=(truth-mu)/sd;control,_=circles(seed=992);control=(control-mu)/sd
            per={str(cl):independent_metrics(truth[tl==cl],generated[tl==cl]) for cl in [0,1]}
            physical=generated*sd+mu;radial=np.r_[np.abs(np.linalg.norm(physical[:2000]-[-1,0],axis=1)-2),np.abs(np.linalg.norm(physical[2000:]-[2.5,0],axis=1)-3.5)]
            extra={'by_class':per,'radial_absolute_error':float(radial.mean())}
        else:
            truth=sine(4000,991);control=sine(4000,992);t=torch.from_numpy(np.random.default_rng(991).uniform(0,2*np.pi,4000).astype('float32'));generated=model(torch.stack([t,torch.randn(4000)],1)).numpy()
            residual=generated[:,1]-np.sin(generated[:,0]);extra={'residual_sd_narrow':float(residual[generated[:,0]<np.pi].std()),'residual_sd_wide':float(residual[generated[:,0]>=np.pi].std()),'out_of_domain_fraction':float(((generated[:,0]<0)|(generated[:,0]>2*np.pi)).mean())}
    # Same balanced 2,000-point pooled evaluation for each method.
    ix=np.random.default_rng(333).permutation(len(truth))[:2000]
    result={'task':a.task,'method':a.method,'seed':a.seed,'steps':steps,'independent':independent_metrics(truth[ix],generated[ix]),'real_real':independent_metrics(truth[ix],control[ix]),**extra}
    np.savez_compressed(out/'samples.npz',real=real,truth=truth,generated=generated);save_json(out/'result.json',result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['circles','sine'],required=True);p.add_argument('--method',choices=TOY_METHODS,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--threads',type=int,default=1);p.add_argument('--steps',type=int);train(p.parse_args())
