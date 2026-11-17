#!/usr/bin/env python3
"""Repeated original-architecture TCGA generation, with resumable local fits."""
import os,sys,argparse,time,json,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from examples.paper_revision.common import FIXED_METHODS as METHODS,SCALES,TissueGenerator,Classifier,glocal_rooted as glocal,independent_metrics,save_json,digest


def checkpoint(path,**state):
    tmp=path.with_suffix(".tmp");torch.save(state,tmp);tmp.replace(path)


def load_data(folder):
    tr=pd.read_csv(folder/"tcga_landmark_train_norm.csv",index_col=0);te=pd.read_csv(folder/"tcga_landmark_test_norm.csv",index_col=0)
    meta=['full_barcode','tissue_type','cancer_type','reduced_barcode','cohort','cancer','barcode','gender','age']
    genes=[c for c in tr if c not in meta];tissues=sorted(tr.tissue_type.unique());mapping={v:i for i,v in enumerate(tissues)}
    x=tr[genes].to_numpy(dtype=np.float32);y=tr.tissue_type.map(mapping).to_numpy(dtype=np.int64)
    u=te[genes].to_numpy(dtype=np.float32);v=te.tissue_type.map(mapping).to_numpy(dtype=np.int64)
    if not np.isfinite(x).all() or not np.isfinite(u).all():raise ValueError("Nonfinite input; preprocessing must be explicit.")
    return x,y,u,v,tissues,genes


def train(args):
    torch.set_num_threads(args.threads);torch.manual_seed(args.seed);np.random.seed(args.seed)
    out=args.out/f"{args.mode}_{args.method.lower()}_seed{args.seed}";out.mkdir(parents=True,exist_ok=True)
    if (out/"result.json").exists():return
    x,y,xt,yt,tissues,genes=load_data(args.data)
    split=json.loads((args.data/'split.json').read_text())
    train_idx=np.asarray(split['fit_indices']);val_idx=np.asarray(split['validation_indices'])
    if len(set(train_idx)&set(val_idx)) or sorted(np.r_[train_idx,val_idx].tolist())!=list(range(len(x))):raise ValueError('Invalid inner split')
    # Fixed training-only Gaussian-kernel scale, without outcome-based tuning.
    scale_rng=np.random.default_rng(7741);subset=x[train_idx][scale_rng.choice(len(train_idx),min(1024,len(train_idx)),replace=False)]
    distances=pdist(subset);bandwidth=float(np.median(distances[distances>0]))
    config={"protocol":"paper-glocal-rooted-fixed-validation-v1","method":args.method,"seed":args.seed,"split_seed":42,
            "mode":args.mode,"threads":args.threads,"max_steps":args.max_steps,"batch":256,"lr":.001,
            "validation_every":25,"patience_validation_checks":200,"validation_bank_batches":4,"validation_bank_seed":args.seed+82000,"validation_noise_seed":args.seed+83000,"root_floor":1e-12,"loss_multiplier":SCALES[args.method],
            "classifier_epochs":args.classifier_epochs,"classifier_patience":50,"synthetic_multiplier":10,
            "bandwidth":bandwidth,"input_hashes":{p.name:digest(p) for p in args.data.glob('tcga_landmark_*_norm.csv')},
            "preprocessing_hashes":{p.name:digest(p) for p in [args.data/"split.json",args.data/"preprocessing.json"]},
            "source_hashes":{str(p.relative_to(Path(__file__).resolve().parents[2])):digest(p) for p in
              [Path(__file__).resolve(),Path(__file__).with_name('common.py'),Path(__file__).resolve().parents[2]/'sigdist/losses.py',Path(__file__).resolve().parents[2]/'sigdist/_quantiles.py']},
            "torch":str(torch.__version__),"numpy":np.__version__,"tissues":tissues,"genes":genes,
            "limits":"Fixed data split and 4x256 validation bank/noise; full empirical profiles; original architecture, maximum budget, and 5000-update patience span; validation every 25 updates. Root floor 1e-12 for numerical stability. Seed SD is training variation, not population uncertainty."}
    if (out/"config.json").exists():
        old=json.loads((out/"config.json").read_text())
        if old!=config:raise ValueError("Resume configuration or source differs from this run.")
    else:save_json(out/"config.json",config)
    model=TissueGenerator(len(tissues),len(genes));opt=torch.optim.Adam(model.parameters(),lr=.001)
    best=copy.deepcopy(model.state_dict());bestval=float('inf');best_step=0;wait=0;start=0;seconds=0.;history=[]
    data_rng=np.random.default_rng(args.seed+100);val_rng=np.random.default_rng(args.seed+200)
    latent_rng=torch.Generator().manual_seed(args.seed+300);vlatent_rng=torch.Generator().manual_seed(args.seed+400)
    # Fixed validation observations and noise, identical across objectives.
    if len(val_idx)<1024:raise ValueError("The fixed paper protocol requires at least 1,024 validation observations.")
    bank_rng=np.random.default_rng(args.seed+82000)
    bank_ids=val_idx[bank_rng.permutation(len(val_idx))[:1024]].reshape(4,256)
    bank_z=torch.randn(4,256,32,generator=torch.Generator().manual_seed(args.seed+83000))
    np.savez_compressed(out/'validation-bank.npz',indices=bank_ids,noise=bank_z.numpy())
    validation_bank=[(torch.from_numpy(x[ids]),torch.from_numpy(y[ids]),z)for ids,z in zip(bank_ids,bank_z)]
    training_done=(out/"generator.pt").exists();ck=out/"training.pt"
    if ck.exists() and not training_done:
        state=torch.load(ck,weights_only=False);model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        best=state['best'];bestval=state['bestval'];best_step=state.get('best_step',0);wait=state['wait'];start=state['step'];seconds=state['seconds'];history=state['history']
        data_rng.bit_generator.state=state['data_rng'];val_rng.bit_generator.state=state['val_rng'];latent_rng.set_state(state['latent_rng']);vlatent_rng.set_state(state['vlatent_rng'])
    begin=time.monotonic()
    if not training_done:
        for step in range(start+1,args.max_steps+1):
            ids=train_idx[data_rng.permutation(len(train_idx))[:256]];real=torch.from_numpy(x[ids]);lab=torch.from_numpy(y[ids])
            gen=model(torch.randn(len(ids),32,generator=latent_rng),lab);value=glocal(gen,real,lab,args.method,bandwidth,args.mode)
            if not torch.isfinite(value):raise RuntimeError(f"Nonfinite loss at step {step}")
            opt.zero_grad();value.backward();opt.step()
            if step==1 or step%25==0:
                with torch.no_grad():
                    v=float(torch.stack([glocal(model(z,lab),real,lab,args.method,bandwidth,args.mode)for real,lab,z in validation_bank]).mean())
                if v<bestval-1e-6:bestval=v;best=copy.deepcopy(model.state_dict());best_step=step;wait=0
                else:wait+=1
            if step%500==0 or step==args.max_steps or wait>=200:
                history.append({'step':step,'train':float(value.detach()),'validation':v,'best':bestval,'wait':wait})
                elapsed=seconds+time.monotonic()-begin
                checkpoint(ck,model=model.state_dict(),optimizer=opt.state_dict(),best=best,bestval=bestval,best_step=best_step,wait=wait,step=step,seconds=elapsed,history=history,
                           data_rng=data_rng.bit_generator.state,val_rng=val_rng.bit_generator.state,latent_rng=latent_rng.get_state(),vlatent_rng=vlatent_rng.get_state())
                save_json(out/"progress.json",{'stage':'generator','step':step,'max_steps':args.max_steps,'seconds':elapsed,'best_validation':bestval})
                print(args.method,args.seed,step,elapsed,flush=True)
            if wait>=200:break
        model.load_state_dict(best)
        checkpoint(out/"generator.pt",model=best,step=step,selected_step=best_step,seconds=seconds+time.monotonic()-begin,history=history)
    saved=torch.load(out/"generator.pt",weights_only=False);model.load_state_dict(saved['model']);model.eval()
    # Fixed post-training streams; evaluation does not depend on stopping time.
    torch.manual_seed(args.seed+10000);syn=[];labels=[]
    with torch.no_grad():
        for t in range(len(tissues)):
            remaining=int(np.sum(y==t))*10;parts=[]
            while remaining:
                n=min(remaining,1024);parts.append(model(torch.randn(n,32),torch.full((n,),t)).numpy());remaining-=n
            syn.append(np.concatenate(parts));labels.append(np.full(len(syn[-1]),t))
    sx=np.concatenate(syn).astype(np.float32);sy=np.concatenate(labels).astype(np.int64)
    # Separate held-out precision/recall; equal budgets and fixed geometry.
    eval_rng=np.random.default_rng(args.seed+30000);n=min(2032,len(xt));ri=eval_rng.permutation(len(xt))[:n];gi=eval_rng.permutation(len(sx))[:n]
    independent=independent_metrics(xt[ri],sx[gi]);controls=independent_metrics(xt[ri],x[eval_rng.permutation(len(x))[:n]])
    by_tissue={}
    for t,name in enumerate(tissues):
        rr=xt[yt==t];gg=sx[sy==t];count_t=min(len(rr),len(gg),256)
        if count_t>=4:by_tissue[name]=independent_metrics(rr[eval_rng.permutation(len(rr))[:count_t]],gg[eval_rng.permutation(len(gg))[:count_t]])
    # Exact full-dimensional nearest neighbors in blocks; avoid high-dimensional tree degeneration.
    xtensor=torch.from_numpy(x);ynorm=(xtensor*xtensor).sum(dim=1);nearest=[];distances=[]
    with torch.no_grad():
        for begin_nn in range(0,len(sx),512):
            block=torch.from_numpy(sx[begin_nn:begin_nn+512])
            sq=((block*block).sum(dim=1)[:,None]+ynorm[None,:]-2*block@xtensor.T).clamp_min(0)
            values,ids=sq.min(dim=1);nearest.append(ids.numpy());distances.append(values.sqrt().numpy())
    indices=np.concatenate(nearest);d=np.concatenate(distances);counts=np.bincount(indices,minlength=len(x));p=counts[counts>0]/len(sx)
    coverage=float(np.mean(counts>0));entropy=float(-(p*np.log2(p)).sum()/np.log2(len(x)));nnd=float(d.mean())
    # The original synthetic-trained classifier endpoint and budgets.
    torch.manual_seed(args.seed+20000);crng=np.random.default_rng(args.seed+21000);clf=Classifier(len(genes),len(tissues));co=torch.optim.Adam(clf.parameters(),lr=.001)
    a,b=train_test_split(np.arange(len(sx)),test_size=.2,stratify=sy,random_state=42)
    bx=torch.from_numpy(sx[b]);by=torch.from_numpy(sy[b]);bestc=copy.deepcopy(clf.state_dict());bestcv=float('inf');cw=0;ce=0
    cp=out/"classifier-training.pt"
    if cp.exists():
        c=torch.load(cp,weights_only=False);clf.load_state_dict(c['model']);co.load_state_dict(c['optimizer']);bestc=c['best'];bestcv=c['bestval'];cw=c['wait'];ce=c['epoch'];crng.bit_generator.state=c['rng']
    epoch=ce
    for epoch in range(ce+1,args.classifier_epochs+1):
        order=crng.permutation(a)
        for i in range(0,len(order)-256,256):
            ids=order[i:i+256];loss=nn.functional.cross_entropy(clf(torch.from_numpy(sx[ids])),torch.from_numpy(sy[ids]));co.zero_grad();loss.backward();co.step()
        with torch.no_grad():cv=float(nn.functional.cross_entropy(clf(bx),by))
        if cv<bestcv-1e-6:bestcv=cv;bestc=copy.deepcopy(clf.state_dict());cw=0
        else:cw+=1
        if epoch%10==0 or cw>=50 or epoch==args.classifier_epochs:
            checkpoint(cp,model=clf.state_dict(),optimizer=co.state_dict(),best=bestc,bestval=bestcv,wait=cw,epoch=epoch,rng=crng.bit_generator.state)
            save_json(out/"progress.json",{'stage':'classifier','epoch':epoch,'selected_step':saved.get('selected_step'),'generator_steps':saved['step'],'best_validation':bestcv})
        if cw>=50:break
    clf.load_state_dict(bestc)
    with torch.no_grad():pred=clf(torch.from_numpy(xt)).argmax(dim=1).numpy()
    result={'method':args.method,'seed':args.seed,'mode':args.mode,'selected_step':saved.get('selected_step'),'generator_steps':saved['step'],'generator_seconds':saved['seconds'],
            'classifier_epochs':epoch,'accuracy':float(accuracy_score(yt,pred)),'macro_f1':float(f1_score(yt,pred,average='macro',zero_division=0)),
            'coverage':coverage,'entropy':entropy,'nn_distance':nnd,'f1_per_tissue':dict(zip(tissues,f1_score(yt,pred,labels=np.arange(len(tissues)),average=None,zero_division=0).tolist())),
            'independent':independent,'real_real':controls,'independent_by_tissue':by_tissue,'total_elapsed_this_process':time.monotonic()-begin}
    checkpoint(out/'classifier.pt',model=bestc)
    np.savez_compressed(out/'evaluation_samples.npz',generated=sx[gi],real_test_indices=ri,predictions=pred)
    save_json(out/'result.json',result);save_json(out/'progress.json',{'stage':'complete'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--method',choices=METHODS,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--mode',choices=['glocal','global'],default='glocal');p.add_argument('--max-steps',type=int,default=100000);p.add_argument('--classifier-epochs',type=int,default=500)
    train(p.parse_args())
