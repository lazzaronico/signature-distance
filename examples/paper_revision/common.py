"""Models, exact objectives and independent evaluation used in paper repeats."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from sigdist import losses

METHODS=["MSE","ED","SD","CSD","GSD","MMD"]
SCALES={name:1. for name in METHODS+["SW1"]}
TOY_METHODS=METHODS+["SW1"]
FIXED_METHODS=["MSE","ED","SD","CSD","GSD"]


def digest(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n");tmp.replace(path)


class TissueGenerator(nn.Module):
    def __init__(self,tissues,genes,noise=32,width=512,depth=5):
        super().__init__();self.noise=noise;self.tissues=tissues;inputs=noise+tissues
        self.layers=nn.ModuleList([nn.Linear(inputs if i==0 else width+inputs,width) for i in range(depth)])
        self.norms=nn.ModuleList([nn.LayerNorm(width) for _ in range(depth)])
        self.fc_out=nn.Linear(width,genes)
    def forward(self,z,labels):
        x=torch.cat([z,F.one_hot(labels,self.tissues).float()],dim=1);h=x
        for i,(lin,norm) in enumerate(zip(self.layers,self.norms)):
            h=F.leaky_relu(norm(lin(x if i==0 else torch.cat([h,x],dim=1))))
        return self.fc_out(h)


class Classifier(nn.Module):
    def __init__(self,genes,tissues):
        super().__init__();layers=[]
        for i in range(3):layers.extend([nn.Linear(genes if i==0 else 256,256),nn.LayerNorm(256),nn.ReLU()])
        layers.append(nn.Linear(256,tissues));self.net=nn.Sequential(*layers)
    def forward(self,x):return self.net(x)


class SineMLP(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(2,128),nn.GELU(),nn.Linear(128,128),nn.GELU(),nn.Linear(128,2))
    def forward(self,x):return self.net(x)


def objective(x,y,method,bandwidth):
    if method=="MSE":return F.mse_loss(x,y)
    if method=="SW1":
        directions=torch.randn(x.shape[1],128,generator=torch.Generator().manual_seed(8273))
        directions/=torch.linalg.vector_norm(directions,dim=0)
        directions=directions.to(device=x.device,dtype=x.dtype)
        return (torch.sort(x@directions,dim=0).values-torch.sort(y@directions,dim=0).values).abs().mean()
    if method=="MMD":
        dist=lambda a,b:torch.cdist(a,b,compute_mode="donot_use_mm_for_euclid_dist")
        return torch.exp(-dist(x,x)**2/(2*bandwidth**2)).mean()+torch.exp(-dist(y,y)**2/(2*bandwidth**2)).mean()-2*torch.exp(-dist(x,y)**2/(2*bandwidth**2)).mean()
    fn={"ED":losses.energy,"SD":losses.sd,"CSD":losses.csd,"GSD":losses.gsd}[method]
    return SCALES[method]*fn(x,y)


def glocal(x,y,labels,method,bandwidth,mode="glocal",rooted=False):
    def evaluate(a,b):
        value=objective(a,b,method,bandwidth)
        return value.clamp_min(1e-12).sqrt() if rooted and method in ["ED","SD"] else value
    global_value=evaluate(x,y)
    if mode=="global":return global_value
    local=[]
    for label in labels.unique():
        mask=labels==label
        if mask.sum()>=2:local.append(evaluate(x[mask],y[mask]))
    return (global_value+torch.stack(local).mean())/2 if local else global_value


def glocal_rooted(x,y,labels,method,bandwidth,mode="glocal"):
    """Root ED and SD in each global/local term, then average the terms."""
    return glocal(x,y,labels,method,bandwidth,mode,rooted=True)


def independent_metrics(real,generated,seed=2026,k=3,projections=128):
    # Equal budgets required; evaluations occur in the supplied full feature space.
    n=min(len(real),len(generated));real=real[:n];generated=generated[:n]
    rr=cdist(real,real);gg=cdist(generated,generated);rg=cdist(real,generated)
    np.fill_diagonal(rr,np.inf);np.fill_diagonal(gg,np.inf)
    kk=min(k,n-1);radii_r=np.partition(rr,kk-1,axis=1)[:,kk-1];radii_g=np.partition(gg,kk-1,axis=1)[:,kk-1]
    precision=np.any(rg<=radii_r[:,None],axis=0).mean();recall=np.any(rg<=radii_g[None,:],axis=1).mean()
    rng=np.random.default_rng(seed);directions=rng.normal(size=(real.shape[1],projections));directions/=np.linalg.norm(directions,axis=0)
    sw=np.abs(np.sort(real@directions,axis=0)-np.sort(generated@directions,axis=0)).mean()
    return {"n":n,"k":kk,"precision":float(precision),"recall":float(recall),"sliced_w1":float(sw)}
