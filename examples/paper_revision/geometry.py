#!/usr/bin/env python3
"""TCGA overview and interpolation with patient-disjoint construction/reference sets."""
import argparse,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from examples.paper_revision.common import save_json,digest
from examples.paper_revision.statistical_tests import score,NAMES


def scores(x,y,bw,directions):
    z=np.r_[x,y];D=cdist(z,z);K=np.exp(-D*D/(2*bw*bw));n=len(x)
    return score(D,K,z@directions,np.arange(n),np.arange(n,2*n))


def main(a):
    a.out.mkdir(parents=True,exist_ok=True)
    selected=pd.read_csv(a.audit/'rnaseq_top256_genes.csv').gene.tolist()
    # Only selected expression columns are loaded; original identifier is retained.
    first=pd.read_csv(a.rnaseq,nrows=0).columns[0];frame=pd.read_csv(a.rnaseq,usecols=[first]+selected,index_col=0)[selected]
    x=frame.to_numpy('float64');ids=frame.index.astype(str).to_numpy();patients=np.array([s[:12] for s in ids]);_,one=np.unique(patients,return_index=True)
    x=x[one];ids=ids[one];patients=patients[one]
    if not np.isfinite(x).all():raise ValueError('Nonfinite input')
    # Unweighted symmetrized kNN graph; simplify reciprocal edges consistently.
    import igraph as ig,leidenalg
    graph=kneighbors_graph(x,n_neighbors=10,include_self=False,mode='connectivity',n_jobs=1);graph=graph.maximum(graph.T)
    src,dst=graph.nonzero();edges=[(int(i),int(j)) for i,j in zip(src,dst) if i<j]
    g=ig.Graph(n=len(x),edges=edges,directed=False)
    partition=leidenalg.find_partition(g,leidenalg.RBConfigurationVertexPartition,resolution_parameter=1.,seed=42)
    labels=np.array(partition.membership);counts=np.bincount(labels);largest=int(counts.argmax())
    pca=PCA(n_components=2,svd_solver='full');coords=pca.fit_transform(x)
    manifest=pd.read_csv(a.landmark/'sample_manifest.csv');lookup={}
    for prefix,group in manifest.groupby(manifest.full_barcode.str[:15]):
        if group.tissue_type.nunique()==1:lookup[prefix]=group.tissue_type.iloc[0]
    tissues=np.array([lookup.get(s[:15],'unmatched') for s in ids])
    pd.DataFrame({'sample_id':ids,'patient':patients,'tissue':tissues,'leiden_cluster':labels}).to_csv(a.out/'sample_manifest.csv',index=False)
    np.savez_compressed(a.out/'geometry.npz',x=x,ids=ids,labels=labels,pca=coords,tissues=tissues,largest=largest)
    rng=np.random.default_rng(711);directions=rng.normal(size=(256,128));directions/=np.linalg.norm(directions,axis=0)
    alpha=np.linspace(0,1,11);values=[];controls=[];examples={}
    for rep in range(20):
        ix=np.random.default_rng(39000+rep).permutation(len(x))[:800];aa,bb,cc,dd=[x[ix[i*200:(i+1)*200]] for i in range(4)]
        _,perm=linear_sum_assignment(cdist(aa,bb));bb=bb[perm]
        D=cdist(np.r_[aa,bb],np.r_[aa,bb]);bw=float(np.median(D[D>0]))
        values.append(np.stack([scores((1-t)*aa+t*bb,cc,bw,directions) for t in alpha]));controls.append(scores(dd,cc,bw,directions))
        if rep==0:examples={'a':aa,'b':bb,'c':cc,'d':dd}
    np.savez_compressed(a.out/'interpolation.npz',alpha=alpha,values=values,real_real=controls,names=NAMES,**examples)
    save_json(a.out/'configuration.json',{'samples_original':len(frame),'patients_retained':len(x),'selection':'One first-listed specimen per patient, then top-variance 256 genes from original supplied matrix; no new expression transform.',
      'feature_selection_scope':'Descriptive full-cohort visualization; features are not held-out selected biomarkers.','graph_neighbors':10,'graph':'Unweighted union-symmetric graph, each edge once','leiden_resolution':1,'leiden_seed':42,'clusters':counts.tolist(),'largest_cluster':largest,
      'mapped_tissue_samples':int(np.sum(tissues!='unmatched')),'unmatched':int(np.sum(tissues=='unmatched')),'pca_explained_variance':pca.explained_variance_ratio_.tolist(),
      'interpolation':{'replicates':20,'n_per_set':200,'alpha':alpha.tolist(),'source_seed':39000,'sets':'Four disjoint patient subsets: two construction endpoints, held-out comparison and real-real control. Replicates can reuse patients.','matching':'Full Euclidean linear assignment','uncertainty':'Descriptive variation across random partitions; no point-level permutation p-values.','MMD_bandwidth':'Median positive distance in construction endpoints; held-out points excluded.'},
      'source_sha256':digest(__file__),'input_sha256':digest(a.rnaseq),'graph_versions':{'igraph':ig.__version__,'leidenalg':getattr(leidenalg,'__version__','unavailable')}})
    print('geometry and interpolation complete',counts.tolist(),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--rnaseq',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--landmark',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
