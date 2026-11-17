#!/usr/bin/env python3
"""Render paper figures and aggregate tables from explicit revision outputs."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from examples.paper_revision.common import METHODS,save_json
from examples.paper_revision.statistical_tests import NAMES,SCENARIOS

COLORS=['#6B7280','#2676B8','#C64D3C','#8E5DA5','#BA7922','#188A72']
MC=dict(zip(METHODS,COLORS));SC=['#2676B8','#C64D3C','#8E5DA5','#188A72','#BA7922'];DISPLAY=['Energy E','Signature S','CSD','Gaussian MMD²','Sliced W₁']
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.labelsize':9,'axes.titlesize':10,'legend.fontsize':8,'figure.dpi':140,'savefig.facecolor':'white','font.family':['Arial','DejaVu Sans'],'mathtext.fontset':'custom','mathtext.rm':'Arial','mathtext.it':'Arial:italic','mathtext.bf':'Arial:bold','mathtext.cal':'Arial:italic','mathtext.sf':'Arial','mathtext.tt':'Arial','pdf.fonttype':42,'ps.fonttype':42})


def save(fig,out,name):
    fig.savefig(out/f'{name}.pdf',bbox_inches='tight');fig.savefig(out/f'{name}.png',dpi=180,bbox_inches='tight');fig.savefig(out/f'{name}.tif',dpi=600,bbox_inches='tight',pil_kwargs={'compression':'tiff_lzw'})
    # Journal TIFFs require RGB without alpha; the source canvas is opaque white.
    with Image.open(out/f'{name}.tif') as source:
        if source.mode=='RGBA':assert source.getchannel('A').getextrema()==(255,255)
        rgb=source.convert('RGB')
    rgb.save(out/f'{name}.tif',dpi=(600,600),compression='tiff_lzw');plt.close(fig)


def wilson(k,n):
    p=k/n;z=1.95996398454;den=1+z*z/n;c=(p+z*z/(2*n))/den;h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return c-h,c+h


def statistics(a):
    rows=[];individual=[];permutation_examples=[];fig,axes=plt.subplots(2,5,figsize=(7.5,4.6),sharey=True)
    titles=['Same Gaussian','Shift = 2','Scale = 0.7','Shift = 0.25','Scale = 0.9']
    for i,n in enumerate([50,200]):
        for j,s in enumerate(SCENARIOS):
            files=sorted((a.work/'statistics').glob(f'n{n}_{s}_*.npz'))
            if len(files)!=50:raise ValueError('Statistical campaign incomplete')
            for rep,f in enumerate(files):
                d=np.load(f)
                for q,name in enumerate(NAMES):
                    individual.append({'n':n,'scenario':s,'replicate':rep,'seed':int(d['seed']),'metric':name,'observed':d['observed'][q],'p':d['p'][q],'holm':d['holm'][q],'exceedances':int(d['exceedances'][q]),'permutations':len(d['null']),'bandwidth':float(d['bandwidth'])})
            p=np.stack([np.load(f)['p'] for f in files]);k=(p<=.05).sum(0);lo,hi=wilson(k,50);ax=axes[i,j];rate=k/50
            ax.errorbar(np.arange(5),rate,yerr=[rate-lo,hi-rate],fmt='none',color='#555',capsize=2,lw=.8)
            ax.scatter(np.arange(5),rate,c=SC,s=28,zorder=3);ax.axhline(.05,color='#999',ls=':',lw=.8);ax.set_ylim(-.03,1.05);ax.set_xticks(range(5),['E','S','CSD','MMD','SW'],rotation=45);ax.set_title(titles[j])
            if j==0:ax.set_ylabel(f'n = {n}\nRejection fraction')
            for q,name in enumerate(NAMES):rows.append({'n':n,'scenario':s,'metric':name,'rejections':int(k[q]),'replicates':50,'fraction':rate[q],'wilson_lower':lo[q],'wilson_upper':hi[q]})
    fig.tight_layout();save(fig,a.out,'fig02_statistics');pd.DataFrame(rows).to_csv(a.out/'statistical_power.csv',index=False);pd.DataFrame(individual).to_csv(a.out/'statistical_individual_pairs.csv',index=False)
    fig,axes=plt.subplots(5,3,figsize=(7.5,9))
    pvalues=[]
    for j,s in enumerate(list(SCENARIOS)[:3]):
        data=np.load(a.work/'statistics'/f'n50_{s}_000.npz')
        for i,name in enumerate(NAMES):
            ax=axes[i,j];ax.hist(data['null'][:,i],bins=32,color=SC[i],alpha=.7);ax.axvline(data['observed'][i],color='black',ls='--',lw=1)
            ax.set_title(f"{titles[j]} · p={data['p'][i]:.3f}")
            if j==0:ax.set_ylabel(DISPLAY[i]+'\nPermutations')
            permutation_examples.extend({'scenario':s,'metric':name,'permutation':b,'value':value} for b,value in enumerate(data['null'][:,i]))
            pvalues.append({'scenario':s,'metric':name,'p':float(data['p'][i]),'holm':float(data['holm'][i])})
    fig.tight_layout();save(fig,a.out,'figS02_permutation_examples');pd.DataFrame(pvalues).to_csv(a.out/'illustrative_pvalues.csv',index=False);pd.DataFrame(permutation_examples).to_csv(a.out/'illustrative_permutation_values.csv',index=False)


def geometry(a):
    manifest=pd.read_csv(a.work/'patient_split/sample_manifest.csv')
    manifest.groupby(['cohort','tissue_type']).size().rename('retained_samples').reset_index().to_csv(a.out/'cohort_label_mapping.csv',index=False)
    z=np.load(a.work/'geometry/geometry.npz');v=np.load(a.work/'geometry/interpolation.npz');cfg=json.loads((a.work/'geometry/configuration.json').read_text());values=v['values'].copy();real_real=v['real_real'].copy()
    # Energy and signature columns are shown on the rooted scale (d_E and SD), as in the main interpolation figure.
    values[:,:,:2]=np.sqrt(np.clip(values[:,:,:2],0,None));real_real[:,:2]=np.sqrt(np.clip(real_real[:,:2],0,None));ratio=values/real_real[:,None,:];titles=['ED','SD']+list(DISPLAY[2:])
    individual=[{'partition':rep,'alpha':alpha,'metric':name,'score':values[rep,j,i],'real_real':real_real[rep,i],'ratio':ratio[rep,j,i]} for rep in range(len(ratio)) for j,alpha in enumerate(v['alpha']) for i,name in enumerate(NAMES)]
    pd.DataFrame(individual).to_csv(a.out/'interpolation_individual_partitions.csv',index=False)
    fig,axes=plt.subplots(1,5,figsize=(7.5,2.8))
    rows=[]
    for i,name in enumerate(NAMES):
        q=ratio[:,:,i];m=q.mean(0);sd=q.std(0,ddof=1);ax=axes[i];ax.plot(v['alpha'],m,c=SC[i]);ax.fill_between(v['alpha'],m-sd,m+sd,color=SC[i],alpha=.2);ax.axhline(1,color='#777',ls=':');ax.set_title(titles[i]);ax.set_xlabel('Interpolation α');
        if i==0:ax.set_ylabel('Score / real–real score')
        for j,t in enumerate(v['alpha']):rows.append({'metric':name,'alpha':t,'mean_ratio':m[j],'partition_sd':sd[j]})
    fig.tight_layout();save(fig,a.out,'fig04_interpolation');pd.DataFrame(rows).to_csv(a.out/'interpolation_summary.csv',index=False)
    fig=plt.figure(figsize=(7.2,6.5));grid=fig.add_gridspec(2,2);ax=[fig.add_subplot(grid[0,0]),fig.add_subplot(grid[:,1]),fig.add_subplot(grid[1,0])];tissues=z['tissues'];counts_t=pd.Series(tissues[tissues!='unmatched']).value_counts();major=counts_t.index[:5].tolist()
    cmap=plt.get_cmap('tab10')
    for t,color in [('Other linked labels','#C8C8C8'),('unmatched','#EEEEEE')]+[(t,cmap(i)) for i,t in enumerate(major)]:
        mask=(~np.isin(tissues,major))&(tissues!='unmatched') if t=='Other linked labels' else tissues==t
        ax[0].scatter(*z['pca'][mask].T,s=3,c=[color],alpha=.7,rasterized=True,label=t)
    handles,labels=ax[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.01),ncol=4,fontsize=8,markerscale=3,frameon=False,title='Linked cohort labels (panel a)',title_fontsize=8)
    ax[0].set_xlabel(f"PC1 ({100*cfg['pca_explained_variance'][0]:.1f}%)");ax[0].set_ylabel(f"PC2 ({100*cfg['pca_explained_variance'][1]:.1f}%)");ax[0].set_title('a. TCGA cohort PCA')
    counts=pd.read_csv(a.work/'patient_split/cohort_counts.csv').set_index('tissue_type');counts=counts.loc[counts.sum(axis=1).sort_values().index];ax[1].barh(counts.index,counts['fit'],label='Fit',color='#2676B8');ax[1].barh(counts.index,counts['validation'],left=counts['fit'],label='Validation',color='#94B7D4');ax[1].barh(counts.index,counts['test'],left=counts['fit']+counts['validation'],label='Test',color='#C64D3C');ax[1].tick_params(axis='y',labelsize=9);ax[1].set_xlabel('Landmark samples');ax[1].set_title('b. Landmark partitions');ax[1].legend(fontsize=8)
    anchor=v['c'][0];u=(np.arange(200)+.5)/200
    for data,label,color in [(v['a'],'Endpoint A','#2676B8'),((v['a']+v['b'])/2,'Interpolated midpoint','#C64D3C'),(v['c'],'Held-out reference','#188A72')]:ax[2].plot(u,np.sort(np.linalg.norm(data-anchor,axis=1)),label=label,c=color)
    ax[2].set_xlabel('Radial quantile');ax[2].set_ylabel('Euclidean distance (256 genes)');ax[2].set_title('c. Radial profiles');ax[2].legend(fontsize=8);fig.tight_layout(rect=(0,.13,1,1));save(fig,a.out,'figS01_tcga_overview')


def main(a):
    a.out.mkdir(parents=True,exist_ok=True)
    if a.part in ['all','statistics']:statistics(a)
    if a.part in ['all','geometry']:geometry(a)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--part',choices=['all','statistics','geometry'],default='all');main(p.parse_args())
