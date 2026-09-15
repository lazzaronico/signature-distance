"""Export the paper's repeated-run tables and accuracy/recall figure."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from examples.paper_revision.common import save_json

GROUPS={'toys':42,'tcga_glocal':25,'tcga_mmd':5,'tcga_global':25,'expansion':15}
METHODS=['MSE','ED','SD','CSD','GSD'];COLORS=['#888888','#4a90d9','#e85d4a','#9b59b6','#e67e22']


def load_records(work,recorded):
    if recorded:
        root=Path(__file__).with_name('reference_results')
        return {key:json.loads((root/f'{key}.json').read_text()) for key in GROUPS}
    groups={}
    for key,count in GROUPS.items():
        paths=sorted((work/key).glob('*_bootstrap*.json' if key=='expansion' else '*/result.json'))
        if len(paths)!=count:raise ValueError(f'{key}: expected {count} results, found {len(paths)}')
        groups[key]=[json.loads(p.read_text()) for p in paths]
    return groups


def frame(records):
    rows=[]
    for r in records:
        row={k:r[k] for k in ['method','seed','task','accuracy','macro_f1','selected_step','bootstrap','radial_absolute_error','residual_sd_narrow','residual_sd_wide','out_of_domain_fraction'] if k in r}
        row.update(r['independent']);rows.append(row)
    return pd.DataFrame(rows)


def main(a):
    a.out.mkdir(parents=True,exist_ok=True);groups=load_records(a.work,a.recorded)
    tables={key:frame(rows) for key,rows in groups.items()}
    for key,df in tables.items():
        df.to_csv(a.out/f'{key}_individual.csv',index=False)
        keys=['task','method'] if key=='toys' else ['method']
        numeric=[c for c in df.select_dtypes(include='number') if c not in ['seed','bootstrap','n','k']]
        df.groupby(keys,sort=False)[numeric].agg(['mean','std']).to_csv(a.out/f'{key}_summary.csv')
    df=tables['tcga_glocal'];summary={}
    for method in METHODS:
        rr=df[df.method==method]
        if sorted(rr.seed.tolist())!=list(range(42,47)):raise ValueError(f'Unexpected seeds for {method}')
        summary[method]={key:{'mean':float(rr[key].mean()),'std':float(rr[key].std(ddof=1))} for key in ['accuracy','macro_f1','precision','recall','sliced_w1']}
    plt.rcdefaults();sns.set_style('whitegrid');plt.rcParams.update({'figure.dpi':150,'font.size':10,'savefig.bbox':'tight','savefig.pad_inches':.1})
    fig,axes=plt.subplots(1,2,figsize=(10,5))
    for ax,key,title,label in zip(axes,['accuracy','recall'],['Downstream Classification Accuracy','Distributional Recall'],['Test Accuracy (%)','Recall (%)']):
        values=np.array([summary[m][key]['mean']*100 for m in METHODS]);sd=np.array([summary[m][key]['std']*100 for m in METHODS]);bars=ax.bar(METHODS,values,color=COLORS)
        ax.set_ylabel(label);ax.set_title(title,fontweight='bold');ax.errorbar(range(5),values,yerr=sd,fmt='none',ecolor='#333333',capsize=3)
        for bar,value,error in zip(bars,values,sd):ax.text(bar.get_x()+bar.get_width()/2,value+error+.5,f'{value:.1f}',ha='center',va='bottom',fontsize=10,fontweight='bold')
        ax.set_ylim(0,max(values+sd)*1.08+1)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(a.out/f'fig12_accuracy_32d.{ext}',dpi=150,bbox_inches='tight')
    plt.close(fig)
    # Per-tissue classifier F1 uses the same 25 fixed-validation glocal runs.
    tissues=sorted(groups['tcga_glocal'][0]['f1_per_tissue'])
    values=np.array([[np.mean([r['f1_per_tissue'][t] for r in groups['tcga_glocal'] if r['method']==m]) for t in tissues] for m in METHODS])
    pd.DataFrame(values,index=METHODS,columns=tissues).to_csv(a.out/'tcga_tissue_f1.csv')
    fig,ax=plt.subplots(figsize=(16,4));sns.heatmap(values,ax=ax,annot=True,fmt='.2f',cmap='YlOrRd',vmin=0,vmax=1,xticklabels=tissues,yticklabels=METHODS,cbar_kws={'label':'F1 score'})
    ax.set_xlabel('Tissue');ax.set_ylabel('Loss');fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(a.out/f'fig12_heatmap_32d.{ext}')
    plt.close(fig);save_json(a.out/'tcga_glocal_summary.json',summary)
    save_json(a.out/'export.json',{'source':'recorded paper results' if a.recorded else 'supplied experiment outputs','counts':{k:len(v) for k,v in groups.items()}})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path);p.add_argument('--recorded',action='store_true');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if not a.recorded and a.work is None:p.error('provide --work or --recorded')
    main(a)
