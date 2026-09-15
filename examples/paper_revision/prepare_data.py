#!/usr/bin/env python3
"""Deduplicate TCGA aliquots, resolve the ESCA/STES overlap, split by patient.

The target labels retain the supplied annotation scheme. They are cohort-derived
labels, not a new clinical reannotation. Combined STES copies of ESCA aliquots
are dropped in favour of the specific ESCA record with identical expression.
"""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from examples.paper_revision.common import save_json,digest
from examples.paper_revision.data_manifest import META


def prepare(source,out):
    out.mkdir(parents=True,exist_ok=True)
    paths=[source/"df_train_tcga_landmark.csv",source/"df_test_tcga_landmark.csv"]
    original=pd.concat([pd.read_csv(p,index_col=0) for p in paths],ignore_index=True)
    genes=[c for c in original if c not in META]
    if not np.isfinite(original[genes].to_numpy()).all():raise ValueError("Nonfinite raw expression")
    keep=[];conflicts=0
    for barcode,g in original.groupby('full_barcode',sort=True):
        if not np.allclose(g[genes],g[genes].iloc[0],rtol=0,atol=1e-10):raise ValueError("Conflicting expression for duplicate aliquot")
        if g.tissue_type.nunique()>1:
            if set(g.cohort)!={'ESCA','STES'} or set(g.tissue_type)!={'esophagus','stomach'}:raise ValueError("Unexpected label conflict")
            keep.append(g[g.cohort=='ESCA'].index[0]);conflicts+=1
        else:keep.append(g.index[0])
    data=original.loc[keep].copy().reset_index(drop=True)
    patients=data.full_barcode.str[:12]
    if data.assign(patient=patients).groupby('patient').tissue_type.nunique().max()!=1:raise ValueError("Conflicting patient labels")
    groups=data.assign(patient=patients).drop_duplicates('patient').sort_values('patient')
    tr,te=train_test_split(groups.patient.to_numpy(),test_size=.2,stratify=groups.tissue_type,random_state=42)
    sub=groups.set_index('patient').loc[tr]
    fit,val=train_test_split(tr,test_size=.3,stratify=sub.tissue_type,random_state=42)
    assignments={**dict.fromkeys(fit,'fit'),**dict.fromkeys(val,'validation'),**dict.fromkeys(te,'test')}
    split=patients.map(assignments);fitmask=split=='fit'
    mu=data.loc[fitmask,genes].mean();sd=data.loc[fitmask,genes].std(ddof=1).clip(lower=1e-8)
    norm=data.copy();norm[genes]=(data[genes]-mu)/sd
    outer=norm.loc[split!='test'].reset_index(drop=True);test=norm.loc[split=='test'].reset_index(drop=True)
    outer.to_csv(out/'tcga_landmark_train_norm.csv');test.to_csv(out/'tcga_landmark_test_norm.csv')
    inner=split[split!='test'].to_numpy()
    save_json(out/'split.json',{'fit_indices':np.flatnonzero(inner=='fit').tolist(),'validation_indices':np.flatnonzero(inner=='validation').tolist(),'seed':42,'unit':'TCGA participant (first 12 barcode characters)'})
    manifest=data[['full_barcode','tissue_type','cohort']].copy();manifest['patient']=patients;manifest['split']=split
    manifest.to_csv(out/'sample_manifest.csv',index=False)
    counts=manifest.groupby(['tissue_type','split']).size().unstack(fill_value=0);counts.to_csv(out/'cohort_counts.csv')
    pd.DataFrame({'gene':genes,'fit_mean':mu.to_numpy(),'fit_sample_sd':sd.to_numpy()}).to_csv(out/'normalization.csv',index=False)
    report={'input_rows':len(original),'unique_aliquots':len(data),'duplicate_rows_removed':len(original)-len(data),'ESCA_STES_label_conflicts_resolved':conflicts,
      'patients':len(groups),'samples_by_split':split.value_counts().to_dict(),'patients_by_split':groups.patient.map(assignments).value_counts().to_dict(),'tissues':len(counts),
      'normalization':'Fit subset only, mean and sample SD ddof=1, clipped at 1e-8; fixed for validation and test.',
      'patient_overlap':0,'source_hashes':{p.name:digest(p) for p in paths},'producing_source':digest(__file__),
      'labels':'Supplied cohort-derived 24-label task retained; ESCA aliquots duplicated in aggregate STES use the specific ESCA annotation. This is not a new clinical reannotation.',
      'upstream_release':'Requires acquisition provenance confirmation by authors.'}
    save_json(out/'preprocessing.json',report)
    print(report)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();prepare(a.source,a.out)
