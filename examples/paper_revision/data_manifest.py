#!/usr/bin/env python3
"""Audit the two TCGA inputs and reproduce landmark normalization from source CSVs.

Outputs contain sample identifiers and belong with the data provenance package;
raw expression and demographic fields are not exported.
"""
import sys,argparse,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from examples.paper_revision.common import save_json,digest

META=['full_barcode','tissue_type','cancer_type','reduced_barcode','cohort','cancer','barcode','gender','age']


def main(args):
    args.out.mkdir(parents=True,exist_ok=True)
    raw_tr=pd.read_csv(args.landmark/'df_train_tcga_landmark.csv',index_col=0);raw_te=pd.read_csv(args.landmark/'df_test_tcga_landmark.csv',index_col=0)
    tr=pd.read_csv(args.landmark/'tcga_landmark_train_norm.csv',index_col=0);te=pd.read_csv(args.landmark/'tcga_landmark_test_norm.csv',index_col=0)
    genes=[c for c in tr if c not in META];mu=raw_tr[genes].mean();sd=raw_tr[genes].std(ddof=1).clip(lower=1e-8)
    errors={split:float(np.nanmax(np.abs(((raw[genes]-mu)/sd).to_numpy()-norm[genes].to_numpy()))) for split,raw,norm in [('train',raw_tr,tr),('test',raw_te,te)]}
    manifest=pd.concat([frame[['full_barcode','tissue_type','cancer_type','cohort']].assign(split=split) for frame,split in [(tr,'train'),(te,'test')]],ignore_index=True)
    manifest.to_csv(args.out/'landmark_sample_manifest.csv',index=False)
    counts=manifest.groupby(['tissue_type','split']).size().unstack(fill_value=0);counts.to_csv(args.out/'landmark_cohort_counts.csv')
    pd.DataFrame({'gene':genes,'train_mean':mu,'train_std_ddof1':sd}).to_csv(args.out/'landmark_normalization.csv',index=False)
    hashes={p.name:digest(p) for p in args.landmark.glob('*.csv')}
    summary={'landmark':{'train_samples':len(tr),'test_samples':len(te),'genes':len(genes),'tissues':len(counts),'normalization':'Per-gene training mean and sample SD (ddof=1), SD clipped at 1e-8.',
               'normalization_max_error':errors,'train_test_barcode_overlap':len(set(tr.full_barcode)&set(te.full_barcode)),
               'patient_prefix_overlap':len(set(tr.full_barcode.astype(str).str[:12])&set(te.full_barcode.astype(str).str[:12])),
               'input_sha256':hashes,'source_accession_status':'Upstream acquisition/release not established by processed CSVs; verify with data owner.'}}
    # Stream the large general RNA matrix to calculate the original global variance ranking.
    count=0;sum_x=None;sum_x2=None;ids=[];nonfinite=0;minimum=float('inf');maximum=-float('inf')
    for chunk in pd.read_csv(args.rnaseq,index_col=0,chunksize=256):
        values=chunk.to_numpy(dtype=np.float64);nonfinite+=int((~np.isfinite(values)).sum());count+=len(chunk);ids.extend(chunk.index.astype(str))
        if sum_x is None:sum_x=np.zeros(values.shape[1]);sum_x2=np.zeros(values.shape[1]);columns=chunk.columns.tolist()
        sum_x+=values.sum(axis=0);sum_x2+=(values*values).sum(axis=0);minimum=min(minimum,float(np.nanmin(values)));maximum=max(maximum,float(np.nanmax(values)))
    variance=(sum_x2-sum_x*sum_x/count)/(count-1);top=np.argsort(-variance)[:256]
    pd.DataFrame({'sample_id':ids}).to_csv(args.out/'rnaseq_sample_manifest.csv',index=False)
    pd.DataFrame({'gene':[columns[i] for i in top],'variance':variance[top]}).to_csv(args.out/'rnaseq_top256_genes.csv',index=False)
    lookup={row.full_barcode:row.tissue_type for row in manifest.itertuples()}
    known=sum(i in lookup for i in ids)
    summary['rnaseq']={'samples':count,'genes':len(columns),'selected_features':256,'selection':'Highest sample variance across the supplied general RNA-seq matrix, reproducing the original illustration selection.',
                       'nonfinite_entries':nonfinite,'range':[minimum,maximum],'sha256':digest(args.rnaseq),'exact_barcode_matches_to_landmark':known,
                       'upstream_processing':'Expression units/transformation and original download release require acquisition provenance; numerical range alone is insufficient.'}
    summary['producing_source_sha256']=digest(__file__);save_json(args.out/'data_audit.json',summary)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--landmark',type=Path,required=True);p.add_argument('--rnaseq',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
