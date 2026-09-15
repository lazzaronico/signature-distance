import json,os,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd


def test_complete_tissue_pipeline_on_small_synthetic_input(tmp_path):
    rng=np.random.default_rng(4);data=tmp_path/'data';data.mkdir()
    for split,n in [('train',60),('test',12)]:
        frame=pd.DataFrame(rng.normal(size=(n,8)),columns=[f'gene{i}' for i in range(8)]);frame['tissue_type']=['a','b','c']*(n//3);frame.to_csv(data/f'tcga_landmark_{split}_norm.csv')
    (data/'split.json').write_text(json.dumps({'fit_indices':list(range(42)),'validation_indices':list(range(42,60))}))
    (data/'preprocessing.json').write_text(json.dumps({'synthetic_test_only':True}))
    script=Path(__file__).parents[1]/'examples/paper_revision/train_tcga.py'
    cmd=[sys.executable,str(script),'--data',str(data),'--out',str(tmp_path/'out'),'--method','SD','--seed','13','--threads','1','--max-steps','2','--classifier-epochs','1']
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    run=subprocess.run(cmd,cwd=tmp_path,env=env,capture_output=True,text=True,timeout=90)
    assert run.returncode==0,run.stderr+run.stdout
    dest=tmp_path/'out/glocal_sd_seed13';result=json.loads((dest/'result.json').read_text());assert np.isfinite(result['accuracy']);assert result['generator_steps']==2
    config=json.loads((dest/'config.json').read_text());assert config['seed']==13;assert config['loss_multiplier']==1.;assert 'split.json' in config['preprocessing_hashes']
