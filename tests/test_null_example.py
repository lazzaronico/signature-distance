import numpy as np
import torch
from examples.paper_revision.statistical_tests import compare,score
from scipy.spatial.distance import cdist
from sigdist.losses import sd,energy,csd


def test_statistics_match_independent_torch_losses_and_exact_common_permutations():
    x=np.array([[0.,1.],[1.,2.],[3.,-1.]]);y=np.array([[2.,0.],[4.,3.],[6.,1.]])
    result=compare(x,y,7,19);z=np.r_[x,y];D=cdist(z,z);bw=result['bandwidth'];K=np.exp(-D**2/(2*bw*bw))
    prng=np.random.default_rng(711);dirs=prng.normal(size=(2,128));dirs/=np.linalg.norm(dirs,axis=0);P=z@dirs
    rng=np.random.default_rng(19);expected=[]
    for _ in range(7):
        p=rng.permutation(6);expected.append(score(D,K,P,p[:3],p[3:]))
    np.testing.assert_allclose(result['null'],expected)
    tx,ty=torch.tensor(x),torch.tensor(y)
    np.testing.assert_allclose(result['observed'][:3],[energy(tx,ty).item(),sd(tx,ty).item(),csd(tx,ty).item()],rtol=1e-12)
    np.testing.assert_allclose(result['p'],(result['exceedances']+1)/8)
    assert np.all(result['holm']>=result['p']) and np.all(result['holm']<=1)
