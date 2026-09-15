import numpy as np
import pytest
import torch
from sigdist import SignatureDistance
from sigdist import losses


def test_point_mass_normalization():
    x=torch.tensor([[0.,0.]],dtype=torch.double);y=torch.tensor([[3.,4.]],dtype=torch.double)
    assert losses.sd(x,y).item()==pytest.approx(10.)
    assert losses.energy(x,y).item()==pytest.approx(10.)
    assert SignatureDistance(mode="signature").fit(x.numpy()).score(y.numpy())**2==pytest.approx(10.)


def test_exact_training_matches_comparison_and_directed_sum():
    torch.manual_seed(31)
    x=torch.randn(5,3,dtype=torch.double,requires_grad=True);y=torch.randn(7,3,dtype=torch.double)
    raw=losses.sd(x,y)
    assert raw.item()==pytest.approx(SignatureDistance(mode="signature").fit(x.detach().numpy()).score(y.numpy())**2,abs=1e-5)
    assert raw.item()==pytest.approx((losses.sdiv(x,y)+losses.sdiv(y,x)).item())
    assert losses.energy(x,y).item()<=raw.item()+1e-10
    for fn in [losses.sd,losses.cd,losses.csd,losses.gsd,losses.ediv]:
        grad=torch.autograd.grad(fn(x,y),x,retain_graph=True)[0]
        assert torch.isfinite(grad).all()


def test_energy_divergence_gradient_and_constant():
    torch.manual_seed(91);x=torch.randn(4,2,dtype=torch.double,requires_grad=True);y=torch.randn(6,2,dtype=torch.double)
    e=losses.energy(x,y);div=losses.ediv(x,y)
    torch.testing.assert_close(torch.autograd.grad(e,x,retain_graph=True)[0],torch.autograd.grad(div,x)[0])
    assert (div-e).item()==pytest.approx(torch.cdist(y,y).mean().item())
