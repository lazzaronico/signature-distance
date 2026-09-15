import numpy as np
import torch
from examples.paper_revision.common import objective,glocal_rooted
from sigdist import losses


def test_rooted_glocal_applies_root_before_averaging():
    x=torch.tensor([[0.,0.],[1.,0.],[3.,2.],[4.,1.]],dtype=torch.double,requires_grad=True)
    y=torch.tensor([[.4,.2],[1.3,.1],[2.1,1.4],[3.3,1.1]],dtype=torch.double)
    labels=torch.tensor([0,0,1,1])
    for method,fn in [('SD',losses.sd),('ED',losses.energy)]:
        terms=[fn(x,y),fn(x[:2],y[:2]),fn(x[2:],y[2:])]
        expected=(terms[0].sqrt()+(terms[1].sqrt()+terms[2].sqrt())/2)/2
        actual=glocal_rooted(x,y,labels,method,1.)
        torch.testing.assert_close(actual,expected)
        torch.testing.assert_close(torch.autograd.grad(actual,x,retain_graph=True)[0],torch.autograd.grad(expected,x,retain_graph=True)[0])


def test_sliced_wasserstein_keeps_the_paper_direction_bank_and_rng_state():
    torch.manual_seed(12);x=torch.randn(9,2,requires_grad=True);y=torch.randn(9,2);state=torch.get_rng_state().clone()
    directions=torch.randn(2,128,generator=torch.Generator().manual_seed(8273));directions/=torch.linalg.vector_norm(directions,dim=0)
    expected=(torch.sort(x@directions,dim=0).values-torch.sort(y@directions,dim=0).values).abs().mean()
    actual=objective(x,y,'SW1',1.)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert torch.equal(state,torch.get_rng_state())
    torch.testing.assert_close(torch.autograd.grad(actual,x,retain_graph=True)[0],torch.autograd.grad(expected,x)[0],rtol=0,atol=0)
