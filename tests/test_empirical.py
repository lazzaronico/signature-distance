import numpy as np
import pytest
import torch
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist
from sigdist import SignatureDistance
from sigdist._quantiles import row_wasserstein


@pytest.mark.parametrize("n,m", [(1, 7), (3, 5), (9, 4), (8, 8)])
def test_quantile_integral_against_scipy(n, m):
    rng = np.random.default_rng(81)
    a = rng.normal(size=(5, n)); b = rng.normal(size=(5, m))
    actual = row_wasserstein(torch.tensor(a), torch.tensor(b)).numpy()
    expected = [wasserstein_distance(x, y) for x, y in zip(a, b)]
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_full_statistic_unequal_sizes_duplicates_and_energy_bound():
    x = np.array([[0., 0.], [0., 0.], [1., 2.]])
    y = np.array([[2., 0.], [1., 1.], [4., 2.], [2., 2.], [-1., 0.]])
    xx, xy, yy = cdist(x,x), cdist(x,y), cdist(y,y)
    raw = (np.mean([wasserstein_distance(a,b) for a,b in zip(xx,xy)]) +
           np.mean([wasserstein_distance(a,b) for a,b in zip(yy,xy.T)]))/2
    model = SignatureDistance(mode="signature").fit(x)
    assert model.score(y)**2 == pytest.approx(2*raw, abs=1e-6)
    assert model.energy_distance(y)**2/2 <= raw + 1e-6
    assert model.score(x) == pytest.approx(0., abs=1e-7)
    assert SignatureDistance(mode="signature").fit(y).score(x) == pytest.approx(model.score(y))
    # Duplicating every atom preserves the empirical distribution.
    assert model.score(np.repeat(y, 2, axis=0)) == pytest.approx(model.score(y), abs=1e-6)


def test_quantile_gradients_away_from_ties():
    a = torch.tensor([[.2, 1.3, 4.1]], dtype=torch.double, requires_grad=True)
    b = torch.tensor([[.4, 2.2]], dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(row_wasserstein, (a, b))


def test_seeded_permutation_null_preserves_both_sample_sizes():
    x = np.array([[0.], [1.], [3.]])
    y = np.array([[2.], [4.], [6.], [7.], [8.]])
    model = SignatureDistance(mode="signature").fit(x)
    null, observed, p = model.pvalue(y, n_permutations=9, random_state=19)
    rng = np.random.default_rng(19); pool = np.vstack([x,y]); expected = []
    for _ in range(9):
        perm = rng.permutation(len(pool))
        expected.append(SignatureDistance(mode="signature").fit(pool[perm[:3]]).score(pool[perm[3:]]))
    np.testing.assert_allclose(null, expected, atol=1e-6)
    assert observed == pytest.approx(model.score(y))
    assert p == pytest.approx((np.sum(null >= observed)+1)/10)
    np.testing.assert_array_equal(null, model.pvalue(y,9,random_state=19)[0])


def test_batching_preserves_score():
    rng = np.random.default_rng(4); x = rng.normal(size=(31,3)); y = rng.normal(size=(27,3))
    a = SignatureDistance(mode="signature").fit(x).score(y)
    b = SignatureDistance(mode="signature",batch_size=7).fit(x).score(y)
    assert a == pytest.approx(b, abs=1e-6)


def test_full_profile_interface_rejects_inactive_truncation_option():
    with pytest.raises(ValueError,match="k=0"):
        SignatureDistance(mode="signature",k=3)
    with pytest.raises(ValueError,match="Unknown"):
        SignatureDistance(mode="unknown")
