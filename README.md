# Signature Distance

Signature Distance compares empirical distributions through radial-distance profiles.
This repository contains the library and experiments for
[Signature Distance: Generalizing Energy Statistics](https://doi.org/10.1101/2024.10.23.619602).

With raw signature statistic S and raw energy expression E, SD = √S and
E ≤ S ≤ 2W₁. SD is symmetric and separating; no general triangle-inequality
guarantee is assumed. See [formulas and conventions](docs/formulation.md).

## Install

```sh
python -m pip install -e '.[examples,test]'
python -m pytest
```

Python 3.10 or later is required. The [experiment guide](examples/paper_revision/README.md)
gives the paper's environment, inputs and reproduction commands.

## Compare empirical distributions

```python
import numpy as np
from sigdist import SignatureDistance

rng = np.random.default_rng(42)
X = rng.normal(size=(100, 3))
Y = rng.normal(size=(150, 3))
comparison = SignatureDistance(mode="signature", device="cpu").fit(X)
sd = comparison.score(Y)                # sqrt(S)
ed = comparison.energy_distance(Y)      # sqrt(E)
null, observed, p = comparison.pvalue(Y, n_permutations=999, random_state=42)
```

All empirical atoms, including zero self-distances, are retained. Unequal sample
sizes use exact integration over empirical quantile intervals. The two anchor
averages are summed, giving the two distributions equal weight irrespective of
sample counts. Permutation tests preserve group sizes and use plus-one p-values.

For comparable sample sizes n in dimension d, direct geometry costs O(n²d),
row sorting O(n² log n), and distance storage O(n²).

## Differentiable objectives

```python
from sigdist.losses import sd, sdiv, csd, gsd, energy, ediv

loss = sd(generated, real)  # raw S
loss.backward()
```

- `sd(X,Y)` returns S; the class above returns √S.
- `sdiv(X,Y)` uses Y anchors, i.e. mathematical SDiv(Y || X).
- `csd` and `gsd` return the rooted composite objectives defined in the paper.
- `energy` returns E = 2C − A − B. `ediv` returns 2C − A, omitting the
  real-only constant while retaining all empirical self-distances.

## Reproduce and cite

[Paper experiments](examples/paper_revision/README.md) include statistical tests,
repeated synthetic and TCGA training, population expansion, plotting commands,
and the recorded per-run results used in the paper. TCGA inputs are supplied
separately; acquisition requirements are described in the guide.

Cite this software using [CITATION.cff](CITATION.cff), and the paper as:
Nicolò Lazzaro et al. *Signature Distance: Generalizing Energy Statistics*.
bioRxiv. https://doi.org/10.1101/2024.10.23.619602.

The software is distributed under the [MIT license](LICENSE).
