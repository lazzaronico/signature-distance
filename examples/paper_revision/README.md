# Paper experiments

Run commands from the repository root after installing `.[examples,test]`.
`requirements-paper.txt` records the paper's Python 3.13 environment. Each runner
writes configurations and results under the supplied output directory. Training
uses complete empirical profiles, including self-distances.

## Statistical comparisons

```sh
python examples/paper_revision/statistical_tests.py --out outputs/statistics --workers 2
python examples/paper_revision/render.py --work outputs --out outputs/figures --part statistics
python -m examples.paper_revision.illustrations.concept --out outputs/figures
python -m examples.paper_revision.illustrations.permutation --statistics outputs/statistics --out outputs/figures
```

The repeated comparisons use 50 independent pairs per setting, n=50 and n=200,
five Gaussian scenarios and 999 shared pooled permutations. Gaussian MMD² uses
the pooled median positive distance as bandwidth; sliced W₁ uses 128 fixed
Gaussian directions. Both remain fixed across label permutations. Outputs
include p-values, Holm adjustments, rejection rates and Wilson intervals.
The main sensitivity illustration displays the first n=50 pair; the complete
repeated results accompany it. The concept figure has two reference-only
self-bootstrap panels and one symmetric SD permutation panel, with bootstrap
confidence intervals for the observed statistics.

## TCGA inputs and preprocessing

Two processed inputs are required: the landmark train/test CSVs before gene
standardization (`df_train_tcga_landmark.csv`, `df_test_tcga_landmark.csv`, 978
gene columns), and `tcga_rnaseq.csv` (9,732 specimens × 17,653 genes). Landmark
metadata fields are listed in `prepare_data.py`; `tissue_type` supplies labels
and `full_barcode` identifies aliquots and patients. These inputs are not bundled
in this software release. The source study uses TCGA data from the
[GDC Data Portal](https://portal.gdc.cancer.gov/). Exact acquisition releases and
expression transformations must accompany the supplied matrices; they cannot
be inferred from their values. The scripts reproduce downstream processing,
not acquisition of equivalent matrices directly from GDC.

```sh
python examples/paper_revision/prepare_data.py --source /path/to/landmark --out outputs/patient_split
python examples/paper_revision/data_manifest.py --landmark /path/to/landmark --rnaseq /path/to/tcga_rnaseq.csv --out outputs/audit
```

The manifest command also reads the supplied standardized landmark CSVs
(`tcga_landmark_train_norm.csv`, `tcga_landmark_test_norm.csv`) to check their
normalization. Subsequent experiments use the output of `prepare_data.py`.
Preprocessing deduplicates aliquots, resolves overlapping ESCA/STES annotations
and partitions patients into fitting, validation and test sets (5,548, 2,378
and 1,994 observations). Gene means and sample standard deviations use fitting
data only. Generated manifests include identifiers and should stay with the
input data rather than in the public code repository.

## Generative comparisons

```sh
for seed in 42 43 44; do
  for method in MSE ED SD CSD GSD MMD SW1; do
    for task in circles sine; do
      python examples/paper_revision/train_toys.py --task "$task" --method "$method" --seed "$seed" --out outputs/toys --threads 1
    done
  done
done
for seed in 42 43 44 45 46; do
  for method in MSE ED SD CSD GSD; do
    python examples/paper_revision/train_tcga_fixed.py --data outputs/patient_split --method "$method" --seed "$seed" --out outputs/tcga_glocal --threads 2
    python examples/paper_revision/train_tcga_fixed.py --data outputs/patient_split --method "$method" --mode global --seed "$seed" --out outputs/tcga_global --threads 2
  done
  python examples/paper_revision/train_tcga.py --data outputs/patient_split --method MMD --mode glocal --seed "$seed" --out outputs/tcga_mmd --threads 2
done
for seed in 42 43 44; do
  for method in ED SD; do
    for task in circles sine; do
      python examples/paper_revision/train_toys.py --task "$task" --method "$method" --rooted --seed "$seed" --out outputs/toys_rooted --threads 1
    done
  done
done
```

These commands produce 109 fits. Independent fits can run concurrently with
bounded CPU threads. TCGA runners resume checkpoints in the same output folder;
keep code and configuration unchanged for a resumed run.

| Experiment | Architecture | Optimizer and budget | Objective / validation |
|---|---|---|---|
| Rings (21 fits) | 5×128 ReLU, latent 2 and class input | Adam 1e-3, batch 256, 10,000 updates | Raw E/S, CSD/GSD, MSE, MMD² or sliced W₁ |
| Sine (21 fits) | 2×128 GELU, two inputs and outputs | Adam 3e-4, 1,000 epochs × 7 full batches | Same seven objectives |
| Main glocal TCGA (25 fits) | 5×512 LayerNorm/leaky-ReLU, latent 32, 24 labels | Adam 1e-3, batch 256, at most 100,000 updates | Rooted ED/SD, CSD/GSD or MSE; fixed validation |
| Rooted synthetic repeat (12 fits) | As rings and sine | As rings and sine | Rooted ED/SD |
| Global-only TCGA (25 fits) | Same generator | Same optimizer and maximum budget | As the main comparison, without per-tissue terms |
| Glocal MMD (5 fits) | Same generator | Same optimizer and maximum budget | MMD²; stochastic validation |
| Synthetic-trained classifier | 3×256 LayerNorm/ReLU | Adam 1e-3, at most 500 epochs, patience 50 | Cross-entropy |

No compensating objective multipliers are used. In the main glocal comparison,
ED and SD are rooted within each global/local term, then averaged. Four fixed
validation batches of 256 points and latent vectors are shared across losses
within each seed, evaluated every 25 updates with patience 200 checks. The
global-only control uses the same objectives, seeds and validation batches and
omits the per-tissue terms. The additional MMD comparison uses fresh validation
points/noise every five updates and patience 1,000 checks; its stopping rule
therefore differs from the main comparison. On the synthetic tasks ED and SD
optimize raw E and S; the rooted repeat gives the same independent measures
within the variation across runs.

MMD bandwidth is fixed from at most 1,024 fitting observations. The synthetic
sliced-W₁ loss uses 128 normalized Gaussian directions (seed 8273). Evaluation
uses a separate 128-direction bank (seed 2026), k=3 nearest-neighbour-ball
precision/recall, equal generated/test budgets and real–real controls in the
full feature space. The classifier trains on 10 times the outer training-pool
size in generated points. Sine evaluation uses held-out covariates shared by
real observations and generator inputs, with fresh latent noise. Figures show
seed 42 and tables retain every repeat; error bars across seeds are sample
standard deviations, not patient confidence intervals.

## Biological geometry and population expansion

```sh
python examples/paper_revision/geometry.py --rnaseq /path/to/tcga_rnaseq.csv --audit outputs/audit --landmark outputs/patient_split --out outputs/geometry
python examples/paper_revision/interpolation_destination.py --rnaseq /path/to/tcga_rnaseq.csv --audit outputs/audit --out outputs/figures
python examples/paper_revision/expansion.py --geometry outputs/geometry --out outputs/expansion
python examples/paper_revision/render.py --work outputs --out outputs/figures --part geometry
python examples/paper_revision/render_samples.py --work outputs --out outputs/figures
```

Main interpolation preserves specimen order, selects 400 points per endpoint,
uses full-dimensional optimal assignment and compares interpolants with the
destination. Each curve is divided by its own starting value and reaches zero
at the destination. Its five-method supplement uses the same construction.
The separate held-out interpolation analysis uses 20 four-way patient-disjoint
partitions, 200 points per set, and a real–real control; its energy and
signature ratios are rooted, as in the main interpolation figure.

Geometry uses one specimen per patient, the 256 most variable genes, a union
symmetric 10-neighbour graph and Leiden resolution 1. The largest cluster has
931 patients. This full-cohort feature/cluster selection is descriptive.
Expansion uses three partitions of 310: a seed set that guides the update, a
validation set that selects the stopping step and a held-out set that verifies
it. Five bootstrap seed populations are each initialized from 31 k-means
centroids. Each trajectory runs 1,000 steps with step size 0.1, noise 0.1,
covariance regularization 1e-6 I and trace normalization. The guidance is the
gradient of raw E, S or S + C on the complete population, multiplied by the
population size so that each point receives the gradient of its own term; a
fourth chain has no guidance. The chains of a bootstrap share their initial
population and noise draws. Each chain stops at the multiple of 25 steps with
the smallest distance to the validation set in the measure of its guidance.
The figure displays bootstrap 0 at its selected step. Held-out distances do not
select anything, and generated points are not removed.

## Recorded results and figure mapping

```sh
python examples/paper_revision/render_training.py --recorded --out outputs/tables
# To summarize a newly completed campaign instead:
python examples/paper_revision/render_training.py --work outputs --out outputs/tables
```

`reference_results` contains the paper's individual-run training and expansion
outcomes and configurations. The first command regenerates their tables,
classification-accuracy/recall figure and per-tissue F1 heatmap without training.
Raw TCGA expression and neural checkpoints are not included. A new campaign
can differ numerically across library versions, hardware and threading settings;
bitwise deterministic multithreaded training is not assumed.

| Paper content | Producing command/module |
|---|---|
| Figure 1: radial profiles and references | `illustrations.concept` |
| Figure 2: finite-sample sensitivity | `illustrations.permutation` after `statistical_tests.py` |
| Figure 3: pointwise component landscapes | `illustrations/legacy/loss_landscape.py` |
| Figure 4 and S5: normalized destination interpolation | `interpolation_destination.py` |
| Figure 5: expansion | `expansion.py`, then `render_samples.py --part expansion` |
| Figures 6–7 and S6–S7: five/seven synthetic losses | `train_toys.py`, then `render_samples.py --part toys` |
| Figure 8 and Appendix tissue heatmap | `render_training.py` |
| Appendix normalization/component illustration | `illustrations/legacy/norm_comparison.py` |
| S1–S4: cohort, repeated tests and held-out interpolation | `render.py` |

The two component illustrations are separate explanatory calculations. Their
isolated `legacy` support preserves the displayed estimators (including the
normalization illustration's truncated/rank-interpolated profiles); it is not
imported by the library or the repeated comparisons. To render them:

```sh
SIGDIST_FIGURES=outputs/figures python examples/paper_revision/illustrations/legacy/loss_landscape.py
SIGDIST_FIGURES=outputs/figures SIGDIST_CACHE=outputs/component-cache python examples/paper_revision/illustrations/legacy/norm_comparison.py
```

The latter runs its normalization/component demonstration if cached synthetic
samples are absent. Plotting scripts preserve panel styles; embedding size and
numbering are controlled by the manuscript template. Fonts can vary by system.
