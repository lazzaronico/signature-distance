# Recorded paper results

These per-run numerical records are the source for the repeated-training and
expansion summaries in the manuscript. `configurations.json` contains their
training and evaluation settings. There are 42 synthetic fits, 25 glocal TCGA
fits, five glocal MMD comparator fits, 25 global-only TCGA fits and 15 expansion
trajectories. TCGA expression values, sample identifiers and model checkpoints
are not included.

Run `render_training.py --recorded --out outputs/tables` to regenerate the
summaries without retraining. To evaluate a fresh campaign, use `--work outputs`
instead. The experiment guide describes how to produce those outputs. The
portable entry points preserve the experimental protocols; source-file hashes
recorded by a new execution refer to that execution's code.
