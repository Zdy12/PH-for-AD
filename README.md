# Interpretable multiscale topological features from fMRI

Reference implementation for:

> Zhao, D., Li, S., Wang, Y., Wang, C., Zhou, Z., Yan, G., & Qi, X. (2026). **Extracting interpretable higher-order topological features across multiple scales from fMRI for Alzheimer's disease classification.** *Biomedical Signal Processing and Control, 116*, 109621. https://doi.org/10.1016/j.bspc.2026.109621

The pipeline converts regional BOLD time series into Pearson functional-connectivity matrices, forms distance matrices as `1 - correlation`, computes Vietoris–Rips persistent homology in dimensions 0, 1, and 2, quantifies the persistence diagrams, concatenates these higher-order descriptors with lower-order functional-connectivity edges, and evaluates an SVM classifier.

## Method-to-code map

| Paper stage | Implementation |
|---|---|
| Weighted brain network | `src/topofmri/connectivity.py` |
| Vietoris–Rips filtration, H0/H1/H2 | `src/topofmri/topology.py` |
| Landscape, Betti curve, heat kernel, entropy | `src/topofmri/topology.py` |
| Lower-order upper-triangular edges | `src/topofmri/connectivity.py` |
| Feature fusion | `src/topofmri/pipeline.py` |
| SVM and leakage-safe evaluation | `src/topofmri/evaluation.py` |

With 90 AAL regions, the paper configuration produces 5,808 features per subject:

- connectivity edges: 4,005;
- persistence landscapes: 300 (one layer) + 600 (two layers);
- Betti curves: 300;
- heat kernels: 300 + 300 (`sigma=1.2` and `1.4`);
- persistence entropy: 3.

## Installation

Python 3.9–3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

## Input data

ADNI data are not redistributed. Place one text or NumPy file per subject under class folders:

```text
data/raw/
├── AD/
│   ├── subject_001.txt
│   └── ...
└── CN/
    ├── subject_101.txt
    └── ...
```

Each file must be either:

- a BOLD array with shape `(time_points, 90)` (default), or
- a precomputed symmetric `90 × 90` Pearson correlation matrix when `--input-kind correlation` is used.

Text files may be whitespace-, tab-, or comma-delimited. `.npy` files are also supported. All subjects must have the same number of ROIs; time-series lengths may differ.

## Run the full pipeline

```bash
python scripts/run_pipeline.py \
  --ad-dir data/raw/AD \
  --cn-dir data/raw/CN \
  --output-dir outputs/paper_run \
  --input-kind timeseries \
  --evaluation holdout \
  --svm-kernel linear \
  --svm-c 1.0
```

This writes `features.npz`, `feature_manifest.json`, `subjects.csv`, `metrics_per_fold.csv`, and `metrics_summary.csv`. Intermediate persistence diagrams can optionally be retained with `--save-diagrams`.

For a quick installation check without ADNI data:

```bash
python scripts/make_synthetic_data.py --output-dir data/synthetic
python scripts/run_pipeline.py \
  --ad-dir data/synthetic/AD \
  --cn-dir data/synthetic/CN \
  --output-dir outputs/smoke_test \
  --homology-dimensions 0 1 \
  --evaluation holdout \
  --svm-kernel linear
```

The smoke test checks execution only; its scores have no scientific meaning.

## Reproducibility notes

- Feature extraction is unsupervised. Scaling is fitted separately inside every training fold to prevent leakage.
- AUC is computed from probability/decision scores, never from hard class predictions.
- The default command follows the supplied SVM implementation: a stratified 80/20 holdout, standardization fitted only on training data, and a linear SVM with `C=1`. The published paper reports five-fold cross-validation and an RBF SVM (`C=0.8`, `gamma=0.001`). To run that published setting, use `--evaluation cv --folds 5 --svm-kernel rbf --svm-c 0.8 --svm-gamma 0.001`.
- The implementation clips Pearson correlations to `[-1, 1]`, replaces non-finite entries, enforces symmetry, and sets the distance diagonal to zero.
- `H2` Vietoris–Rips computation can be expensive. Start with the synthetic smoke test and then run full data on a sufficiently provisioned machine.
- The supplied legacy script used heat-kernel sigmas `1.4` and `2.8`; the published paper reports `1.2` and `1.4`. This repository defaults to the published values. Use `--heat-sigmas 1.4 2.8` only to reproduce the older local experiment.

## Citation

Please cite the paper above if this code is useful. A machine-readable record is available in `CITATION.cff`.

## Data and license

ADNI data access is governed by ADNI's data-use terms and is not included here. The code is released under the MIT License; third-party datasets and packages retain their own terms.
