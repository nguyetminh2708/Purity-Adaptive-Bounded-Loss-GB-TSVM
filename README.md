# PA-BL-GBTSVM

Purity-Adaptive Bounded-Loss Granular-Ball Twin SVM.

Twin SVM over granular balls, hinge loss swapped for the Wave loss (bounded, smooth) for robustness to label noise.
From the essay "The Twin SVM family: SVM, TWSVM, LS-TWSVM, GBTSVM" (Maths4AI, 2026).

## Idea

- GBTSVM = twin SVM on ball centers/radii with hinge loss (unbounded). One noisy ball can drag the hyperplane far.
- Swap hinge for Wave loss:

  ```
  L(u) = u^2 e^{a u} / (1 + lam * u^2 e^{a u})
  ```

  - bounded by 1/lam, smooth, redescending. The harder a ball violates, the less it counts, so noisy balls self-discount.
  - for small violations it is almost independent of lam, so lam reshapes the penalty, not its volume.

What we actually found:

1. Main: Wave @ pur=1 beats hinge and nearest-centroid (10 tabular sets, 4 noise types; confirmed on real CIFAR-10N noise, Part A).
   - noise ceiling: past ~40% noise on a hard pair the signal is gone and nobody wins (Parts C, D).
2. Extension: rescue + nearest-centroid fallback when a class collapses to 1 ball (there the 2-center twin SVM equals nearest-centroid anyway).
3. Utility: auto hyperparam search with WOA (mealpy). Its value is stability under heavy noise.

## Install

```
pip install -r requirements.txt
```

Heads up on extras (requirements.txt is incomplete):

- requirements.txt says `cvxpy`, but the baselines import `cvxopt`. So: `pip install cvxopt` (needed for the hard GBTSVM / LGBTSVM / PinGBTSVM).
- `pip install mealpy` only for `tune_meta.py` (auto-tune).
- `pip install torch torchvision` only for the CIFAR real-noise runs.

## Data

Tabular (UCI/KEEL): already in `data/raw/` (haberman, heart, ionosphere, sonar, banknote, australian, german, breast_cancer, SPECTF, ...).
Synthetic noise = 4 styles: symmetric, asymmetric, boundary, far.

Real noise (CIFAR-10N / CIFAR-100N, Wei et al. 2022) needs two things, images and human labels:

- Images: auto-downloaded. `exp_realnoise.py` uses torchvision `download=True`, so the first run pulls CIFAR-10 by itself, nothing to do (CIFAR-100 same via `datasets.CIFAR100`).
  - By hand only if the network is blocked (proxy / sandbox): get `cifar-10-python.tar.gz` or `cifar-100-python.tar.gz` from cs.toronto.edu/~kriz, extract into `data/raw/cifar10/...` and `data/raw/cifar100/...`.
- Human labels: not in torchvision, place by hand. `CIFAR-10_human.pt`, `CIFAR-100_human.pt` from UCSC-REAL/cifar-10-100n (its `data/`) or noisylabels.com, drop into `data/raw/`.

Noise levels:

- `CIFAR-10_human.pt`: aggre 9%, random1 17%, worse 40%.
- `CIFAR-100_human.pt`: fine 40%, coarse 26%.

## Layout

```
src/gb/balls.py             ball generation (recursive 2-means until purity >= rho), arrays/viable
src/gb/rescue.py            rescue + nearest-centroid + predict_with_fallback
src/loss/wave.py            Wave loss, gradient, numerical check
src/models/wave_gbtsvm.py   main model, WaveGBTSVM (Adam solver)
src/models/registry.py      baselines GBTSVM / LGBTSVM / PinGBTSVM, OVO/OVR
src/models/baselines/       the 3 baselines (QP via cvxopt)
src/data/datasets.py        load UCI/KEEL + inject noise
experiments/                experiment scripts (below)
results/                    raw csv, kept to rerun stats
paper/                      manuscript (banthao/) + the report workbook ThucNghiem_GBTSVM_BaoCao.xlsx (tracked)
docs/                       full working workbook + slides (local only, gitignored)
```

## Run

Each script has `--help`.

```bash
# Main: Wave vs hinge vs nearest-centroid at pur=1
python experiments/exp_pur1_nc.py --datasets haberman heart breast_cancer ionosphere --seeds 3

# Sweep purity
python experiments/exp_seeds_pur.py --seeds 5 --kernel linear --purity 0.8

# Sweep synthetic noise types
python experiments/exp_noise_kinds.py --seeds 5 --kernel linear

# Degeneration: rescue vs nearest-centroid
python experiments/exp_rescue_knn.py --purity 0.6 --seeds 5

# Auto-tune with WOA (csv checkpoint, resumable)
python experiments/tune_meta.py --datasets heart sonar --algo WOA --epoch 25 --pop 20 --out results/tune.csv
```

Real CIFAR-10N noise (`exp_realnoise.py`). One `--purity` per run; the pur=1 column always comes with it, so change `--purity` to get the 0.7 / 0.6 / 0.5 steps.

```bash
# Part A, moderate noise: automobile vs truck, aggre_label
python experiments/exp_realnoise.py --classes automobile truck --noise aggre_label --pca 100 --seeds 5 --purity 0.7

# Part C, high noise: cat vs dog, worse_label
python experiments/exp_realnoise.py --classes cat dog --noise worse_label --pca 50 --seeds 5 --purity 0.6
```

CIFAR-100N: no script yet. Same recipe but read `CIFAR-100_human.pt` (`noisy_label` = fine 40%, `noisy_coarse_label` = coarse 26%) with images in `data/raw/cifar100`. A script like `exp_realnoise.py` can be added to reproduce Part D.

## Results

Shipped workbook: `paper/ThucNghiem_GBTSVM_BaoCao.xlsx` (trimmed report version).
Tổng kết = summary, Sheet 17 = real CIFAR noise (Parts A to D), Sheet 18 = vs nearest-centroid.
The full working workbook stays in `docs/` (local only, gitignored).

## Refs

GBTSVM (IEEE TNNLS 2025, arXiv 2410.04774, github.com/mtanveer1/GBTSVM), GBFTSVM (IEEE TFS 2025), GB-Pin-TSVM (IEEE TCSS), Wave loss (Pattern Recognition 2024, arXiv 2404.18101), CIFAR-10N / CIFAR-100N (Wei et al., ICLR 2022).
