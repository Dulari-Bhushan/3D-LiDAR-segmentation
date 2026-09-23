# Interpretable AI and Geometry Fusion in RangeViT for 3D LiDAR Semantic Segmentation

This repository contains an academic reproduction-and-extension study of **RangeViT**
(Ando et al., CVPR 2023) for 3D LiDAR semantic segmentation on SemanticKITTI. The
baseline training pipeline is first stabilized, then extended with two coupled
contributions: **decoder-side geometry fusion** and a **prototype-based interpretable
objective** with attention-entropy regularization.

Full technical details, derivations, and the complete set of results/figures are in the
project report: [`AI25MTECH14009 final report in ICLR format.pdf`](AI25MTECH14009%20final%20report%20in%20ICLR%20format.pdf).
Accompanying slide decks: [`AI25MTECH14009 ppt 1 Visual_Computing.pdf`](AI25MTECH14009%20ppt%201%20Visual_Computing.pdf)
and [`AI25MTECH14009 ppt 2 Visual_Computing.pdf`](AI25MTECH14009%20ppt%202%20Visual_Computing.pdf).

## Abstract

RangeViT is studied under a reproduction-and-extension setting. The baseline training
pipeline was stabilized first, then extended with prototype-based interpretability
regularization, decoder-side geometry fusion, and attention entropy control. Across the
main ablation family, prototype-only training reduced mIoU from 0.3629 to 0.2767, while
adding geometry recovered and improved performance to 0.3861, and the full configuration
reached 0.3922. A broader registry of experiments also shows that geometry-fusion-only
exploration can reach 0.4797 mIoU in the same workspace, indicating that geometry
injection is the strongest individual driver while auxiliary regularizers need careful
balancing.

The report documents both successful and unsuccessful runs, including split-shift
experiments and Cityscapes-initialized trials, and ties the observed behavior back to
optimization sensitivity, class imbalance, and structural cues in range-view
representation. The main technical result is that interpretability constraints are most
useful when they are anchored by geometry at the decoder stage rather than applied in
isolation.

## Motivation

LiDAR semantic segmentation is challenging because classes are imbalanced, point
density decreases with distance, and scene geometry is highly anisotropic. Range-view
projection enables efficient dense computation with transformer architectures, but
geometric structure that is explicit in 3D becomes implicit in 2D range coordinates.
This project asks two questions: can segmentation quality be improved without
introducing training instability, and can interpretability be improved beyond scalar
performance metrics? The answer explored here couples explicit geometry fusion near the
decoder with prototype-driven interpretable supervision in feature space.

## Method

### RangeViT baseline

The baseline (Ando et al., 2023) projects a LiDAR point cloud to a range image, encodes
it with a convolutional stem + ViT encoder, decodes to a 2D feature map, and applies a
3D refiner layer for point-wise predictions. Baseline segmentation training uses:

$$\mathcal{L}_{seg} = \mathcal{L}_{focal} + \mathcal{L}_{lovasz}$$

### Contribution 1 — Geometry Fusion

Projected geometric channels are fused with decoder features right before final
classification, via a compact 1×1 projection, concatenation, a fusion convolution, and a
residual connection:

$$\tilde{d} = d + \phi([d;\ \psi(g)])$$

This reintroduces boundary-critical structural cues that are otherwise blurred deep in
the encoded representation, while keeping the backbone unchanged.

### Contribution 2 — Interpretable Objective

Prototype compactness/separation, a geometry-consistency term, and attention-entropy
regularization are optimized jointly with the segmentation loss:

$$\mathcal{L}_{total} = \mathcal{L}_{seg} + \lambda_p \mathcal{L}_{proto} + \lambda_g \mathcal{L}_{geo} + \lambda_a \mathcal{L}_{attn}$$

- `L_proto = L_compact + L_sep` — pulls features toward their class prototype and pushes
  them away from competing-class prototypes with a margin.
- `L_attn = (H̄ − H_t)²` — regularizes mean attention entropy toward a target value.
- `L_geo` — penalizes excessive within-class spread of geometry channels per prototype.

The key design intent is that these are **coupled**, not independent add-ons: geometry
fusion improves the stability and usefulness of prototype-based representation shaping,
and prototype/attention regularization is only performance-positive once it is anchored
to geometry at the decoder.

## Results

Canonical ablation family (SemanticKITTI, in-domain 00+01 split, 80/20, ViT-S/16
encoder, ImageNet21k pretraining):

| Variant       | Best Acc | Best mIoU | Best Recall |
|---------------|----------|-----------|-------------|
| Baseline      | 0.4668   | 0.3629    | 0.4746      |
| Proto Only    | 0.4481   | 0.2767    | 0.3506      |
| Geo + Proto   | 0.4348   | 0.3861    | 0.4394      |
| Full Method   | 0.4425   | 0.3922    | 0.4413      |

A broader, non-canonical experiment registry (different setups, not directly comparable)
shows geometry-fusion-only runs reaching up to **0.4797 mIoU** — the single strongest
result in the workspace, which is why geometry fusion is treated as the central
contribution rather than a minor auxiliary tweak.

Key findings:
- Prototype-only regularization **hurts** mIoU relative to baseline when not anchored to
  geometry (0.3629 → 0.2767).
- Adding geometry fusion recovers and exceeds baseline (Geo+Proto: 0.3861, Full: 0.3922).
- Geometry fusion is the strongest individual performance driver; the full composite
  objective trades a little peak mIoU for richer, more interpretable structure.
- Large/frequent classes (road, building, vegetation) dominate mIoU; rare object classes
  (bicycle, motorcycle, person, etc.) remain at ~0 IoU under this reproduction setup —
  consistent with SemanticKITTI's long-tailed class distribution.

Full ablation curves, per-class IoU, prototype-purity/attention-entropy dashboards, and
interpretability visualizations (margin maps, centroid-distance heatmaps, single-scan
interpretable maps) are in the report and in
[`submission_package/results/`](submission_package/results/).

## Repository structure

```
.
├── AI25MTECH14009 final report in ICLR format.pdf   # Full report (method, results, discussion, references)
├── AI25MTECH14009 ppt 1 Visual_Computing.pdf         # Presentation slides
├── AI25MTECH14009 ppt 2 Visual_Computing.pdf         # Presentation slides
├── RangeViT_ 3D LiDAR Segmentation.pdf               # Reference copy of the report/paper
│
├── rangevit-main/            # RangeViT codebase (baseline + geometry fusion + interpretable objective)
│   ├── main.py, train.py, option.py
│   ├── config_*.yaml         # Baseline / ablation / full-method run configs
│   ├── models/, dataset/, kernels/, utils/, data/
│   └── README.md             # Upstream RangeViT usage instructions
│
├── submission_package/       # Curated, reviewer-facing results package
│   ├── docs/                 # Script used to generate curated charts/tables
│   ├── reproducibility/      # Environment setup, dataset layout, exact run commands
│   └── results/
│       ├── tables/           # Curated CSV result tables
│       ├── figures/          # Ablation, convergence, and interpretability figures
│       └── log_summaries/    # Experiment-wise notes, including failed runs
│
└── final_submission_bundle/  # Self-contained bundle combining the above (rangevit-main +
                               # submission_package + report PDF) for archival/submission
```

## Reproducing the experiments

### Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r rangevit-main/requirements.txt
pip install nuscenes-devkit
```

### Dataset

SemanticKITTI is **not** included in this repository. Download it separately and point
`--data_root` at the sequences directory:

```
<dataset_root>/semantic_kitti/dataset/sequences/
```

### Training

Each ablation variant has its own config file in `rangevit-main/`:

| Variant             | Config                                    |
|---------------------|--------------------------------------------|
| Baseline            | `config_run_baseline.yaml`                 |
| Prototype only       | `config_run_proto_only.yaml`               |
| Geometry + Prototype | `config_run_geo_proto.yaml`                |
| Full method          | `config_run_full.yaml`                     |
| Geometry fusion only | `config_run_geom_fusion_only.yaml`         |
| 00+01 split          | `config_kitti_00_01_split.yaml`            |

```bash
cd rangevit-main
python -m torch.distributed.launch --nproc_per_node=2 --master_port=63545 \
  --use_env main.py config_run_full.yaml \
  --data_root <dataset_root>/semantic_kitti/dataset/sequences \
  --save_path <path_to_log_dir>
```

### Evaluation from a checkpoint

```bash
cd rangevit-main
python -m torch.distributed.launch --nproc_per_node=1 --master_port=63545 \
  --use_env main.py config_kitti.yaml \
  --data_root <dataset_root>/semantic_kitti/dataset/sequences \
  --save_path <path_to_eval_out> \
  --checkpoint <path_to_checkpoint.pth> --val_only
```

See [`submission_package/reproducibility/`](submission_package/reproducibility/) for the
exact commands used for every run family, and
[`submission_package/results/log_summaries/`](submission_package/results/log_summaries/)
for notes on runs that failed or were unstable.

## Limitations

- Prototype regularization is sensitive to margin and pixel-sampling thresholds, which
  can destabilize gradients when class support is sparse.
- Geometry fusion only helps when geometric cues are well-aligned with the decoder
  feature stream; weak integration does not guarantee improvement.
- A fixed attention-entropy target is not necessarily optimal across all classes or
  training stages.
- Run families differ in split policy and initialization, so cross-family comparisons
  (e.g. Cityscapes-initialized or 00+01-split runs) are reported separately from the
  canonical in-domain ablation rather than averaged together.
- Distributed multi-GPU training suffered occasional process-level interruptions, which
  limited exhaustive hyperparameter sweeping.

## Future work

- Adaptive, uncertainty-aware prototype updates and class-frequency-aware margin
  weighting to counter long-tail imbalance.
- Sequence-shift-robust training with explicit domain adaptation for cross-split
  generalization.
- Multi-seed statistical robustness and run-family uncertainty reporting.
- Staged optimization: stabilize geometry fusion first, then ramp interpretable losses
  gradually (coefficient annealing, gradient-norm balancing, dynamic loss routing).
- A lightweight geometry-token cross-attention module as a successor to late fusion.
- Multi-objective checkpoint selection (mIoU + interpretability stability), rather than
  scalar mIoU alone.

## Acknowledgements and citation

This project builds directly on **RangeViT**:

> Angelika Ando, Spyros Gidaris, Andrei Bursuc, Gilles Puy, Alexandre Boulch, Renaud
> Marlet. *RangeViT: Towards Vision Transformers for 3D Semantic Segmentation in
> Autonomous Driving.* CVPR 2023. [[arXiv](https://arxiv.org/abs/2301.10222)] [[code](https://github.com/valeoai/rangevit)]

```bibtex
@inproceedings{RangeViT,
  title={RangeViT: Towards Vision Transformers for 3D Semantic Segmentation in Autonomous Driving},
  author={Ando, Angelika and Gidaris, Spyros and Bursuc, Andrei and Puy, Gilles and Boulch, Alexandre and Marlet, Renaud},
  booktitle={CVPR},
  year={2023}
}
```

Trained on **SemanticKITTI**: Behley et al., *SemanticKITTI: A Dataset for Semantic
Scene Understanding of LiDAR Sequences*, ICCV 2019.

The full reference list is in the project report.

## License

The RangeViT codebase in [`rangevit-main/`](rangevit-main/) is distributed under the
Apache License 2.0 (see [`rangevit-main/LICENSE`](rangevit-main/LICENSE)), inherited
from the upstream [valeoai/rangevit](https://github.com/valeoai/rangevit) repository.
