# Final Submission Package (RangeViT Project)

This folder contains a cleaned and organized submission package for faculty review.

## 1) What is included

- Codebase and dependencies (dataset not included)
- Environment setup and run instructions
- Curated experiment logs and summaries
- Reproduced result tables and figures
- Report PDF is kept in the final bundle root (not inside this folder)

## 2) Folder structure

- `docs/`
  - `generate_submission_assets.py`: script used to generate curated charts/tables.
- `reproducibility/`
  - `environment_setup.md`
  - `dataset_and_paths.md`
  - `run_commands.md`
- `results/`
  - `tables/`: curated CSV tables
  - `figures/`: curated and newly generated charts
  - `log_summaries/`: experiment-wise notes, including failed runs

Note: the old `report/` source folder was intentionally removed during final cleanup after generating the report PDF.

## 3) Archived old presentation files

Original presentation files were moved safely (not deleted) to:

- `/home/ai25mtech14009/rangevit-main/archive/presentation_assets_2026-04-05`

This archive includes old slide outlines, draft plots, and intermediate notes.

## 4) Code location

Primary codebase remains at:

- `/home/ai25mtech14009/rangevit-main/rangevit-main`

No dataset is included in this package.

## 5) Final bundle to submit

Use the prepared final submission archive:

- `/home/ai25mtech14009/rangevit-main/final_submission_bundle.zip`

This archive includes:

- `rangevit-main/` (all code and configs, dataset payload excluded)
- `submission_package/` (reproducibility + results)
- report PDF file(s) at bundle root

## 6) Key generated assets

Main curated tables:

- `/home/ai25mtech14009/rangevit-main/submission_package/results/tables/ablation_best_results_curated.csv`
- `/home/ai25mtech14009/rangevit-main/submission_package/results/tables/experiment_registry_curated.csv`

Main curated figures:

- `/home/ai25mtech14009/rangevit-main/submission_package/results/figures/ablation_grouped_metrics.png`
- `/home/ai25mtech14009/rangevit-main/submission_package/results/figures/ablation_iou_delta_curated.png`
- `/home/ai25mtech14009/rangevit-main/submission_package/results/figures/validation_iou_curves_curated.png`
- `/home/ai25mtech14009/rangevit-main/submission_package/results/figures/selected_experiment_iou_comparison.png`
- `/home/ai25mtech14009/rangevit-main/submission_package/results/figures/experiment_outcome_counts.png`
- `/home/ai25mtech14009/rangevit-main/submission_package/results/figures/failure_reason_distribution.png`
- `/home/ai25mtech14009/rangevit-main/submission_package/results/figures/novelty_architecture_diagram.png`

## 7) Report note

Overleaf source files are no longer part of this cleaned folder. Submission should use the generated report PDF from the final bundle.
