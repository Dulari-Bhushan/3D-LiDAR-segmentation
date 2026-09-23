# Experiment Failure and Progress Notes

## Why include this file

The submission explicitly documents all experiment attempts, including unsuccessful runs, to show complete research effort and engineering iterations.

## Major attempted tracks

1. Baseline stable training
2. Prototype-only objective
3. Geo+Proto objective
4. Full method (Geo+Proto+attention regularization)
5. Geo-fusion-only stress test
6. 00+01 split experiments (both in-domain and cross-sequence attempts)
7. Cityscapes-initialized trials and resume trials

## Important failed/partial outcomes

1. Cross-sequence 00+01 train and 08 validation runs showed instability and failures linked to distributed process issues.
2. Multiple logs show NCCL watchdog timeout and elastic ChildFailedError during multi-GPU execution.
3. Some early Cityscapes-initialized runs converged poorly (very low best IoU), indicating domain mismatch and unstable transfer for this setup.
4. A few runs have no final best-result line, indicating interrupted execution or incomplete run lifecycle.

## Main evidence files

- `/home/ai25mtech14009/rangevit-main/runs/experiment_registry_auto.csv`
- `/home/ai25mtech14009/rangevit-main/submission_package/results/tables/experiment_registry_curated.csv`
- `/home/ai25mtech14009/rangevit-main/submission_package/results/figures/experiment_outcome_counts.png`
- `/home/ai25mtech14009/rangevit-main/submission_package/results/figures/failure_reason_distribution.png`

## Main conclusion from failures

- Problems were not only algorithmic; infrastructure-level distributed training issues also affected reproducibility.
- Geometry-aware designs produced stronger and more stable performance compared with prototype-only setup.
- Novel objectives should be tuned only after stable distributed training settings are ensured.
