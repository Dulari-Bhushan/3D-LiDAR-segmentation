#!/usr/bin/env python3
import csv
import os
import shutil
from collections import Counter

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

ROOT = "/home/ai25mtech14009/rangevit-main"
ARCHIVE = os.path.join(ROOT, "archive", "presentation_assets_2026-04-05")
SUBMISSION = os.path.join(ROOT, "submission_package")
RESULTS_FIG = os.path.join(SUBMISSION, "results", "figures")
RESULTS_TABLE = os.path.join(SUBMISSION, "results", "tables")
REPORT_FIG = os.path.join(SUBMISSION, "report", "figures")
RUNS = os.path.join(ROOT, "runs")

os.makedirs(RESULTS_FIG, exist_ok=True)
os.makedirs(RESULTS_TABLE, exist_ok=True)
os.makedirs(REPORT_FIG, exist_ok=True)


def read_csv(path):
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def float_or_none(v):
    try:
        return float(v)
    except Exception:
        return None


# 1) Curate ablation table
ablation_src = os.path.join(ARCHIVE, "ablation_best_results.csv")
ablation = read_csv(ablation_src)
for row in ablation:
    row["best_acc"] = float(row["best_acc"])
    row["best_iou"] = float(row["best_iou"])
    row["best_recall"] = float(row["best_recall"])

base_iou = next(r["best_iou"] for r in ablation if r["run"].lower() == "baseline")
curated_ablation = []
for row in ablation:
    curated_ablation.append(
        {
            "run": row["run"],
            "best_acc": f"{row['best_acc']:.4f}",
            "best_iou": f"{row['best_iou']:.4f}",
            "best_recall": f"{row['best_recall']:.4f}",
            "iou_delta_vs_baseline": f"{(row['best_iou'] - base_iou):+.4f}",
        }
    )

write_csv(
    os.path.join(RESULTS_TABLE, "ablation_best_results_curated.csv"),
    curated_ablation,
    ["run", "best_acc", "best_iou", "best_recall", "iou_delta_vs_baseline"],
)

# 2) Curate experiment registry with categories
registry_src = os.path.join(RUNS, "experiment_registry_auto.csv")
registry = read_csv(registry_src)
curated_registry = []
for row in registry:
    fname = row["run_file"].lower()
    category = "General"
    notes = ""

    if "00_01" in fname and "val08" in fname:
        category = "Cross-sequence split (00+01 train, 08 val)"
        notes = "Failed mostly due to distributed NCCL timeout/elastic child failure."
    elif "00_01" in fname and "in_domain" in fname:
        category = "In-domain split (00+01, 80/20)"
        notes = "Completed; strong in-domain score, not directly comparable to seq08 transfer."
    elif "cityscapes" in fname:
        category = "Cityscapes-initialized trials"
        notes = "Observed unstable or low gains; initialization mismatch likely for range-view LiDAR domain."
    elif "geom_fusion_only" in fname:
        category = "Geo-fusion only"
        notes = "Strongest standalone result in this log family."
    elif "geom_fusion_full" in fname:
        category = "Geo-fusion + interpretable objective"
        notes = "Best balanced novelty run in final method family."
    elif "full_method" in fname:
        category = "Full method (ablation family)"
        notes = "Final run used for main ablation summary."
    elif "kitti_train_2gpu" in fname:
        category = "Early baseline training"
        notes = "Low baseline quality from early setup."

    curated_registry.append(
        {
            "run_file": row["run_file"],
            "category": category,
            "status": row["status"],
            "failure_reason": row["failure_reason"] or "none",
            "best_acc": row["best_acc"] or "NA",
            "best_iou": row["best_iou"] or "NA",
            "best_recall": row["best_recall"] or "NA",
            "notes": notes,
        }
    )

write_csv(
    os.path.join(RESULTS_TABLE, "experiment_registry_curated.csv"),
    curated_registry,
    [
        "run_file",
        "category",
        "status",
        "failure_reason",
        "best_acc",
        "best_iou",
        "best_recall",
        "notes",
    ],
)

# 3) Figure: ablation grouped metrics
runs = [r["run"] for r in ablation]
accs = [r["best_acc"] for r in ablation]
ious = [r["best_iou"] for r in ablation]
recs = [r["best_recall"] for r in ablation]

x = list(range(len(runs)))
width = 0.25
plt.figure(figsize=(10, 5.2))
plt.bar([i - width for i in x], accs, width=width, label="Acc")
plt.bar(x, ious, width=width, label="mIoU")
plt.bar([i + width for i in x], recs, width=width, label="Recall")
plt.xticks(x, runs, rotation=15)
plt.ylabel("Score")
plt.title("Best Metrics by Ablation Variant")
plt.ylim(0, max(max(accs), max(ious), max(recs)) + 0.08)
plt.grid(axis="y", alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_FIG, "ablation_grouped_metrics.png"), dpi=220)
plt.close()

# 4) Figure: IoU delta from baseline
deltas = [r["best_iou"] - base_iou for r in ablation]
colors = ["#6baed6" if d >= 0 else "#fb6a4a" for d in deltas]
plt.figure(figsize=(8.6, 4.8))
plt.bar(runs, deltas, color=colors)
plt.axhline(0.0, color="black", linewidth=1)
plt.ylabel("mIoU Delta")
plt.title("Ablation mIoU Delta vs Baseline")
plt.grid(axis="y", alpha=0.25)
for i, d in enumerate(deltas):
    plt.text(i, d + (0.002 if d >= 0 else -0.006), f"{d:+.3f}", ha="center", va="bottom" if d >= 0 else "top", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_FIG, "ablation_iou_delta_curated.png"), dpi=220)
plt.close()

# 5) Figure: status counts
status_counts = Counter(r["status"] for r in curated_registry)
labels = sorted(status_counts.keys())
vals = [status_counts[k] for k in labels]
plt.figure(figsize=(6.8, 4.4))
plt.bar(labels, vals, color=["#31a354" if s == "completed" else "#de2d26" for s in labels])
plt.ylabel("Number of Runs")
plt.title("Experiment Outcome Counts")
plt.grid(axis="y", alpha=0.25)
for i, v in enumerate(vals):
    plt.text(i, v + 0.05, str(v), ha="center", va="bottom")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_FIG, "experiment_outcome_counts.png"), dpi=220)
plt.close()

# 6) Figure: failure reasons
failure_counts = Counter(r["failure_reason"] for r in curated_registry if r["status"] == "failed")
if failure_counts:
    labels = sorted(failure_counts.keys())
    vals = [failure_counts[k] for k in labels]
    plt.figure(figsize=(7.8, 4.4))
    plt.bar(labels, vals, color="#ef6548")
    plt.ylabel("Failed Runs")
    plt.title("Failure Reason Distribution")
    plt.grid(axis="y", alpha=0.25)
    for i, v in enumerate(vals):
        plt.text(i, v + 0.05, str(v), ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FIG, "failure_reason_distribution.png"), dpi=220)
    plt.close()

# 7) Figure: selected experiment mIoU comparison
selected = []
for r in curated_registry:
    iou = float_or_none(r["best_iou"]) if r["best_iou"] != "NA" else None
    if iou is None:
        continue
    name = r["run_file"]
    keep = any(k in name for k in [
        "geom_fusion_only",
        "geom_fusion_full",
        "full_method_2gpu_20260331_100054",
        "kitti_00_01_in_domain",
        "kitti_train_3gpu_cityscapes_20260329_171347",
        "kitti_train_3gpu_cityscapes_resume_20260329_232826",
        "kitti_train_2gpu_20260328_144458",
    ])
    if keep:
        selected.append((name.replace("_2gpu", "").replace("_3gpu", ""), iou))

selected = sorted(selected, key=lambda t: t[1], reverse=True)
if selected:
    plt.figure(figsize=(12, 5.2))
    labels = [s[0] for s in selected]
    vals = [s[1] for s in selected]
    plt.bar(range(len(labels)), vals, color="#3182bd")
    plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
    plt.ylabel("Best mIoU")
    plt.title("Selected Experiment mIoU Comparison")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FIG, "selected_experiment_iou_comparison.png"), dpi=220)
    plt.close()

# 8) Figure: validation mIoU curves by run
val_rows = read_csv(os.path.join(ARCHIVE, "validation_metrics_by_epoch.csv"))
curves = {}
for r in val_rows:
    run = r["run"]
    epoch = int(r["epoch"])
    iou = float_or_none(r["val_iou"])
    if iou is None:
        continue
    curves.setdefault(run, []).append((epoch, iou))

plt.figure(figsize=(9.2, 5.2))
for run, pts in curves.items():
    pts = sorted(pts)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    plt.plot(xs, ys, label=run, linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Validation mIoU")
plt.title("Validation mIoU Curves by Variant")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_FIG, "validation_iou_curves_curated.png"), dpi=220)
plt.close()

# 9) Novel architecture diagram for report
fig, ax = plt.subplots(figsize=(12, 4.8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 6)
ax.axis("off")

blocks = [
    (0.6, 2.5, 2.2, 1.2, "Range Image\\nInput"),
    (3.1, 2.5, 2.4, 1.2, "ViT Encoder\\n+ Decoder"),
    (5.9, 3.8, 2.4, 1.2, "Geometry\\nProjection"),
    (5.9, 1.2, 2.4, 1.2, "Prototype\\nMemory"),
    (8.8, 2.5, 2.4, 1.2, "Fusion +\\nInterpretable Loss"),
    (11.6, 2.5, 2.0, 1.2, "Segmentation\\nLogits"),
]

for x0, y0, w, h, text in blocks:
    ax.add_patch(Rectangle((x0, y0), w, h, fill=False, linewidth=1.6))
    ax.text(x0 + w / 2, y0 + h / 2, text, ha="center", va="center", fontsize=10)

arrows = [
    ((2.8, 3.1), (3.1, 3.1)),
    ((5.5, 3.1), (8.8, 3.1)),
    ((8.3, 4.4), (8.8, 3.4)),
    ((8.3, 1.8), (8.8, 2.8)),
    ((11.2, 3.1), (11.6, 3.1)),
]
for (x1, y1), (x2, y2) in arrows:
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", mutation_scale=12, linewidth=1.4))

ax.text(7.0, 5.4, "Proposed RangeViT Extension: Geometry Fusion + Prototype Interpretability", ha="center", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_FIG, "novelty_architecture_diagram.png"), dpi=220)
plt.close(fig)

# 9b) Figure: metric trade-off scatter (Acc vs IoU, marker size by Recall)
plt.figure(figsize=(7.2, 5.4))
for r in ablation:
    x = r["best_acc"]
    y = r["best_iou"]
    s = 800 * r["best_recall"]
    plt.scatter(x, y, s=s, alpha=0.7, label=r["run"])
    plt.text(x + 0.002, y + 0.002, r["run"], fontsize=9)
plt.xlabel("Best Accuracy")
plt.ylabel("Best mIoU")
plt.title("Ablation Trade-off: Accuracy vs mIoU (bubble size = Recall)")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_FIG, "ablation_tradeoff_scatter.png"), dpi=220)
plt.close()

# 9c) Figure: best mIoU by experiment category
cat_best = {}
for r in curated_registry:
    iou = float_or_none(r["best_iou"]) if r["best_iou"] != "NA" else None
    if iou is None:
        continue
    c = r["category"]
    cat_best[c] = max(cat_best.get(c, -1.0), iou)

if cat_best:
    items = sorted(cat_best.items(), key=lambda t: t[1], reverse=True)
    labels = [i[0] for i in items]
    vals = [i[1] for i in items]
    plt.figure(figsize=(11.2, 5.2))
    plt.bar(range(len(labels)), vals, color="#756bb1")
    plt.xticks(range(len(labels)), labels, rotation=25, ha="right")
    plt.ylabel("Best mIoU")
    plt.title("Best mIoU by Experiment Category")
    plt.grid(axis="y", alpha=0.25)
    for i, v in enumerate(vals):
        plt.text(i, v + 0.003, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FIG, "category_best_iou.png"), dpi=220)
    plt.close()

# 9d) Figure: chronological experiment status timeline
def parse_time_from_name(name):
    m = re.search(r"(\d{8})_(\d{6})", name)
    if not m:
        return None
    return m.group(1) + m.group(2)

import re
timeline = []
for r in curated_registry:
    t = parse_time_from_name(r["run_file"])
    if t is None:
        continue
    timeline.append((t, r["status"], r["run_file"]))

if timeline:
    timeline = sorted(timeline, key=lambda x: x[0])
    xs = list(range(len(timeline)))
    ys = [1 if t[1] == "completed" else 0 for t in timeline]
    colors = ["#2ca25f" if y == 1 else "#de2d26" for y in ys]
    plt.figure(figsize=(11.4, 3.8))
    plt.scatter(xs, ys, c=colors, s=45)
    plt.yticks([0, 1], ["Failed", "Completed"])
    plt.xlabel("Chronological Run Index")
    plt.ylabel("Outcome")
    plt.title("Experiment Outcome Timeline")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FIG, "experiment_timeline_status.png"), dpi=220)
    plt.close()

# 10) Copy selected legacy visuals into report figures with cleaner names
copy_map = {
    "fig7_single_scan_interpretability.png": "single_scan_interpretability_map.png",
    "fig8_margin_class_stats_single_scan.png": "single_scan_margin_statistics.png",
    "fig9_centroid_distance_heatmap_single_scan.png": "single_scan_centroid_distance_heatmap.png",
    "fig6_interpretable_signals.png": "interpretable_signal_dashboard.png",
    "geo_fusion_focus_bar.png": "geo_fusion_focus_bar.png",
    "RangeViT_method.png": "original_rangevit_architecture.png",
}

for src_name, dst_name in copy_map.items():
    if src_name == "RangeViT_method.png":
        src = os.path.join(ROOT, "rangevit-main", "RangeViT_method.png")
    else:
        src = os.path.join(ARCHIVE, src_name)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(REPORT_FIG, dst_name))

# Also copy all newly generated figures to report figures
for fname in os.listdir(RESULTS_FIG):
    src = os.path.join(RESULTS_FIG, fname)
    if os.path.isfile(src):
        shutil.copy2(src, os.path.join(REPORT_FIG, fname))

print("Generated assets in", SUBMISSION)
print("Results figures:")
for n in sorted(os.listdir(RESULTS_FIG)):
    print(" -", n)
print("Report figures:")
for n in sorted(os.listdir(REPORT_FIG)):
    print(" -", n)
