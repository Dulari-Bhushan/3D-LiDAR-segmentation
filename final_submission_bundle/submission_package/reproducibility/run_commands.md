# Run Commands

All commands below are reproducibility references using absolute paths.

## Baseline run

```bash
cd /home/ai25mtech14009/rangevit-main/rangevit-main
python -m torch.distributed.launch --nproc_per_node=2 --master_port=63545 \
  --use_env main.py /home/ai25mtech14009/rangevit-main/rangevit-main/config_run_baseline.yaml \
  --data_root /home/ai25mtech14009/rangevit-main/rangevit-main/data/semantic_kitti/dataset/sequences \
  --save_path /home/ai25mtech14009/rangevit-main/runs/log_run_baseline_stable
```

## Prototype-only run

```bash
cd /home/ai25mtech14009/rangevit-main/rangevit-main
python -m torch.distributed.launch --nproc_per_node=2 --master_port=63545 \
  --use_env main.py /home/ai25mtech14009/rangevit-main/rangevit-main/config_run_proto_only.yaml \
  --data_root /home/ai25mtech14009/rangevit-main/rangevit-main/data/semantic_kitti/dataset/sequences \
  --save_path /home/ai25mtech14009/rangevit-main/runs/log_run_proto_only_stable
```

## Geo+Proto run

```bash
cd /home/ai25mtech14009/rangevit-main/rangevit-main
python -m torch.distributed.launch --nproc_per_node=2 --master_port=63545 \
  --use_env main.py /home/ai25mtech14009/rangevit-main/rangevit-main/config_run_geo_proto.yaml \
  --data_root /home/ai25mtech14009/rangevit-main/rangevit-main/data/semantic_kitti/dataset/sequences \
  --save_path /home/ai25mtech14009/rangevit-main/runs/log_run_geo_proto
```

## Full method run

```bash
cd /home/ai25mtech14009/rangevit-main/rangevit-main
python -m torch.distributed.launch --nproc_per_node=2 --master_port=63545 \
  --use_env main.py /home/ai25mtech14009/rangevit-main/rangevit-main/config_run_full.yaml \
  --data_root /home/ai25mtech14009/rangevit-main/rangevit-main/data/semantic_kitti/dataset/sequences \
  --save_path /home/ai25mtech14009/rangevit-main/runs/log_run_full_method
```

## Geo-fusion-only run

```bash
cd /home/ai25mtech14009/rangevit-main/rangevit-main
python -m torch.distributed.launch --nproc_per_node=2 --master_port=63545 \
  --use_env main.py /home/ai25mtech14009/rangevit-main/rangevit-main/config_run_geom_fusion_only.yaml \
  --data_root /home/ai25mtech14009/rangevit-main/rangevit-main/data/semantic_kitti/dataset/sequences \
  --save_path /home/ai25mtech14009/rangevit-main/runs/log_run_geom_fusion_only
```

## 00+01 split run (in-domain validation)

```bash
cd /home/ai25mtech14009/rangevit-main/rangevit-main
python -m torch.distributed.launch --nproc_per_node=2 --master_port=63545 \
  --use_env main.py /home/ai25mtech14009/rangevit-main/rangevit-main/config_kitti_00_01_split.yaml \
  --data_root /home/ai25mtech14009/rangevit-main/rangevit-main/data/semantic_kitti/dataset/sequences \
  --save_path /home/ai25mtech14009/rangevit-main/runs/log_exp_kitti_00_01_split
```

## Evaluation from checkpoint

```bash
cd /home/ai25mtech14009/rangevit-main/rangevit-main
python -m torch.distributed.launch --nproc_per_node=1 --master_port=63545 \
  --use_env main.py /home/ai25mtech14009/rangevit-main/rangevit-main/config_kitti.yaml \
  --data_root /home/ai25mtech14009/rangevit-main/rangevit-main/data/semantic_kitti/dataset/sequences \
  --save_path /home/ai25mtech14009/rangevit-main/runs/eval_out \
  --checkpoint /path/to/checkpoint.pth --val_only
```
