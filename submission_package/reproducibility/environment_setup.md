# Environment Setup

## System used

- OS: Linux
- Python: 3.8 (project virtual environment)
- PyTorch with CUDA

## Recommended environment steps

```bash
cd /home/ai25mtech14009/rangevit-main/rangevit-main
python3 -m venv /home/ai25mtech14009/venvs/rangevit
source /home/ai25mtech14009/venvs/rangevit/bin/activate
pip install --upgrade pip
pip install -r /home/ai25mtech14009/rangevit-main/rangevit-main/requirements.txt
pip install nuscenes-devkit
```

## Notes

- Do not include dataset files in the submission.
- Keep run outputs under `/home/ai25mtech14009/rangevit-main/runs`.
- Generated submission artifacts are in `/home/ai25mtech14009/rangevit-main/submission_package`.
