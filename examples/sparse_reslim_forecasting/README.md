# Sparse-Reslim Forecasting Example

This is a proof-of-principle example for deterministic weather forecasting with
Sparse-Reslim. It predicts ERA5 2-meter temperature (`T2m`) 120 hours (5 days)
ahead on the global 1.0-degree grid. EDM is not included.

## Install

(1) Create the Frontier environment:

```bash
conda create -n sparse-reslim python=3.11 -y
conda activate sparse-reslim
pip install torch==2.8.0+rocm6.4 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/rocm6.4
```

(2) From the ORBIT-2 repository root, install the package and dependencies:

```bash
pip install -e .
pip install -r examples/sparse_reslim_forecasting/requirements.txt
```

## Data

ERA5 hourly 1.0-degree data on Frontier:

```text
/lustre/orion/world-shared/lrn036/jyc/frontier/ClimaX-v2/data/ERA5-1hr-superres/1.0_deg/
```

## Run

(3) Submit the one-GPU job:

```bash
sbatch examples/sparse_reslim_forecasting/launch.sh
```

The launch script runs a 30-epoch, single-step 120-hour `T2m` forecast with a
Sparse-Reslim keep ratio of 0.25. A quick CPU check is also available:

```bash
python examples/sparse_reslim_forecasting/train.py --smoke-test
```

No pretrained weights are required or included. For the complete ECCV code,
multi-variable experiments, distributed training, and EDM model, see the
[full Sparse-Reslim repository](https://github.com/janet-sw/Sparse-Reslim).
