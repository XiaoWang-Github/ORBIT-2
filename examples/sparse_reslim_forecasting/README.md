# Sparse-Reslim Weather Forecasting Proof of Principle

This example demonstrates the deterministic forecasting part of Sparse-Reslim
inside ORBIT-2. It predicts the global ERA5 2-meter temperature (`T2m`) 120
hours (5 days) ahead on the 1.0-degree grid. Only the forecasting example is
included here; EDM/diffusion code and paper-scale distributed training are not
part of this directory.

Sparse-Reslim processes every spatial token in the first and last Transformer
blocks, while only a random 25% of tokens pass through the middle blocks. The
sparse residual updates are scattered back to their original positions, so the
model still produces a dense forecast at every grid cell.

## Environment and dependencies

(1) Create and activate a conda environment on Frontier:

```bash
conda create -n sparse-reslim python=3.11 -y
conda activate sparse-reslim
pip install torch==2.8.0+rocm6.4 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/rocm6.4
```

(2) From the ORBIT-2 repository root, install ORBIT-2 and the small set of
packages used directly by this example:

```bash
pip install -e .
pip install -r examples/sparse_reslim_forecasting/requirements.txt
```

The direct dependencies are PyTorch, NumPy, and PyTorch Lightning. This example
does not require xFormers, MPI, DDStore, EDM, or pretrained model weights.

## ERA5 task and data

The provided launch script runs a direct, single-step **120-hour T2m forecast**
using hourly ERA5 data at **1.0-degree resolution**. The data are available on
Frontier at:

```text
/lustre/orion/world-shared/lrn036/jyc/frontier/ClimaX-v2/data/ERA5-1hr-superres/1.0_deg/
```

The directory must contain `normalize_mean.npz`, `normalize_std.npz`, and the
`train/`, `val/`, and `test/` NPZ splits. The paper uses 1979--2017 for
training, 2018--2019 for validation, and 2020 for testing.

## Run the forecasting example

(3) Activate the environment above and submit the one-GPU Frontier job from the
repository root:

```bash
sbatch examples/sparse_reslim_forecasting/launch.sh
```

The launch script runs the equivalent command:

```bash
python examples/sparse_reslim_forecasting/train.py \
  /lustre/orion/world-shared/lrn036/jyc/frontier/ClimaX-v2/data/ERA5-1hr-superres/1.0_deg/ \
  --max-epochs 30 \
  --batch-size 1 \
  --pred-range 120 \
  --input-vars 2m_temperature \
  --output-vars 2m_temperature \
  --keep-ratio 0.25 \
  --accelerator gpu \
  --devices 1
```

Here, `--pred-range 120` is a 120-hour (5-day) lead because this dataset is
hourly. `--keep-ratio 0.25` sends 25% of spatial tokens through the sparse
middle blocks. To use another compatible copy of the data, override the path
at submission time:

```bash
ERA5_DIR=/path/to/ERA5-1hr/1.0_deg sbatch \
  examples/sparse_reslim_forecasting/launch.sh
```

(4) Before requesting a GPU, the model and backward pass can be checked with
synthetic data:

```bash
python examples/sparse_reslim_forecasting/train.py --smoke-test
```

This compact job is intended to make the forecasting method easy to inspect
and launch. For the full ECCV work, including the paper's multi-variable ERA5
experiments, 1.40625-degree and 1.0-degree configurations, distributed runs,
and EDM model, see the
[full Sparse-Reslim ECCV repository](https://github.com/janet-sw/Sparse-Reslim).
