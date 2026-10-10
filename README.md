# ORBIT-2: Scaling Exascale Vision Foundation Models for Weather and Climate Downscaling and Forecasting

<div align="center">
  <img src="docs/figs/example_downscale.png" width="540px">
</div>

<div align="center">
  <img src="docs/figs/example_downscale2.png" width="640px">
</div>

This repository contains code accompanying the paper [**ORBIT-2: Scaling Exascale Vision Foundation Models for Weather and Climate Downscaling**](https://arxiv.org/pdf/2505.04802). In addition to downscaling, this repository includes a minimal example demonstrating how the ORBIT-2 innovations can be applied to weather forecasting.

## Overview
ORBIT-2 is a scalable vision foundation model for weather and climate applications, supporting both global hyper-resolution downscaling and deterministic weather forecasting. ORBIT-2 is designed to efficiently process large-scale spatiotemporal data and scale to extreme model sizes and sequence lengths on modern GPU supercomputers.

ORBIT-2 incorporates two key innovations:
 (1) Residual Slim ViT (Reslim), a lightweight architecture with residual learning and Bayesian regularization for efficient, robust prediction; and 
 (2) TILES, a tile-wise sequence scaling algorithm that reduces self-attention complexity from quadratic to linear, enabling long-sequence processing and massive parallelism. 
 ORBIT-2 scales to 10 billion parameters across 65,536 GPUs, achieving up to 4.1 ExaFLOPS sustained throughput and 74–98% strong scaling efficiency. It supports downscaling to 0.9 km global resolution and processes sequences up to 4.2 billion tokens. On 7 km resolution benchmarks, ORBIT-2 achieves high accuracy with 𝑅2 scores in range of 0.98–0.99 against observation data.

 The ORBIT-2 framework can be applied to different weather and climate learning tasks:
(1) Weather and climate downscaling: Learn mappings from coarse-resolution climate/weather fields to high-resolution predictions.
(2) Weather forecasting: Predict future atmospheric states directly from historical weather observations. A minimal Sparse-Reslim example predicts global ERA5 2-meter temperature (T2m) 120 hours (5 days) ahead at 1.0-degree resolution.

## Motivation
Conventional weather and climate modeling approaches face fundamental challenges in computational cost, resolution, and scalability. Dynamical approaches, including numerical weather prediction and nested climate models, require massive supercomputers and can take hours to weeks to generate high-resolution forecasts or downscaled climate data. Statistical and conventional machine learning approaches can be substantially faster, but often have limited generalization across regions, variables, and forecasting horizons and may not efficiently scale to the massive spatial and temporal sequences required for global modeling. ORBIT-2 addresses these challenges by combining scalable vision foundation models with exascale computing, enabling efficient learning and prediction for both weather forecasting and global hyper-resolution downscaling. Once trained, ORBIT-2 can perform predictions efficiently on modest hardware, enabling rapid inference for both forecasting and downscaling. By providing a unified, scalable foundation model that can learn from large-scale weather and climate data, ORBIT-2 offers a general framework for high-resolution prediction across diverse regions, variables, and weather and climate applications.

## Reslim Architecture
Reslim is a vision transformer (ViT) architecture that operates and trains directly on adaptively compressed spatial inputs, significantly reducing sequence length while preserving critical information. It preserves accuracy and reduces uncertainty through a lightweight residual learning architecture, enabling efficient, low-overhead predictions.

<div align="center">
  <img src="docs/figs/reslim.png" width="640px">
</div>



## TILES Sequence Scaling Algorithm
TILES is a ViT training algorithm that reduces ViT's self-attention complexity from quadratic to linear. It works by dividing images into overlapping tiles, each processed in parallel on separate Graphical Process Units (GPUs) using localized self-attention. Each tile's downscaled or forecasted outputs are then seamlessly merged to the full image.

<div align="center">
  <img src="docs/figs/TILES.png" width="400px">
</div>


## Installation
In order to run the code, we provide the instructions below for creating a conda environment with the necessary packages. Follow the instructions according to the respective GPU type (AMD or NVIDIA) that you will be utilizing.

### Frontier (Systems with AMD GPUs)
```
conda create -n orbit python=3.11 -y
conda activate orbit
pip install torch==2.8.0+rocm6.4 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/rocm6.4
pip install -U xformers==0.0.32 --index-url https://download.pytorch.org/whl/rocm6.4
MPICC="cc -shared" pip install --no-cache-dir --no-binary=mpi4py mpi4py
pip install -e .
```

### DGX (System with NVIDIA GPUs)
```
conda create -n orbit python=3.11 -y
conda activate orbit
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu129
pip3 install -U xformers==0.0.32 --index-url https://download.pytorch.org/whl/cu129
conda install -c conda-forge mpi4py mpich
pip install -e .

```


## Tutorial Example for Downscaling

### Frontier
#### Prerequisites
- Access to Frontier supercomputer at ORNL
- Allocated compute hours
- Conda environment setup (see Installation section above)

#### Step 1: Configure Your Experiment
Choose an appropriate configuration file from `configs/`:
- `interm_8m.yaml`: 8M parameter model for testing
- `interm_117m.yaml`: 117M parameter model
- `interm_1b.yaml`: 1B parameter model
- `interm_10b.yaml`: 10B parameter model (requires more nodes)

**Important:** Before running, review and modify the config file for your needs. See the [Hyperparameter Configuration](#hyperparameter-configuration) section below for detailed explanations of all parameters.

Key settings for Frontier:

**GPU Configuration:**
```yaml
trainer:
  gpu_type: "amd"      # AMD GPUs on Frontier
```

**Parallelism Configuration:**
```yaml
parallelism:
  fsdp: 4              # Fully Sharded Data Parallel
  simple_ddp: 4        # Data Parallel
  tensor_par: 1        # Tensor parallelism
  seq_par: 1           # Sequence parallelism
# Note: fsdp × simple_ddp × tensor_par_ranks should equal total number of GPUs
```

**TILES Configuration (for very large images):**
```yaml
tiling:
  do_tiling: False     # Enable TILES for processing very large images/sequences
  div: 4               # Division factor: splits image into div×div tiles
                       # div=2 → 4 tiles (2×2), div=4 → 16 tiles (4×4)
  overlap: 3           # Number of pixel rows/columns to overlap between adjacent tiles
                       # Overlap ensures smooth reconstruction when stitching tiles
                       # Important: Total tile size must be divisible by patch_size
                       # If not, you'll get an error asking to adjust overlap
```

#### Step 2: Submit Training Job
First, edit `launch_intermediate.sh` to update:
- `#SBATCH -A` → your project allocation
- `conda activate` path → your conda environment
- `../configs/interm_8m.yaml` → your chosen config file (e.g., `interm_117m.yaml`, `interm_1b.yaml`)

Then submit the job:
```bash
cd examples/
sbatch launch_intermediate.sh
```

The script will:
- Load required modules (ROCm 6.3.1, libfabric, aws-ofi-rccl)
- Set up environment variables for optimal AMD GPU performance
- Launch training with 8 GPUs per node
- Output logs to `flash-{JOBID}.out`

#### Step 3: Monitor Training
```bash
# Check job status
squeue -u $USER

# Monitor training progress
tail -f flash-{JOBID}.out
```

Key log indicators:
- Epoch progress and loss values
- GPU memory usage per rank
- Training throughput (samples/sec)
- Checkpoint save confirmations

#### Step 4: Visualize Results
After training completes or reaches a checkpoint, edit `launch_visualize.sh` to update:
- `#SBATCH -A` → your project allocation
- `conda activate` path → your conda environment
- `../configs/interm_8m_ft.yaml` → your config file

For visualization, you have two options to specify the checkpoint:

**Option 1: Using config file (default)**
```yaml
trainer:
  pretrain: /path/to/your/checkpoint.ckpt  # Path to trained model checkpoint
```

**Option 2: Using command line argument (overrides config)**
```bash
# Edit launch_visualize.sh to add --checkpoint argument:
python ./visualize.py ../configs/interm_8m_ft.yaml --checkpoint /path/to/custom/checkpoint.ckpt
```

Then submit the visualization job:
```bash
sbatch launch_visualize.sh
```

This will generate visualization outputs for:
- Input low-resolution data
- ORBIT-2 downscaled predictions at high resolution
- Comparison metrics

Additional visualization options:
- `--checkpoint PATH`: Override checkpoint path from config file

### DGX
1. Modify your [CONFIG_FILE] making sure that fsdp x simple_ddp x tensor_par_ranks = [NUM_GPUS]
- Several Example config files are available in the `configs` folder
2. Set gpu_type to `nvidia` in your config file
3. Enter the examples directory
4. Run the the training script with `mpirun -n [NUM_GPUS] python -u intermediate_downscaling.py [CONFIG_FILE] MPI` 



## Hyperparameter Configuration
To run the examples in the previous section we use a YAML config file to encapsulate all of the different available options for the model and training. Below is a description of the different available options.

- `max_epochs`: Int.
Number of epochs to be done before the training script ends. 

- `checkpoint`: Str.
Checkpoint file to use for continued training of a model. Expects the same configuration setting as the previously trained model. Starts at the epoch where the previous model stopped training.

- `pretrain`: Str.
Checkpoint file to use as a pretrained state for the model. Use this when training configurations such as data_type and amount of GPUs is changed.

- `batch_size`: Int.
Per GPU batch size.

- `buffer_size`: Int.
The size the dataloader buffer is to reach before batches are served to the training loop.

- `num_workers`: Int.
Number of data loader workers.

- `data_type`: Str (float32, bfloat16).
Data type to be used during training.

- `gpu_type`: Str (amd, nvidia).
Type of GPU being use for training.

- `train_loss`: Str (various options avaiable check `src/climate_learn/metrics`).
Loss function to use during training.

- `fsdp`: Int.
Number of Fully Sharded Data Parallel ranks to use, for sharding model states.

- `simple_ddp`: Int.
Number of Data Parallel ranks to use, for distributing different data to ranks.

- `tensor_par`: Int.
Number of Tensor Parallel ranks to use, for distributing tensor across multiple ranks.

- `do_tiling`: Bool.
Whether to perform tiling of the input data.

- `div`: Int.
Number of tiles to divide the x and y dimensions of the data into, e.g if data is (180,90) and div=2, each image is split into 2x2=4 (90,45) tiles.

- `overlap`: Int.
Number of pixel rows/columns to overlap between adjacent tiles in the TILES algorithm. Overlap regions ensure smooth reconstruction when tiles are stitched back together. Note: Due to the 2:1 aspect ratio of climate data (longitude:latitude), the actual horizontal overlap is 2x the vertical overlap


- `lr`: Float.
Initial learning rate for the optimizer.

- `weight_decay`: Float.
Weight decay for the Adam optimizer.

- `beta_1,beta_2`: Float.
Beta coefficients for the Adam optimizer.

- `warmup_epochs`: Int.
Number of warmup epochs for learnining rate scheduler.

- `warmup_start_lr`: Float.
Learning rate to use for warm up.

- `eta_min`: Float.
Coefficient for linear warmup cosine annealing

- `supperres_mag`: Int.
Scale to magnify the input data to, if supperres_mag=4 and data is (180,90) output will be (720,360).

- `cnn_ratio`: Int.
How to scale the size of the output for the convolutional neural network skip connection
**To Do: Xiao, more detailed description**

- `patch_size`: Int.  
Size of patches. Input data must be divisible by `patch_size`.

- `embed_dim`: Int.  
Embedding dimension for Transformer Inputs

- `depth`: Int.
Number of Transformer blocks.

- `decoder_depth`: Int.
Number of MLP blocks to use in the decoder.

- `num_heads`: Int.  
Number of heads in Multi-head Attention layer.

- `mlp_ratio`: Int.
Ratio of MLP hidden dimension to embedding dimension, used to set the dimension of the MLP (FeedForward) layer.

- `drop_path_rate`: Float (0,1).
Stochastic depth dropout rate for dropping random layers during training.

- `drop_rate`: Float (0,1).
Stochastic dropout rate for dropping random values from attention input in transformer computations.

- `low_res_dir`: Dictionary of Str.
Dictionary with each entry containing folder locations for files corresponding to different dataset's low resolution data. 

- `high_res_dir`: Dictionary of Str.
Dictionary with each entry containing folder locations for files corresponding to different dataset's high resolution data. 

- `spatial_resolution`: Dictionary of Int. 
Dictionary with each entry containing spatial resolution information corresponding to the data used in `low_res_dir` 

- `default_vars`: List[str].
List of different potential modalities to be used as input. This list contains the available input channels.

- `dict_in_variables`: Dictionary of Lists of Str.
Variables corresponding to the different channels used as input data for the model training.

- `dict_out_variables`: Dictionary of Lists of Str.
Variables corresponding to the different channels in the output dataset that will be used as targets from the high resolution dataset for model prediction.



## Datsets and Model Checkpoints
Training dataset and its usage description can be found at [Constellation Data Storage Archive](https://doi.ccs.ornl.gov/dataset/e4c2db1f-e88c-5ad0-bb96-59be0ef7c772).

## Pretraining and Fine-Tuning Datasets on Frontier
See Table 1 of the orbit-2 paper at [ORBIT-2 Paper on arXiv](https://arxiv.org/pdf/2505.04802) , for datasets trained for checkpoints

## Performance
### Strong Scaling on Frontier

<div align="center">
  <img src="docs/figs/strong_scaling.png" width="640px">
</div>


### Maximum Sequence Length Scaling
See Table 3 of the paper. ORBIT-2 uses Reslim architecture and TILES sequence scaling algorithm to scale up to 4.2 billion token sequence length.


### Model Inference Speed
See Table 5 of the paper. A key advantage of AI foundation models is their efficiency at inference. Once trained, they can be deployed on edge devices with limited resources and deliver near real-time predictions. We evaluate inference performance on 8 GPUs, using ERA5 to ERA5 downscaling from 112 km to 28 km. For the 9.5M-parameter model, downscaling each global sample requires only 4 millisecond. For the 10B-parameter model, it takes 0.55 second. In contrast, dynamic numerical downscaling approaches require days or weeks of computation on a large supercomputer. This highlights the unmatched prediction speed of AI, enabling deployment in resource-limited environments with near real-time performance.


### Downscaling Accuracy
See Table 6 for both U.S. regional and global downscaling accuracy. On 7 km resolution benchmarks, ORBIT-2 achieves high accuracy with 𝑅2 scores in range of 0.98–0.99 against observation data. 

### Climate Analysis
ORBIT-2 achieves high-fidelity precipitation downscaling (1998–2021) across 58 IPCC climate regions. Key metrics include R² correlation, SSIM, monsoon onset/withdrawal timing, precipitation seasonality, entropy, and Hovmöller diagnostics. The figure on the left shows ERA5 at 28 km resolution, exhibiting moderate skill scores across metrics and regions, whereas ORBIT-2 7 km downscaling (figure on the right) consistently achieves much higher skill scores across all metrics and monsoon regions. These results highlight the effectiveness of ORBIT-2 in enhancing the spatiotemporal fidelity of precipitation, especially in regions governed by complex climatic processes.

<div align="center">
  <img src="docs/figs/climate_analysis.png" width="640px">
</div>



### Animation
ERA5 input at 28km and ORBIT-2 downscaled output at 7km for global precipitation.
[ORBIT-2 animation](https://www.youtube.com/watch?v=Iahsl1L_1jQ)

### Pre-trained Models on Hugging Face
Pre-trained ORBIT-2 checkpoints and their corresponding configuration files are available on Hugging Face. These models are fine-tuned for specific regions and climate variables.

#### US Regional Models
The [US fine-tuned models](https://huggingface.co/jychoi-hpc/ORBIT-2/tree/main/us-finetune) include checkpoints and configuration files for 9.5M and 126M parameter models targeting precipitation and temperature:
- `us_9.5m_precipitation.ckpt` / `us_9.5m_precipitation.yaml`
- `us_9.5m_temperature.ckpt` / `us_9.5m_temperature.yaml`
- `us_126m_precipitation.ckpt` / `us_126m_precipitation.yaml`
- `us_126m_temperature.ckpt` / `us_126m_temperature.yaml`

#### Global Models
The [global fine-tuned models](https://huggingface.co/jychoi-hpc/ORBIT-2/tree/main/global-finetune) include checkpoints and configuration files for 9.5M and 126M parameter models targeting global precipitation:
- `global_9.5m_precipitation.ckpt` / `global_9.5m_precipitation.yaml`
- `global_126m_precipitation.ckpt` / `global_126m_precipitation.yaml`

### Citation
If you use ORBIT-2 in your research, please cite both the software and the paper:

```bibtex 
@software{Wang_ORBIT-2_Scaling_Exascale,
  author       = {Wang, Xiao and Choi, Jong-Youl and Lyngaas, Isaac and Yoon, Hong-Jun 
                  and Aji, Ashwin and Kurihana, Takuya and Fan, Ming and Balaprakash, Prasanna 
                  and Lu, Dan},
  title        = {ORBIT-2 Weather and Climate Downscaling Software Repository},
  url          = {https://github.com/XiaoWang-Github/ORBIT-2}
}
```

```bibtex
@misc{wang2025orbit2scalingexascalevision,
  title         = {ORBIT-2: Scaling Exascale Vision Foundation Models for Weather and Climate Downscaling}, 
  author        = {Xiao Wang and Jong-Youl Choi and Takuya Kurihaya and Isaac Lyngaas and Hong-Jun Yoon 
                   and Xi Xiao and David Pugmire and Ming Fan and Nasik M. Nafi and Aristeidis Tsaris 
                   and Ashwin M. Aji and Maliha Hossain and Mohamed Wahib and Dali Wang and Peter Thornton 
                   and Prasanna Balaprakash and Moetasim Ashfaq and Dan Lu},
  year          = {2025},
  eprint        = {2505.04802},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2505.04802}
}
```

## Tutorial Example for Weather Forecasting
### Data

The example uses hourly ERA5 data at 1.0-degree resolution available on Frontier:
```
/lustre/orion/world-shared/lrn036/jyc/frontier/ClimaX-v2/data/ERA5-1hr-superres/1.0_deg/
```
The data path, variables, forecast history, history window, forecast range, and
training hyperparameters are configured in
`configs/sparse_reslim_forecasting.yaml`. If you are using a different system,
update `data.data_dir` in that file.

### Run
To run the forecasting example on one GPU:
```
sbatch examples/sparse_reslim_forecasting/launch.sh
```
The launch script uses a native PyTorch training loop to run a 30-epoch,
single-step 120-hour T2m forecast with a Sparse-Reslim keep ratio of 0.25.
The best validation checkpoint is saved under
`outputs/sparse_reslim_forecasting/checkpoints/best.pt`.

Input compression and middle-layer token dropping are independent options.
Set `compression.enabled` and `model.token_dropping` in the YAML to test either
method alone, both together, or the dense baseline. Use a separate
`trainer.output_dir` for each architecture because their checkpoints are not
interchangeable.

TILES is controlled by `tiling.do_tiling`, `tiling.div`, and
`tiling.overlap`. Training streams the overlapping tiles as samples; inference
processes every tile and crops the overlap before stitching a full-resolution
forecast. The YAML values are chosen so enabling the switch is compatible with
the configured ERA5 grid, compression ratio, and patch size.

Native PyTorch distributed training is controlled by `trainer.devices` and the
`parallelism` section. Their process counts must agree with the Slurm task
count. `simple_ddp > 1` selects DDP, `fsdp > 1` selects FSDP `FULL_SHARD`, and
using both selects FSDP `HYBRID_SHARD`. Transformer activation checkpointing is
an independent YAML switch. For example, after setting `trainer.devices: 2`
and `parallelism.fsdp: 2`, launch a two-GPU smoke run with:

```bash
sbatch --nodes=1 --ntasks-per-node=2 --gpus-per-node=2 \
  examples/sparse_reslim_forecasting/launch.sh
```

Each process receives a disjoint subset of ERA5 files. Metrics are reduced
across ranks, while FSDP checkpoints are gathered to a portable full state dict
and written only by rank zero. `trainer.batch_size` is per process, so the
effective global batch size is `batch_size * trainer.devices`.

To use another forecasting configuration, set `CONFIG_PATH` when submitting:
```
CONFIG_PATH=/path/to/forecast.yaml sbatch examples/sparse_reslim_forecasting/launch.sh
```

To check the complete input-to-checkpoint pipeline using one train, validation,
and test batch on Frontier:
```
CONFIG_PATH=configs/sparse_reslim_forecasting_smoke.yaml sbatch examples/sparse_reslim_forecasting/launch.sh
```

For a quick sanity check:
```
python examples/sparse_reslim_forecasting/train.py --smoke-test
```

After training, visualize one test forecast together with its input, target,
and signed error on Frontier:
```
CONFIG_PATH=configs/sparse_reslim_forecasting_smoke.yaml \
  sbatch examples/sparse_reslim_forecasting/launch_visualize.sh
```
The figure is saved under
`outputs/sparse_reslim_forecasting_smoke/visualizations/` by default.

### Full Sparse-Reslim implementation

This example is a compact deterministic forecasting workflow built around the
ORBIT-2 Reslim and TILES components. The
[ECCV Sparse-Reslim repository](https://github.com/janet-sw/Sparse-Reslim)
contains the complete research implementation, including multi-variable
experiments, EDM probabilistic forecasting, paper configurations, and the
Sparse Residual Routing experiments.


## Citation
If you use ORBIT-2 in your research, please cite both the software and the paper:

```bibtex 
@software{Wang_ORBIT-2_Scaling_Exascale,
  author       = {Wang, Xiao and Choi, Jong-Youl and Lyngaas, Isaac and Yoon, Hong-Jun 
                  and Aji, Ashwin and Kurihana, Takuya and Fan, Ming and Balaprakash, Prasanna 
                  and Wang, Janet and Lu, Dan},
  title        = {ORBIT-2 Weather and Climate Downscaling Software Repository},
  url          = {https://github.com/XiaoWang-Github/ORBIT-2}
}
```
If you use ORBIT-2 for downscaling, cite the following paper:

```bibtex
@misc{wang2025orbit2scalingexascalevision,
  title         = {ORBIT-2: Scaling Exascale Vision Foundation Models for Weather and Climate Downscaling}, 
  author        = {Xiao Wang and Jong-Youl Choi and Takuya Kurihaya and Isaac Lyngaas and Hong-Jun Yoon 
                   and Xi Xiao and David Pugmire and Ming Fan and Nasik M. Nafi and Aristeidis Tsaris 
                   and Ashwin M. Aji and Maliha Hossain and Mohamed Wahib and Dali Wang and Peter Thornton 
                   and Prasanna Balaprakash and Moetasim Ashfaq and Dan Lu},
  year          = {2025},
  eprint        = {2505.04802},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2505.04802}
}
```
If you use ORBIT-2 for weather forecasting, cite the following paper:

```bibtex
@misc{wang2026tokensbetterforecastssparse,
      title={Less Tokens, Better Forecasts: Sparse Residual Routing for Efficient Weather Prediction}, 
      author={Janet Wang and Yunbei Zhang and Lin Zhao and Xi Xiao and Jihun Hamm and Xiao Wang},
      year={2026},
      eprint={2607.02829},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2607.02829}, 
}
```
