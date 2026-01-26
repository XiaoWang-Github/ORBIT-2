# ORBIT-2 PTQ 퀀타이징 사용 가이드

## 개요

Res-slim-VIT 모델에 Post-Training Quantization (PTQ)를 적용하여 attention 레이어를 INT8로 퀀타이징하고, CNN 레이어는 FP16/32로 유지하는 하이브리드 전략을 구현했습니다.

## 빠른 시작

### 1. 기본 사용법 (하이브리드 퀀타이징)

```bash
# Attention 레이어만 INT8로 퀀타이징 (권장)
python visualize.py ../configs/interm_8m.yaml --quantize --index 0
```

### 2. 전체 모델 퀀타이징

```bash
# 모든 레이어를 INT8로 퀀타이징 (권장하지 않음)
python visualize.py ../configs/interm_8m.yaml --quantize-all --index 0
```

### 3. 체크포인트 지정

```bash
# 특정 체크포인트 사용
python visualize.py ../configs/interm_8m.yaml \
    --quantize \
    --checkpoint /path/to/checkpoint.ckpt \
    --index 0
```

## 명령행 옵션

- `--quantize`: Attention 레이어만 INT8 퀀타이징 (하이브리드 전략)
- `--quantize-all`: 모든 레이어 INT8 퀀타이징 (비권장)
- `--checkpoint`: 체크포인트 파일 경로
- `--index`: 시각화할 샘플 인덱스 (기본: 0)
- `--variable`: 시각화할 변수 (기본: total_precipitation_24hr)

## Frontier 슈퍼컴퓨터에서 실행

### Slurm 스크립트 예제

기존 `launch_visualize.sh`를 수정하여 퀀타이징을 적용:

```bash
#!/bin/bash
#SBATCH -A <your_project>
#SBATCH -J orbit2_quant
#SBATCH -o quant-%j.out
#SBATCH -t 00:30:00
#SBATCH -p batch
#SBATCH -N 1

# Load modules (Frontier)
module load PrgEnv-gnu
module load rocm/6.3.1
module load libfabric/1.15.2.0
module load aws-ofi-rccl/rocm6.3.1_2a27110

# Activate environment
source /path/to/conda/etc/profile.d/conda.sh
conda activate orbit

# Run with quantization
srun -N1 -n8 -c7 --gpus-per-task=1 --gpu-bind=closest \
    python ./visualize.py ../configs/interm_8m.yaml \
    --quantize \
    --index 0 \
    --variable total_precipitation_24hr
```

## 출력 해석

퀀타이징 적용 시 다음과 같은 정보가 출력됩니다:

```
================================================================================
POST-TRAINING QUANTIZATION (PTQ) ENABLED
================================================================================

Environment Information:
  PyTorch version: 2.8.0+rocm6.4
  CUDA available: True
  ROCm available: True
  ROCm version: 6.4
  Device: AMD INSTINCT MI250X
  Quantization available: True

✓ Environment check passed

================================================================================
APPLYING DYNAMIC QUANTIZATION
================================================================================
Strategy: Hybrid (Attention INT8, CNN FP16/32)

Quantizing X Linear layers in attention modules...
  Quantizing: blocks.0.attn.qkv
  Quantizing: blocks.0.attn.proj
  ...

✓ Dynamic quantization applied successfully
================================================================================

================================================================================
MODEL QUANTIZATION SUMMARY
================================================================================
Layer Name                                         Precision      Parameters
--------------------------------------------------------------------------------
blocks.0.attn.qkv                                  INT8               ...
blocks.0.attn.proj                                 INT8               ...
path2.0                                            FP32               ...
conv_out                                           FP32               ...
...
--------------------------------------------------------------------------------
Total parameters: XXX
Quantized parameters (INT8): YYY
Quantization ratio: ZZ.ZZ%
================================================================================
```

## 예상 성능 (AMD MI250X)

- **모델 크기 감소**: ~75% (INT8 attention + FP16 CNN)
- **추론 속도**: 1.2-2x 향상 (보수적 예측)
- **정확도 손실**: < 5% 목표

## 문제 해결

### Quantization not available 에러

```
WARNING: PyTorch quantization not available!
```

**해결책**: PyTorch 2.8.0+rocm6.4가 올바르게 설치되었는지 확인

```bash
python -c "import torch; print(torch.__version__)"
python -c "import torch.quantization; print('OK')"
```

### Import 에러

```
ModuleNotFoundError: No module named 'climate_learn.utils.quantization_utils'
```

**해결책**: 프로젝트를 editable mode로 재설치

```bash
pip install -e .
```

## 다음 단계

1. **성능 벤치마크**: 퀀타이징 전후 속도 비교
2. **정확도 평가**: 원본 모델 대비 메트릭 비교
3. **최적화**: AMD Quark 라이브러리 시도 (성능 향상 미흡 시)
4. **QAT**: 정확도 손실이 클 경우 Quantization-Aware Training 고려

## 참고

- Implementation Plan: `implementation_plan.md`
- 코드: `src/climate_learn/utils/quantization_utils.py`
- Slurm 스크립트: `examples/launch_visualize.sh`
