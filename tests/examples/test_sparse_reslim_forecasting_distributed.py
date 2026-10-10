import importlib.util
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
DISTRIBUTED_MODULE_PATH = (
    REPO_ROOT / "examples" / "sparse_reslim_forecasting" / "distributed.py"
)

SPEC = importlib.util.spec_from_file_location(
    "forecast_distributed", DISTRIBUTED_MODULE_PATH
)
forecast_distributed = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = forecast_distributed
SPEC.loader.exec_module(forecast_distributed)


@pytest.mark.parametrize(
    ("fsdp", "ddp", "expected"),
    (
        (1, 1, "single"),
        (1, 2, "ddp"),
        (2, 1, "fsdp"),
        (2, 2, "hybrid_fsdp"),
    ),
)
def test_parallelism_mode(fsdp, ddp, expected):
    parallelism = {
        "fsdp": fsdp,
        "simple_ddp": ddp,
        "tensor_par": 1,
        "seq_par": 1,
    }

    assert forecast_distributed.parallelism_mode(parallelism) == expected
    assert forecast_distributed.expected_world_size(parallelism) == fsdp * ddp


def test_parallelism_mode_rejects_tensor_parallelism():
    parallelism = {
        "fsdp": 1,
        "simple_ddp": 1,
        "tensor_par": 2,
        "seq_par": 1,
    }

    with pytest.raises(ValueError, match="tensor_par=1"):
        forecast_distributed.parallelism_mode(parallelism)
