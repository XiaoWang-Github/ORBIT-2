import importlib.util
from pathlib import Path
import sys

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
TILING_MODULE_PATH = (
    REPO_ROOT / "examples" / "sparse_reslim_forecasting" / "tiling.py"
)

SPEC = importlib.util.spec_from_file_location("forecast_tiling", TILING_MODULE_PATH)
forecast_tiling = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = forecast_tiling
SPEC.loader.exec_module(forecast_tiling)


def test_tiles_round_trip_without_seams():
    image = torch.arange(8 * 16).reshape(1, 1, 8, 16)
    specs = forecast_tiling.build_tile_specs((8, 16), div=2, overlap=2)

    assert len(specs) == 4
    assert {spec.shape for spec in specs} == {(6, 12)}
    tiles = [forecast_tiling.extract_tile(image, spec) for spec in specs]
    stitched = forecast_tiling.stitch_tiles(tiles, specs, (8, 16))

    assert torch.equal(stitched, image)

    era5_specs = forecast_tiling.build_tile_specs(
        (180, 360), div=2, overlap=6
    )
    assert {spec.shape for spec in era5_specs} == {(96, 192)}


def test_tiles_require_even_spatial_division():
    with pytest.raises(ValueError, match="not divisible"):
        forecast_tiling.build_tile_specs((9, 16), div=2, overlap=2)
