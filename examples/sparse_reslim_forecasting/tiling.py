"""TILES geometry for the minimal forecasting example.

The overlap convention mirrors ORBIT-2's ``NpyReader`` and ``TileProcessor``:
longitude overlap is twice latitude overlap for 2:1 climate grids.  Models see
the complete overlapping tile, while stitching keeps only each tile's core.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TileSpec:
    """Global extraction, local crop, and final placement for one tile."""

    row: int
    column: int
    y_start: int
    y_end: int
    x_start: int
    x_end: int
    core_y_start: int
    core_y_end: int
    core_x_start: int
    core_x_end: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.y_end - self.y_start, self.x_end - self.x_start

    @property
    def crop(self) -> tuple[slice, slice]:
        return (
            slice(self.core_y_start - self.y_start, self.core_y_end - self.y_start),
            slice(self.core_x_start - self.x_start, self.core_x_end - self.x_start),
        )

    @property
    def placement(self) -> tuple[slice, slice]:
        return (
            slice(self.core_y_start, self.core_y_end),
            slice(self.core_x_start, self.core_x_end),
        )


def calculate_tile_overlap(overlap: int) -> tuple[int, int, int, int]:
    """Return left, right, top, and bottom overlap using ORBIT-2 rules."""

    if overlap % 2 == 0:
        top = bottom = overlap // 2
        left = right = overlap
    else:
        top = overlap // 2
        bottom = overlap // 2 + 1
        left = overlap // 2 * 2
        right = (overlap // 2 + 1) * 2
    return left, right, top, bottom


def calculate_tile_bounds(
    tile_index: int,
    total_tiles: int,
    dimension_size: int,
    overlap_start: int,
    overlap_end: int,
) -> tuple[int, int]:
    """Reproduce the overlap-aware bounds used by ORBIT-2 ``NpyReader``."""

    if total_tiles == 1:
        return 0, dimension_size

    tile_size = dimension_size // total_tiles
    start = tile_size * tile_index
    end = tile_size * (tile_index + 1)

    if tile_index == 0:
        start += overlap_start
        end += overlap_start
    elif tile_index == total_tiles - 1:
        start -= overlap_end
        end -= overlap_end

    start -= overlap_start
    end += overlap_end
    return start, end


def build_tile_specs(
    image_size: tuple[int, int], div: int, overlap: int
) -> tuple[TileSpec, ...]:
    """Build a raster-ordered TILES plan for a same-resolution forecast."""

    if isinstance(div, bool) or not isinstance(div, int) or div < 1:
        raise ValueError("div must be a positive integer")
    if isinstance(overlap, bool) or not isinstance(overlap, int) or overlap < 0:
        raise ValueError("overlap must be a non-negative integer")

    height, width = image_size
    if height % div or width % div:
        raise ValueError(f"image size {image_size} is not divisible by div={div}")

    left, right, top, bottom = calculate_tile_overlap(overlap)
    core_height = height // div
    core_width = width // div
    specs = []
    for row in range(div):
        for column in range(div):
            y_start, y_end = calculate_tile_bounds(
                row, div, height, top, bottom
            )
            x_start, x_end = calculate_tile_bounds(
                column, div, width, left, right
            )
            core_y_start = row * core_height
            core_y_end = (row + 1) * core_height
            core_x_start = column * core_width
            core_x_end = (column + 1) * core_width
            if y_start < 0 or x_start < 0 or y_end > height or x_end > width:
                raise ValueError("overlap is too large for the requested tile grid")
            specs.append(
                TileSpec(
                    row=row,
                    column=column,
                    y_start=y_start,
                    y_end=y_end,
                    x_start=x_start,
                    x_end=x_end,
                    core_y_start=core_y_start,
                    core_y_end=core_y_end,
                    core_x_start=core_x_start,
                    core_x_end=core_x_end,
                )
            )

    shapes = {spec.shape for spec in specs}
    if len(shapes) != 1:
        raise ValueError(f"TILES produced inconsistent tile shapes: {sorted(shapes)}")
    return tuple(specs)


def extract_tile(tensor: torch.Tensor, spec: TileSpec) -> torch.Tensor:
    """Extract one tile from the final two spatial dimensions."""

    return tensor[..., spec.y_start : spec.y_end, spec.x_start : spec.x_end]


def stitch_tiles(
    tiles: Sequence[torch.Tensor],
    specs: Sequence[TileSpec],
    image_size: tuple[int, int],
) -> torch.Tensor:
    """Crop tile overlaps and reconstruct the full image in raster order."""

    if not tiles or len(tiles) != len(specs):
        raise ValueError("tiles and specs must have the same non-zero length")
    prefix_shape = tiles[0].shape[:-2]
    output = tiles[0].new_zeros((*prefix_shape, *image_size))
    for tile, spec in zip(tiles, specs):
        if tile.shape[:-2] != prefix_shape or tile.shape[-2:] != spec.shape:
            raise ValueError(
                f"tile {spec.row},{spec.column} has shape {tuple(tile.shape)}; "
                f"expected {(*prefix_shape, *spec.shape)}"
            )
        crop_y, crop_x = spec.crop
        place_y, place_x = spec.placement
        output[..., place_y, place_x] = tile[..., crop_y, crop_x]
    return output
