"""Refactored visualization utilities for climate model outputs with tiling support."""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any
import logging
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.stats import rankdata

import copy
from ..data.processing.era5_constants import VAR_TO_UNIT as ERA5_VAR_TO_UNIT
from ..data.processing.cmip6_constants import VAR_TO_UNIT as CMIP6_VAR_TO_UNIT
from climate_learn.data.processing.era5_constants import CONSTANTS

# Configure logging
logger = logging.getLogger(__name__)

# Constants
FIGURE_DPI = 100
DEFAULT_COLORMAP = "coolwarm"
FLIP_REQUIRED_SOURCES = {"ERA5", "PRISM", "DAYMET"}


@dataclass
class TileCoordinates:
    """Container for tile coordinate mappings.

    Manages three coordinate systems:
    1. Global: Position in full image
    2. Tile: Portion to extract from processed tile
    3. Result: Placement in final stitched image
    """

    # Global coordinates in the full image
    xi1: int  # Start x coordinate in input image
    xi2: int  # End x coordinate in input image
    yi1: int  # Start y coordinate in input image
    yi2: int  # End y coordinate in input image

    # Global coordinates in the full output image
    xo1: int  # Start x coordinate in output image
    xo2: int  # End x coordinate in output image
    yo1: int  # Start y coordinate in output image
    yo2: int  # End y coordinate in output image

    # Tile extraction coordinates (accounting for overlap)
    xi1t: int  # Start x to extract from input tile
    xi2t: int  # End x to extract from input tile
    yi1t: int  # Start y to extract from input tile
    yi2t: int  # End y to extract from input tile
    xo1t: int  # Start x to extract from output tile
    xo2t: int  # End x to extract from output tile
    yo1t: int  # Start y to extract from output tile
    yo2t: int  # End y to extract from output tile

    # Final placement coordinates in stitched result
    xi1r: int  # Start x in final input result
    xi2r: int  # End x in final input result
    yi1r: int  # Start y in final input result
    yi2r: int  # End y in final input result
    xo1r: int  # Start x in final output result
    xo2r: int  # End x in final output result
    yo1r: int  # Start y in final output result
    yo2r: int  # End y in final output result


@dataclass
class VisualizationConfig:
    """Configuration for visualization parameters."""

    figure_dpi: int = FIGURE_DPI
    colormap: str = DEFAULT_COLORMAP
    save_intermediate: bool = False
    compute_metrics: bool = True
    flip_for_source: bool = True
    save_numpy: bool = True
    verbose: bool = True


class TileProcessor:
    """Handles tile-based processing for large images."""

    def __init__(
        self,
        div: int,
        overlap: int,
        input_shape: Tuple[int, int],
        output_shape: Tuple[int, int],
        superres_mag: int,
    ):
        """Initialize tile processor with given parameters."""
        self.div = div  # Number of divisions per dimension
        self.overlap = overlap  # Overlap between tiles
        self.yinp, self.xinp = input_shape  # Input height and width
        self.yout, self.xout = output_shape  # Output height and width
        self.hmul = self.xout // self.xinp
        self.vmul = self.yout // self.yinp

        # Calculate overlap values
        if overlap % 2 == 0:
            self.top = self.bottom = overlap // 2
            self.left = self.right = overlap // 2 * 2
        else:
            self.left = overlap // 2 * 2
            self.right = (overlap // 2 + 1) * 2
            self.top = overlap // 2
            self.bottom = overlap // 2 + 1

        logger.debug(
            f"TileProcessor initialized: div={div}, overlap={overlap}, "
            f"input_shape={input_shape}, output_shape={output_shape}"
        )

    def get_tile_coordinates(self, hindex: int, vindex: int) -> TileCoordinates:
        """Calculate coordinates for a specific tile."""
        if self.div == 1:
            # No tiling - use full image
            return TileCoordinates(
                xi1=0,
                xi2=self.xinp,
                yi1=0,
                yi2=self.yinp,
                xo1=0,
                xo2=self.xout,
                yo1=0,
                yo2=self.yout,
                xi1t=0,
                xi2t=self.xinp,
                yi1t=0,
                yi2t=self.yinp,
                xo1t=0,
                xo2t=self.xout,
                yo1t=0,
                yo2t=self.yout,
                xi1r=0,
                xi2r=self.xinp,
                yi1r=0,
                yi2r=self.yinp,
                xo1r=0,
                xo2r=self.xout,
                yo1r=0,
                yo2r=self.yout,
            )

        # Calculate base coordinates
        coords = self._calculate_base_coords(hindex, vindex)
        coords = self._apply_overlap_adjustments(coords, hindex, vindex)
        coords = self._calculate_tile_internal_coords(coords, hindex, vindex)

        return coords

    def _calculate_base_coords(self, hindex: int, vindex: int) -> TileCoordinates:
        """Calculate base tile coordinates without overlap."""
        xi1 = self.xinp // self.div * hindex
        xi2 = self.xinp // self.div * (hindex + 1)
        yi1 = self.yinp // self.div * vindex
        yi2 = self.yinp // self.div * (vindex + 1)

        xo1 = self.xout // self.div * hindex
        xo2 = self.xout // self.div * (hindex + 1)
        yo1 = self.yout // self.div * vindex
        yo2 = self.yout // self.div * (vindex + 1)

        # Result placement is same as base for now
        return TileCoordinates(
            xi1=xi1,
            xi2=xi2,
            yi1=yi1,
            yi2=yi2,
            xo1=xo1,
            xo2=xo2,
            yo1=yo1,
            yo2=yo2,
            xi1t=0,
            xi2t=0,
            yi1t=0,
            yi2t=0,
            xo1t=0,
            xo2t=0,
            yo1t=0,
            yo2t=0,
            xi1r=xi1,
            xi2r=xi2,
            yi1r=yi1,
            yi2r=yi2,
            xo1r=xo1,
            xo2r=xo2,
            yo1r=yo1,
            yo2r=yo2,
        )

    def _apply_overlap_adjustments(
        self, coords: TileCoordinates, hindex: int, vindex: int
    ) -> TileCoordinates:
        """Apply overlap adjustments to tile coordinates."""
        # Horizontal adjustments
        if hindex == 0:
            coords.xi2 += self.left
            coords.xo2 += self.left * self.hmul
        else:
            coords.xi1 -= self.left
            coords.xo1 -= self.left * self.hmul

        if hindex == self.div - 1:
            coords.xi1 -= self.right
            coords.xo1 -= self.right * self.hmul
        else:
            coords.xi2 += self.right
            coords.xo2 += self.right * self.hmul

        # Vertical adjustments
        if vindex == 0:
            coords.yi2 += self.top
            coords.yo2 += self.top * self.vmul
        else:
            coords.yi1 -= self.top
            coords.yo1 -= self.top * self.vmul

        if vindex == self.div - 1:
            coords.yi1 -= self.bottom
            coords.yo1 -= self.bottom * self.vmul
        else:
            coords.yi2 += self.bottom
            coords.yo2 += self.bottom * self.vmul

        return coords

    def _calculate_tile_internal_coords(
        self, coords: TileCoordinates, hindex: int, vindex: int
    ) -> TileCoordinates:
        """Calculate internal tile coordinates for extraction."""
        # Horizontal internal coords
        if hindex == 0:
            coords.xi1t = 0
            coords.xi2t = self.xinp // self.div
            coords.xo1t = 0
            coords.xo2t = self.xout // self.div
        elif hindex == self.div - 1:
            coords.xi1t = self.left + self.right
            coords.xi2t = coords.xi1t + self.xinp // self.div
            coords.xo1t = (self.left + self.right) * self.hmul
            coords.xo2t = coords.xo1t + self.xout // self.div
        else:
            coords.xi1t = self.left
            coords.xi2t = coords.xi1t + self.xinp // self.div
            coords.xo1t = self.left * self.hmul
            coords.xo2t = coords.xo1t + self.xout // self.div

        # Vertical internal coords
        if vindex == 0:
            coords.yi1t = 0
            coords.yi2t = self.yinp // self.div
            coords.yo1t = 0
            coords.yo2t = self.yout // self.div
        elif vindex == self.div - 1:
            coords.yi1t = self.top + self.bottom
            coords.yi2t = coords.yi1t + self.yinp // self.div
            coords.yo1t = (self.top + self.bottom) * self.vmul
            coords.yo2t = coords.yo1t + self.yout // self.div
        else:
            coords.yi1t = self.top
            coords.yi2t = coords.yi1t + self.yinp // self.div
            coords.yo1t = self.top * self.vmul
            coords.yo2t = coords.yo1t + self.yout // self.div

        return coords


def min_max_normalize(data: np.ndarray) -> np.ndarray:
    """Normalize data to [0, 1] range.

    Handles edge case where all values are identical (returns zeros).
    """
    min_val = data.min()
    max_val = data.max()
    if max_val - min_val == 0:
        return np.zeros_like(data)
    return (data - min_val) / (max_val - min_val)


def clip_replace_constant(
    y: torch.Tensor, yhat: torch.Tensor, out_variables: List[str]
) -> torch.Tensor:
    """Clip precipitation values and replace constants with ground truth."""
    # Ensure precipitation is non-negative (physical constraint)
    try:
        prcp_index = out_variables.index("total_precipitation_24hr")
        torch.clamp_(yhat[:, prcp_index, :, :], min=0.0)
    except ValueError:
        logger.warning("total_precipitation_24hr not found in output variables")

    # Replace predicted constants with ground truth (e.g., land_sea_mask)
    for i, var in enumerate(out_variables):
        if var in CONSTANTS:
            yhat[:, i] = y[:, i]

    return yhat


def should_flip_image(src: str) -> bool:
    """Check if image should be flipped vertically based on data source."""
    return any(source in src for source in FLIP_REQUIRED_SOURCES)


def get_variable_with_units(variable: str, src: str) -> str:
    """Get variable name with units based on data source."""
    if "ERA5" in src:
        return f"{variable} ({ERA5_VAR_TO_UNIT.get(variable, '')})"
    elif src == "CMIP6":
        return f"{variable} ({CMIP6_VAR_TO_UNIT.get(variable, '')})"
    else:
        return variable


def load_test_sample(dm_vis, index: int) -> Tuple[torch.Tensor, torch.Tensor, Any, Any]:
    """Load a specific test sample from the data module."""
    counter = 0
    adj_index = None

    for batch in dm_vis.test_dataloader():
        x, y = batch[:2]
        in_variables = batch[2]
        out_variables = batch[3]

        batch_size = x.shape[0]
        if index in range(counter, counter + batch_size):
            adj_index = index - counter
            return x, y, adj_index, (in_variables, out_variables)
        counter += batch_size

    raise RuntimeError(f"Index {index} not found in test dataset")


def process_single_tile(
    model,
    x_tile: torch.Tensor,
    y_tile: torch.Tensor,
    in_variables: List[str],
    out_variables: List[str],
    out_list: List[str],
    in_channel: int,
    out_channel: int,
    in_transform,
    out_transform,
    device: torch.device,
    src: str,
    adj_index: int,
    coords: TileCoordinates,
    processor: "TileProcessor",
) -> Dict[str, np.ndarray]:
    """Process a single tile and return the results."""
    # Move to device and run inference - ensure input dtype matches model dtype
    model_dtype = next(model.parameters()).dtype
    x_tile = x_tile.to(device=device, dtype=model_dtype)
    pred = model.forward(x_tile, in_variables, out_variables)
    pred = clip_replace_constant(y_tile, pred, out_variables)

    # Process input - extract single channel and expand
    xx = x_tile[adj_index]
    temp = xx[in_channel]
    temp = temp.repeat(len(out_list), 1, 1)
    img = in_transform(temp)[out_channel].detach().cpu().numpy()

    # Process prediction
    ppred = out_transform(pred.squeeze(0))
    ppred = ppred[out_channel].detach().cpu().numpy()

    # Process ground truth
    yy = out_transform(y_tile[adj_index])
    yy = yy[out_channel].detach().cpu().numpy()

    # Apply flips if needed
    if should_flip_image(src):
        img = np.flip(img, 0)
        ppred = np.flip(ppred, 0)
        yy = np.flip(yy, 0)

        # Adjust coordinates for flipped images
        coords = adjust_coords_for_flip(coords, processor)

    return {"input": img, "prediction": ppred, "ground_truth": yy, "coords": coords}


def adjust_coords_for_flip(
    coords: TileCoordinates, processor: "TileProcessor"
) -> TileCoordinates:
    """Adjust coordinates after vertical image flip."""
    # Adjust input tile extraction coordinates after flip
    # Calculate flipped y-coordinates for tile extraction
    tile_height_with_overlap = processor.yinp // processor.div + (
        processor.top + processor.bottom
    )
    yi2tp = tile_height_with_overlap - coords.yi1t
    yi1tp = tile_height_with_overlap - coords.yi2t
    coords.yi1t = yi1tp
    coords.yi2t = yi2tp

    # Calculate flipped y-coordinates for result placement
    yi2rp = processor.yinp - coords.yi1r
    yi1rp = processor.yinp - coords.yi2r
    coords.yi1r = yi1rp
    coords.yi2r = yi2rp

    # Adjust output tile extraction coordinates after flip
    output_tile_height_with_overlap = (
        processor.yout // processor.div
        + (processor.top + processor.bottom) * processor.vmul
    )
    yo2tp = output_tile_height_with_overlap - coords.yo1t
    yo1tp = output_tile_height_with_overlap - coords.yo2t
    coords.yo1t = yo1tp
    coords.yo2t = yo2tp

    yo2rp = processor.yout - coords.yo1r
    yo1rp = processor.yout - coords.yo2r
    coords.yo1r = yo1rp
    coords.yo2r = yo2rp

    return coords


def stitch_tiles(
    tiles: List[Dict[str, Any]], processor: TileProcessor, has_ground_truth: bool
) -> Dict[str, np.ndarray]:
    """Stitch tiles together into complete images.

    Reconstructs full image from processed tiles, handling overlap regions.
    """
    # Initialize output arrays
    inputs = np.zeros((processor.yinp, processor.xinp), dtype=np.float32)
    preds = np.zeros((processor.yout, processor.xout), dtype=np.float32)
    groundtruths = None

    if has_ground_truth:
        groundtruths = np.zeros((processor.yout, processor.xout), dtype=np.float32)

    # Place each tile in the correct position
    for tile_data in tiles:
        coords = tile_data["coords"]

        # Place input
        inputs[coords.yi1r : coords.yi2r, coords.xi1r : coords.xi2r] = tile_data[
            "input"
        ][coords.yi1t : coords.yi2t, coords.xi1t : coords.xi2t]

        # Place prediction
        preds[coords.yo1r : coords.yo2r, coords.xo1r : coords.xo2r] = tile_data[
            "prediction"
        ][coords.yo1t : coords.yo2t, coords.xo1t : coords.xo2t]

        # Place ground truth if available
        if has_ground_truth and groundtruths is not None:
            groundtruths[coords.yo1r : coords.yo2r, coords.xo1r : coords.xo2r] = (
                tile_data["ground_truth"][
                    coords.yo1t : coords.yo2t, coords.xo1t : coords.xo2t
                ]
            )

    return {"input": inputs, "prediction": preds, "ground_truth": groundtruths}


def save_visualization(
    images: Dict[str, np.ndarray], config: VisualizationConfig, rank: int = 0
) -> None:
    """Save visualization images and numpy arrays.

    Only rank 0 saves to avoid file conflicts.
    """
    if rank != 0:
        return  # Only rank 0 saves files

    # Calculate min/max for consistent colormap scaling
    img_min = np.min(images["input"])
    img_max = np.max(images["input"])

    plt.figure(
        figsize=(
            images["input"].shape[1] / config.figure_dpi,
            images["input"].shape[0] / config.figure_dpi,
        )
    )
    plt.imshow(images["input"], cmap=config.colormap, vmin=img_min, vmax=img_max)
    plt.show()
    plt.savefig("0_input.png")
    plt.close()

    # Save prediction
    plt.figure(
        figsize=(
            images["prediction"].shape[1] / config.figure_dpi,
            images["prediction"].shape[0] / config.figure_dpi,
        )
    )
    plt.imshow(images["prediction"], cmap=config.colormap, vmin=img_min, vmax=img_max)
    plt.show()
    plt.savefig("0_prediction.png")
    plt.close()

    if config.save_numpy:
        np.save("0_preds.npy", images["prediction"])

    # Save ground truth if available
    if images["ground_truth"] is not None:
        plt.figure(
            figsize=(
                images["ground_truth"].shape[1] / config.figure_dpi,
                images["ground_truth"].shape[0] / config.figure_dpi,
            )
        )
        plt.imshow(
            images["ground_truth"], cmap=config.colormap, vmin=img_min, vmax=img_max
        )
        plt.show()
        plt.savefig("0_truth.png")
        plt.close()

        if config.save_numpy:
            np.save("0_truth.npy", images["ground_truth"])


def compute_metrics(
    prediction: np.ndarray, ground_truth: np.ndarray
) -> Dict[str, float]:
    """Compute evaluation metrics between prediction and ground truth."""
    if ground_truth is None:
        return {}

    # Ensure same shape
    if ground_truth.shape != prediction.shape:
        min_h = min(ground_truth.shape[0], prediction.shape[0])
        min_w = min(ground_truth.shape[1], prediction.shape[1])
        ground_truth = ground_truth[:min_h, :min_w]
        prediction = prediction[:min_h, :min_w]

    data_range = ground_truth.max() - ground_truth.min()

    metrics = {
        "psnr": peak_signal_noise_ratio(
            ground_truth, prediction, data_range=data_range
        ),
        "ssim": structural_similarity(ground_truth, prediction, data_range=data_range),
    }

    return metrics


def visualize_at_index(
    mm,
    dm,
    dm_vis,
    out_list,
    in_transform,
    out_transform,
    variable,
    src,
    device,
    div,
    overlap,
    index=0,
    tensor_par_size=1,
    tensor_par_group=None,
    config: Optional[VisualizationConfig] = None,
):
    """Main visualization function with tiling support."""
    if config is None:
        config = VisualizationConfig()

    # Setup visualization parameters
    lat, lon = dm.get_lat_lon()
    extent = [
        lon.min(),
        lon.max(),
        lat.min(),
        lat.max(),
    ]  # Geographic extent for plotting
    out_channel = dm.out_vars.index(variable)  # Index of variable in output channels
    in_channel = dm.in_vars.index(variable)  # Index of variable in input channels

    # Calculate image dimensions based on data and model configuration
    yout = len(lat)  # Output height from latitude points
    xout = len(lon)  # Output width from longitude points

    # When input and output directories are same, we need to scale output dimensions
    if dm.inp_root_dir == dm.out_root_dir:
        yout = yout * mm.superres_mag
        xout = xout * mm.superres_mag

    # Input dimensions are always lower resolution
    yinp = yout // mm.superres_mag  # Input height
    xinp = xout // mm.superres_mag  # Input width

    # Initialize tile processor
    processor = TileProcessor(div, overlap, (yinp, xinp), (yout, xout), mm.superres_mag)

    # Load test sample
    x, y, adj_index, (in_variables, out_variables) = load_test_sample(dm_vis, index)

    # Process tiles in a grid pattern
    tiles = []
    for vindex in range(div):  # Vertical tile index
        for hindex in range(div):  # Horizontal tile index
            # Get tile coordinates with overlap handling
            coords = processor.get_tile_coordinates(hindex, vindex)

            # Extract tile data from full tensors
            # x_tile: [batch, channels, height, width] for input
            x_tile = x[:, :, coords.yi1 : coords.yi2, coords.xi1 : coords.xi2]
            # y_tile: Ground truth at higher resolution
            y_tile = y[:, :, coords.yo1 : coords.yo2, coords.xo1 : coords.xo2]

            # Process tile
            tile_result = process_single_tile(
                mm,
                x_tile,
                y_tile,
                in_variables,
                out_variables,
                out_list,
                in_channel,
                out_channel,
                in_transform,
                out_transform,
                device,
                src,
                adj_index,
                coords,
                processor,
            )

            tiles.append(tile_result)

            if config.verbose:
                logger.info(f"Processed tile {vindex},{hindex}")

    # Stitch processed tiles back into full images
    # Ground truth exists when input and output directories differ
    has_ground_truth = dm.inp_root_dir != dm.out_root_dir
    images = stitch_tiles(tiles, processor, has_ground_truth)

    # Log information
    if dist.get_rank() == 0:
        logger.info(
            f"Input shape: {images['input'].shape}, "
            f"range: [{images['input'].min():.4f}, {images['input'].max():.4f}]"
        )
        logger.info(
            f"Prediction shape: {images['prediction'].shape}, "
            f"range: [{images['prediction'].min():.4f}, {images['prediction'].max():.4f}]"
        )

    # Save visualizations
    save_visualization(images, config, dist.get_rank())

    # Compute metrics if requested
    if config.compute_metrics and has_ground_truth:
        metrics = compute_metrics(images["prediction"], images["ground_truth"])
        if dist.get_rank() == 0:
            logger.info(
                f"Metrics - PSNR: {metrics['psnr']:.4f}, SSIM: {metrics['ssim']:.4f}"
            )
            print(
                f"Goodness of fit: PSNR {metrics['psnr']:.6f}, SSIM {metrics['ssim']:.6f}"
            )

    # Print shape info for backward compatibility with original code
    if dist.get_rank() == 0:
        print(
            f"img.shape {images['input'].shape}, min {images['input'].min()}, max {images['input'].max()}"
        )
        print(
            f"ppred.shape {images['prediction'].shape}, min {images['prediction'].min()}, max {images['prediction'].max()}"
        )

    return metrics if config.compute_metrics and has_ground_truth else {}


def visualize_batch(
    mm,
    batch,
    dm,
    out_list,
    in_transform,
    out_transform,
    variable,
    src,
    device,
    div,
    overlap,
    config: Optional[VisualizationConfig] = None,
    batch_idx: int = 0,
):
    """Visualize and evaluate a whole batch of samples."""
    if config is None:
        config = VisualizationConfig()

    x, y = batch[:2]
    if len(batch) >= 4:
        in_variables = batch[2]
        out_variables = batch[3]
    else:
        raise ValueError("Batch structure not supported")

    batch_size = x.shape[0]

    # Setup visualization parameters
    lat, lon = dm.get_lat_lon()
    out_channel = dm.out_vars.index(variable)
    in_channel = dm.in_vars.index(variable)

    yout = len(lat)
    xout = len(lon)

    if dm.inp_root_dir == dm.out_root_dir:
        yout = yout * mm.superres_mag
        xout = xout * mm.superres_mag

    yinp = yout // mm.superres_mag
    xinp = xout // mm.superres_mag

    processor = TileProcessor(div, overlap, (yinp, xinp), (yout, xout), mm.superres_mag)

    # Store tiles for each sample
    sample_tiles = [[] for _ in range(batch_size)]

    for vindex in range(div):
        for hindex in range(div):
            coords = processor.get_tile_coordinates(hindex, vindex)

            x_tile = x[:, :, coords.yi1 : coords.yi2, coords.xi1 : coords.xi2]
            y_tile = y[:, :, coords.yo1 : coords.yo2, coords.xo1 : coords.xo2]

            # Inference - ensure input dtype matches model dtype
            # Get model dtype from the first parameter
            model_dtype = next(mm.parameters()).dtype
            x_tile = x_tile.to(device=device, dtype=model_dtype)
            y_tile_dev = y_tile.to(device=device, dtype=model_dtype)
            
            with torch.no_grad():
                pred = mm.forward(x_tile, in_variables, out_variables)
                pred = clip_replace_constant(y_tile_dev, pred, out_variables)

            # Process each sample
            for i in range(batch_size):
                # Input processing
                xx = x_tile[i]
                temp = xx[in_channel]
                temp = temp.repeat(len(out_list), 1, 1)
                img = in_transform(temp)[out_channel].detach().cpu().numpy()

                # Prediction processing
                ppred = out_transform(pred[i])
                ppred = ppred[out_channel].detach().cpu().numpy()

                # Ground truth processing
                yy = out_transform(y_tile[i])
                yy = yy[out_channel].detach().cpu().numpy()

                current_coords = copy.deepcopy(coords)
                if should_flip_image(src):
                    img = np.flip(img, 0)
                    ppred = np.flip(ppred, 0)
                    yy = np.flip(yy, 0)
                    current_coords = adjust_coords_for_flip(current_coords, processor)

                tile_result = {
                    "input": img,
                    "prediction": ppred,
                    "ground_truth": yy,
                    "coords": current_coords
                }
                sample_tiles[i].append(tile_result)

    # Stitch and compute metrics
    batch_metrics = []
    has_ground_truth = dm.inp_root_dir != dm.out_root_dir
    
    for i in range(batch_size):
        images = stitch_tiles(sample_tiles[i], processor, has_ground_truth)
        
        # Save visualization only for the very first sample of the first batch
        if batch_idx == 0 and i == 0:
            save_visualization(images, config, dist.get_rank())

        if config.compute_metrics and has_ground_truth:
            metrics = compute_metrics(images["prediction"], images["ground_truth"])
            batch_metrics.append(metrics)
            
    return batch_metrics


# Backward compatibility functions - maintain API compatibility with original code
def visualize_sample(img, extent, title, vmin=-1, vmax=-1):
    """Legacy visualization function for single sample."""
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    cmap = plt.cm.coolwarm
    cmap.set_bad("black", 1)

    if vmin != -1 and vmax != -1:
        im = ax.imshow(img, extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        im = ax.imshow(img, extent=extent, cmap=cmap)

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig


def visualize_mean_bias(dm, mm, out_transform, variable, src):
    """Visualize mean bias between predictions and observations."""
    from tqdm import tqdm

    lat, lon = dm.get_lat_lon()
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    channel = dm.out_vars.index(variable)

    if src == "era5":
        variable_with_units = f"{variable} ({ERA5_VAR_TO_UNIT[variable]})"
    elif src == "cmip6":
        variable_with_units = f"{variable} ({CMIP6_VAR_TO_UNIT[variable]})"
    elif src == "prism":
        variable_with_units = f"Daily Max Temperature (C)"
    else:
        raise NotImplementedError(f"{src} is not a supported source")

    all_biases = []
    for batch in tqdm(dm.test_dataloader()):
        x, y = batch[:2]
        x = x.to(mm.device)
        y = y.to(mm.device)
        pred = mm.forward(x)
        pred = out_transform(pred)[:, channel].detach().cpu().numpy()
        obs = out_transform(y)[:, channel].detach().cpu().numpy()
        bias = pred - obs
        all_biases.append(bias)

    fig, ax = plt.subplots()
    all_biases = np.concatenate(all_biases)
    mean_bias = np.mean(all_biases, axis=0)

    if src == "era5":
        mean_bias = np.flip(mean_bias, 0)

    ax.imshow(mean_bias, cmap=plt.cm.coolwarm, extent=extent)
    ax.set_title(f"Mean Bias: {variable_with_units}")

    cax = fig.add_axes(
        [
            ax.get_position().x1 + 0.02,
            ax.get_position().y0,
            0.02,
            ax.get_position().y1 - ax.get_position().y0,
        ]
    )
    fig.colorbar(ax.get_images()[0], cax=cax)
    plt.show()


# based on https://github.com/oliverangelil/rankhistogram/tree/master
def rank_histogram(obs, ensemble, channel):
    """Create rank histogram for ensemble predictions."""
    obs = obs.numpy()[:, channel]
    ensemble = ensemble.numpy()[:, :, channel]
    combined = np.vstack((obs[np.newaxis], ensemble))
    ranks = np.apply_along_axis(lambda x: rankdata(x, method="min"), 0, combined)
    ties = np.sum(ranks[0] == ranks[1:], axis=0)
    ranks = ranks[0]
    tie = np.unique(ties)

    for i in range(1, len(tie)):
        idx = ranks[ties == tie[i]]
        ranks[ties == tie[i]] = [
            np.random.randint(idx[j], idx[j] + tie[i] + 1, tie[i])[0]
            for j in range(len(idx))
        ]

    hist = np.histogram(
        ranks, bins=np.linspace(0.5, combined.shape[0] + 0.5, combined.shape[0] + 1)
    )
    plt.bar(range(1, ensemble.shape[0] + 2), hist[0])
    plt.show()
