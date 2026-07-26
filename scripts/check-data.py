"""
Inspect an ORBIT-2 NPZ file and optionally plot variables.

Usage:
    python check-data.py <file.npz>
    python check-data.py <file.npz> --plot 2m_temperature --time 0

The script prints a summary table (name, shape, min, max, dtype) for every
variable in the file.  With --plot, it saves a 2D imshow PNG for each named
variable at the requested time index.  Arrays with shape (time, 1, lat, lon)
are handled automatically: the level dimension is squeezed out and the
(lat, lon) slice at the given time index is plotted.
"""

import numpy as np
import argparse
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


def plot_variables(data, var_names, out_dir=".", time=0):
    """Plot a 2D imshow for each variable in var_names at the given time index.

    Args:
        data: NpzFile object returned by np.load.
        var_names: List of variable names to plot.
        out_dir: Directory where PNG files are saved.
        time: Time index to extract from the first array dimension.
    """
    for var_name in var_names:
        if var_name not in data.files:
            print(f"Warning: '{var_name}' not found in file, skipping.")
            continue

        arr = data[var_name]

        # Extract 2D slice: time=0, squeeze out any size-1 dims, take last two as (lat, lon)
        if arr.ndim < 2:
            print(
                f"Warning: '{var_name}' has shape {arr.shape}, need at least 2D to imshow, skipping."
            )
            continue

        img = arr[time] if arr.ndim >= 3 else arr
        img = np.squeeze(img)  # drop size-1 dims (e.g. the level dim)
        if img.ndim != 2:
            img = img.reshape(img.shape[-2], img.shape[-1])

        doy = (
            int(data["days_of_year"][time, 0, 0, 0])
            if "days_of_year" in data.files
            else "?"
        )
        tod = (
            int(data["time_of_day"][time, 0, 0, 0])
            if "time_of_day" in data.files
            else "?"
        )

        suffix = f"_t{time}"
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(
            img,
            origin="lower",
            aspect="auto",
            interpolation="None",
            vmin=np.nanmin(img),
            vmax=np.nanmax(img),
        )
        fig.colorbar(im, ax=ax)
        ax.set_title(f"{var_name}  |  days of year={doy}  time of day={tod}")
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        out_path = os.path.join(out_dir, f"{var_name}{suffix}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect an NPZ file and optionally plot variables"
    )
    parser.add_argument("file", type=str, help="Path to the .npz file")
    parser.add_argument(
        "--plot", nargs="+", metavar="VAR", help="Variable name(s) to plot with imshow"
    )
    parser.add_argument(
        "--out",
        type=str,
        default=".",
        help="Output directory for plot images (default: current dir)",
    )
    parser.add_argument(
        "--time", type=int, default=0, help="Time index to plot (default: 0)"
    )
    args = parser.parse_args()

    data = np.load(args.file)

    # Print summary table
    col_w = max(len(v) for v in data.files)
    header = (
        f"{'Variable':<{col_w}}  {'Shape':<20}  {'Min':>14}  {'Max':>14}  {'Dtype':<10}"
    )
    print("=" * len(header))
    print(header)
    print("=" * len(header))
    for var_name in data.files:
        v = data[var_name]
        print(
            f"{var_name:<{col_w}}  {str(v.shape):<20}  {v.min():>14.6g}  {v.max():>14.6g}  {str(v.dtype):<10}"
        )
    print("=" * len(header))

    if args.plot:
        os.makedirs(args.out, exist_ok=True)
        plot_variables(data, args.plot, out_dir=args.out, time=args.time)
