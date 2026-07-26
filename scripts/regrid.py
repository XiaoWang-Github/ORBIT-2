# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Run WeatherBench 2 regridding pipeline.

Only rectalinear grids (one dimensional lat/lon coordinates) on the input Zarr
file are supported, but irregular spacing is OK.

Example Usage:
  ```
  export BUCKET=my-bucket
  export PROJECT=my-project
  export REGION=us-central1

  python scripts/regrid.py \
    --input_path=gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr \
    --output_path=gs://$BUCKET/datasets/era5/$USER/1959-2022-6h-64x33.zarr \
    --output_chunks="time=100" \
    --longitude_nodes=64 \
    --latitude_nodes=33 \
    --latitude_spacing=equiangular_with_poles \
    --regridding_method=conservative \
    --runner=DataflowRunner \
    -- \
    --project=$PROJECT \
    --region=$REGION \
    --temp_location=gs://$BUCKET/tmp/ \
    --setup_file=./setup.py \
    --requirements_file=./scripts/dataflow-requirements.txt \
    --job_name=regrid-$USER
  ```
"""
from absl import app
from absl import flags
import apache_beam as beam
import numpy as np
from weatherbench2 import flag_utils
from weatherbench2 import regridding
import xarray_beam
import pandas as pd
import os
from tqdm import tqdm
from dask.diagnostics import ProgressBar
import xarray as xr
from filelock import FileLock, SoftFileLock, Timeout
import time

from datetime import datetime, timedelta
import sys

xr.set_options(display_max_rows=1000)


def fix_invalid_date(date_str):
    try:
        return pd.to_datetime(date_str)
    except ValueError:
        # Split and check components
        year, month, day = map(int, date_str.split("-"))
        # Get the last valid day of the given month
        last_valid_day = (datetime(year, month % 12 + 1, 1) - timedelta(days=1)).day
        # Correct the day
        corrected_day = min(day, last_valid_day)
        return pd.to_datetime(f"{year}-{month:02d}-{corrected_day:02d}")


def macro_replace(outfile, xa):
    if "%{grid_shape}" in outfile:
        nlon = len(xa["longitude"])
        nlat = len(xa["latitude"])
        grid_shape = f"{nlon}x{nlat}"

        outfile = outfile.replace("%{grid_shape}", grid_shape)

    if "time" in xa.dims:
        tmin = xa.time.min()
        tmax = xa.time.max()
        if "%{year}" in outfile:
            timestr0 = tmin.dt.strftime("%Y").item()
            outfile = outfile.replace("%{year}", f"{timestr0}")

        if "%{yearmon}" in outfile:
            timestr0 = tmin.dt.strftime("%Y%m").item()
            outfile = outfile.replace("%{yearmon}", f"{timestr0}")

        if "%{yearmonday}" in outfile:
            timestr0 = tmin.dt.strftime("%Y%m%d").item()
            outfile = outfile.replace("%{yearmonday}", f"{timestr0}")

        if "%{year_range}" in outfile:
            timestr0 = tmin.dt.strftime("%Y").item()
            timestr1 = (tmax + np.timedelta64(1, "Y").astype("timedelta64[D]")).dt.strftime("%Y").item()
            outfile = outfile.replace("%{year_range}", f"{timestr0}-{timestr1}")

        if "%{yearmon_range}" in outfile:
            timestr0 = tmin.dt.strftime("%Y%m").item()
            timestr1 = (tmax + np.timedelta64(1, "M").astype("timedelta64[D]")).dt.strftime("%Y%m").item()
            outfile = outfile.replace("%{yearmon_range}", f"{timestr0}-{timestr1}")

        if "%{yearmonday_range}" in outfile:
            timestr0 = tmin.dt.strftime("%Y%m%d").item()
            nhourly = (xa.time[1] - xa.time[0]).item() / 3600 / 1e9
            assert nhourly.is_integer()
            nhourly = int(nhourly)
            timestr1 = (
                (tmax + np.timedelta64(nhourly, "h")).dt.strftime("%Y%m%d").item()
            )
            outfile = outfile.replace("%{yearmonday_range}", f"{timestr0}-{timestr1}")

    return outfile


INPUT_PATH = flags.DEFINE_string("input_path", None, help="zarr inputs")
OUTPUT_PATH = flags.DEFINE_string("output_path", None, help="zarr outputs")
OUTPUT_CHUNKS = flag_utils.DEFINE_chunks(
    "output_chunks", "", help="desired chunking of output zarr"
)
LATITUDE_NODES = flags.DEFINE_integer(
    "latitude_nodes", None, help="number of desired latitude nodes"
)
LONGITUDE_NODES = flags.DEFINE_integer(
    "longitude_nodes", None, help="number of desired longitude nodes"
)
# REGRIDDING_SCALE = flags.DEFINE_float("scale", None, help="regridding_scale")
REGRIDDING_UNIT = flags.DEFINE_string(
    "regridding_unit", None, help="regridding unit (e.g., 0.25_deg or 15.0_arcmin)"
)
REGRIDDING_USA_ONLY = flags.DEFINE_bool("usa", False, "usa")
LATITUDE_SPACING = flags.DEFINE_enum(
    "latitude_spacing",
    "equiangular_with_poles",
    ["equiangular_with_poles", "equiangular_without_poles"],
    help="desired latitude spacing",
)
REGRIDDING_METHOD = flags.DEFINE_enum(
    "regridding_method",
    "conservative",
    ["nearest", "bilinear", "conservative"],
    help="regridding method",
)
LATITUDE_NAME = flags.DEFINE_string(
    "latitude_name", "latitude", help="Name of latitude dimension in dataset"
)
LONGITUDE_NAME = flags.DEFINE_string(
    "longitude_name", "longitude", help="Name of longitude dimension in dataset"
)
NUM_THREADS = flags.DEFINE_integer(
    "num_threads",
    None,
    help="Number of chunks to read/write in parallel per worker.",
)
RUNNER = flags.DEFINE_string("runner", None, "beam.runners.Runner")
YEAR = flags.DEFINE_integer("year", None, help="year")
MONTH = flags.DEFINE_integer("month", None, help="month")
DAY_BEGIN = flags.DEFINE_integer("day_begin", None, help="day begin")
DAY_END = flags.DEFINE_integer("day_end", None, help="day end")  ## inclusive
FILLNA = flags.DEFINE_bool("fillna", False, "fillna")
PREONLY = flags.DEFINE_bool("preonly", False, "preonly")


class ProgressDoFn(beam.DoFn):
    """A custom DoFn to track progress with tqdm."""

    def __init__(self, total_chunks):
        self.total_chunks = total_chunks
        self.progress_bar = None

    def setup(self):
        # Initialize tqdm once per worker
        if self.progress_bar is None:
            self.progress_bar = tqdm(
                total=self.total_chunks,
                desc="Processing Chunks",
                position=0,
                leave=True,
            )

    def process(self, element):
        # Update progress bar
        self.progress_bar.update(1)
        yield element

    def teardown(self):
        # Close tqdm when done
        if self.progress_bar is not None:
            self.progress_bar.close()
            self.progress_bar = None

def check_negative(v):
    k = "total_precipitation_24hr"
    if k in v:
        v[k] = v[k].where(v[k] > 0, 0.0)
    return v

def main(argv):
    # ds = xr.open_dataset("datasets/orography/GTOPO_DEM_30s.nc")
    # ds = ds.rename({"lat": "latitude", "lon": "longitude", "z": "orography"})
    # ds["orography"] = ds["orography"].astype(np.float32)
    # ds.to_zarr("datasets/orography/orography-43200x21600.zarr", mode="w")

    t0 = time.time()
    source_ds, input_chunks = xarray_beam.open_zarr(INPUT_PATH.value)
    if YEAR.value is not None:
        time0 = time1 = f"{YEAR.value}"
        if MONTH.value is not None:
            time0 = time1 = f"{YEAR.value}-{MONTH.value:02d}"
            if DAY_BEGIN.value is not None:
                time0 = f"{YEAR.value}-{MONTH.value:02d}-{DAY_BEGIN.value:02d}"
                time1 = f"{YEAR.value}-{MONTH.value:02d}-{DAY_END.value:02d}"
                time0 = fix_invalid_date(time0)
                time1 = fix_invalid_date(time1) + np.timedelta64(23, "h")  ## inclusive
        time_slice = slice(time0, time1)
        source_ds = source_ds.sel(time=time_slice)

    print("elapsed:", time.time() - t0)
    if ("2m_temperature" in source_ds and "2m_temperature_min" not in source_ds) or (
        "total_precipitation" in source_ds
        and "total_precipitation_24hr" not in source_ds
    ):
        full_ds, _ = xarray_beam.open_zarr(INPUT_PATH.value)
        nhourly = (source_ds.time[1] - source_ds.time[0]).item() / 3600 / 1e9
        assert nhourly.is_integer()
        nhourly = int(nhourly)
        nsamples = 24 // nhourly

        if source_ds.time[0] > source_ds.time[0]:
            extra = full_ds.sel(
                time=slice(
                    source_ds.time[0] - np.timedelta64(24 - nhourly, "h"),
                    source_ds.time[0] - 1,
                )
            )
        else:
            extra = full_ds.sel(
                time=slice(
                    source_ds.time[0],
                    source_ds.time[0] + np.timedelta64(24 - nhourly, "h") - 1,
                )
            )
            extra = extra.assign_coords(
                time=extra.time - np.timedelta64(24 - nhourly, "h")
            )

        var_list = list()
        for var in ["sea_surface_temperature", "2m_temperature", "total_precipitation"]:
            if var in source_ds:
                var_list.append(var)

        combined = xr.concat(
            [
                extra[var_list],
                source_ds[var_list],
            ],
            dim="time",
        )

        ## handle nan values in sea_surface_temperature
        if "sea_surface_temperature" in combined:
            combined["2m_temperature_combined"] = combined[
                "sea_surface_temperature"
            ].combine_first(combined["2m_temperature"])
        elif "2m_temperature" in combined:
            combined["2m_temperature_combined"] = combined["2m_temperature"]

        if "2m_temperature_combined" in combined:
            source_ds["2m_temperature_min"] = (
                combined["2m_temperature_combined"]
                .rolling(time=nsamples, center=False)
                .min()
                .dropna("time")
                .compute()
            )

            source_ds["2m_temperature_max"] = (
                combined["2m_temperature_combined"]
                .rolling(time=nsamples, center=False)
                .max()
                .dropna("time")
                .compute()
            )

        if "total_precipitation" in combined:
            source_ds["total_precipitation_24hr"] = (
                combined["total_precipitation"]
                .rolling(time=nsamples, center=False)
                .sum()
                .dropna("time")
                .compute()
            )

    print("source_ds:", source_ds)
    print("elapsed:", time.time() - t0)

    # ## Temporary (extra only)
    # selected_vars = [
    #     "sea_surface_temperature",
    #     "2m_temperature",
    #     "total_precipitation_24hr",
    #     "2m_temperature_min",
    #     "2m_temperature_max",
    #     "volumetric_soil_water_layer_1",
    #     "10m_u_component_of_wind",
    #     "10m_v_component_of_wind",
    # ]
    # selected_vars_ = list()
    # for var in selected_vars:
    #     if var in source_ds:
    #         selected_vars_.append(var)
    # source_ds = source_ds[selected_vars_]
    # del input_chunks["level"]

    # if PREONLY.value:
    #     source_ds["2m_temperature_combined"] = source_ds[
    #         "sea_surface_temperature"
    #     ].combine_first(source_ds["2m_temperature"])

    #     source_ds = source_ds[
    #         ["2m_temperature_min", "2m_temperature_max", "total_precipitation_24hr", "2m_temperature_combined"]
    #     ]

    #     output_chunks = OUTPUT_CHUNKS.value
    #     print("OUTPUT_CHUNKS:", repr(output_chunks))
    #     source_ds = source_ds.chunk(output_chunks)

    #     output_path = OUTPUT_PATH.value
    #     output_path = macro_replace(output_path, source_ds)
    #     print("output_path:", output_path)

    #     with ProgressBar():
    #         source_ds.to_zarr(output_path, mode="w")
    #     sys.exit()

    # Rename latitude/longitude names
    renames = {
        LONGITUDE_NAME.value: "longitude",
        LATITUDE_NAME.value: "latitude",
    }
    source_ds = source_ds.rename(renames)
    input_chunks = {renames.get(k, k): v for k, v in input_chunks.items()}

    # Lat/lon must be single chunk for regridding.
    input_chunks["longitude"] = -1
    input_chunks["latitude"] = -1
    if "level" in input_chunks:
        input_chunks["level"] = -1

    us_bounds = (
        24,
        53,
        235,
        293.5,
    )  # (lat_min, lat_max, lon_min, lon_max) or (125W, 66.5W)
    us_lat_min, _, us_lon_min, _ = us_bounds

    ## Common
    ## handle nan values in sea_surface_temperature
    if "sea_surface_temperature" in source_ds:
        source_ds["2m_temperature"] = source_ds["sea_surface_temperature"] = source_ds[
            "sea_surface_temperature"
        ].combine_first(source_ds["2m_temperature"])

    ## Global
    print("elapsed:", time.time() - t0)
    if REGRIDDING_USA_ONLY.value is not True:
        print("Global")
        ## orography
        if (source_ds.longitude < 0).any():
            source_ds = source_ds.assign_coords(
                longitude=(source_ds.longitude + 360) % 360
            )

        if LATITUDE_SPACING.value == "equiangular_with_poles":
            lat_start = -90
            lat_stop = 90
        else:
            assert LATITUDE_SPACING.value == "equiangular_without_poles"
            lat_start = -90 + 0.5 * 180 / LATITUDE_NODES.value
            lat_stop = 90 - 0.5 * 180 / LATITUDE_NODES.value

        source_ds = source_ds.sortby(["longitude", "latitude"])
        old_lon = source_ds.coords["longitude"].data
        old_lat = source_ds.coords["latitude"].data

        new_lon = np.linspace(0, 360, num=LONGITUDE_NODES.value, endpoint=False)
        new_lat = np.linspace(
            lat_start, lat_stop, num=LATITUDE_NODES.value, endpoint=True
        )

        if "era5-imerg" in OUTPUT_PATH.value:
            ## temporary for ERA5-IMERGE
            ## Filter variables
            const_variables = ["geopotential_at_surface", "orography", "landcover"]
            era5_selected_variables = [
                "u_component_of_wind",
                "v_component_of_wind",
                "temperature",
                "specific_humidity",
                "geopotential",
                "sea_surface_temperature",
                "geopotential_at_surface",
                "2m_temperature",
                "total_precipitation_24hr",
                "2m_temperature_min",
                "2m_temperature_max",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "volumetric_soil_water_layer_1",
            ]
            era5_selected_levels = [200, 500, 850]

            data_variables = set(list(source_ds.data_vars))
            selected_const_variables = data_variables & set(const_variables)
            selected_var_variables = data_variables & set(era5_selected_variables)
            selected_variables = selected_const_variables | selected_var_variables

            source_ds = source_ds[selected_variables]

            ## Filter levels
            if "level" in source_ds.coords:
                source_ds = source_ds.sel(level=era5_selected_levels)

            ## Filter hours
            if "time" in source_ds:
                ## Convert n-hourly (n<24) to daily
                nhourly = (source_ds.time[1] - source_ds.time[0]).item() / 3600 / 1e9
                assert nhourly.is_integer()
                nhourly = int(nhourly)
                if nhourly < 24:
                    samples_per_day = 24 // nhourly
                    source_ds_1dy = source_ds.coarsen(
                        time=samples_per_day
                    ).mean()  ## fixed daily bins
                    source_ds_1dy["time"] = source_ds_1dy.time.dt.floor("D")
                    source_ds = source_ds_1dy

    else:
        print("USA only")
        ## USA region only
        ## prism, daymet
        ## convert to ERA5 longitude 0-360
        if (source_ds.longitude < 0).any():
            source_ds = source_ds.assign_coords(
                longitude=(source_ds.longitude + 360) % 360
            )

        ## Filter variables
        const_variables = ["geopotential_at_surface", "orography", "landcover"]
        prism_selected_variables = ["land_sea_mask", "prcp", "tmax", "tmin"]
        era5_selected_variables = [
            "u_component_of_wind",
            "v_component_of_wind",
            "temperature",
            "specific_humidity",
            "geopotential",
            "sea_surface_temperature",
            "geopotential_at_surface",
            "2m_temperature",
            "total_precipitation_24hr",
            "2m_temperature_min",
            "2m_temperature_max",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "volumetric_soil_water_layer_1",
        ]
        era5_selected_levels = [200, 500, 850]

        data_variables = set(list(source_ds.data_vars))
        selected_const_variables = data_variables & set(const_variables)
        if "prism" in INPUT_PATH.value or "daymet" in INPUT_PATH.value:
            selected_var_variables = data_variables & set(prism_selected_variables)
        elif "era5" in INPUT_PATH.value:
            selected_var_variables = data_variables & set(era5_selected_variables)
        else:
            selected_var_variables = data_variables
        selected_variables = selected_const_variables | selected_var_variables

        source_ds = source_ds[selected_variables]

        # ## Temporary (extra only)
        # selected_vars = [
        #         "sea_surface_temperature",
        #         "2m_temperature",
        #         "total_precipitation_24hr",
        #         # "2m_temperature_min",
        #         # "2m_temperature_max",
        #         # "volumetric_soil_water_layer_1",
        #         # "10m_u_component_of_wind",
        #         # "10m_v_component_of_wind",
        # ]
        # source_ds = source_ds[selected_vars]
        # del input_chunks["level"]

        ## Filter levels
        if "level" in source_ds.coords:
            source_ds = source_ds.sel(level=era5_selected_levels)

        ## Filter hours
        # source_ds = source_ds.sel(time=source_ds.time.dt.hour == 0)
        # source_ds = source_ds.resample(time="1D").mean() ## slow
        # source_ds = source_ds.groupby(source_ds.time.dt.floor("D")).mean() ## time var changes
        if "time" in source_ds and ("cmip6" not in INPUT_PATH.value):
            ## Convert n-hourly (n<24) to daily
            nhourly = (source_ds.time[1] - source_ds.time[0]).item() / 3600 / 1e9
            assert nhourly.is_integer()
            nhourly = int(nhourly)
            if nhourly < 24:
                samples_per_day = 24 // nhourly
                source_ds_1dy = source_ds.coarsen(
                    time=samples_per_day
                ).mean()  ## fixed daily bins
                source_ds_1dy["time"] = source_ds_1dy.time.dt.floor("D")
                source_ds = source_ds_1dy

        source_ds = source_ds.sortby(["longitude", "latitude"])
        old_lon = np.sort(source_ds.coords["longitude"].data)
        old_lat = np.sort(source_ds.coords["latitude"].data)

        ## arcmin
        regrid_unit, unit = REGRIDDING_UNIT.value.split("_")
        regrid_unit = float(regrid_unit)
        if unit == "deg":
            regrid_unit = regrid_unit * 60

        lon_start = us_lon_min
        lon_interval = regrid_unit / 60  # arcmin to deg
        new_lon = lon_start + lon_interval * np.arange(LONGITUDE_NODES.value)

        lat_start = us_lat_min
        lat_interval = regrid_unit / 60  # arcmin to deg
        new_lat = lat_start + lat_interval * np.arange(LATITUDE_NODES.value)

        print("lon start:", lon_start)
        print("lat start:", lat_start)
        print("interval:", lon_interval * 60, lat_interval * 60)
        print("gridshape:", f"{len(new_lon)}x{len(new_lat)}")

    output_chunks = OUTPUT_CHUNKS.value
    print("OUTPUT_CHUNKS:", repr(output_chunks))
    print("elapsed:", time.time() - t0)

    output_path = OUTPUT_PATH.value
    if "%{grid_shape}" in output_path:
        output_path = output_path.replace(
            "%{grid_shape}", f"{len(new_lon)}x{len(new_lat)}"
        )

    output_path = macro_replace(output_path, source_ds)

    print("output_path:", output_path)
    if os.path.exists(output_path):
        print("Skip:", output_path)
        return

    dirname = os.path.dirname(output_path)
    basename = os.path.basename(output_path)
    lockfile = os.path.join(dirname, "." + basename + ".lock")
    lock = SoftFileLock(lockfile, timeout=0)
    try:
        lock.acquire()
    except Timeout:
        ## someone is working
        return

    regridder_cls = {
        "nearest": regridding.NearestRegridder,
        "bilinear": regridding.BilinearRegridder,
        "conservative": regridding.ConservativeRegridder,
    }[REGRIDDING_METHOD.value]

    source_grid = regridding.Grid.from_degrees(lon=old_lon, lat=old_lat)
    target_grid = regridding.Grid.from_degrees(lon=new_lon, lat=new_lat)
    regridder = regridder_cls(source_grid, target_grid)

    print("source_ds:", source_ds)
    print("input_chunks:", input_chunks)
    print("output_chunks:", output_chunks)

    template = (
        xarray_beam.make_template(source_ds)
        .isel(longitude=0, latitude=0, drop=True)
        .expand_dims(longitude=new_lon, latitude=new_lat)
        .transpose(..., "longitude", "latitude")
    )

    chunked_ds = source_ds.chunk(input_chunks)
    # Calculate total number of chunks
    total_chunks = sum(
        np.prod([len(c) for c in da.chunks]) for da in chunked_ds.data_vars.values()
    )
    print(f"Total number of chunks across all variables: {total_chunks}")    

    with ProgressBar():
        with beam.Pipeline(runner=RUNNER.value, argv=argv) as root:
            _ = (
                root
                | xarray_beam.DatasetToChunks(
                    source_ds,
                    input_chunks,
                    split_vars=True,
                    num_threads=NUM_THREADS.value,
                )
                | "Progress" >> beam.ParDo(ProgressDoFn(total_chunks))
                | "Regrid"
                >> beam.MapTuple(lambda k, v: (k, regridder.regrid_dataset(v)))
                | "ResetNegative"
                >> beam.MapTuple(lambda k, v: (k, check_negative(v)))
                | xarray_beam.ConsolidateChunks(output_chunks)
                | xarray_beam.ChunksToZarr(
                    output_path,
                    template,
                    output_chunks,
                    num_threads=NUM_THREADS.value,
                )
            )

        # ds = xr.open_zarr(output_path)
        # print("NULL?:", ds.isnull().any().compute())

    lock.release()
    print("Done.")
    print("elapsed:", time.time() - t0)


if __name__ == "__main__":
    from dask.distributed import Client

    # Create a Dask client
    client = Client()
    print(f"Number of Dask workers: {len(client.nthreads())}")

    app.run(main)
