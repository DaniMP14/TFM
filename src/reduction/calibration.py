from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from astropy.nddata import CCDData
import astropy.units as u
import ccdproc

from reduction.io import array_to_ccd, ccd_median, load_ccd


def _seconds(exposure: float) -> u.Quantity:
    return float(exposure) * u.second


def combine_ccds(paths: Iterable[str | Path], method: str = "median") -> CCDData:
    ccds = [load_ccd(path) for path in paths]
    if not ccds:
        raise ValueError("No input frames were provided.")

    if method == "median":
        return array_to_ccd(ccd_median(ccds), unit=str(ccds[0].unit))

    raise ValueError(f"Unsupported combination method: {method}")


def build_master_bias(bias_paths: Iterable[str | Path]) -> CCDData:
    return combine_ccds(bias_paths, method="median")


def build_master_dark(
    dark_paths: Iterable[str | Path],
    master_bias: CCDData | None = None,
) -> CCDData:
    corrected_darks: list[CCDData] = []
    for path in dark_paths:
        dark_ccd = load_ccd(path)
        if master_bias is not None:
            dark_ccd = ccdproc.subtract_bias(dark_ccd, master_bias)
        corrected_darks.append(dark_ccd)

    if not corrected_darks:
        raise ValueError("No dark frames were provided.")

    return array_to_ccd(ccd_median(corrected_darks), unit=str(corrected_darks[0].unit))


def normalize_flat_frame(flat_ccd: CCDData) -> CCDData:
    median_value = float(np.median(flat_ccd.data))
    if median_value == 0:
        raise ValueError("Flat median is zero; cannot normalize.")
    return CCDData(flat_ccd.data / median_value, unit=flat_ccd.unit)


def build_master_flat(
    flat_paths: Iterable[str | Path],
    master_bias: CCDData | None = None,
    master_dark: CCDData | None = None,
    dark_exposure: float | None = None,
    flat_exposure: float | None = None,
) -> CCDData:
    normalized_flats: list[CCDData] = []
    for path in flat_paths:
        flat_ccd = load_ccd(path)
        if master_bias is not None:
            flat_ccd = ccdproc.subtract_bias(flat_ccd, master_bias)
        if master_dark is not None and dark_exposure is not None and flat_exposure is not None:
            flat_ccd = ccdproc.subtract_dark(
                flat_ccd,
                master_dark,
                dark_exposure=_seconds(dark_exposure),
                data_exposure=_seconds(flat_exposure),
                scale=True,
            )
        normalized_flats.append(normalize_flat_frame(flat_ccd))

    if not normalized_flats:
        raise ValueError("No flat frames were provided.")

    return array_to_ccd(ccd_median(normalized_flats), unit=str(normalized_flats[0].unit))


def calibrate_light(
    light_path: str | Path,
    master_bias: CCDData | None = None,
    master_dark: CCDData | None = None,
    master_flat: CCDData | None = None,
    dark_exposure: float | None = None,
) -> CCDData:
    light_ccd = load_ccd(light_path)

    if master_bias is not None:
        light_ccd = ccdproc.subtract_bias(light_ccd, master_bias)

    if master_dark is not None and dark_exposure is not None:
        light_exposure = light_ccd.header.get("EXPTIME", light_ccd.header.get("EXPOSURE"))
        if light_exposure is None:
            raise ValueError("Light frame has no EXPTIME/EXPOSURE in header.")
        light_ccd = ccdproc.subtract_dark(
            light_ccd,
            master_dark,
            dark_exposure=_seconds(dark_exposure),
            data_exposure=_seconds(float(light_exposure)),
            scale=True,
        )

    if master_flat is not None:
        light_ccd = ccdproc.flat_correct(light_ccd, master_flat)

    return light_ccd
