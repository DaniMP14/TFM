from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from astropy.nddata import CCDData
import astropy.units as u
from astropy.stats import sigma_clipped_stats, mad_std
import ccdproc

from reduction.io import array_to_ccd, ccd_median, load_ccd


def _seconds(exposure: float) -> u.Quantity:
    return float(exposure) * u.second


def _header_float(ccd: CCDData, *keys: str, default: float | None = None) -> float | None:
    for key in keys:
        value = ccd.header.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return default


def _combine_loaded_ccds(
    ccds: Iterable[CCDData],
    method: str = "median",
    sigma_clip: bool = True,
    sigma_clip_threshold: float = 3.0,
    sigma_clip_maxiters: int = 3,
) -> CCDData:
    ccd_list = list(ccds)
    if not ccd_list:
        raise ValueError("No input frames were provided.")

    if method != "median":
        raise ValueError(f"Unsupported combination method: {method}")

    if not sigma_clip:
        return array_to_ccd(ccd_median(ccd_list), unit=str(ccd_list[0].unit))

    combiner = ccdproc.Combiner(ccd_list)
    # Note: maxiters is NOT passed to sigma_clipping() because ccdproc internally
    # extracts it from **kwd without popping it, causing a duplicate keyword error
    # when forwarding **kwd to astropy.stats.sigma_clip().
    combiner.sigma_clipping(
        low_thresh=sigma_clip_threshold,
        high_thresh=sigma_clip_threshold,
        func=np.ma.median,
        dev_func=mad_std,
    )
    return combiner.median_combine()


def _sanitize_master_flat(master_flat: CCDData, clip_low: float = 0.2, clip_high: float = 5.0) -> CCDData:
    data = np.array(master_flat.data, dtype=float, copy=True)
    finite_mask = np.isfinite(data)
    if not np.any(finite_mask):
        raise ValueError("Master flat has no finite pixels.")

    median_value = float(np.median(data[finite_mask]))
    if median_value <= 0:
        raise ValueError("Master flat median is not positive.")

    # Avoid pathological divisions in flat correction (dead/hot pixels and invalid values).
    data[~finite_mask] = median_value
    data = np.clip(data, clip_low * median_value, clip_high * median_value)
    return CCDData(data, unit=master_flat.unit)


def combine_ccds(
    paths: Iterable[str | Path],
    method: str = "median",
    sigma_clip: bool = True,
    sigma_clip_threshold: float = 3.0,
    sigma_clip_maxiters: int = 3,
) -> CCDData:
    ccds = [load_ccd(path) for path in paths]
    return _combine_loaded_ccds(
        ccds,
        method=method,
        sigma_clip=sigma_clip,
        sigma_clip_threshold=sigma_clip_threshold,
        sigma_clip_maxiters=sigma_clip_maxiters,
    )


def build_master_bias(bias_paths: Iterable[str | Path]) -> CCDData:
    return combine_ccds(
        bias_paths,
        method="median",
        sigma_clip=True,
        sigma_clip_threshold=3.0,
        sigma_clip_maxiters=3,
    )


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

    return _combine_loaded_ccds(
        corrected_darks,
        method="median",
        sigma_clip=True,
        sigma_clip_threshold=3.0,
        sigma_clip_maxiters=3,
    )


def normalize_flat_frame(flat_ccd: CCDData) -> CCDData:
    _, median_value, _ = sigma_clipped_stats(flat_ccd.data, sigma=3.0, maxiters=3)
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

    master_flat = _combine_loaded_ccds(
        normalized_flats,
        method="median",
        sigma_clip=True,
        sigma_clip_threshold=3.0,
        sigma_clip_maxiters=3,
    )
    return _sanitize_master_flat(master_flat)


def calibrate_light(
    light_path: str | Path,
    master_bias: CCDData | None = None,
    master_dark: CCDData | None = None,
    master_flat: CCDData | None = None,
    dark_exposure: float | None = None,
    remove_cosmics: bool = False,
    cosmic_sigclip: float = 6.0,
    cosmic_objlim: float = 7.0,
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

    if remove_cosmics:
        gain = _header_float(light_ccd, "GAIN", default=1.0)
        readnoise = _header_float(light_ccd, "RDNOISE", "READNOIS", default=6.5)
        light_ccd = ccdproc.cosmicray_lacosmic(
            light_ccd,
            sigclip=cosmic_sigclip,
            objlim=cosmic_objlim,
            gain=gain,
            readnoise=readnoise,
        )
        # cosmicray_lacosmic attaches an uncertainty with electron units that is
        # incompatible with ADU data units and causes a ValueError on FITS write.
        # The uncertainty array is not needed downstream, so we drop it here.
        light_ccd.uncertainty = None

    return light_ccd
