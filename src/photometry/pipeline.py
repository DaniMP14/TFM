from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.table import Table
from numpy import median

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from photometry.aperture import (  # noqa: E402
    choose_comparison_sources,
    choose_target_source,
    detect_sources,
    measure_aperture_flux,
)
from photometry.differential import build_differential_light_curve  # noqa: E402
from photometry.io import index_aligned_frames, resolve_target_xy  # noqa: E402
from reduction.io import load_ccd  # noqa: E402


@dataclass(frozen=True)
class PhotometryPaths:
    aligned_dir: Path
    output_dir: Path
    alignment_summary_path: Path | None = None


@dataclass(frozen=True)
class PhotometryConfig:
    target_xy: tuple[float, float] | None = None
    object_name: str = "Qatar-1"  # used for SIMBAD lookup if target_xy is not provided
    pixel_scale_arcsec: float | None = None  # arcsec/px; optional if FITS has XPIXSZ/FOCALLEN
    prefer_wcs: bool = True
    wcs_solver: str = "auto"  # auto | none | local | online
    wcs_cache_dir: Path | None = None
    astrometry_api_key: str | None = None
    wcs_solver_timeout_sec: int = 180
    persist_solved_wcs: bool = True
    detection_fwhm: float = 4.0
    detection_threshold_sigma: float = 5.0
    exclude_border: int = 15
    brightest_sources: int | None = 200
    max_comparisons: int = 5
    min_source_separation: float = 20.0
    min_edge_distance: float = 25.0
    aperture_radius: float | None = None
    aperture_scale: float = 1.8
    annulus_scale_inner: float = 3.0
    annulus_scale_outer: float = 5.0


def _build_reference_catalog(
    first_frame_path: Path,
    config: PhotometryConfig,
    min_separation_override: float | None = None,
) -> tuple[list[dict[str, float]], dict[str, float], list[dict[str, float]]]:
    reference_ccd = load_ccd(first_frame_path)
    sources = detect_sources(
        reference_ccd.data,
        fwhm=config.detection_fwhm,
        threshold_sigma=config.detection_threshold_sigma,
        exclude_border=config.exclude_border,
        brightest=config.brightest_sources,
    )
    if not sources:
        raise ValueError("No sources were detected in the reference aligned frame.")

    target_source = choose_target_source(
        sources,
        image_shape=reference_ccd.data.shape,
        target_xy=config.target_xy,
    )

    # Warn if snap is far from the requested position (possible wrong target)
    if config.target_xy is not None:
        snap_dist = float(np.hypot(
            target_source["x"] - config.target_xy[0],
            target_source["y"] - config.target_xy[1],
        ))
        if snap_dist > 3.0 * config.detection_fwhm:
            print(
                f"[photometry] WARNING: snap distance to target = {snap_dist:.1f} px "
                f"({snap_dist * config.detection_fwhm:.1f}× FWHM). "
                "Qatar-1 may not have been detected — consider lowering detection_threshold_sigma."
            )
        else:
            print(f"[photometry] Target snap OK: distance = {snap_dist:.1f} px")

    effective_min_sep = max(
        config.min_source_separation,
        min_separation_override if min_separation_override is not None else 0.0,
    )
    comparison_sources = choose_comparison_sources(
        sources,
        target_source=target_source,
        image_shape=reference_ccd.data.shape,
        max_comparisons=config.max_comparisons,
        min_separation=effective_min_sep,
        min_edge_distance=config.min_edge_distance,
    )
    if not comparison_sources:
        raise ValueError("No suitable comparison stars were found in the reference frame.")

    return sources, target_source, comparison_sources


def _resolve_aperture_radii(
    target_source: dict[str, float],
    comparison_sources: list[dict[str, float]],
    config: PhotometryConfig,
) -> tuple[float, float, float]:
    if config.aperture_radius is not None:
        aperture_radius = float(config.aperture_radius)
    else:
        fwhm_samples = [target_source.get("fwhm", config.detection_fwhm)]
        fwhm_samples.extend(source.get("fwhm", config.detection_fwhm) for source in comparison_sources)
        base_fwhm = float(median(fwhm_samples))
        aperture_radius = max(config.aperture_scale * base_fwhm, 2.0)

    annulus_r_in = max(config.annulus_scale_inner * aperture_radius, aperture_radius + 2.0)
    annulus_r_out = max(config.annulus_scale_outer * aperture_radius, annulus_r_in + 2.0)
    return aperture_radius, annulus_r_in, annulus_r_out


def run_photometry_pipeline(
    paths: PhotometryPaths,
    config: PhotometryConfig | None = None,
) -> Table:
    config = config or PhotometryConfig()

    summary_path = paths.alignment_summary_path or (paths.aligned_dir / "alignment_summary.csv")
    frame_records = index_aligned_frames(paths.aligned_dir, summary_path=summary_path)
    if not frame_records:
        raise ValueError("No aligned frames were found for photometry.")

    paths.output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve target position: explicit coords > SIMBAD lookup > image centre fallback
    resolved_target_xy = config.target_xy
    if resolved_target_xy is None and config.object_name:
        try:
            resolved_target_xy = resolve_target_xy(
                reference_frame_path=frame_records[0].path,
                object_name=config.object_name,
                pixel_scale_arcsec=config.pixel_scale_arcsec,
                prefer_wcs=config.prefer_wcs,
                wcs_solver=config.wcs_solver,
                wcs_cache_dir=config.wcs_cache_dir,
                astrometry_api_key=config.astrometry_api_key,
                wcs_solver_timeout_sec=config.wcs_solver_timeout_sec,
                persist_solved_wcs=config.persist_solved_wcs,
            )
            print(f"[photometry] Target '{config.object_name}' resolved to pixel ({resolved_target_xy[0]:.1f}, {resolved_target_xy[1]:.1f})")
        except Exception as exc:
            print(f"[photometry] SIMBAD lookup failed ({exc}). Falling back to image centre.")

    # Build a temporary config with the resolved position so _build_reference_catalog uses it
    if resolved_target_xy is not None and resolved_target_xy != config.target_xy:
        import dataclasses
        config = dataclasses.replace(config, target_xy=resolved_target_xy)

    # First pass: build catalog with default separation to compute annulus radii
    _, target_source, comparison_sources = _build_reference_catalog(frame_records[0].path, config)
    aperture_radius, annulus_r_in, annulus_r_out = _resolve_aperture_radii(
        target_source,
        comparison_sources,
        config,
    )

    # Second pass: enforce min_separation >= annulus_r_out + 5 px to avoid annulus overlap
    effective_min_sep = annulus_r_out + 5.0
    if effective_min_sep > config.min_source_separation:
        print(f"[photometry] Enforcing min_separation={effective_min_sep:.1f} px (ann_r_out={annulus_r_out:.1f} px)")
        _, target_source, comparison_sources = _build_reference_catalog(
            frame_records[0].path, config, min_separation_override=effective_min_sep
        )
        aperture_radius, annulus_r_in, annulus_r_out = _resolve_aperture_radii(
            target_source, comparison_sources, config
        )

    selected_sources: list[dict[str, float | str]] = [
        {
            "source_id": "target",
            "role": "target",
            **target_source,
        }
    ]
    selected_sources.extend(
        {
            "source_id": f"comp_{index + 1}",
            "role": "comparison",
            **source,
        }
        for index, source in enumerate(comparison_sources)
    )

    source_catalog = Table(rows=selected_sources)
    source_catalog.write(paths.output_dir / "source_catalog.csv", format="csv", overwrite=True)

    measurement_rows: list[dict[str, float | int | str | None]] = []
    positions = [(float(source["x"]), float(source["y"])) for source in selected_sources]
    for record in frame_records:
        ccd = load_ccd(record.path)
        frame_measurements = measure_aperture_flux(
            ccd.data,
            positions=positions,
            aperture_radius=aperture_radius,
            annulus_r_in=annulus_r_in,
            annulus_r_out=annulus_r_out,
        )

        for source, measurement in zip(selected_sources, frame_measurements, strict=True):
            measurement_rows.append(
                {
                    "frame_index": int(record.frame_index),
                    "file": record.path.name,
                    "time_jd": record.time_jd,
                    "exptime": record.exptime,
                    "source_id": str(source["source_id"]),
                    "role": str(source["role"]),
                    **measurement,
                }
            )

    measurement_df = pd.DataFrame(measurement_rows)
    measurement_table = Table.from_pandas(measurement_df)
    measurement_table.write(paths.output_dir / "aperture_photometry.csv", format="csv", overwrite=True)

    light_curve_df = build_differential_light_curve(measurement_df)
    light_curve_table = Table.from_pandas(light_curve_df)
    light_curve_table.write(paths.output_dir / "differential_light_curve.csv", format="csv", overwrite=True)
    return light_curve_table


if __name__ == "__main__":
    default_paths = PhotometryPaths(
        aligned_dir=Path("data/aligned"),
        output_dir=Path("data/photometry"),
    )
    run_photometry_pipeline(default_paths)