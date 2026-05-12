from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.stats import sigma_clipped_stats
from astropy.table import Table

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
    detection_fwhm: float = 3.5 # in pixels; used for source detection and as a reference for aperture size
    detection_threshold_sigma: float = 5.0
    exclude_border: int = 15
    brightest_sources: int | None = 200
    max_comparisons: int = 12
    min_source_separation: float = 20.0
    min_edge_distance: float = 25.0
    min_neighbor_distance: float | None = None
    comparison_brightness_range: tuple[float, float] = (0.3, 3.0)
    aperture_radius: float | None = None
    aperture_scale: float = 1.5
    annulus_scale_inner: float = 1.7
    annulus_scale_outer: float = 2.4
    adaptive_photometry: bool = True
    recenter_sources_per_frame: bool = True
    centroid_box_size: int = 15
    aperture_radius_min: float = 2.0
    aperture_radius_max: float = 20.0
    fwhm_clip_range: tuple[float, float] = (1.5, 12.0) # in pixels; used to filter out sources with unrealistic FWHM estimates


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
    ap_r_est = max(config.aperture_scale * config.detection_fwhm, 2.0)
    ann_r1_est = max(config.annulus_scale_inner * ap_r_est, ap_r_est + 2.0)
    ann_r2_est = max(config.annulus_scale_outer * ap_r_est, ann_r1_est + 2.0)
    effective_min_neighbor_distance = (
        config.min_neighbor_distance
        if config.min_neighbor_distance is not None
        else ann_r2_est + ap_r_est + 2.0
    )
    comparison_sources = choose_comparison_sources(
        sources,
        target_source=target_source,
        image_shape=reference_ccd.data.shape,
        max_comparisons=config.max_comparisons,
        min_separation=effective_min_sep,
        min_edge_distance=config.min_edge_distance,
        min_neighbor_distance=effective_min_neighbor_distance,
        brightness_range=config.comparison_brightness_range,
    )
    if not comparison_sources:
        raise ValueError("No suitable comparison stars were found in the reference frame.")

    return sources, target_source, comparison_sources


def _resolve_aperture_radii(
    target_source: dict[str, float],
    config: PhotometryConfig,
) -> tuple[float, float, float]:
    if config.aperture_radius is not None:
        aperture_radius = float(config.aperture_radius)
    else:
        base_fwhm = float(target_source.get("fwhm", config.detection_fwhm))
        aperture_radius = max(config.aperture_scale * base_fwhm, 2.0)

    annulus_r_in = max(config.annulus_scale_inner * aperture_radius, aperture_radius + 2.0)
    annulus_r_out = max(config.annulus_scale_outer * aperture_radius, annulus_r_in + 2.0)
    return aperture_radius, annulus_r_in, annulus_r_out


# La función _recenter_position calcula el centroide de una fuente en una imagen dada una posición inicial, 
# utilizando un método de centroide de momento dentro de una caja alrededor de la posición inicial. 
# Esto ayuda a mejorar la precisión de la fotometría al ajustar las posiciones de las fuentes para cada cuadro.
def _recenter_position(
    data: np.ndarray,
    x: float,
    y: float,
    box_size: int,
) -> tuple[float, float]:
    half = max(int(box_size // 2), 2)
    height, width = data.shape
    ix = int(round(x))
    iy = int(round(y))
    x0, x1 = max(ix - half, 0), min(ix + half + 1, width)
    y0, y1 = max(iy - half, 0), min(iy + half + 1, height)
    patch = data[y0:y1, x0:x1]
    if patch.size == 0:
        return x, y

    # Robust background and hot-pixel suppression using sigma clipping.
    _, background_median, background_std = sigma_clipped_stats(patch, sigma=3.0, maxiters=5)
    if not np.isfinite(background_median):
        background_median = float(np.median(patch))
    if not np.isfinite(background_std) or background_std <= 0.0:
        background_std = float(np.std(patch))

    hot_threshold = float(background_median + 6.0 * max(background_std, 1e-6))
    clipped_patch = np.clip(patch, a_min=None, a_max=hot_threshold)
    signal = np.clip(clipped_patch - float(background_median), a_min=0.0, a_max=None)
    total = float(signal.sum())
    if total <= 0.0:
        return x, y

    yy, xx = np.indices(signal.shape)
    cx = float((signal * (xx + x0)).sum() / total)
    cy = float((signal * (yy + y0)).sum() / total)
    return cx, cy


# La función _estimate_fwhm estima el FWHM de una fuente en una imagen utilizando un método de momento para calcular el centroide y la dispersión de la señal, 
# y luego convierte esa dispersión a FWHM asumiendo una forma gaussiana. Esto permite obtener una medida de la extensión de la fuente en la imagen.
def _estimate_fwhm(
    data: np.ndarray,
    x: float,
    y: float,
    box_size: int,
) -> float | None:
    half = max(int(box_size // 2), 3)
    height, width = data.shape
    ix = int(round(x))
    iy = int(round(y))
    x0, x1 = max(ix - half, 0), min(ix + half + 1, width)
    y0, y1 = max(iy - half, 0), min(iy + half + 1, height)
    patch = data[y0:y1, x0:x1]
    if patch.size == 0:
        return None

    background = float(np.median(patch))
    signal = np.clip(patch - background, a_min=0.0, a_max=None)
    total = float(signal.sum())
    if total <= 0.0:
        return None

    yy, xx = np.indices(signal.shape)
    x_abs = xx + x0
    y_abs = yy + y0
    cx = float((signal * x_abs).sum() / total)
    cy = float((signal * y_abs).sum() / total)
    var_x = float((signal * (x_abs - cx) ** 2).sum() / total)
    var_y = float((signal * (y_abs - cy) ** 2).sum() / total)
    sigma = float(np.sqrt(max((var_x + var_y) / 2.0, 0.0)))
    if not np.isfinite(sigma) or sigma <= 0.0:
        return None
    return 2.355 * sigma


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

    # Resolve target position: explicit coords > WCS > SIMBAD lookup > image center fallback
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
            target_source, config
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
    reference_positions = [(float(source["x"]), float(source["y"])) for source in selected_sources]
    for record in frame_records:
        ccd = load_ccd(record.path)

        frame_positions = reference_positions
        if config.recenter_sources_per_frame:
            frame_positions = [
                _recenter_position(
                    ccd.data,
                    x=pos[0],
                    y=pos[1],
                    box_size=config.centroid_box_size,
                )
                for pos in reference_positions
            ]

        per_source_fwhm: list[float | None] = [None] * len(frame_positions)
        if config.adaptive_photometry:
            for idx, (x, y) in enumerate(frame_positions):
                fwhm = _estimate_fwhm(
                    ccd.data,
                    x=x,
                    y=y,
                    box_size=config.centroid_box_size,
                )
                if fwhm is None:
                    continue
                if config.fwhm_clip_range[0] <= fwhm <= config.fwhm_clip_range[1]:
                    per_source_fwhm[idx] = float(fwhm)

        for idx, (source, position) in enumerate(zip(selected_sources, frame_positions, strict=True)):
            if config.aperture_radius is not None:
                source_aperture_radius = float(config.aperture_radius)
            else:
                source_ref_fwhm = float(source.get("fwhm", config.detection_fwhm))
                source_fwhm = per_source_fwhm[idx] if per_source_fwhm[idx] is not None else source_ref_fwhm
                source_aperture_radius = float(
                    np.clip(
                        config.aperture_scale * source_fwhm,
                        config.aperture_radius_min,
                        config.aperture_radius_max,
                    )
                )

            source_annulus_r_in = max(config.annulus_scale_inner * source_aperture_radius, source_aperture_radius + 2.0)
            source_annulus_r_out = max(config.annulus_scale_outer * source_aperture_radius, source_annulus_r_in + 2.0)

            measurement = measure_aperture_flux(
                ccd.data,
                positions=[position],
                aperture_radius=source_aperture_radius,
                annulus_r_in=source_annulus_r_in,
                annulus_r_out=source_annulus_r_out,
            )[0]

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