from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from astropy.table import Table

from reduction.calibration import (
    build_master_bias,
    build_master_dark,
    build_master_flat,
    calibrate_light,
)
from reduction.io import group_by_frame_type, group_flats_by_filter, index_frames
from reduction.qc import compute_qc_metrics


@dataclass(frozen=True)
class ReductionPaths:
    raw_dir: Path
    output_dir: Path
    background_median_max: float = 70.0


def run_reduction_pipeline(paths: ReductionPaths) -> Table:
    records = index_frames(paths.raw_dir)
    grouped = group_by_frame_type(records)

    bias_records = grouped.get("bias", [])
    dark_records = grouped.get("dark", [])
    flat_records = grouped.get("flat", [])
    light_records = grouped.get("light", [])

    if not light_records:
        raise ValueError("No light frames were found in the input directory.")

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    masters_dir = paths.output_dir / "masters"
    calibrated_dir = paths.output_dir / "calibrated"
    masters_dir.mkdir(exist_ok=True)
    calibrated_dir.mkdir(exist_ok=True)

    master_bias = build_master_bias([record.path for record in bias_records]) if bias_records else None
    if master_bias is not None:
        master_bias.write(masters_dir / "master_bias.fits", overwrite=True)

    master_dark = build_master_dark(
        [record.path for record in dark_records],
        master_bias=master_bias,
    ) if dark_records else None
    if master_dark is not None:
        master_dark.write(masters_dir / "master_dark.fits", overwrite=True)

    dark_exposure = dark_records[0].exptime if dark_records else None
    master_flats: dict[str, object] = {}
    for filter_name, filter_records in group_flats_by_filter(flat_records).items():
        flat_exposure = filter_records[0].exptime
        master_flat = build_master_flat(
            [record.path for record in filter_records],
            master_bias=master_bias,
            master_dark=master_dark,
            dark_exposure=dark_exposure,
            flat_exposure=flat_exposure,
        )
        master_flats[filter_name] = master_flat
        master_flat.write(masters_dir / f"master_flat_{filter_name}.fits", overwrite=True)

    rows: list[dict[str, float | str]] = []
    for light_record in light_records:
        filter_key = light_record.filter_name or "NO_FILTER"
        calibrated = calibrate_light(
            light_record.path,
            master_bias=master_bias,
            master_dark=master_dark,
            master_flat=master_flats.get(filter_key),
            dark_exposure=dark_exposure,
        )
        output_path = calibrated_dir / f"cal_{light_record.path.name}"
        calibrated.write(output_path, overwrite=True)

        qc_result = compute_qc_metrics(
            calibrated,
            high_background_threshold=paths.background_median_max,
        )
        row = {
            "file": light_record.path.name,
            "filter": filter_key,
            "quality_flag": qc_result.quality_flag,
            **qc_result.to_dict(),
        }
        rows.append(row)

    result_table = Table(rows=rows)
    result_table.write(paths.output_dir / "qc_summary.csv", format="csv", overwrite=True)
    return result_table


if __name__ == "__main__":
    example_paths = ReductionPaths(
        raw_dir=Path("data/raw"),
        output_dir=Path("data/reduced"),
    )
    run_reduction_pipeline(example_paths)
