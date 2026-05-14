from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.table import Table

# Ensure src/ is on sys.path so sibling packages (reduction, alignment) resolve
# whether this module is imported from a notebook or run directly.
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from reduction.io import load_ccd  # noqa: E402
from alignment.registration import apply_shift, estimate_subpixel_shift  # noqa: E402


@dataclass(frozen=True)
class AlignmentPaths:
    calibrated_dir: Path
    output_dir: Path


@dataclass(frozen=True)
class AlignmentConfig:
    max_shift: int = 100  # maximum pixel shift to search in each direction
    interpolation_order: int = 3  # cubic interpolation
    reference_strategy: str = "first"  # "first" or "median"
    shift_mode: str = "previous"  # "previous" (incremental) or "global"
    include_reference_in_output: bool = True
    two_pass: bool = True
    corr_hard_threshold: float = 0.4
    corr_soft_threshold: float = 0.7
    jump_threshold: float = 5.0


def _discover_calibrated_files(calibrated_dir: Path) -> list[Path]:
    paths = sorted(calibrated_dir.glob("*.fit")) + sorted(calibrated_dir.glob("*.fits")) + sorted(calibrated_dir.glob("*.fts")) + sorted(calibrated_dir.glob("*.fz"))
    return sorted(set(paths))


def _resolve_aligned_dir(output_dir: Path) -> Path:
    # If output_dir already points to data/aligned, do not create aligned/aligned.
    if output_dir.name.lower() == "aligned":
        return output_dir
    return output_dir / "aligned"


def _safe_write_ccd(ccd: Any, target_path: Path) -> Path:
    """Write a FITS file and gracefully handle Windows file locks.

    If the target file is locked by another process (e.g., notebook viewer), the
    function writes to a fallback name with suffix _newN.
    """
    try:
        ccd.write(target_path, overwrite=True)
        return target_path
    except PermissionError:
        for i in range(1, 1000):
            alt_path = target_path.with_name(f"{target_path.stem}_new{i}{target_path.suffix}")
            try:
                ccd.write(alt_path, overwrite=True)
                print(
                    f"Warning: could not overwrite locked file '{target_path.name}'. "
                    f"Wrote '{alt_path.name}' instead."
                )
                return alt_path
            except PermissionError:
                continue
        raise


def _build_reference(paths: list[Path], strategy: str) -> np.ndarray:
    if strategy == "first":
        return load_ccd(paths[0]).data.astype(float)

    if strategy == "median":
        stack = np.stack([load_ccd(path).data.astype(float) for path in paths], axis=0)
        return np.median(stack, axis=0)

    raise ValueError(f"Unsupported reference strategy: {strategy}")


def _estimate_shifts(
    files: list[Path],
    reference_data: np.ndarray,
    config: AlignmentConfig,
) -> list[dict[str, Any]]:
    estimates: list[dict[str, Any]] = []
    previous_data: np.ndarray | None = None
    cumulative_dx = 0.0
    cumulative_dy = 0.0

    for index, file_path in enumerate(files):
        ccd = load_ccd(file_path)
        current_data = ccd.data.astype(float)

        shift_error = False
        shift_error_msg = ""

        if index == 0 and config.reference_strategy == "first":
            dx, dy, peak_value = 0.0, 0.0, 1.0
        else:
            try:
                if config.shift_mode == "previous" and previous_data is not None:
                    reference_for_shift = previous_data
                else:
                    reference_for_shift = reference_data

                shift = estimate_subpixel_shift(
                    reference=reference_for_shift,
                    moving=current_data,
                    max_shift=config.max_shift,
                )

                if config.shift_mode == "previous":
                    cumulative_dx += float(shift.dx)
                    cumulative_dy += float(shift.dy)
                    dx, dy = cumulative_dx, cumulative_dy
                else:
                    dx, dy = float(shift.dx), float(shift.dy)

                peak_value = float(shift.peak_value)
            except ValueError as exc:
                # Keep the pipeline running: this frame will be flagged/rejected downstream.
                dx, dy, peak_value = 0.0, 0.0, -1.0
                shift_error = True
                shift_error_msg = str(exc)
                print(f"Warning: shift estimation failed for '{file_path.name}': {exc}")

        previous_data = current_data

        estimates.append(
            {
                "file": file_path,
                "dx": dx,
                "dy": dy,
                "corr_peak": peak_value,
                "shift_error": shift_error,
                "shift_error_msg": shift_error_msg,
            }
        )
    return estimates


def _compute_jumps(estimates: list[dict[str, Any]]) -> np.ndarray:
    dx = np.array([float(item["dx"]) for item in estimates], dtype=float)
    dy = np.array([float(item["dy"]) for item in estimates], dtype=float)
    ddx = np.abs(np.diff(dx, prepend=dx[0]))
    ddy = np.abs(np.diff(dy, prepend=dy[0]))
    return np.sqrt(ddx ** 2 + ddy ** 2)


def _detect_bad_indices(
    estimates: list[dict[str, Any]],
    corr_hard_threshold: float,
    corr_soft_threshold: float,
    jump_threshold: float,
) -> tuple[set[int], np.ndarray, np.ndarray, float]:
    corr = np.array([float(item["corr_peak"]) for item in estimates], dtype=float)
    jumps = _compute_jumps(estimates)

    low_corr = corr < corr_soft_threshold

    # Always reject frames where shift estimation itself failed.
    bad_indices: set[int] = {
        i for i, item in enumerate(estimates) if bool(item.get("shift_error", False))
    }

    # Reject very low-correlation frames regardless of jump.
    for i in range(len(estimates)):
        if corr[i] < corr_hard_threshold:
            bad_indices.add(i)

    # Reject low-correlation frames when they also show a large alignment jump.
    for i in range(len(estimates)):
        if corr[i] < corr_soft_threshold and jumps[i] > jump_threshold:
            bad_indices.add(i)

    return bad_indices, jumps, low_corr, corr_soft_threshold


def _write_aligned_frames(
    output_dir: Path,
    estimates: list[dict[str, Any]],
    config: AlignmentConfig,
) -> None:
    for index, item in enumerate(estimates):
        file_path = Path(item["file"])
        ccd = load_ccd(file_path)
        dx = float(item["dx"])
        dy = float(item["dy"])

        if bool(item.get("shift_error", False)):
            print(f"Warning: skipping '{file_path.name}' due to shift estimation failure.")
            continue

        if index == 0 and config.reference_strategy == "first":
            if config.include_reference_in_output:
                _safe_write_ccd(ccd, output_dir / f"ali_{file_path.name}")
            continue

        aligned = apply_shift(ccd, dx=dx, dy=dy, order=config.interpolation_order)
        _safe_write_ccd(aligned, output_dir / f"ali_{file_path.name}")


def run_alignment_pipeline(paths: AlignmentPaths, config: AlignmentConfig | None = None) -> Table:
    config = config or AlignmentConfig()

    files = _discover_calibrated_files(paths.calibrated_dir)
    if not files:
        raise ValueError("No calibrated frames were found for alignment.")

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    aligned_dir = _resolve_aligned_dir(paths.output_dir)
    aligned_dir.mkdir(exist_ok=True)

    pass1_reference = _build_reference(files, config.reference_strategy)
    pass1_estimates = _estimate_shifts(files, pass1_reference, config)

    bad_indices: set[int] = set()
    pass1_jumps = _compute_jumps(pass1_estimates)
    pass1_low_corr = np.zeros(len(files), dtype=bool)
    corr_threshold = float("nan")

    if config.two_pass:
        bad_indices, pass1_jumps, pass1_low_corr, corr_threshold = _detect_bad_indices(
            pass1_estimates,
            corr_hard_threshold=config.corr_hard_threshold,
            corr_soft_threshold=config.corr_soft_threshold,
            jump_threshold=config.jump_threshold,
        )

    kept_files = [file_path for i, file_path in enumerate(files) if i not in bad_indices]
    if not kept_files:
        raise ValueError("All frames were rejected in pre-alignment screening.")

    pass2_reference = _build_reference(kept_files, config.reference_strategy)
    pass2_estimates = _estimate_shifts(kept_files, pass2_reference, config)
    _write_aligned_frames(aligned_dir, pass2_estimates, config)

    pass2_by_name = {Path(item["file"]).name: item for item in pass2_estimates}

    rows: list[dict[str, float | str | bool]] = []
    for i, pass1_item in enumerate(pass1_estimates):
        file_name = Path(pass1_item["file"]).name
        aligned_item = pass2_by_name.get(file_name)
        was_rejected = i in bad_indices

        if aligned_item is None:
            dx = float(pass1_item["dx"])
            dy = float(pass1_item["dy"])
            corr_peak = float(pass1_item["corr_peak"])
            status = "rejected_prealign"
        elif bool(aligned_item.get("shift_error", False)):
            dx = float(aligned_item["dx"])
            dy = float(aligned_item["dy"])
            corr_peak = float(aligned_item["corr_peak"])
            status = "failed_alignment"
        else:
            dx = float(aligned_item["dx"])
            dy = float(aligned_item["dy"])
            corr_peak = float(aligned_item["corr_peak"])
            status = "aligned"

        rows.append(
            {
                "file": file_name,
                "dx": dx,
                "dy": dy,
                "corr_peak": corr_peak,
                "reference_strategy": config.reference_strategy,
                "status": status,
                "pass1_jump": float(pass1_jumps[i]),
                "pass1_low_corr": bool(pass1_low_corr[i]),
                "rejected": bool(was_rejected),
                "corr_threshold": corr_threshold,
                "pass1_shift_error": bool(pass1_item.get("shift_error", False)),
            }
        )

    result_table = Table(rows=rows)
    result_table.write(paths.output_dir / "alignment_summary.csv", format="csv", overwrite=True)
    return result_table


if __name__ == "__main__":
    default_paths = AlignmentPaths(
        calibrated_dir=Path("../../data/reduced/calibrated"),
        output_dir=Path("../../data/aligned"),
    )
    run_alignment_pipeline(default_paths)
