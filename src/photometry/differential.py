from __future__ import annotations

import numpy as np
import pandas as pd

def _robust_sigma(values: np.ndarray) -> float:
    if values.size == 0:
        return np.nan
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = float(np.std(values))
    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = 1e-4
    return sigma


def build_differential_light_curve(frame_measurements: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"frame_index", "file", "source_id", "role", "net_flux"}
    missing_columns = required_columns - set(frame_measurements.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing columns for differential photometry: {missing}")

    # Precompute robust weights for comparison stars based on their temporal stability.
    comp_all = frame_measurements[
        (frame_measurements["role"] == "comparison") & (frame_measurements["net_flux"] > 0.0)
    ].copy()
    comp_pivot = comp_all.pivot_table(
        index="frame_index",
        columns="source_id",
        values="net_flux",
        aggfunc="median",
    ).sort_index()

    comp_scale = comp_pivot.median(axis=0, skipna=True)
    comp_norm = comp_pivot.divide(comp_scale.replace(0.0, np.nan), axis=1)
    ensemble_median = comp_norm.median(axis=1, skipna=True)

    comp_weights: dict[str, float] = {}
    for source_id in comp_norm.columns:
        series = comp_norm[source_id]
        valid = series.notna() & ensemble_median.notna()
        if valid.sum() < 10:
            continue
        residuals = (series[valid] - ensemble_median[valid]).to_numpy(dtype=float)
        sigma = _robust_sigma(residuals)
        comp_weights[str(source_id)] = 1.0 / (sigma * sigma)

    if not comp_weights:
        # Fallback to equal weighting if stability could not be estimated.
        comp_weights = {str(col): 1.0 for col in comp_norm.columns}

    target_all = frame_measurements[
        (frame_measurements["role"] == "target") & (frame_measurements["net_flux"] > 0.0)
    ]
    target_median_flux = float(target_all["net_flux"].median()) if not target_all.empty else 1.0
    target_median_flux = max(target_median_flux, 1e-12)

    rows: list[dict[str, float | int | str | None]] = []
    grouped = frame_measurements.groupby("frame_index", sort=True)
    for frame_index, group in grouped:
        target_rows = group[group["role"] == "target"]
        comp_rows = group[group["role"] == "comparison"]
        if target_rows.empty or comp_rows.empty:
            continue

        target_flux = float(target_rows["net_flux"].iloc[0])
        good_comps = comp_rows[comp_rows["net_flux"] > 0.0]
        if good_comps.empty:
            continue

        comp_norm_values: list[float] = []
        comp_weight_values: list[float] = []
        for _, row in good_comps.iterrows():
            source_id = str(row["source_id"])
            if source_id not in comp_weights:
                continue
            scale = float(comp_scale.get(source_id, np.nan))
            if not np.isfinite(scale) or scale <= 0.0:
                continue
            norm_flux = float(row["net_flux"]) / scale
            if not np.isfinite(norm_flux) or norm_flux <= 0.0:
                continue
            comp_norm_values.append(norm_flux)
            comp_weight_values.append(float(comp_weights[source_id]))

        if not comp_norm_values:
            continue

        norm_array = np.array(comp_norm_values, dtype=float)
        weight_array = np.array(comp_weight_values, dtype=float)

        weight_sum = float(weight_array.sum())
        if weight_sum <= 0.0:
            continue

        weighted_norm_comparison = float(np.sum(norm_array * weight_array) / weight_sum)
        if weighted_norm_comparison <= 0.0 or not np.isfinite(weighted_norm_comparison):
            continue

        target_norm_flux = target_flux / target_median_flux
        differential_flux = target_norm_flux / weighted_norm_comparison
        comparison_flux = weighted_norm_comparison
        differential_mag = -2.5 * np.log10(max(differential_flux, 1e-12))

        first_row = group.iloc[0]
        rows.append(
            {
                "frame_index": int(frame_index),
                "file": str(first_row["file"]),
                "time_jd": float(first_row["time_jd"]) if pd.notna(first_row.get("time_jd")) else None,
                "target_flux": target_flux,
                "comparison_flux": comparison_flux,
                "differential_flux": float(differential_flux),
                "differential_mag": float(differential_mag),
                "n_comparisons_used": int(len(norm_array)),
            }
        )

    light_curve = pd.DataFrame(rows)
    if not light_curve.empty:
        baseline = float(light_curve["differential_flux"].median())
        light_curve["relative_flux"] = light_curve["differential_flux"] / baseline
    return light_curve