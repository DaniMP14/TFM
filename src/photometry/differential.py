from __future__ import annotations

import numpy as np
import pandas as pd

# La función build_differential_light_curve construye una curva de luz diferencial a partir de las mediciones de fotometría de apertura, 
# normalizando el flujo diferencial al valor mediano para obtener una curva de luz relativa. Se agrupan las mediciones por índice de cuadro, 
# se calcula el flujo diferencial para cada cuadro y se filtran los cuadros sin fuentes objetivo o de comparación válidas.
def build_differential_light_curve(frame_measurements: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"frame_index", "file", "source_id", "role", "net_flux"}
    missing_columns = required_columns - set(frame_measurements.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing columns for differential photometry: {missing}")

    rows: list[dict[str, float | int | str | None]] = []
    grouped = frame_measurements.groupby("frame_index", sort=True)
    for frame_index, group in grouped:
        target_rows = group[group["role"] == "target"]
        comp_rows = group[group["role"] == "comparison"]
        if target_rows.empty or comp_rows.empty:
            continue

        target_flux = float(target_rows["net_flux"].iloc[0])
        good_comps = comp_rows[comp_rows["net_flux"] > 0.0]
        comparison_flux = float(good_comps["net_flux"].sum())
        if comparison_flux <= 0.0:
            continue

        differential_flux = target_flux / comparison_flux
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
                "n_comparisons_used": int(len(good_comps)),
            }
        )

    light_curve = pd.DataFrame(rows)
    if not light_curve.empty:
        baseline = float(light_curve["differential_flux"].median())
        light_curve["relative_flux"] = light_curve["differential_flux"] / baseline
    return light_curve