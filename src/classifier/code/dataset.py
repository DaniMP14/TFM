from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from classifier.code.features import extract_curve_features


REQUIRED_MANIFEST_COLUMNS = {"sample_id", "label", "curve_path", "source_type"}


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)

    missing_columns = REQUIRED_MANIFEST_COLUMNS - set(manifest.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Manifest is missing required columns: {missing}")

    manifest = manifest.copy()
    manifest["label"] = manifest["label"].astype(int)

    invalid_labels = manifest.loc[~manifest["label"].isin([0, 1]), "label"]
    if not invalid_labels.empty:
        raise ValueError("All labels must be binary values: 0 or 1")

    return manifest


def _resolve_curve_path(manifest_path: Path, curve_path: str) -> Path:
    candidate = Path(curve_path)
    if candidate.is_absolute():
        return candidate
    return (manifest_path.parent / candidate).resolve()


def _read_curve(curve_csv_path: Path, flux_column: str, time_column: str) -> tuple[np.ndarray, np.ndarray]:
    curve_df = pd.read_csv(curve_csv_path)
    if flux_column not in curve_df.columns:
        raise ValueError(f"Curve '{curve_csv_path.name}' does not include column '{flux_column}'")

    if time_column in curve_df.columns:
        time = curve_df[time_column].to_numpy(dtype=float)
    else:
        time = np.arange(len(curve_df), dtype=float)

    flux = curve_df[flux_column].to_numpy(dtype=float)
    return time, flux


def build_feature_dataset(
    manifest_path: Path,
    flux_column: str = "detrended_flux",
    time_column: str = "time_jd",
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    manifest = load_manifest(manifest_path)

    feature_rows: list[dict[str, float]] = []
    metadata_rows: list[dict[str, str | int]] = []

    for row in manifest.itertuples(index=False):
        curve_path = _resolve_curve_path(manifest_path, row.curve_path)
        if not curve_path.exists():
            raise FileNotFoundError(f"Curve file does not exist: {curve_path}")

        time, flux = _read_curve(
            curve_path,
            flux_column=flux_column,
            time_column=time_column,
        )
        features = extract_curve_features(time=time, flux=flux)
        feature_rows.append(features)
        metadata_rows.append(
            {
                "sample_id": str(row.sample_id),
                "label": int(row.label),
                "source_type": str(row.source_type),
                "curve_path": str(curve_path),
            }
        )

    x_df = pd.DataFrame(feature_rows)
    meta_df = pd.DataFrame(metadata_rows)
    y = meta_df["label"].to_numpy(dtype=int)
    return x_df, y, meta_df
