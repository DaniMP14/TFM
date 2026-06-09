from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_RAW_MANIFEST_COLUMNS = {"raw_path", "label", "source_type"}


def _safe_sample_id(path: Path) -> str:
    return path.stem.replace(" ", "_")


def _read_three_column_numeric(path: Path) -> pd.DataFrame:
    data = np.loadtxt(path, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise ValueError(f"Expected at least 2 numeric columns in {path.name}")

    result = pd.DataFrame(
        {
            "time_jd": data[:, 0],
            "detrended_flux": data[:, 1],
        }
    )
    if data.shape[1] >= 3:
        result["flux_err"] = data[:, 2]
    return result


def _read_hops_like_text(path: Path) -> pd.DataFrame:
    # HOPS/TRESCA style: comments start with '#', data are whitespace-separated.
    # Typical columns: time, raw_flux, raw_err, trend, detrended_flux, detrended_err, model, residuals
    table = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        engine="python",
    )
    if table.shape[1] >= 5:
        result = pd.DataFrame(
            {
                "time_jd": table.iloc[:, 0].astype(float),
                "detrended_flux": table.iloc[:, 4].astype(float),
            }
        )
        if table.shape[1] >= 6:
            result["flux_err"] = table.iloc[:, 5].astype(float)
        return result

    # Fallback for compact text with two/three columns.
    if table.shape[1] >= 2:
        result = pd.DataFrame(
            {
                "time_jd": table.iloc[:, 0].astype(float),
                "detrended_flux": table.iloc[:, 1].astype(float),
            }
        )
        if table.shape[1] >= 3:
            result["flux_err"] = table.iloc[:, 2].astype(float)
        return result

    raise ValueError(f"Could not parse HOPS-like text file: {path}")


def _read_csv_with_columns(
    path: Path,
    flux_column: str | None,
    time_column: str | None,
    error_column: str | None,
) -> pd.DataFrame:
    df = pd.read_csv(path)

    flux_candidates = [
        flux_column,
        "detrended_flux",
        "normalized_flux",
        "relative_flux",
        "flux",
        "raw_flux",
    ]
    flux_name = next((c for c in flux_candidates if c and c in df.columns), None)
    if flux_name is None:
        raise ValueError(
            f"Could not find a flux column in '{path.name}'. Available columns: {list(df.columns)}"
        )

    time_candidates = [
        time_column,
        "time_jd",
        "time",
        "bjd_tdb",
        "jd",
        "frame_index",
    ]
    time_name = next((c for c in time_candidates if c and c in df.columns), None)

    result = pd.DataFrame()
    if time_name is not None:
        result["time_jd"] = pd.to_numeric(df[time_name], errors="coerce")
    else:
        result["time_jd"] = np.arange(len(df), dtype=float)

    result["detrended_flux"] = pd.to_numeric(df[flux_name], errors="coerce")

    err_candidates = [error_column, "flux_err", "detrended_flux_uncertainty", "raw_flux_uncertainty"]
    err_name = next((c for c in err_candidates if c and c in df.columns), None)
    if err_name is not None:
        result["flux_err"] = pd.to_numeric(df[err_name], errors="coerce")

    return result


def parse_curve_file(
    curve_path: Path,
    format_hint: str | None = None,
    flux_column: str | None = None,
    time_column: str | None = None,
    error_column: str | None = None,
) -> pd.DataFrame:
    hint = (format_hint or "").strip().lower()
    suffix = curve_path.suffix.lower()

    if hint == "threecol":
        parsed = _read_three_column_numeric(curve_path)
    elif hint in {"hops", "tresca", "commented_txt"}:
        parsed = _read_hops_like_text(curve_path)
    elif hint == "csv":
        parsed = _read_csv_with_columns(curve_path, flux_column, time_column, error_column)
    else:
        if suffix == ".csv":
            parsed = _read_csv_with_columns(curve_path, flux_column, time_column, error_column)
        elif suffix in {".txt", ".dat", ".tsv"}:
            try:
                parsed = _read_hops_like_text(curve_path)
            except Exception:
                parsed = _read_three_column_numeric(curve_path)
        else:
            raise ValueError(
                f"Unsupported extension '{curve_path.suffix}' for {curve_path.name}. "
                "Use format_hint=csv|hops|threecol."
            )

    parsed = parsed.copy()
    parsed = parsed.replace([np.inf, -np.inf], np.nan)
    parsed = parsed.dropna(subset=["time_jd", "detrended_flux"])
    parsed = parsed.sort_values("time_jd").reset_index(drop=True)

    if len(parsed) < 20:
        raise ValueError(f"Parsed curve '{curve_path.name}' has fewer than 20 valid points")

    return parsed


def convert_raw_manifest_to_training_dataset(
    raw_manifest_path: Path,
    processed_dir: Path,
    output_manifest_path: Path,
) -> pd.DataFrame:
    raw_manifest = pd.read_csv(raw_manifest_path)
    missing = REQUIRED_RAW_MANIFEST_COLUMNS - set(raw_manifest.columns)
    if missing:
        raise ValueError(f"Raw manifest is missing required columns: {sorted(missing)}")

    processed_dir.mkdir(parents=True, exist_ok=True)

    training_rows: list[dict[str, str | int]] = []
    for row in raw_manifest.itertuples(index=False):
        raw_path = Path(str(row.raw_path))
        if not raw_path.is_absolute():
            raw_path = (raw_manifest_path.parent / raw_path).resolve()

        if not raw_path.exists():
            raise FileNotFoundError(f"Raw curve not found: {raw_path}")

        sample_id = (
            str(getattr(row, "sample_id"))
            if hasattr(row, "sample_id") and pd.notna(getattr(row, "sample_id"))
            else _safe_sample_id(raw_path)
        )
        label = int(row.label)
        if label not in (0, 1):
            raise ValueError(f"Invalid label {label} for sample '{sample_id}'. Use 0 or 1.")

        parsed = parse_curve_file(
            curve_path=raw_path,
            format_hint=str(getattr(row, "format_hint")) if hasattr(row, "format_hint") and pd.notna(getattr(row, "format_hint")) else None,
            flux_column=str(getattr(row, "flux_column")) if hasattr(row, "flux_column") and pd.notna(getattr(row, "flux_column")) else None,
            time_column=str(getattr(row, "time_column")) if hasattr(row, "time_column") and pd.notna(getattr(row, "time_column")) else None,
            error_column=str(getattr(row, "error_column")) if hasattr(row, "error_column") and pd.notna(getattr(row, "error_column")) else None,
        )

        processed_path = processed_dir / f"{sample_id}.csv"
        parsed.to_csv(processed_path, index=False)

        training_rows.append(
            {
                "sample_id": sample_id,
                "label": label,
                "source_type": str(row.source_type),
                "curve_path": str(processed_path.resolve()),
            }
        )

    training_manifest = pd.DataFrame(training_rows)
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    training_manifest.to_csv(output_manifest_path, index=False)
    return training_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert mixed raw light-curve formats to standardized training CSV + manifest.",
    )
    parser.add_argument(
        "--raw-manifest",
        type=Path,
        required=True,
        help="CSV with columns raw_path,label,source_type and optional sample_id,format_hint,flux_column,time_column,error_column.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("src/classifier/data/training/processed"),
        help="Directory where standardized per-curve CSV files will be written.",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("src/classifier/data/training/manifest_training.csv"),
        help="Output training manifest consumed by classifier.code.train.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_df = convert_raw_manifest_to_training_dataset(
        raw_manifest_path=args.raw_manifest,
        processed_dir=args.processed_dir,
        output_manifest_path=args.output_manifest,
    )
    print(f"Generated {len(manifest_df)} standardized curves")
    print(f"Training manifest: {args.output_manifest}")


if __name__ == "__main__":
    main()
