from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from classifier.code.features import extract_curve_features
from classifier.code.prepare_dataset import parse_curve_file


@dataclass
class ModelBundle:
    """Agrupa el modelo, las columnas de features esperadas y el umbral de decision."""

    name: str
    model: Any
    feature_columns: list[str]
    decision_threshold: float
    threshold_strategy: str
    metrics_train: dict[str, object]


def load_model_bundle(artifact_dir: Path, name: str | None = None) -> ModelBundle:
    """Carga un modelo serializado desde una carpeta de artefactos.

    Busca automáticamente el primer .pkl que encuentre en el directorio.
    Lee feature_columns.json (o feature_columns_*.json) y metrics.json para
    reconstruir el umbral de decisión.
    """
    artifact_dir = Path(artifact_dir)
    if not artifact_dir.exists():
        raise FileNotFoundError(f"Directorio de artefactos no encontrado: {artifact_dir}")

    pkl_candidates = sorted(artifact_dir.glob("*.pkl"))
    if not pkl_candidates:
        raise FileNotFoundError(f"No se encontro ningun .pkl en {artifact_dir}")
    pkl_path = pkl_candidates[0]

    try:
        with pkl_path.open("rb") as fp:
            model = pickle.load(fp)
    except ValueError as exc:
        msg = str(exc)
        # Common case: scikit-learn tree dtype mismatch between training and inference versions.
        if "incompatible dtype" in msg and "node array" in msg:
            raise RuntimeError(
                "No se pudo cargar el modelo por incompatibilidad de versiones de scikit-learn. "
                f"Archivo: {pkl_path}. Solucion: reentrenar el modelo con la version actual o usar la misma "
                "version de scikit-learn con la que se entreno."
            ) from exc
        raise

    # Feature columns: admite nombre fijo o sufijo numérico (ej: feature_columns_8.json)
    fc_candidates = sorted(artifact_dir.glob("feature_columns*.json"))
    if not fc_candidates:
        raise FileNotFoundError(f"No se encontro feature_columns*.json en {artifact_dir}")
    feature_columns: list[str] = json.loads(fc_candidates[0].read_text(encoding="utf-8"))

    # Métricas de entrenamiento (para umbral y metadatos)
    metrics_candidates = sorted(artifact_dir.glob("metrics*.json"))
    metrics_train: dict[str, object] = {}
    decision_threshold = 0.5
    threshold_strategy = "default_0.5"
    if metrics_candidates:
        metrics_train = json.loads(metrics_candidates[0].read_text(encoding="utf-8"))
        decision_threshold = float(metrics_train.get("decision_threshold", 0.5))
        threshold_strategy = str(metrics_train.get("threshold_strategy", "default_0.5"))

    bundle_name = name or artifact_dir.name
    return ModelBundle(
        name=bundle_name,
        model=model,
        feature_columns=feature_columns,
        decision_threshold=decision_threshold,
        threshold_strategy=threshold_strategy,
        metrics_train=metrics_train,
    )


def extract_features_from_csv(
    curve_path: Path,
    flux_column: str = "detrended_flux",
    time_column: str = "time_jd",
) -> dict[str, float]:
    """Lee una curva normalizada (CSV) y extrae el vector de features."""
    df = pd.read_csv(curve_path)
    if flux_column not in df.columns:
        raise ValueError(f"Columna '{flux_column}' no encontrada en {curve_path.name}")
    time = df[time_column].to_numpy(dtype=float) if time_column in df.columns else np.arange(len(df), dtype=float)
    flux = df[flux_column].to_numpy(dtype=float)
    return extract_curve_features(time=time, flux=flux)


def extract_features_from_raw(
    raw_path: Path,
    flux_column: str = "detrended_flux",
    time_column: str = "time_jd",
) -> dict[str, float]:
    """Lee un archivo raw (cualquier formato soportado por parse_curve_file) y extrae features."""
    df = parse_curve_file(raw_path)
    time = df[time_column].to_numpy(dtype=float) if time_column in df.columns else np.arange(len(df), dtype=float)
    flux = df[flux_column].to_numpy(dtype=float)
    return extract_curve_features(time=time, flux=flux)


def predict_single(
    bundle: ModelBundle,
    features: dict[str, float],
) -> dict[str, float]:
    """Infiere clase y probabilidad para una curva dada su vector de features."""
    row = pd.DataFrame([features])[bundle.feature_columns]
    prob_pos = float(bundle.model.predict_proba(row)[0, 1])
    label = int(prob_pos >= bundle.decision_threshold)
    return {
        "prob_positive": round(prob_pos, 6),
        "label_pred": label,
        "threshold": bundle.decision_threshold,
    }


def predict_manifest(
    bundle: ModelBundle,
    manifest_path: Path,
    flux_column: str = "detrended_flux",
    time_column: str = "time_jd",
    true_label_col: str | None = "label",
) -> pd.DataFrame:
    """Aplica el modelo a todas las curvas del manifest y devuelve un DataFrame de resultados."""
    manifest = pd.read_csv(manifest_path)

    rows: list[dict[str, object]] = []
    for record in manifest.itertuples(index=False):
        curve_path = Path(str(record.curve_path))
        if not curve_path.is_absolute():
            curve_path = (manifest_path.parent / curve_path).resolve()

        try:
            features = extract_features_from_csv(
                curve_path,
                flux_column=flux_column,
                time_column=time_column,
            )
            result = predict_single(bundle, features)
            row: dict[str, object] = {
                "sample_id": getattr(record, "sample_id", curve_path.stem),
                "curve_path": str(curve_path),
                "source_type": getattr(record, "source_type", "unknown"),
                f"prob_pos_{bundle.name}": result["prob_positive"],
                f"label_pred_{bundle.name}": result["label_pred"],
            }
            if true_label_col and hasattr(record, true_label_col):
                row["label_true"] = int(getattr(record, true_label_col))
        except Exception as exc:
            row = {
                "sample_id": getattr(record, "sample_id", curve_path.stem),
                "curve_path": str(curve_path),
                "source_type": getattr(record, "source_type", "unknown"),
                f"prob_pos_{bundle.name}": float("nan"),
                f"label_pred_{bundle.name}": -1,
                "error": str(exc),
            }
            if true_label_col and hasattr(record, true_label_col):
                row["label_true"] = int(getattr(record, true_label_col))

        rows.append(row)

    return pd.DataFrame(rows)


def compare_models(
    bundles: list[ModelBundle],
    manifest_path: Path,
    flux_column: str = "detrended_flux",
    time_column: str = "time_jd",
    true_label_col: str | None = "label",
) -> pd.DataFrame:
    """Aplica todos los modelos al mismo manifest y devuelve tabla comparativa unificada.

    Optimizacion: cada curva se lee y se featuriza una sola vez, y sobre ese vector
    se evalúan todos los modelos. Esto evita repetir I/O y extracción de features
    por cada modelo.
    """
    manifest = pd.read_csv(manifest_path)
    rows: list[dict[str, object]] = []

    for record in manifest.itertuples(index=False):
        curve_path = Path(str(record.curve_path))
        if not curve_path.is_absolute():
            curve_path = (manifest_path.parent / curve_path).resolve()

        row: dict[str, object] = {
            "sample_id": getattr(record, "sample_id", curve_path.stem),
            "curve_path": str(curve_path),
            "source_type": getattr(record, "source_type", "unknown"),
        }
        if true_label_col and hasattr(record, true_label_col):
            row["label_true"] = int(getattr(record, true_label_col))

        try:
            features = extract_features_from_csv(
                curve_path,
                flux_column=flux_column,
                time_column=time_column,
            )
        except Exception as exc:
            for bundle in bundles:
                row[f"prob_pos_{bundle.name}"] = float("nan")
                row[f"label_pred_{bundle.name}"] = -1
            row["error"] = str(exc)
            rows.append(row)
            continue

        model_errors: list[str] = []
        for bundle in bundles:
            try:
                result = predict_single(bundle, features)
                row[f"prob_pos_{bundle.name}"] = result["prob_positive"]
                row[f"label_pred_{bundle.name}"] = result["label_pred"]
            except Exception as exc:
                row[f"prob_pos_{bundle.name}"] = float("nan")
                row[f"label_pred_{bundle.name}"] = -1
                model_errors.append(f"{bundle.name}: {exc}")

        if model_errors:
            row["error"] = " | ".join(model_errors)

        rows.append(row)

    return pd.DataFrame(rows)


def compute_comparison_metrics(
    comparison_df: pd.DataFrame,
    bundles: list[ModelBundle],
) -> pd.DataFrame:
    """Calcula accuracy, precision, recall, F1 por modelo sobre el DataFrame comparativo."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    if "label_true" not in comparison_df.columns:
        raise ValueError("El DataFrame no contiene columna 'label_true'. No se pueden calcular métricas.")

    y_true = comparison_df["label_true"].to_numpy(dtype=int)
    rows: list[dict[str, object]] = []

    for bundle in bundles:
        col = f"label_pred_{bundle.name}"
        if col not in comparison_df.columns:
            continue
        valid = comparison_df[col] >= 0
        y_pred = comparison_df.loc[valid, col].to_numpy(dtype=int)
        y_t = y_true[valid.to_numpy()]

        n_errors = int((~valid).sum())
        rows.append({
            "model": bundle.name,
            "n_evaluated": int(valid.sum()),
            "n_errors": n_errors,
            "threshold": bundle.decision_threshold,
            "accuracy": round(float(accuracy_score(y_t, y_pred)), 4),
            "precision": round(float(precision_score(y_t, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_t, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_t, y_pred, zero_division=0)), 4),
        })

    return pd.DataFrame(rows)
