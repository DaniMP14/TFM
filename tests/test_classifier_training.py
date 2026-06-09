from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from classifier.code.dataset import build_feature_dataset
from classifier.code.training import RandomForestTrainingConfig, train_random_forest


def _make_synthetic_curve(
    n_points: int,
    noise_std: float,
    depth: float,
    width: int,
    center: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    flux = 1.0 + rng.normal(0.0, noise_std, size=n_points)
    start = max(center - width // 2, 0)
    end = min(center + width // 2, n_points)
    if depth > 0.0:
        flux[start:end] -= depth
    time = np.linspace(0.0, 1.0, n_points)
    return pd.DataFrame({"time_jd": time, "detrended_flux": flux})


def test_random_forest_training_from_manifest_like_dataset(tmp_path: Path) -> None:
    curves_dir = tmp_path / "curves"
    curves_dir.mkdir(parents=True)

    manifest_rows: list[dict[str, str | int]] = []
    n_samples_per_class = 16

    for idx in range(n_samples_per_class):
        df_pos = _make_synthetic_curve(
            n_points=220,
            noise_std=0.002,
            depth=0.015,
            width=36,
            center=110,
            seed=idx,
        )
        file_pos = curves_dir / f"pos_{idx:03d}.csv"
        df_pos.to_csv(file_pos, index=False)
        manifest_rows.append(
            {
                "sample_id": f"pos_{idx:03d}",
                "label": 1,
                "source_type": "synthetic",
                "curve_path": str(file_pos),
            }
        )

        df_neg = _make_synthetic_curve(
            n_points=220,
            noise_std=0.0035,
            depth=0.0,
            width=0,
            center=110,
            seed=1000 + idx,
        )
        file_neg = curves_dir / f"neg_{idx:03d}.csv"
        df_neg.to_csv(file_neg, index=False)
        manifest_rows.append(
            {
                "sample_id": f"neg_{idx:03d}",
                "label": 0,
                "source_type": "field",
                "curve_path": str(file_neg),
            }
        )

    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    x_df, y, _ = build_feature_dataset(manifest_path=manifest_path)

    model, metrics, importance_df = train_random_forest(
        x_df=x_df,
        y=y,
        config=RandomForestTrainingConfig(n_estimators=200, random_state=7),
    )

    assert len(x_df) == 2 * n_samples_per_class
    assert model.n_estimators == 200
    assert metrics["accuracy"] >= 0.75
    assert not importance_df.empty
