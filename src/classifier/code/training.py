from __future__ import annotations

import json
import pickle
import importlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from classifier.code.dataset import build_feature_dataset


@dataclass(frozen=True)
class RandomForestTrainingConfig:
    model_type: str = "rf"
    test_size: float = 0.25
    random_state: int = 42
    n_estimators: int = 200
    max_depth: int = None
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    class_weight: dict[int, float] = field(default_factory=lambda: {0: 1.0, 1: 1.0})
    svm_c: float = 1.0
    svm_kernel: str = "rbf"
    svm_gamma: str = "scale"
    xgb_n_estimators: int = 600
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.03
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.7
    xgb_min_child_weight: int = 3
    xgb_reg_lambda: float = 3.0


def _compute_metrics(
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float | list[int] | list[list[int]]]:
    return {
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "class_distribution_train": [int(np.sum(y_train == 0)), int(np.sum(y_train == 1))],
        "class_distribution_test": [int(np.sum(y_test == 0)), int(np.sum(y_test == 1))],
        "confusion_matrix": confusion_matrix(y_test, y_pred).astype(int).tolist(),
    }


def _safe_feature_importances(model: Any, n_features: int) -> np.ndarray:
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
        if values.size == n_features:
            return values
    return np.full(n_features, np.nan, dtype=float)


def _youden_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    thresholds = np.unique(np.asarray(y_score, dtype=float))
    if thresholds.size == 0:
        return 0.5

    best_threshold = 0.5
    best_score = -np.inf

    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))

        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        youden = tpr - fpr

        if youden > best_score:
            best_score = youden
            best_threshold = float(threshold)

    return best_threshold


def train_random_forest(
    x_df: pd.DataFrame,
    y: np.ndarray,
    config: RandomForestTrainingConfig | None = None,
    use_grid_search: bool = False,
    cv_folds: int = 5,
    grid_scoring: str = "f1",
    use_youden_threshold: bool = False,
    threshold_validation_size: float = 0.2,
) -> tuple[RandomForestClassifier, dict[str, float | list[int] | list[list[int]]], pd.DataFrame]:
    config = config or RandomForestTrainingConfig()
    model_type = config.model_type.lower().strip()

    if model_type not in {"rf", "svm", "xgb"}:
        raise ValueError("model_type must be one of: rf, svm, xgb")

    x_train, x_test, y_train, y_test = train_test_split(
        x_df,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    x_fit = x_train
    y_fit = y_train
    x_threshold = None
    y_threshold = None
    threshold_value = 0.5

    if use_youden_threshold:
        x_fit, x_threshold, y_fit, y_threshold = train_test_split(
            x_train,
            y_train,
            test_size=threshold_validation_size,
            random_state=config.random_state,
            stratify=y_train,
        )

    if model_type == "svm" and use_grid_search:
        raise ValueError("Grid search is currently implemented for rf model_type only")
    if model_type == "xgb" and use_grid_search:
        raise ValueError("Grid search is currently implemented for rf model_type only")

    base_model = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_split=config.min_samples_split,
        min_samples_leaf=config.min_samples_leaf,
        class_weight=config.class_weight,
        random_state=config.random_state,
        n_jobs=-1,
    )

    model: Any
    grid_meta: dict[str, object] | None = None

    if model_type == "rf" and use_grid_search:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=config.random_state)
        param_grid = {
            "n_estimators": [200, 300, 500, 600, 800],
            "max_depth": [14, 28, None],
            "min_samples_split": [2, 3, 5, 10],
            "min_samples_leaf": [1, 2, 3, 5],
            "class_weight": [config.class_weight, "balanced"],
        }
        search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring=grid_scoring,
            cv=cv,
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
        search.fit(x_fit, y_fit)
        model = search.best_estimator_
        grid_meta = {
            "grid_search_enabled": True,
            "grid_scoring": grid_scoring,
            "cv_folds": int(cv_folds),
            "grid_best_score": float(search.best_score_),
            "grid_best_params": search.best_params_,
        }
    elif model_type == "rf":
        model = base_model
        model.fit(x_fit, y_fit)

    elif model_type == "svm":
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "svc",
                    SVC(
                        C=config.svm_c,
                        kernel=config.svm_kernel,
                        gamma=config.svm_gamma,
                        class_weight=config.class_weight,
                        probability=True,
                        random_state=config.random_state,
                    ),
                ),
            ]
        )
        model.fit(x_fit, y_fit)

    else:
        try:
            xgb_module = importlib.import_module("xgboost")
            XGBClassifier = getattr(xgb_module, "XGBClassifier")
        except Exception as exc:
            raise ImportError(
                "xgboost is not installed. Install it with 'pip install xgboost' or add it to environment.yml."
            ) from exc

        if isinstance(config.class_weight, dict):
            w0 = float(config.class_weight.get(0, 1.0))
            w1 = float(config.class_weight.get(1, 1.0))
            scale_pos_weight = w1 / max(w0, 1e-8)
        else:
            scale_pos_weight = 1.0

        model = XGBClassifier(
            n_estimators=config.xgb_n_estimators,
            max_depth=config.xgb_max_depth,
            learning_rate=config.xgb_learning_rate,
            subsample=config.xgb_subsample,
            colsample_bytree=config.xgb_colsample_bytree,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=config.random_state,
            n_jobs=-1,
            tree_method="hist",
            scale_pos_weight=scale_pos_weight,
            min_child_weight=config.xgb_min_child_weight,
            reg_lambda=config.xgb_reg_lambda,
        )
        model.fit(x_fit, y_fit)

    if use_youden_threshold:
        if x_threshold is None or y_threshold is None:
            raise RuntimeError("Threshold validation split was not created correctly")
        threshold_scores = model.predict_proba(x_threshold)[:, 1]
        threshold_value = _youden_threshold(y_threshold, threshold_scores)

    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]

    if use_youden_threshold:
        y_pred = (y_prob >= threshold_value).astype(int)

    metrics = _compute_metrics(y_train=y_train, y_test=y_test, y_pred=y_pred, y_prob=y_prob)
    metrics["model_type"] = model_type
    metrics["decision_threshold"] = float(threshold_value)
    metrics["threshold_strategy"] = "youden" if use_youden_threshold else "default_0.5"

    if grid_meta is not None:
        metrics.update(grid_meta)

    if model_type == "rf":
        importances = _safe_feature_importances(model, n_features=x_df.shape[1])
    elif model_type == "svm":
        importances = np.full(x_df.shape[1], np.nan, dtype=float)
    else:
        importances = _safe_feature_importances(model, n_features=x_df.shape[1])

    importance_df = pd.DataFrame(
        {
            "feature": x_df.columns,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)

    return model, metrics, importance_df


def train_random_forest_from_manifest(
    manifest_path: Path,
    output_dir: Path,
    config: RandomForestTrainingConfig | None = None,
    flux_column: str = "detrended_flux",
    time_column: str = "time_jd",
    use_grid_search: bool = False,
    cv_folds: int = 5,
    grid_scoring: str = "f1",
    use_youden_threshold: bool = False,
    threshold_validation_size: float = 0.2,
) -> dict[str, object]:
    config = config or RandomForestTrainingConfig()

    x_df, y, meta_df = build_feature_dataset(
        manifest_path=manifest_path,
        flux_column=flux_column,
        time_column=time_column,
    )
    model, metrics, importance_df = train_random_forest(
        x_df=x_df,
        y=y,
        config=config,
        use_grid_search=use_grid_search,
        cv_folds=cv_folds,
        grid_scoring=grid_scoring,
        use_youden_threshold=use_youden_threshold,
        threshold_validation_size=threshold_validation_size,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = config.model_type.lower().strip()
    model_path = output_dir / f"{model_name}_quality_classifier.pkl"
    with model_path.open("wb") as fp:
        pickle.dump(model, fp)

    feature_columns_path = output_dir / "feature_columns.json"
    feature_columns_path.write_text(json.dumps(list(x_df.columns), indent=2), encoding="utf-8")

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    config_path = output_dir / "training_config.json"
    config_path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    importance_path = output_dir / "feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)

    features_path = output_dir / "training_features.csv"
    pd.concat([meta_df, x_df], axis=1).to_csv(features_path, index=False)

    summary = {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "feature_importance_path": str(importance_path),
        "model_type": model_name,
        "n_samples": int(len(y)),
    }
    return summary
