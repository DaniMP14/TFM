from __future__ import annotations

import argparse
import json
from pathlib import Path

from classifier.code.training import (
    RandomForestTrainingConfig,
    train_random_forest_from_manifest,
)


def parse_args() -> argparse.Namespace:
    defaults = RandomForestTrainingConfig()

    parser = argparse.ArgumentParser(
        description="Train a light-curve quality classifier using Random Forest.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to manifest CSV file with labels and light-curve paths.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/classifier/artifacts"),
        help="Directory where model and metrics will be stored.",
    )
    parser.add_argument(
        "--flux-column",
        type=str,
        default="detrended_flux",
        help="Flux column name present in each light-curve CSV.",
    )
    parser.add_argument(
        "--time-column",
        type=str,
        default="time_jd",
        help="Time column name present in each light-curve CSV.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["rf", "svm", "xgb"],
        default=defaults.model_type,
        help="Modelo a entrenar: rf, svm o xgb.",
    )
    parser.add_argument("--n-estimators", type=int, default=defaults.n_estimators)
    parser.add_argument("--max-depth", type=int, default=defaults.max_depth)
    parser.add_argument("--test-size", type=float, default=defaults.test_size)
    parser.add_argument("--random-state", type=int, default=defaults.random_state)
    parser.add_argument("--min-samples-split", type=int, default=defaults.min_samples_split)
    parser.add_argument("--min-samples-leaf", type=int, default=defaults.min_samples_leaf)
    parser.add_argument("--svm-c", type=float, default=defaults.svm_c)
    parser.add_argument("--svm-kernel", type=str, default=defaults.svm_kernel)
    parser.add_argument("--svm-gamma", type=str, default=defaults.svm_gamma)
    parser.add_argument("--xgb-n-estimators", type=int, default=defaults.xgb_n_estimators)
    parser.add_argument("--xgb-max-depth", type=int, default=defaults.xgb_max_depth)
    parser.add_argument("--xgb-learning-rate", type=float, default=defaults.xgb_learning_rate)
    parser.add_argument("--xgb-subsample", type=float, default=defaults.xgb_subsample)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=defaults.xgb_colsample_bytree)
    parser.add_argument("--xgb-min-child-weight", type=int, default=defaults.xgb_min_child_weight)
    parser.add_argument("--xgb-reg-lambda", type=float, default=defaults.xgb_reg_lambda)
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Activa busqueda de hiperparametros con GridSearchCV sobre train split.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Numero de folds para GridSearchCV (solo si --grid-search).",
    )
    parser.add_argument(
        "--grid-scoring",
        type=str,
        default="f1",
        help="Metrica objetivo de GridSearchCV (ej: f1, precision, recall, roc_auc).",
    )
    parser.add_argument(
        "--youden-threshold",
        action="store_true",
        help="Calcula el umbral de decision con el indice de Youden usando un split interno de validacion.",
    )
    parser.add_argument(
        "--threshold-validation-size",
        type=float,
        default=0.2,
        help="Proporcion del train reservado para ajustar el umbral de Youden.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RandomForestTrainingConfig(
        model_type=args.model,
        test_size=args.test_size,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        svm_c=args.svm_c,
        svm_kernel=args.svm_kernel,
        svm_gamma=args.svm_gamma,
        xgb_n_estimators=args.xgb_n_estimators,
        xgb_max_depth=args.xgb_max_depth,
        xgb_learning_rate=args.xgb_learning_rate,
        xgb_subsample=args.xgb_subsample,
        xgb_colsample_bytree=args.xgb_colsample_bytree,
        xgb_min_child_weight=args.xgb_min_child_weight,
        xgb_reg_lambda=args.xgb_reg_lambda,
    )
    summary = train_random_forest_from_manifest(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        config=config,
        flux_column=args.flux_column,
        time_column=args.time_column,
        use_grid_search=args.grid_search,
        cv_folds=args.cv_folds,
        grid_scoring=args.grid_scoring,
        use_youden_threshold=args.youden_threshold,
        threshold_validation_size=args.threshold_validation_size,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
