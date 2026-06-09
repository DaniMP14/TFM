from __future__ import annotations

from pathlib import Path
from typing import Any


def run_bayesian_fit(
    curve_path: Path,
    target_name: str,
    output_dir: Path,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Punto de entrada del ajuste bayesiano (placeholder).

    En esta etapa se crea solo la interfaz y la estructura de carpetas.
    """
    raise NotImplementedError(
        "run_bayesian_fit aun no esta implementado. "
        "Completa src/bayesian/pipeline.py con el ajuste bayesiano."
    )
