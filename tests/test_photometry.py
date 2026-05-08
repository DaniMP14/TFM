from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def load_module(module_name: str, relative_path: str):
    module_path = SRC_DIR / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


differential_module = load_module("photometry.differential", "photometry/differential.py")
build_differential_light_curve = differential_module.build_differential_light_curve


def test_build_differential_light_curve_normalizes_to_median() -> None:
    frame_measurements = pd.DataFrame(
        [
            {"frame_index": 0, "file": "f0.fit", "time_jd": 1.0, "source_id": "target", "role": "target", "net_flux": 1000.0},
            {"frame_index": 0, "file": "f0.fit", "time_jd": 1.0, "source_id": "comp_1", "role": "comparison", "net_flux": 500.0},
            {"frame_index": 0, "file": "f0.fit", "time_jd": 1.0, "source_id": "comp_2", "role": "comparison", "net_flux": 500.0},
            {"frame_index": 1, "file": "f1.fit", "time_jd": 2.0, "source_id": "target", "role": "target", "net_flux": 900.0},
            {"frame_index": 1, "file": "f1.fit", "time_jd": 2.0, "source_id": "comp_1", "role": "comparison", "net_flux": 500.0},
            {"frame_index": 1, "file": "f1.fit", "time_jd": 2.0, "source_id": "comp_2", "role": "comparison", "net_flux": 500.0},
        ]
    )

    light_curve = build_differential_light_curve(frame_measurements)

    assert len(light_curve) == 2
    assert np.isclose(light_curve.loc[0, "differential_flux"], 1.0)
    assert np.isclose(light_curve.loc[1, "differential_flux"], 0.9)
    assert np.isclose(light_curve["relative_flux"].median(), 1.0)