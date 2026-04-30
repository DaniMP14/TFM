from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from astropy.nddata import CCDData
import astropy.units as u

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


calibration_module = load_module("reduction.calibration", "reduction/calibration.py")
qc_module = load_module("reduction.qc", "reduction/qc.py")

normalize_flat_frame = calibration_module.normalize_flat_frame
classify_frame_quality = qc_module.classify_frame_quality


def test_normalize_flat_frame_scales_median_to_one() -> None:
    flat = CCDData(np.array([[2.0, 4.0], [6.0, 8.0]]), unit=u.adu)
    normalized = normalize_flat_frame(flat)

    assert np.isclose(np.median(normalized.data), 1.0)


def test_classify_frame_quality_rejects_saturated_frames() -> None:
    quality_flag = classify_frame_quality(
        std=12.5,
        saturation_fraction=0.02,
        bad_pixel_fraction=0.0,
    )

    assert quality_flag == "reject_saturated"
