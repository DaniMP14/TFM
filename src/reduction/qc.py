from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from astropy.nddata import CCDData


@dataclass(frozen=True)
class QCResult:
    mean: float
    median: float
    std: float
    min_value: float
    max_value: float
    saturation_fraction: float
    bad_pixel_fraction: float
    quality_flag: str

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


def compute_qc_metrics(
    ccd: CCDData,
    saturation_level: float = 60000.0,
    bad_low_threshold: float = 0.0,
) -> QCResult:
    data = np.asarray(ccd.data, dtype=float)
    saturation_fraction = float(np.mean(data >= saturation_level))
    bad_pixel_fraction = float(np.mean(data <= bad_low_threshold))
    quality_flag = classify_frame_quality(
        std=float(np.std(data)),
        saturation_fraction=saturation_fraction,
        bad_pixel_fraction=bad_pixel_fraction,
    )

    return QCResult(
        mean=float(np.mean(data)),
        median=float(np.median(data)),
        std=float(np.std(data)),
        min_value=float(np.min(data)),
        max_value=float(np.max(data)),
        saturation_fraction=saturation_fraction,
        bad_pixel_fraction=bad_pixel_fraction,
        quality_flag=quality_flag,
    )


def classify_frame_quality(
    std: float,
    saturation_fraction: float,
    bad_pixel_fraction: float,
) -> str:
    if saturation_fraction > 0.01:
        return "reject_saturated"
    if bad_pixel_fraction > 0.05:
        return "reject_bad_pixels"
    if std <= 0:
        return "reject_invalid"
    return "ok"
