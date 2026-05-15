from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.wcs import FITSFixedWarning
import astropy.units as u

SUPPORTED_SUFFIXES = {".fits", ".fit", ".fts", ".fz"}
FRAME_ALIASES = {
    "bias": {"bias", "zero"},
    "dark": {"dark", "darks"},
    "flat": {"flat", "flats", "field", "skyflat", "flatfield", "fff", "flat_frame"},
    "light": {"light", "object", "science", "target"},
}
DEFAULT_FILTER = "R"


@dataclass(frozen=True)
class FrameRecord:
    path: Path
    frame_type: str
    filter_name: str | None
    exptime: float | None


def discover_fits_files(root_dir: str | Path) -> list[Path]:
    root_path = Path(root_dir)
    return sorted(
        path for path in root_path.rglob("*") if path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def infer_frame_type(path: str | Path, header: fits.Header | None = None) -> str:
    candidates: list[str] = []

    if header is not None:
        for key in ("IMAGETYP", "OBSTYPE", "FRAME", "TYPE"):
            value = header.get(key)
            if value:
                # Hacer split por si hay varias palabras (e.g. "Light Frame")
                candidates.extend(str(value).strip().lower().split())

    path_tokens = Path(path).stem.lower().replace("-", "_").split("_")
    candidates.extend(path_tokens)

    for frame_type, aliases in FRAME_ALIASES.items():
        # Buscar coincidencia exacta O si el alias está contenido en el candidate
        if any(candidate in aliases or any(alias in candidate for alias in aliases) for candidate in candidates):
            return frame_type

    return "light" # Asumir que es un light si no se puede inferir


def load_header(path: str | Path) -> fits.Header:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        return fits.getheader(path)


def load_ccd(path: str | Path, unit: str = "adu") -> CCDData:
    # Algunos FITS reducidos guardan la imagen en una extensión distinta del PRIMARY.
    # CCDData.read() intenta por defecto leer la primera HDU, así que resolvemos
    # automáticamente la primera extensión que realmente contiene datos de imagen.
    ccd_logger = logging.getLogger("astropy.nddata.ccddata")
    previous_level = ccd_logger.level
    ccd_logger.setLevel(logging.ERROR)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        with fits.open(path) as hdul:
            hdu_index = next((i for i, hdu in enumerate(hdul) if getattr(hdu, "data", None) is not None), 0)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            return CCDData.read(path, unit=unit, hdu=hdu_index)
    finally:
        ccd_logger.setLevel(previous_level)


def get_filter_name(header: fits.Header) -> str | None:
    for key in ("FILTER", "FILTER1", "FILTNAM"):
        value = header.get(key)
        if value not in (None, ""):
            return str(value).strip()
    return DEFAULT_FILTER


def get_exposure_time(header: fits.Header) -> float | None:
    for key in ("EXPTIME", "EXPOSURE"):
        value = header.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def index_frames(root_dir: str | Path) -> list[FrameRecord]:
    records: list[FrameRecord] = []
    for path in discover_fits_files(root_dir):
        header = load_header(path)
        records.append(
            FrameRecord(
                path=path,
                frame_type=infer_frame_type(path, header),
                filter_name=get_filter_name(header),
                exptime=get_exposure_time(header),
            )
        )
    return records


def group_by_frame_type(records: Iterable[FrameRecord]) -> dict[str, list[FrameRecord]]:
    grouped: dict[str, list[FrameRecord]] = {}
    for record in records:
        grouped.setdefault(record.frame_type, []).append(record)
    return grouped


def group_flats_by_filter(records: Iterable[FrameRecord]) -> dict[str, list[FrameRecord]]:
    grouped: dict[str, list[FrameRecord]] = {}
    for record in records:
        filter_name = record.filter_name or "NO_FILTER"
        grouped.setdefault(filter_name, []).append(record)
    return grouped


def ccd_median(ccds: Iterable[CCDData]) -> np.ndarray:
    stack = np.stack([ccd.data.astype(float) for ccd in ccds], axis=0)
    return np.median(stack, axis=0)


def array_to_ccd(data: np.ndarray, unit: str = "adu") -> CCDData:
    return CCDData(data.astype(float), unit=u.Unit(unit))
