from __future__ import annotations

import pandas as pd
from pathlib import Path

import os
import shutil
import subprocess
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from astropy.wcs import FITSFixedWarning

from reduction.io import get_exposure_time


@dataclass(frozen=True)
class AlignedFrameRecord:
    path: Path
    frame_index: int
    time_jd: float | None
    exptime: float | None


def discover_aligned_files(aligned_dir: str | Path) -> list[Path]:
    aligned_path = Path(aligned_dir)
    paths = (
        sorted(aligned_path.glob("ali_*.fit"))
        + sorted(aligned_path.glob("ali_*.fits"))
        + sorted(aligned_path.glob("ali_*.fts"))
        + sorted(aligned_path.glob("ali_*.fz"))
    )
    return sorted(set(paths))


def load_alignment_summary(summary_path: str | Path | None) -> pd.DataFrame | None:
    if summary_path is None:
        return None

    path = Path(summary_path)
    if not path.exists():
        return None

    return pd.read_csv(path)


def _read_header_silently(path: str | Path) -> fits.Header:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        return fits.getheader(path)


def extract_time_jd(header: fits.Header) -> float | None:
    if header.get("JD") is not None:
        return float(header["JD"])

    if header.get("MJD-OBS") is not None:
        return Time(float(header["MJD-OBS"]), format="mjd").jd

    date_obs = header.get("DATE-OBS")
    time_obs = header.get("TIME-OBS")
    if date_obs and time_obs:
        return Time(f"{date_obs}T{time_obs}", format="isot", scale="utc").jd

    if date_obs:
        return Time(str(date_obs), format="isot", scale="utc").jd

    return None


def index_aligned_frames(
    aligned_dir: str | Path,
    summary_path: str | Path | None = None,
    include_rejected: bool = False,
) -> list[AlignedFrameRecord]:
    aligned_files = discover_aligned_files(aligned_dir)
    summary = load_alignment_summary(summary_path)

    allowed_files: set[str] | None = None
    if summary is not None and "file" in summary.columns and not include_rejected:
        if "status" in summary.columns:
            allowed_files = set(summary.loc[summary["status"] == "aligned", "file"].astype(str))
        elif "rejected" in summary.columns:
            allowed_files = set(summary.loc[~summary["rejected"], "file"].astype(str))

    records: list[AlignedFrameRecord] = []
    for frame_index, path in enumerate(aligned_files):
        original_name = path.name.removeprefix("ali_")
        if allowed_files is not None and original_name not in allowed_files:
            continue

        header = _read_header_silently(path)
        records.append(
            AlignedFrameRecord(
                path=path,
                frame_index=len(records),
                time_jd=extract_time_jd(header),
                exptime=get_exposure_time(header),
            )
        )

    return records


def _parse_sexagesimal_coord(ra_str: str, dec_str: str) -> tuple[float, float]:
    """Parse OBJCTRA/OBJCTDEC strings (e.g. '20 14 00', '+65 13 54') to degrees."""
    coord = SkyCoord(ra=ra_str, dec=dec_str, unit=("hourangle", "deg"))
    return float(coord.ra.deg), float(coord.dec.deg)


def _header_to_celestial_wcs(header: fits.Header) -> WCS | None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            wcs = WCS(header)
    except Exception:
        return None
    return wcs if wcs.has_celestial else None


def _wcs_sidecar_path(reference_frame_path: str | Path, wcs_cache_dir: str | Path | None) -> Path:
    frame_path = Path(reference_frame_path)
    sidecar_dir = Path(wcs_cache_dir) if wcs_cache_dir is not None else frame_path.parent / "wcs"
    return sidecar_dir / f"{frame_path.stem}_wcs.fits"


def _load_sidecar_wcs(reference_frame_path: str | Path, wcs_cache_dir: str | Path | None) -> WCS | None:
    sidecar_path = _wcs_sidecar_path(reference_frame_path, wcs_cache_dir)
    if not sidecar_path.exists():
        return None

    try:
        sidecar_header = _read_header_silently(sidecar_path)
    except Exception:
        return None

    return _header_to_celestial_wcs(sidecar_header)


def estimate_pixel_scale_arcsec(header: fits.Header) -> float | None:
    """Estimate pixel scale in arcsec/pixel from FITS optical metadata.

    Uses:
    - XPIXSZ or YPIXSZ in microns
    - FOCALLEN in millimeters
    - XBINNING/YBINNING (optional, defaults to 1)
    """
    focal_len_mm = header.get("FOCALLEN") or header.get("FOCALLENGTH") or header.get("FOCAL")
    pixel_um = header.get("XPIXSZ") or header.get("YPIXSZ")
    if focal_len_mm is None or pixel_um is None:
        return None

    try:
        focal_len_mm = float(focal_len_mm)
        pixel_um = float(pixel_um)
        xbin = float(header.get("XBINNING", 1.0))
    except (TypeError, ValueError):
        return None

    if focal_len_mm <= 0.0 or pixel_um <= 0.0 or xbin <= 0.0:
        return None

    effective_pixel_um = pixel_um * xbin
    return float(206.265 * effective_pixel_um / focal_len_mm)


# La función _simbad_result_to_coord hace un parsing robusto de los resultados de SIMBAD para extraer las coordenadas del objeto,
# manejando diferentes formatos y unidades que pueden presentarse en la respuesta.
def _simbad_result_to_coord(result_table) -> SkyCoord:
    for ra_col, dec_col in (("ra", "dec"), ("RA", "DEC")):
        if ra_col not in result_table.colnames or dec_col not in result_table.colnames:
            continue

        ra_value = result_table[ra_col][0]
        dec_value = result_table[dec_col][0]
        ra_unit = getattr(result_table[ra_col], "unit", None)
        dec_unit = getattr(result_table[dec_col], "unit", None)

        if ra_unit is not None and dec_unit is not None:
            return SkyCoord(ra=float(ra_value), dec=float(dec_value), unit=(ra_unit, dec_unit))

        ra_text = str(ra_value).strip()
        dec_text = str(dec_value).strip()
        return SkyCoord(ra=ra_text, dec=dec_text, unit=("hourangle", "deg"))

    raise ValueError("SIMBAD response does not contain RA/DEC columns.")


def _coord_to_pixel_with_wcs(wcs: WCS, target_coord: SkyCoord) -> tuple[float, float]:
    x, y = wcs.world_to_pixel(target_coord)
    return float(x), float(y)


def _solve_wcs_locally(
    reference_frame_path: str | Path,
    pixel_scale_arcsec: float | None,
    timeout_sec: int,
) -> tuple[WCS | None, fits.Header | None]:
    solve_field = shutil.which("solve-field")
    if solve_field is None:
        return None, None

    frame_path = Path(reference_frame_path)
    with tempfile.TemporaryDirectory(prefix="wcs_solve_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        cmd = [
            solve_field,
            str(frame_path),
            "--overwrite",
            "--no-plots",
            "--dir",
            str(tmp_path),
            "--cpulimit",
            str(max(30, int(timeout_sec))),
        ]

        if pixel_scale_arcsec is not None:
            scale_low = max(0.1, float(pixel_scale_arcsec) * 0.8)
            scale_high = float(pixel_scale_arcsec) * 1.2
            cmd.extend(
                [
                    "--scale-units",
                    "arcsecperpix",
                    "--scale-low",
                    f"{scale_low:.6f}",
                    "--scale-high",
                    f"{scale_high:.6f}",
                ]
            )

        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                timeout=max(30, int(timeout_sec) + 30),
            )
        except Exception:
            return None, None

        if proc.returncode != 0:
            return None, None

        solved_candidates = sorted(tmp_path.glob("*.new"))
        if not solved_candidates:
            return None, None

        solved_path = solved_candidates[0]
        solved_header = fits.getheader(solved_path)
        solved_wcs = _header_to_celestial_wcs(solved_header)
        if solved_wcs is None:
            return None, None
        return solved_wcs, solved_header


def _solve_wcs_online(reference_frame_path: str | Path, api_key: str | None) -> tuple[WCS | None, fits.Header | None]:
    if not api_key:
        api_key = os.environ.get("ASTROMETRY_NET_API_KEY")
    if not api_key:
        return None, None

    try:
        from astroquery.astrometry_net import AstrometryNet
    except Exception:
        return None, None

    ast = AstrometryNet()
    ast.api_key = api_key

    try:
        solved_header = ast.solve_from_image(str(reference_frame_path))
    except Exception:
        return None, None

    solved_wcs = _header_to_celestial_wcs(solved_header)
    if solved_wcs is None:
        return None, None
    return solved_wcs, solved_header


def _resolve_target_xy_approx(
    target_coord: SkyCoord,
    header: fits.Header,
    pixel_scale_arcsec: float | None,
) -> tuple[float, float]:
    objctra = header.get("OBJCTRA")  # TODO: regex con mas claves posibles (OBJRA, OBJ-RA, etc)
    objctdec = header.get("OBJCTDEC")
    if objctra is None or objctdec is None:
        raise ValueError(
            "FITS header is missing OBJCTRA / OBJCTDEC pointing coordinates. "
            "Provide target_xy manually in PhotometryConfig."
        )

    if pixel_scale_arcsec is None:
        pixel_scale_arcsec = estimate_pixel_scale_arcsec(header)
    if pixel_scale_arcsec is None:
        raise ValueError(
            "Could not estimate pixel scale from FITS header. "
            "Provide pixel_scale_arcsec in PhotometryConfig."
        )

    center_ra_deg, center_dec_deg = _parse_sexagesimal_coord(str(objctra).strip(), str(objctdec).strip())
    center_coord = SkyCoord(ra=center_ra_deg, dec=center_dec_deg, unit="deg")

    delta_ra_arcsec = float((target_coord.ra.deg - center_coord.ra.deg) * np.cos(np.radians(center_coord.dec.deg)) * 3600.0)
    delta_dec_arcsec = float((target_coord.dec.deg - center_coord.dec.deg) * 3600.0)

    naxis1 = int(header.get("NAXIS1", 0))
    naxis2 = int(header.get("NAXIS2", 0))
    center_x = naxis1 / 2.0
    center_y = naxis2 / 2.0

    target_x = center_x - delta_ra_arcsec / float(pixel_scale_arcsec)
    target_y = center_y + delta_dec_arcsec / float(pixel_scale_arcsec)
    print(f"[photometry] Target resolved with approximate pointing model: ({target_x:.1f}, {target_y:.1f})")
    return float(target_x), float(target_y)


# Selecciona el frame de referencia con mejor calidad para fotometría.
# Priorizamos bajo bad_pixel_fraction/saturation/background/std y señal suficiente.
def select_best_qc_frame(aligned_dir: str | Path, qc_summary_path: str | Path | None = None) -> Path | None:
    aligned_dir = Path(aligned_dir)
    qc_path = Path(qc_summary_path) if qc_summary_path is not None else aligned_dir.parent / "reduced" / "qc_summary.csv"
    if not qc_path.exists():
        return None

    df = pd.read_csv(qc_path)
    if "file" not in df.columns or "quality_flag" not in df.columns:
        return None

    df_ok = df[df["quality_flag"].astype(str).str.lower() == "ok"].copy()
    if df_ok.empty:
        return None

    score = np.zeros(len(df_ok), dtype=float)
    weights = {
        "bad_pixel_fraction": -0.45,
        "saturation_fraction": -0.25,
        "background_median": -0.15,
        "std": -0.10,
        "median": 0.05,
    }
    for column, weight in weights.items():
        if column in df_ok.columns:
            rank = pd.to_numeric(df_ok[column], errors="coerce").rank(method="average", pct=True)
            rank = rank.fillna(0.5)
            score += weight * rank.to_numpy(dtype=float)

    df_ok = df_ok.assign(_reference_score=score).sort_values("_reference_score", ascending=False)
    for fname in df_ok["file"].astype(str):
        name = fname.strip()
        if not name:
            continue
        fpath = aligned_dir / f"ali_cal_{name}"
        if fpath.exists():
            return fpath
    return None

def resolve_target_xy(
    reference_frame_path: str | Path,
    object_name: str,
    pixel_scale_arcsec: float | None = None,
    prefer_wcs: bool = True,
    wcs_solver: str = "auto",
    wcs_cache_dir: str | Path | None = None,
    astrometry_api_key: str | None = None,
    wcs_solver_timeout_sec: int = 180,
    persist_solved_wcs: bool = True,
) -> tuple[float, float]:
    """Return target position (x, y) for object_name.

    Priority:
    1) Existing FITS WCS
    2) Cached sidecar WCS
    3) Newly solved WCS (local astrometry.net then online)
    4) Approximate pointing model from OBJCTRA/OBJCTDEC + plate scale
    """
    from astroquery.simbad import Simbad

    result = Simbad.query_object(object_name)
    if result is None or len(result) == 0:
        raise ValueError(f"SIMBAD could not resolve object name: '{object_name}'")
    target_coord = _simbad_result_to_coord(result)

    header = _read_header_silently(reference_frame_path)

    if prefer_wcs:
        if wcs_solver not in {"auto", "none", "local", "online"}:
            raise ValueError("wcs_solver must be one of: 'auto', 'none', 'local', 'online'.")

        header_wcs = _header_to_celestial_wcs(header)
        if header_wcs is not None:
            x, y = _coord_to_pixel_with_wcs(header_wcs, target_coord)
            print(f"[photometry] Target resolved with FITS WCS: ({x:.1f}, {y:.1f})")
            return x, y

        sidecar_wcs = _load_sidecar_wcs(reference_frame_path, wcs_cache_dir)
        if sidecar_wcs is not None:
            x, y = _coord_to_pixel_with_wcs(sidecar_wcs, target_coord)
            print(f"[photometry] Target resolved with cached WCS sidecar: ({x:.1f}, {y:.1f})")
            return x, y

        solver_order: list[str] = []
        if wcs_solver == "auto":
            solver_order = ["local", "online"]
        elif wcs_solver in {"local", "online"}:
            solver_order = [wcs_solver]

        for mode in solver_order:
            if mode == "local":
                solved_wcs, solved_header = _solve_wcs_locally(
                    reference_frame_path,
                    pixel_scale_arcsec=pixel_scale_arcsec,
                    timeout_sec=wcs_solver_timeout_sec,
                )
                if solved_wcs is None:
                    continue

                if persist_solved_wcs and solved_header is not None:
                    with fits.open(reference_frame_path) as original_hdu:
                        original_hdu[0].header.update(solved_header)
                        sidecar_path = _wcs_sidecar_path(reference_frame_path, wcs_cache_dir)
                        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
                        original_hdu.writeto(sidecar_path, overwrite=True)
                        print(f"[photometry] Cached solved WCS at {sidecar_path}")

                x, y = _coord_to_pixel_with_wcs(solved_wcs, target_coord)
                print(f"[photometry] Target resolved with local astrometry.net WCS: ({x:.1f}, {y:.1f})")
                return x, y

            if mode == "online":
                solved_wcs, solved_header = _solve_wcs_online(
                    reference_frame_path,
                    api_key=astrometry_api_key,
                )
                if solved_wcs is None or solved_header is None:
                    continue

                if persist_solved_wcs:
                    with fits.open(reference_frame_path) as original_hdu:
                        original_hdu[0].header.update(solved_header)
                        sidecar_path = _wcs_sidecar_path(reference_frame_path, wcs_cache_dir)
                        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
                        original_hdu.writeto(sidecar_path, overwrite=True)
                        print(f"[photometry] Cached solved WCS at {sidecar_path}")

                x, y = _coord_to_pixel_with_wcs(solved_wcs, target_coord)
                print(f"[photometry] Target resolved with online astrometry.net WCS: ({x:.1f}, {y:.1f})")
                return x, y

    return _resolve_target_xy_approx(
        target_coord=target_coord,
        header=header,
        pixel_scale_arcsec=pixel_scale_arcsec,
    )
