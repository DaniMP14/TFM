from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.nddata import CCDData
from scipy import ndimage
from skimage.registration import phase_cross_correlation


@dataclass(frozen=True)
class ShiftEstimate:
    dx: float
    dy: float
    peak_value: float


def _similarity_score(reference: np.ndarray, moving: np.ndarray, dx: float, dy: float) -> float:
    """Compute a robust similarity score in [0, 1] after applying the estimated shift."""
    # Align moving image to reference using the estimated shift.
    shifted = ndimage.shift(
        moving,
        shift=(dy, dx),
        order=1,
        mode="constant",
        cval=float(np.median(moving)),
        prefilter=False,
    )

    # Ignore borders where interpolation artifacts dominate.
    margin_y = max(8, int(0.1 * reference.shape[0]))
    margin_x = max(8, int(0.1 * reference.shape[1]))
    ref_crop = reference[margin_y:-margin_y, margin_x:-margin_x]
    mov_crop = shifted[margin_y:-margin_y, margin_x:-margin_x]

    if ref_crop.size == 0 or mov_crop.size == 0:
        return 0.0

    ref_vec = ref_crop.ravel().astype(float)
    mov_vec = mov_crop.ravel().astype(float)
    ref_std = float(np.std(ref_vec))
    mov_std = float(np.std(mov_vec))
    if ref_std <= 1e-12 or mov_std <= 1e-12:
        return 0.0

    corr = float(np.corrcoef(ref_vec, mov_vec)[0, 1])
    if not np.isfinite(corr):
        return 0.0

    # Map Pearson correlation [-1, 1] to [0, 1].
    return float(np.clip((corr + 1.0) * 0.5, 0.0, 1.0))

# Calcula el desplazamiento subpixel entre dos imágenes usando phase_cross_correlation
# (correlación cruzada en espacio de frecuencias con upsampling). Precisión ~1/upsample_factor px.
# Rechaza desplazamientos que superen max_shift para evitar saltos erróneos.
def estimate_subpixel_shift(
    reference: np.ndarray,
    moving: np.ndarray,
    max_shift: int = 50,
    upsample_factor: int = 10,
) -> ShiftEstimate:
    ref = np.asarray(reference, dtype=float)
    mov = np.asarray(moving, dtype=float)

    if ref.shape != mov.shape:
        raise ValueError("Reference and moving frames must have the same shape.")

    ref = ref - np.median(ref)
    mov = mov - np.median(mov)

    result = phase_cross_correlation(
        ref, mov,
        upsample_factor=upsample_factor,
        normalization='phase',
    )
    # skimage >= 0.19 devuelve (shift, error, phasediff); >= 0.21 puede devolver solo shift
    shift_arr = result[0] if isinstance(result, tuple) else result
    dy, dx = float(shift_arr[0]), float(shift_arr[1])

    if abs(dx) > max_shift or abs(dy) > max_shift:
        raise ValueError(
            f"Desplazamiento estimado ({dx:.2f}, {dy:.2f}) supera max_shift={max_shift} px."
        )

    # Use post-shift image similarity as quality score to avoid API/version-dependent
    # behavior of phase_cross_correlation error values.
    peak_value = _similarity_score(ref, mov, dx=dx, dy=dy)

    return ShiftEstimate(dx=dx, dy=dy, peak_value=peak_value)


# Alias de compatibilidad
estimate_integer_shift = estimate_subpixel_shift

# Toma un CCDData y aplica el desplazamiento usando interpolación (por defecto cúbica) para alinear la imagen. 
# Preserva la unidad y el encabezado, añadiendo información sobre el desplazamiento aplicado.
def apply_shift(
    ccd: CCDData,
    dx: float,
    dy: float,
    order: int = 1,
) -> CCDData:
    shifted = ndimage.shift(
        ccd.data.astype(float),
        shift=(dy, dx),
        order=order,
        mode="constant",
        cval=float(np.median(ccd.data)),
        prefilter=True,
    )

    aligned = CCDData(shifted, unit=ccd.unit)
    aligned.header = ccd.header.copy()
    aligned.header["ALIDX"] = dx
    aligned.header["ALIDY"] = dy
    return aligned
