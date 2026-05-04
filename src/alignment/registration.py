from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.nddata import CCDData
from scipy import ndimage, signal


@dataclass(frozen=True)
class ShiftEstimate:
    dx: float
    dy: float
    peak_value: float

# Toma dos imágenes (referencia y móvil) y calcula la correlación cruzada usando FFT (Transformada Rápida de Fourier) para encontrar el desplazamiento que maximiza la correlación.
# Limita la búsqueda a un área alrededor del centro para evitar desplazamientos excesivos.
# Devuelve el desplazamiento estimado en píxeles (dx, dy) y el valor de la correlación en ese punto.
def estimate_integer_shift(
    reference: np.ndarray,
    moving: np.ndarray,
    max_shift: int = 50,
) -> ShiftEstimate:
    ref = np.asarray(reference, dtype=float)
    mov = np.asarray(moving, dtype=float)

    if ref.shape != mov.shape:
        raise ValueError("Reference and moving frames must have the same shape.")

    ref = ref - np.median(ref)
    mov = mov - np.median(mov)

    corr = signal.fftconvolve(ref, mov[::-1, ::-1], mode="same")

    center_y = corr.shape[0] // 2
    center_x = corr.shape[1] // 2

    y_min = max(center_y - max_shift, 0)
    y_max = min(center_y + max_shift + 1, corr.shape[0])
    x_min = max(center_x - max_shift, 0)
    x_max = min(center_x + max_shift + 1, corr.shape[1])

    local_corr = corr[y_min:y_max, x_min:x_max]
    peak_y_local, peak_x_local = np.unravel_index(np.argmax(local_corr), local_corr.shape)

    peak_y = y_min + peak_y_local
    peak_x = x_min + peak_x_local

    dy = float(peak_y - center_y)
    dx = float(peak_x - center_x)

    return ShiftEstimate(dx=dx, dy=dy, peak_value=float(corr[peak_y, peak_x]))

# Toma un CCDData y aplica el desplazamiento usando interpolación (por defecto cúbica) para alinear la imagen. 
# Preserva la unidad y el encabezado, añadiendo información sobre el desplazamiento aplicado.
def apply_shift(
    ccd: CCDData,
    dx: float,
    dy: float,
    order: int = 3,
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
