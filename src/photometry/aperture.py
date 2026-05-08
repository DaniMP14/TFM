from __future__ import annotations

from typing import Iterable

import numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.aperture import ApertureStats, CircularAnnulus, CircularAperture, aperture_photometry
from photutils.detection import DAOStarFinder


# La función detect_sources se encarga de detectar fuentes en una imagen dada, utilizando el algoritmo DAOStarFinder de photutils. 
# Se aplican estadísticas sigma-clipped para estimar el fondo y el ruido, y se filtran las fuentes detectadas según su posición y brillo.
def detect_sources(
    data: np.ndarray,
    fwhm: float = 4.0,
    threshold_sigma: float = 5.0,
    exclude_border: int = 15,
    brightest: int | None = 200,
) -> list[dict[str, float]]:
    mean_value, median_value, std_value = sigma_clipped_stats(data, sigma=3.0)
    finder = DAOStarFinder(
        fwhm=fwhm,
        threshold=threshold_sigma * std_value,
        exclude_border=True,
        brightest=brightest,
    )
    table = finder(data - median_value)
    if table is None or len(table) == 0:
        return []

    height, width = data.shape
    rows: list[dict[str, float]] = []
    for row in table:
        x = float(row["xcentroid"])
        y = float(row["ycentroid"])
        if x < exclude_border or x > (width - exclude_border):
            continue
        if y < exclude_border or y > (height - exclude_border):
            continue

        rows.append(
            {
                "x": x,
                "y": y,
                "flux": float(row["flux"]),
                "peak": float(row["peak"]),
                "sharpness": float(row["sharpness"]),
                "roundness1": float(row["roundness1"]),
                "roundness2": float(row["roundness2"]),
                "fwhm": fwhm,
            }
        )

    rows.sort(key=lambda item: item["flux"], reverse=True)
    return rows


# La función choose_target_source selecciona la fuente objetivo más cercana a una posición dada (o al centro de la imagen si no se proporciona una posición). 
def choose_target_source(
    sources: Iterable[dict[str, float]],
    image_shape: tuple[int, int],
    target_xy: tuple[float, float] | None = None,
) -> dict[str, float]:
    source_list = list(sources)
    if not source_list:
        raise ValueError("No sources were detected in the reference frame.")

    if target_xy is not None:
        tx, ty = target_xy
    else:
        height, width = image_shape
        tx = width / 2.0
        ty = height / 2.0

    return min(source_list, key=lambda item: (item["x"] - tx) ** 2 + (item["y"] - ty) ** 2) # devuelve la fuente más cercana a la posición objetivo, utilizando la distancia euclidiana al cuadrado para evitar cálculos de raíz cuadrada innecesarios.


# La función choose_comparison_sources selecciona fuentes de comparación basándose en su separación del objetivo, su brillo relativo y su distancia a los bordes de la imagen.
def choose_comparison_sources(
    sources: Iterable[dict[str, float]],
    target_source: dict[str, float],
    image_shape: tuple[int, int],
    max_comparisons: int = 5,
    min_separation: float = 20.0,
    min_edge_distance: float = 25.0,
) -> list[dict[str, float]]:
    height, width = image_shape
    target_flux = max(target_source["flux"], 1e-12)

    candidates: list[tuple[float, dict[str, float]]] = []
    for source in sources:
        if source is target_source:
            continue

        if source["x"] < min_edge_distance or source["x"] > (width - min_edge_distance):
            continue
        if source["y"] < min_edge_distance or source["y"] > (height - min_edge_distance):
            continue

        distance = float(np.hypot(source["x"] - target_source["x"], source["y"] - target_source["y"]))
        if distance < min_separation:
            continue

        brightness_ratio = abs(np.log10(max(source["flux"], 1e-12) / target_flux))
        score = brightness_ratio + 0.001 * distance
        candidates.append((score, source))

    candidates.sort(key=lambda item: item[0])
    return [source for _, source in candidates[:max_comparisons]]


# La función measure_aperture_flux mide el flujo neto de las fuentes en posiciones dadas utilizando fotometría de apertura, 
# calculando el fondo local a través de un anillo alrededor de cada fuente.
def measure_aperture_flux(
    data: np.ndarray,
    positions: list[tuple[float, float]],
    aperture_radius: float,
    annulus_r_in: float,
    annulus_r_out: float,
) -> list[dict[str, float]]:
    aperture = CircularAperture(positions, r=aperture_radius)
    annulus = CircularAnnulus(positions, r_in=annulus_r_in, r_out=annulus_r_out)
    phot_table = aperture_photometry(data, aperture)
    annulus_stats = ApertureStats(data, annulus)
    background_median = np.atleast_1d(annulus_stats.median)

    rows: list[dict[str, float]] = []
    for index, (x, y) in enumerate(positions):
        aperture_sum = float(phot_table["aperture_sum"][index])
        local_background = float(background_median[index])
        background_total = local_background * aperture.area
        rows.append(
            {
                "x": float(x),
                "y": float(y),
                "aperture_sum": aperture_sum,
                "background_median": local_background,
                "background_total": float(background_total),
                "net_flux": float(aperture_sum - background_total),
                "aperture_radius": float(aperture_radius),
                "annulus_r_in": float(annulus_r_in),
                "annulus_r_out": float(annulus_r_out),
            }
        )

    return rows