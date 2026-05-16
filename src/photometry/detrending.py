"""
Normalización y detrending de curvas de luz diferenciales.

Funciones para:
1. Normalizar la curva a 1.0 en baseline pre-tránsito
2. Remover tendencias (polinomio, correlación con FWHM/background/airmass)
3. Enmascarar región de tránsito para fit robusto
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial


def normalize_to_baseline( # TODO: baseline no tiene por que ser pre-transito, puede ser post-transito o ambos
    light_curve: pd.DataFrame,
    relative_flux_column: str = "relative_flux",
    baseline_percentile_range: tuple[float, float] = (0.0, 30.0),
    time_column: str | None = None,
) -> pd.DataFrame:
    """
    Normaliza la curva de luz a 1.0 usando la mediana de la baseline pre-tránsito.
    
    Args:
        light_curve: DataFrame con curva de luz
        relative_flux_column: nombre de columna con flujo relativo
        baseline_percentile_range: (inicio, fin) en % del tiempo para definir pre-tránsito
        time_column: columna de tiempo para definir baseline (si None, usa primeros N puntos)
    
    Returns:
        DataFrame con flujo normalizado en 'normalized_flux'
    """
    result = light_curve.copy()
    
    n_points = len(result)
    start_idx = int(n_points * baseline_percentile_range[0] / 100.0)
    end_idx = int(n_points * baseline_percentile_range[1] / 100.0)
    
    if start_idx >= end_idx or end_idx > n_points:
        raise ValueError(f"Invalid baseline range: ({start_idx}, {end_idx}) for {n_points} points")
    
    baseline_flux = float(result[relative_flux_column].iloc[start_idx:end_idx].median())
    
    if baseline_flux <= 0:
        raise ValueError(f"Baseline flux is {baseline_flux} — check for bad data")
    
    result["normalized_flux"] = result[relative_flux_column] / baseline_flux
    
    return result


def fit_polynomial_trend(
    x: np.ndarray,
    y: np.ndarray,
    degree: int = 2,
    sigma: np.ndarray | None = None,
    robust: bool = True,
) -> Polynomial:
    """
    Ajusta un polinomio a los datos usando mínimos cuadrados.
    
    Args:
        x: coordenada independiente (ej: índice de frame, tiempo)
        y: datos a ajustar
        degree: grado del polinomio
        sigma: errores (pesos inversos)
        robust: si True, usa sigma-clipping iterativo para remover outliers
    
    Returns:
        Polinomio de numpy.polynomial.Polynomial
    """
    if len(x) < degree + 1:
        raise ValueError(f"Not enough points ({len(x)}) for polynomial degree {degree}")
    
    if robust:
        # Iterative sigma-clipping: fit, compute residuals, mask outliers, refit
        mask = np.ones(len(x), dtype=bool)
        for _ in range(3):
            p = Polynomial.fit(
                x[mask],
                y[mask],
                deg=degree,
                w=sigma[mask] if sigma is not None else None,
            )
            residuals = np.abs(y - p(x))
            threshold = 2.5 * np.std(residuals)
            mask = residuals < threshold
            if mask.sum() < degree + 1:
                mask[:] = True
                break
    else:
        p = Polynomial.fit(x, y, deg=degree, w=sigma if sigma is not None else None)
    
    return p


def detrend_polynomial( #TODO: mejorar, provoca sobreajuste (tendencia lineal)
    light_curve: pd.DataFrame,
    normalized_flux_column: str = "normalized_flux",
    index_column: str = "frame_index",
    degree: int = 1,
    mask_out_of_transit: bool = False,
    transit_mask_column: str | None = None,
    robust: bool = True,
    weight_column: str | None = None,
) -> pd.DataFrame:
    """
    Remueve tendencias polinomiales de la curva de luz.
    
    Args:
        light_curve: DataFrame con curva de luz normalizada
        normalized_flux_column: columna con flujo normalizado
        index_column: columna con índice/tiempo para el fit
        degree: grado del polinomio (típicamente 2–3)
        mask_out_of_transit: si True, ajusta solo en puntos fuera de tránsito
        transit_mask_column: nombre de columna booleana con True en tránsito (si mask_out_of_transit=True)
        robust: si True, usa sigma-clipping iterativo
        weight_column: columna con pesos positivos para WLS; pesos bajos reducen la
            influencia de frames con peor calidad (ej: background lunar alto)
    
    Returns:
        DataFrame con 'detrended_flux' = normalized_flux / trend
    """
    result = light_curve.copy()
    
    x = result[index_column].values.astype(float)
    y = result[normalized_flux_column].values.astype(float)
    
    # Si se pasa una máscara de tránsito, ajustar solo fuera del tránsito
    if mask_out_of_transit and transit_mask_column is not None:
        if transit_mask_column not in result.columns:
            raise ValueError(f"Transit mask column '{transit_mask_column}' not found")
        mask = ~result[transit_mask_column].astype(bool)
    else:
        mask = np.ones(len(x), dtype=bool)
    
    if mask.sum() < degree + 1:
        raise ValueError(f"Not enough unmasked points ({mask.sum()}) for polynomial degree {degree}")

    weights = None
    if weight_column is not None:
        if weight_column not in result.columns:
            raise ValueError(f"Weight column '{weight_column}' not found")
        weights = result[weight_column].values.astype(float)
        weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 1.0)
    
    # Ajustar polinomio
    p = fit_polynomial_trend(
        x[mask],
        y[mask],
        degree=degree,
        sigma=weights[mask] if weights is not None else None,
        robust=robust,
    )
    
    # Evaluar tendencia en todos los puntos
    trend = p(x)
    
    # Detrended = normalized / trend
    result["trend"] = trend
    result["detrended_flux"] = result[normalized_flux_column] / np.maximum(trend, 1e-12)
    
    return result


def detrend_with_auxilliary(
    light_curve: pd.DataFrame,
    auxilliary_columns: list[str],
    normalized_flux_column: str = "normalized_flux",
    degree: int = 1,
    mask_out_of_transit: bool = False,
    transit_mask_column: str | None = None,
) -> pd.DataFrame:
    """
    Detrending usando correlación con variables auxiliares (ej: FWHM, background, airmass).
    
    Implementa un modelo lineal:
        flux_corrected = flux_obs / (1 + sum(coeff_i * (aux_i - median(aux_i))))
    
    Args:
        light_curve: DataFrame con curva de luz normalizada + columnas auxiliares
        auxilliary_columns: lista de nombres de columnas auxiliares
        normalized_flux_column: columna con flujo normalizado
        degree: grado del fit (1=lineal, 2=cuadrático, etc.)
        mask_out_of_transit: si True, ajusta solo fuera de tránsito
        transit_mask_column: máscara de tránsito (si mask_out_of_transit=True)
    
    Returns:
        DataFrame con 'detrended_flux' y 'auxiliary_correction'
    """
    result = light_curve.copy()
    
    # Validar que las columnas auxiliares existan y sean numéricas
    missing = [c for c in auxilliary_columns if c not in result.columns]
    if missing:
        raise ValueError(f"Missing auxiliary columns: {missing}")
    
    # Preparar datos para ajuste
    y = result[normalized_flux_column].values.astype(float)
    
    if mask_out_of_transit and transit_mask_column is not None:
        if transit_mask_column not in result.columns:
            raise ValueError(f"Transit mask column '{transit_mask_column}' not found")
        fit_mask = ~result[transit_mask_column].astype(bool)
    else:
        fit_mask = np.ones(len(y), dtype=bool)
    
    # Normalizar variables auxiliares a mediana
    aux_normalized = {}
    for col in auxilliary_columns:
        median_val = float(result[col].median())
        if median_val == 0:
            median_val = 1.0
        aux_normalized[col] = (result[col].values - median_val) / median_val
    
    # Construir matriz de features
    if degree == 1:
        # Lineal: solo términos de primer orden
        X = np.column_stack([aux_normalized[col][fit_mask] for col in auxilliary_columns])
        y_fit = y[fit_mask]
    else:
        # Incluir términos polinomiales
        features = [aux_normalized[col] for col in auxilliary_columns]
        if degree >= 2:
            for col in auxilliary_columns:
                features.append(aux_normalized[col] ** 2)
        X = np.column_stack([features[i][fit_mask] for i in range(len(features))])
        y_fit = y[fit_mask]
    
    # Ajuste lineal: log(flux) = A0 + sum(Ai * aux_i) -> flux = exp(A0 + ...)
    try:
        coeffs = np.linalg.lstsq(np.column_stack([np.ones(len(y_fit)), X]), np.log(y_fit), rcond=None)[0]
        a0 = coeffs[0]
        trend_log = a0 + X.dot(coeffs[1:]) if X.shape[1] > 0 else a0
        correction = np.exp(trend_log)
    except Exception:
        # Si falla, usar mínimos cuadrados directo en escala lineal
        coeffs = np.linalg.lstsq(np.column_stack([np.ones(len(y_fit)), X]), y_fit, rcond=None)[0]
        a0 = coeffs[0]
        trend_lin = a0 + X.dot(coeffs[1:]) if X.shape[1] > 0 else a0
        correction = np.maximum(trend_lin, 1e-12)
    
    result["auxiliary_correction"] = correction
    result["detrended_flux"] = y / np.maximum(correction, 1e-12)
    
    return result


def estimate_transit_mask(
    light_curve: pd.DataFrame,
    normalized_flux_column: str = "normalized_flux",
    window_buffer_points: int = 5,
    min_transit_points: int = 5,
) -> np.ndarray:
    """
     Estima una máscara de tránsito basada en cruces al 10% de profundidad:
     1) Suavizado robusto de la curva normalizada.
     2) Cruce izquierdo/derecho del nivel 10%.
     3) Máscara final = [cruce_izq_10%, cruce_der_10%] +/- buffer (en puntos).

    Args:
        light_curve: DataFrame con curva normalizada.
        normalized_flux_column: columna con flujo normalizado.
        window_buffer_points: número de puntos extra por lado (simétrico).
            Se usa tal cual (si es negativo, se fuerza a 0).
        time_column: columna temporal (opcional, para prior temporal).
        expected_mid_time: instante esperado del centro de tránsito (opcional).
        expected_duration: duración esperada total del tránsito (opcional).
        min_transit_points: número mínimo de puntos para aceptar una máscara refinada.

    Returns:
        Array booleano con True en puntos de tránsito.
    """
    flux = light_curve[normalized_flux_column].to_numpy(dtype=float)
    n_points = len(flux)
    finite_flux = np.isfinite(flux)
    if np.sum(finite_flux) < max(min_transit_points, 3):
        return np.zeros(n_points, dtype=bool)

    # Serie suavizada para estimar cruces de nivel de profundidad.
    flux_filled = flux.copy()
    flux_valid = flux[finite_flux]
    median_flux = np.median(flux_valid)
    flux_filled[~finite_flux] = median_flux
    smooth_window = max(7, (n_points // 25) | 1)
    flux_smooth = pd.Series(flux_filled).rolling(
        smooth_window,
        center=True,
        min_periods=1,
    ).median().to_numpy()

    i_min = int(np.argmin(flux_smooth))
    x = np.arange(n_points, dtype=float)
    # Si existe una máscara previa, usar su OOT para el nivel base mejora la
    # estabilidad frente a pendientes largas del baseline.
    if "in_transit_mask" in light_curve.columns:
        prev_mask = light_curve["in_transit_mask"].astype(bool).to_numpy()
    else:
        prev_mask = np.zeros(n_points, dtype=bool)

    if prev_mask.any() and (~prev_mask).sum() >= max(20, n_points // 5):
        oot_level = float(np.median(flux[~prev_mask]))
    else:
        oot_level = float(np.percentile(flux_smooth[finite_flux], 75))

    depth = float(max(oot_level - np.min(flux_smooth[finite_flux]), 0.0))

    buffer_pts = max(0, int(window_buffer_points))

    def _interp_cross(x1: float, y1: float, x2: float, y2: float, level: float) -> float:
        if y2 == y1:
            return float(x1)
        frac = (level - y1) / (y2 - y1)
        frac = float(np.clip(frac, 0.0, 1.0))
        return float(x1 + frac * (x2 - x1))

    if depth <= 0.0:
        return np.zeros(n_points, dtype=bool)

    level_10 = oot_level - 0.10 * depth

    left_10_x = None
    for i in range(i_min - 1, -1, -1):
        if flux_smooth[i] > level_10 and flux_smooth[i + 1] <= level_10:
            left_10_x = _interp_cross(x[i], flux_smooth[i], x[i + 1], flux_smooth[i + 1], level_10)
            break

    right_10_x = None
    for i in range(i_min, n_points - 1):
        if flux_smooth[i] <= level_10 and flux_smooth[i + 1] > level_10:
            right_10_x = _interp_cross(x[i], flux_smooth[i], x[i + 1], flux_smooth[i + 1], level_10)
            break

    if left_10_x is None or right_10_x is None:
        return np.zeros(n_points, dtype=bool)

    i_start = max(0, int(np.floor(left_10_x)) - buffer_pts)
    i_end = min(n_points - 1, int(np.ceil(right_10_x)) + buffer_pts)

    if i_end < i_start:
        return np.zeros(n_points, dtype=bool)

    transit_mask = np.zeros(n_points, dtype=bool)
    transit_mask[i_start : i_end + 1] = True
    transit_mask &= finite_flux

    if np.sum(transit_mask) < min_transit_points:
        return np.zeros(n_points, dtype=bool)

    return transit_mask
