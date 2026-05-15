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


def estimate_transit_mask( #TODO: hace transitos tardios, mejorar detección de ingreso/egreso (actualmente depende de umbral fijo, no data-driven)
    light_curve: pd.DataFrame,
    normalized_flux_column: str = "normalized_flux",
    threshold_sigma: float = 2.0,
    window_buffer_points: int = 0,
    time_column: str | None = None,
    expected_mid_time: float | None = None,
    expected_duration: float | None = None,
    prior_half_window_factor: float = 1.5,
    min_prior_overlap_fraction: float = 0.2,
    edge_level_frac: float | None = None,
    edge_sigma_factor: float | None = None,
    edge_smoothing_points: int = 7,
    right_trim_quantile: float | None = None,
    min_transit_points: int = 5,
) -> np.ndarray:
    """
    Estima una máscara de tránsito de forma robusta combinando:
    1) detección fotométrica por umbral robusto (MAD),
    2) expansión temporal opcional,
    3) (opcional) refinado por bordes de baja profundidad,
    4) (opcional) prior temporal suave para usar efemérides sin colapsar casos con O-C grande.

    Por defecto, el comportamiento es compatible con la versión previa.

    Args:
        light_curve: DataFrame con curva normalizada.
        normalized_flux_column: columna con flujo normalizado.
        threshold_sigma: número de MADs por debajo de la mediana para el núcleo.
        window_buffer_points: número de puntos extra a enmascarar a cada lado
            del segmento detectado (0 = sin expansión).
        time_column: columna temporal (necesaria para usar prior temporal).
        expected_mid_time: instante esperado del centro de tránsito (misma unidad que time_column).
        expected_duration: duración esperada total del tránsito (misma unidad que time_column).
        prior_half_window_factor: factor multiplicativo del semiancho esperado para construir
            la ventana prior temporal: half_window = factor * expected_duration / 2.
        min_prior_overlap_fraction: fracción mínima de solape entre máscara detectada y prior
            para activar la fusión suave con efemérides.
        edge_level_frac: si se define (ej. 0.02-0.05), refina bordes con un nivel de baja
            profundidad relativo al OOT para capturar mejor ingreso/egreso.
        edge_sigma_factor: alternativa data-driven a edge_level_frac. Si se define
            (ej. 0.8-1.5), usa un nivel de borde basado en dispersión OOT:
            edge_level = OOT - edge_sigma_factor * sigma_OOT.
            Esto evita adelantar/atrasar bordes "a mano".
        edge_smoothing_points: ventana (impar) para suavizar flujo al detectar cruces
            de borde. Reduce sensibilidad a ruido puntual.
        right_trim_quantile: si se define (ej. 0.8-0.9), recorta la cola derecha de la máscara
            al cuantil indicado para evitar egresos artificialmente tardíos.
        min_transit_points: número mínimo de puntos para aceptar una máscara refinada.

    Returns:
        Array booleano con True en puntos de tránsito.
    """
    flux = light_curve[normalized_flux_column].to_numpy(dtype=float)
    finite_flux = np.isfinite(flux)
    if int(np.sum(finite_flux)) < max(min_transit_points, 3):
        return np.zeros(len(flux), dtype=bool)

    flux_valid = flux[finite_flux]
    median_flux = float(np.median(flux_valid))
    mad = float(np.median(np.abs(flux_valid - median_flux)))

    if not np.isfinite(mad) or mad == 0.0:
        mad = float(np.std(flux_valid))

    if not np.isfinite(mad) or mad <= 0.0:
        return np.zeros(len(flux), dtype=bool)

    # Núcleo del tránsito: puntos significativamente por debajo de la mediana
    transit_mask = flux < (median_flux - threshold_sigma * mad)
    transit_mask &= finite_flux

    # Expansión temporal: si hay puntos detectados, ampliar la ventana
    if window_buffer_points > 0 and transit_mask.any():
        indices = np.where(transit_mask)[0]
        i_first = max(0, indices[0] - window_buffer_points)
        i_last = min(len(flux) - 1, indices[-1] + window_buffer_points)
        transit_mask[i_first : i_last + 1] = True

    # Ajuste adicional para garantizar ingreso/egreso
    if transit_mask.any():
        idx = np.where(transit_mask)[0]
        i_start, i_end = idx[0], idx[-1]

        # Expandir bordes si no están bien definidos
        if i_start > 0:
            transit_mask[:i_start] = flux[:i_start] < (median_flux - threshold_sigma * mad)
        if i_end < len(flux) - 1:
            transit_mask[i_end + 1 :] = flux[i_end + 1 :] < (median_flux - threshold_sigma * mad)

        # Revalidar bordes tras expansión
        idx = np.where(transit_mask)[0]
        if len(idx) > 0:
            i_start, i_end = idx[0], idx[-1]
            transit_mask[i_start : i_end + 1] = True

    # Refinado opcional de bordes (captura mejor ingreso/egreso):
    # - por fracción de profundidad (edge_level_frac), o
    # - de forma data-driven con ruido OOT (edge_sigma_factor).
    if transit_mask.any() and (
        (edge_level_frac is not None and edge_level_frac > 0.0)
        or (edge_sigma_factor is not None and edge_sigma_factor > 0.0)
    ):
        idx = np.where(transit_mask)[0]
        i_start_base = int(idx[0])
        i_end_base = int(idx[-1])

        if int(np.sum(~transit_mask & finite_flux)) >= 10:
            oot_level = float(np.median(flux[(~transit_mask) & finite_flux]))
        else:
            oot_level = float(np.percentile(flux_valid, 75.0))

        depth = float(max(oot_level - np.min(flux_valid), 0.0))

        oot_flux_vals = flux[(~transit_mask) & finite_flux]
        sigma_oot = np.nan
        if oot_flux_vals.size >= 8:
            oot_med = float(np.median(oot_flux_vals))
            oot_mad = float(np.median(np.abs(oot_flux_vals - oot_med)))
            if np.isfinite(oot_mad) and oot_mad > 0.0:
                sigma_oot = 1.4826 * oot_mad
            if not np.isfinite(sigma_oot) or sigma_oot <= 0.0:
                sigma_oot = float(np.std(oot_flux_vals))

        edge_level = np.nan
        if edge_sigma_factor is not None and np.isfinite(sigma_oot) and sigma_oot > 0.0:
            edge_level = oot_level - float(edge_sigma_factor) * sigma_oot
        elif edge_level_frac is not None and np.isfinite(depth) and depth > 0.0:
            edge_level = oot_level - float(edge_level_frac) * depth

        if np.isfinite(edge_level):
            smooth_window = max(1, int(edge_smoothing_points))
            if smooth_window % 2 == 0:
                smooth_window += 1
            flux_edge = pd.Series(flux).rolling(
                window=smooth_window,
                center=True,
                min_periods=1,
            ).median().to_numpy(dtype=float)

            i_min = int(np.argmin(np.where(finite_flux, flux_edge, np.inf)))

            i_left_edge = None
            for i in range(i_min - 1, -1, -1):
                if (
                    finite_flux[i]
                    and finite_flux[i + 1]
                    and flux_edge[i] > edge_level
                    and flux_edge[i + 1] <= edge_level
                ):
                    i_left_edge = i
                    break

            i_right_edge = None
            for i in range(i_min, len(flux) - 1):
                if (
                    finite_flux[i]
                    and finite_flux[i + 1]
                    and flux_edge[i] <= edge_level
                    and flux_edge[i + 1] > edge_level
                ):
                    i_right_edge = i + 1
                    break

            i_start = min(i_start_base, i_left_edge if i_left_edge is not None else i_start_base)
            i_end = i_end_base if i_right_edge is None else min(i_end_base, i_right_edge)

            # Recorte opcional del ala derecha (útil cuando el egreso queda sistemáticamente tardío).
            if right_trim_quantile is not None and 0.0 < right_trim_quantile < 1.0:
                i_right_q = int(np.quantile(idx, right_trim_quantile))
                i_end = min(i_end, i_right_q)

            if i_end > i_start and (i_end - i_start + 1) >= min_transit_points:
                refined = np.zeros_like(transit_mask, dtype=bool)
                refined[i_start : i_end + 1] = True
                refined &= finite_flux
                transit_mask = refined

    # Prior temporal suave (NO duro):
    # - Si hay buena detección y solape razonable con prior, fusiona para estabilizar bordes.
    # - Si el solape es pobre (potencial O-C grande), conserva la máscara data-driven.
    # - Si no hay detección y existe prior válido, usa prior como fallback.
    prior_mask = None
    if (
        time_column is not None
        and expected_mid_time is not None
        and expected_duration is not None
        and expected_duration > 0.0
        and time_column in light_curve.columns
    ):
        t = light_curve[time_column].to_numpy(dtype=float)
        finite_t = np.isfinite(t)
        half_window = float(prior_half_window_factor) * float(expected_duration) / 2.0
        if np.isfinite(half_window) and half_window > 0.0:
            t_start = float(expected_mid_time) - half_window
            t_end = float(expected_mid_time) + half_window
            prior_mask = (t >= t_start) & (t <= t_end) & finite_t & finite_flux

    if prior_mask is not None and int(np.sum(prior_mask)) >= min_transit_points:
        if transit_mask.any():
            overlap = float(np.sum(transit_mask & prior_mask)) / float(np.sum(transit_mask))
            if overlap >= float(min_prior_overlap_fraction):
                fused = transit_mask | prior_mask
                idx_f = np.where(fused)[0]
                i0, i1 = int(idx_f[0]), int(idx_f[-1])
                fused_contiguous = np.zeros_like(transit_mask, dtype=bool)
                fused_contiguous[i0 : i1 + 1] = True
                fused_contiguous &= finite_flux
                if int(np.sum(fused_contiguous)) >= min_transit_points:
                    transit_mask = fused_contiguous
        else:
            transit_mask = prior_mask.copy()

    if int(np.sum(transit_mask)) < min_transit_points:
        return np.zeros(len(flux), dtype=bool)

    return transit_mask
