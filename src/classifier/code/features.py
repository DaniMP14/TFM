from __future__ import annotations

from scipy.stats import skew, kurtosis
import numpy as np


def _robust_autocorr_lag1(values: np.ndarray) -> float:
    if values.size < 3:
        return 0.0
    x = values[:-1]
    y = values[1:]
    x_std = float(np.std(x, ddof=1))
    y_std = float(np.std(y, ddof=1))
    if x_std <= 0.0 or y_std <= 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _estimate_ingress_egress_balance(flux: np.ndarray, depth_threshold: float) -> float:
    in_transit = flux < depth_threshold
    if np.sum(in_transit) < 4:
        return 0.0

    indices = np.where(in_transit)[0]
    start, end = int(indices[0]), int(indices[-1])
    midpoint = (start + end) // 2

    left = flux[start : midpoint + 1]
    right = flux[midpoint + 1 : end + 1]
    if left.size == 0 or right.size == 0:
        return 0.0

    left_depth = float(np.median(1.0 - left))
    right_depth = float(np.median(1.0 - right))
    denom = max(abs(left_depth), abs(right_depth), 1e-8)
    return float((left_depth - right_depth) / denom)


def extract_curve_features(
    time: np.ndarray,
    flux: np.ndarray,
) -> dict[str, float]:
    if time.size != flux.size:
        raise ValueError("time and flux arrays must have the same length")
    if flux.size < 20:
        raise ValueError("At least 20 points are required to extract features")

    finite_mask = np.isfinite(time) & np.isfinite(flux)
    time = np.asarray(time[finite_mask], dtype=float)
    flux = np.asarray(flux[finite_mask], dtype=float)
    if flux.size < 20:
        raise ValueError("Not enough finite points after filtering")

    median_flux = float(np.median(flux))
    if median_flux <= 0.0:
        raise ValueError("Median flux must be positive")
    norm_flux = flux / median_flux

    n = len(norm_flux)

    p05 = float(np.percentile(norm_flux, 5.0))
    p10 = float(np.percentile(norm_flux, 10.0))
    p25 = float(np.percentile(norm_flux, 25.0))
    p50 = float(np.percentile(norm_flux, 50.0))
    p75 = float(np.percentile(norm_flux, 75.0))
    p95 = float(np.percentile(norm_flux, 95.0))

    # lower_tail_mean: media del 5-10% más bajo en vez de percentil puntual.
    lower_tail_mean = float(np.mean(norm_flux[norm_flux <= p10]))


    scatter = float(np.std(norm_flux, ddof=1))
    mad = float(np.median(np.abs(norm_flux - p50)))

    depth_robust = float(max(1 - lower_tail_mean, 0.0))
    depth_robust_to_mad = depth_robust / mad
    depth_snr = float(depth_robust / max(scatter, 1e-8))

    flux_min_idx = int(np.argmin(norm_flux))
    min_time = float(time[flux_min_idx])
    time_center = float(np.median(time))

    threshold = 1.0 - 0.35 * depth_robust if depth_robust > 0.0 else 1.0
    in_transit_fraction = float(np.mean(norm_flux < threshold))

    p2p_rms = float(np.std(np.diff(norm_flux), ddof=1))

    puntos_bin = 5
    n_completos = (n // puntos_bin) * puntos_bin
    flux_recortado = norm_flux[:n_completos]
    binned_flux = np.mean(flux_recortado.reshape(-1, puntos_bin), axis=1)
    std_binned = np.std(binned_flux)
    # Si fuera ruido blanco puro, este ratio debería estar cerca de 1/sqrt(5) = 0.44
    binning_std_ratio = float(std_binned / (scatter if scatter > 0 else 1e-6))

    idx_15 = int(n * 0.15)
    if idx_15 > 2:
        wings = np.concatenate([norm_flux[:idx_15], norm_flux[n - idx_15:]])
        wings_flatness = float(np.std(wings))
    else:
        wings_flatness = scatter
    
    zona_central = norm_flux[int(n*0.35) : int(n*0.65)]
    std_local = np.std(zona_central) if len(zona_central) > 0 else scatter
    local_global_std_ratio = float(std_local / (scatter if scatter > 0 else 1e-6))

    features = {
        # "n_points": float(norm_flux.size),
        # "std_flux": scatter,
        # "mad_flux": mad,
        # "p05_flux": p05,
        # "p10_flux": p10,
        # "p25_flux": p25,
        # "p75_flux": p75,
        # "p95_flux": p95,
        # "iqr_flux": float(p75 - p25),
        "transit_depth_est": depth_robust,
        "transit_depth_snr": depth_snr,
        "in_transit_fraction_est": in_transit_fraction,
        "autocorr_lag1": _robust_autocorr_lag1(norm_flux),
        "slope_linear": float(np.polyfit(time, norm_flux, deg=1)[0]),
        "min_time_offset": float(min_time - time_center),
        "skewness_flux": float(skew(norm_flux)),
        "kurtosis_flux": float(kurtosis(norm_flux)),
        "depth_to_mad_ratio": float(depth_robust_to_mad),
        "ingress_egress_balance": _estimate_ingress_egress_balance(norm_flux, threshold),
        "p2p_to_std_ratio": float(p2p_rms / max(scatter, 1e-8)),
        "binning_std_ratio": binning_std_ratio,
        # "wings_flatness": wings_flatness,
        "local_global_std_ratio": local_global_std_ratio,
    }
    return features
