import os
import csv
import glob
import json
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


# -----------------------
# Core numerical utilities
# -----------------------

def ensure_monotone_cdf(F: np.ndarray) -> np.ndarray:
    F = np.asarray(F, dtype=float)
    F = np.clip(F, 0.0, 1.0)
    return np.maximum.accumulate(F)

def quantile_from_cdf(x_grid: np.ndarray, F: np.ndarray, p: float) -> float:
    return float(np.interp(p, F, x_grid))

def cdf_at_x(x_grid: np.ndarray, F: np.ndarray, x0: float) -> float:
    return float(np.interp(x0, x_grid, F))

def pdf_from_cdf(x_grid: np.ndarray, F: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    dx = float(x_grid[1] - x_grid[0])
    f = np.gradient(F, dx)
    f = np.maximum(f, 0.0)
    area = float(np.trapz(f, x_grid))
    if not np.isfinite(area) or area < eps:
        span = max(float(x_grid[-1] - x_grid[0]), eps)
        f = np.ones_like(f) / span
        area = float(np.trapz(f, x_grid))
    f = f / max(area, eps)
    return f

def moments_from_pdf(x_grid: np.ndarray, f: np.ndarray, eps: float = 1e-12):
    mu = float(np.trapz(x_grid * f, x_grid))
    var = float(np.trapz(((x_grid - mu) ** 2) * f, x_grid))
    var = max(var, eps)
    sd = float(np.sqrt(var))
    skew = float(np.trapz((((x_grid - mu) / sd) ** 3) * f, x_grid))
    kurt = float(np.trapz((((x_grid - mu) / sd) ** 4) * f, x_grid))
    ent = float(-np.trapz(f * np.log(f + eps), x_grid))
    return mu, sd, skew, kurt, ent

def expected_shortfall(x_grid: np.ndarray, f: np.ndarray, q_alpha: float, eps: float = 1e-12) -> float:
    mask = x_grid <= q_alpha
    if not np.any(mask):
        return float("nan")
    p = float(np.trapz(f[mask], x_grid[mask]))
    if p < eps:
        return float("nan")
    return float(np.trapz(x_grid[mask] * f[mask], x_grid[mask]) / p)


# -----------------------
# Quantile-only robust shape features (fast)
# -----------------------

def bowley_skew(q25, q50, q75, eps=1e-12):
    denom = (q75 - q25)
    if abs(denom) < eps:
        return float("nan")
    return float((q75 + q25 - 2.0*q50) / denom)

def moors_kurtosis(q125, q375, q625, q875, q25, q75, eps=1e-12):
    denom = (q75 - q25)
    if abs(denom) < eps:
        return float("nan")
    return float((q875 - q625 + q375 - q125) / denom)


# -----------------------
# Feature presets
# -----------------------

PRESET_QUANTILES: Dict[str, Tuple[float, ...]] = {
    "core": (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99),
    "exhaustive_fast": (
        0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50,
        0.75, 0.90, 0.95, 0.98, 0.99, 0.995, 0.999
    ),
}

PRESET_TAIL_THRESHOLDS: Dict[str, Tuple[float, ...]] = {
    # default assumes returns are in decimals (e.g. -0.01 = -1%)
    "exhaustive_fast": (-0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02),
    "core": (0.0,),
}


def compute_features_from_cdf(
    x_grid: np.ndarray,
    F_in: np.ndarray,
    features: str,
    quantiles: Tuple[float, ...],
    tail_thresholds: Tuple[float, ...],
    entropy_eps: float = 1e-12,
) -> Dict[str, float]:
    """
    Extract features from a single CDF on a grid.
    Complexity: O(grid_size).
    """
    x_grid = np.asarray(x_grid, dtype=float)
    F = ensure_monotone_cdf(F_in)

    out: Dict[str, float] = {}

    # quantiles
    q_vals = {p: quantile_from_cdf(x_grid, F, p) for p in quantiles}
    for p, v in q_vals.items():
        out[f"q{int(round(1000*p)):04d}"] = v  # q0001 for 0.001, q0050 for 0.05, etc.

    # sign probability
    out["p_le_0"] = cdf_at_x(x_grid, F, 0.0)
    out["p_ge_0"] = 1.0 - out["p_le_0"]

    if features == "core":
        return out

    # helper for on-demand quantiles
    def q(p: float) -> float:
        if p not in q_vals:
            q_vals[p] = quantile_from_cdf(x_grid, F, p)
            out[f"q{int(round(1000*p)):04d}"] = q_vals[p]
        return q_vals[p]

    # widths/spreads
    out["width_50"] = q(0.75) - q(0.25)
    out["width_80"] = q(0.90) - q(0.10)
    out["width_90"] = q(0.95) - q(0.05)
    out["width_96"] = q(0.98) - q(0.02)
    out["width_98"] = q(0.99) - q(0.01)

    # robust shape
    out["bowley_skew"] = bowley_skew(q(0.25), q(0.50), q(0.75))
    out["moors_kurtosis"] = moors_kurtosis(q(0.125), q(0.375), q(0.625), q(0.875), q(0.25), q(0.75))

    # tail probs at fixed thresholds
    for thr in tail_thresholds:
        out[f"p_le_thr_{thr:g}"] = cdf_at_x(x_grid, F, float(thr))
        out[f"p_ge_thr_{thr:g}"] = 1.0 - out[f"p_le_thr_{thr:g}"]

    # tail asymmetry (quantile-based)
    med = q(0.50)
    out["tail_asym_q01_q99"] = (med - q(0.01)) / max((q(0.99) - med), 1e-12)
    out["tail_asym_q05_q95"] = (med - q(0.05)) / max((q(0.95) - med), 1e-12)

    # pdf-derived stats (grid-dependent, still O(G))
    f = pdf_from_cdf(x_grid, F, eps=entropy_eps)
    mu, sd, skew, kurt, ent = moments_from_pdf(x_grid, f, eps=entropy_eps)
    out["mean"] = mu
    out["std"] = sd
    out["skew"] = skew
    out["kurtosis"] = kurt
    out["entropy"] = ent

    # ES at 10%, 5%, 1%
    out["es_10"] = expected_shortfall(x_grid, f, q(0.10), eps=entropy_eps)
    out["es_05"] = expected_shortfall(x_grid, f, q(0.05), eps=entropy_eps)
    out["es_01"] = expected_shortfall(x_grid, f, q(0.01), eps=entropy_eps)

    return out


# -----------------------
# Output file loading (NW multi-k outputs)
# -----------------------

def _safe_item(z, key, default=None):
    if key not in z:
        return default
    arr = np.asarray(z[key])
    if arr.shape == ():
        try:
            return arr.item()
        except Exception:
            return str(arr)
    return arr


@dataclass
class OutputFile:
    path: str
    ticker: str
    snapshot_id: str
    forecast_origin_date: str
    forecast_time: str
    x_grid: np.ndarray
    cond_cdf: np.ndarray         # (K,G)
    k_values: np.ndarray         # (K,)
    bandwidth_h: Optional[np.ndarray]  # (K,) or None


def load_nw_multik_output(path: str) -> OutputFile:
    z = np.load(path, allow_pickle=True)

    x_grid = np.asarray(z["x_grid"], dtype=float)
    C = np.asarray(z["cond_cdf"], dtype=float)
    k_values = _safe_item(z, "k_values", None)
    h_used = _safe_item(z, "bandwidth_h", None)

    ticker = str(_safe_item(z, "ticker", "UNKNOWN"))
    snapshot_id = str(_safe_item(z, "snapshot_id", os.path.splitext(os.path.basename(path))[0]))
    origin = str(_safe_item(z, "forecast_origin_date", ""))
    ftime = str(_safe_item(z, "forecast_time", ""))

    if C.ndim == 1:
        C = C[None, :]
        if k_values is None:
            k_values = np.array([np.nan], dtype=float)
    else:
        if k_values is None:
            k_values = np.arange(C.shape[0], dtype=int)

    return OutputFile(
        path=path,
        ticker=ticker,
        snapshot_id=snapshot_id,
        forecast_origin_date=origin,
        forecast_time=ftime,
        x_grid=x_grid,
        cond_cdf=C,
        k_values=np.asarray(k_values),
        bandwidth_h=np.asarray(h_used) if h_used is not None else None,
    )


# -----------------------
# Main extraction routine
# -----------------------

def extract_features_from_outputs(
    pattern: str,
    out_csv: str,
    recursive: bool = False,
    features: str = "exhaustive_fast",
    quantiles_override: Optional[Tuple[float, ...]] = None,
    tail_thresholds_override: Optional[Tuple[float, ...]] = None,
    bad_log: Optional[str] = None,
) -> str:
    """
    Scan NW multi-k output .npz files and write one CSV with features per (file,k).
    Returns the output path.
    """
    paths = sorted(glob.glob(pattern, recursive=recursive))
    if not paths:
        raise FileNotFoundError(f"No files match: {pattern}")

    if features not in PRESET_QUANTILES:
        raise ValueError(f"Unknown features preset: {features}. Options: {list(PRESET_QUANTILES.keys())}")

    quantiles = quantiles_override if quantiles_override is not None else PRESET_QUANTILES[features]
    if any((q <= 0.0 or q >= 1.0) for q in quantiles):
        raise ValueError("All quantiles must be in (0,1).")

    tail_thresholds = (
        tail_thresholds_override if tail_thresholds_override is not None else PRESET_TAIL_THRESHOLDS.get(features, (0.0,))
    )

    out_dir = os.path.dirname(out_csv) or "."
    os.makedirs(out_dir, exist_ok=True)
    bad_log = bad_log or os.path.join(out_dir, "cdf_features_bad_files.log")

    rows: List[Dict] = []
    n_bad = 0

    for p in paths:
        try:
            obj = load_nw_multik_output(p)

            for i in range(obj.cond_cdf.shape[0]):
                F = obj.cond_cdf[i]
                if not np.all(np.isfinite(F)):
                    continue

                feats = compute_features_from_cdf(
                    x_grid=obj.x_grid,
                    F_in=F,
                    features=features,
                    quantiles=quantiles,
                    tail_thresholds=tail_thresholds,
                )

                k_val = obj.k_values[i].item() if obj.k_values.ndim == 1 else obj.k_values[i]
                h_val = ""
                if obj.bandwidth_h is not None:
                    try:
                        hv = float(np.asarray(obj.bandwidth_h)[i])
                        if np.isfinite(hv):
                            h_val = hv
                    except Exception:
                        pass

                row = {
                    "file": obj.path,
                    "ticker": obj.ticker,
                    "snapshot_id": obj.snapshot_id,
                    "forecast_origin_date": obj.forecast_origin_date,
                    "forecast_time": obj.forecast_time,
                    "k": int(k_val) if np.isfinite(k_val) else "",
                    "bandwidth_h": h_val,
                    "grid_min": float(obj.x_grid[0]),
                    "grid_max": float(obj.x_grid[-1]),
                    "grid_size": int(len(obj.x_grid)),
                }
                row.update(feats)
                rows.append(row)

        except Exception as e:
            n_bad += 1
            with open(bad_log, "a") as f:
                f.write(f"{p} :: {type(e).__name__}: {e}\n")

    if not rows:
        raise RuntimeError("No features extracted. Check input glob or bad log.")

    base_cols = [
        "file", "ticker", "snapshot_id", "forecast_origin_date", "forecast_time",
        "k", "bandwidth_h", "grid_min", "grid_max", "grid_size",
    ]
    feat_cols = sorted([c for c in rows[0].keys() if c not in base_cols])
    fieldnames = base_cols + feat_cols

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    return out_csv
