import os
import csv
import glob
import json
import numpy as np


def _ensure_monotone_cdf(F: np.ndarray) -> np.ndarray:
    # clip + force nondecreasing
    F = np.clip(F, 0.0, 1.0)
    return np.maximum.accumulate(F)


def _quantile_from_cdf(x_grid: np.ndarray, F: np.ndarray, p: float) -> float:
    # assumes F is monotone increasing
    return float(np.interp(p, F, x_grid))


def _cdf_at_x(x_grid: np.ndarray, F: np.ndarray, x0: float) -> float:
    return float(np.interp(x0, x_grid, F))


def _pdf_from_cdf(x_grid: np.ndarray, F: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    dx = float(x_grid[1] - x_grid[0])
    f = np.gradient(F, dx)
    f = np.maximum(f, 0.0)
    area = float(np.trapz(f, x_grid))
    if not np.isfinite(area) or area < eps:
        # fallback: nearly-degenerate -> uniform-ish tiny
        f = np.ones_like(f) / max(float(x_grid[-1] - x_grid[0]), eps)
        area = float(np.trapz(f, x_grid))
    f = f / max(area, eps)
    return f


def _moments_from_pdf(x_grid: np.ndarray, f: np.ndarray, eps: float = 1e-12):
    mu = float(np.trapz(x_grid * f, x_grid))
    var = float(np.trapz(((x_grid - mu) ** 2) * f, x_grid))
    var = max(var, eps)
    sd = float(np.sqrt(var))
    skew = float(np.trapz((((x_grid - mu) / sd) ** 3) * f, x_grid))
    kurt = float(np.trapz((((x_grid - mu) / sd) ** 4) * f, x_grid))
    ent = float(-np.trapz(f * np.log(f + eps), x_grid))
    return mu, sd, skew, kurt, ent


def _expected_shortfall(x_grid: np.ndarray, f: np.ndarray, q: float, eps: float = 1e-12) -> float:
    mask = x_grid <= q
    if not np.any(mask):
        return float("nan")
    p = float(np.trapz(f[mask], x_grid[mask]))
    if p < eps:
        return float("nan")
    es = float(np.trapz(x_grid[mask] * f[mask], x_grid[mask]) / p)
    return es


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


def compute_stats_for_file(path: str):
    z = np.load(path, allow_pickle=True)

    x_grid = np.asarray(z["x_grid"], dtype=float)
    C = np.asarray(z["cond_cdf"], dtype=float)  # (K,G) or (G,)
    k_values = _safe_item(z, "k_values", None)
    h_used = _safe_item(z, "bandwidth_h", None)

    ticker = _safe_item(z, "ticker", "UNKNOWN")
    snapshot_id = _safe_item(z, "snapshot_id", os.path.splitext(os.path.basename(path))[0])
    origin = _safe_item(z, "forecast_origin_date", "")
    ftime = _safe_item(z, "forecast_time", "")

    meta = {}
    meta_json = _safe_item(z, "meta_json", None)
    if meta_json is not None:
        try:
            meta = json.loads(meta_json)
        except Exception:
            meta = {}

    if C.ndim == 1:
        C = C[None, :]
        if k_values is None:
            k_values = np.array([np.nan], dtype=float)
    else:
        if k_values is None:
            k_values = np.arange(C.shape[0], dtype=int)

    k_values = np.asarray(k_values)

    rows = []
    dx = float(x_grid[1] - x_grid[0])

    for i in range(C.shape[0]):
        F = C[i].astype(float)

        if not np.all(np.isfinite(F)) or np.all(np.isnan(F)):
            continue

        F = _ensure_monotone_cdf(F)

        # quantiles
        qs = {}
        for p in (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99):
            qs[p] = _quantile_from_cdf(x_grid, F, p)

        width_50 = qs[0.75] - qs[0.25]
        width_80 = qs[0.90] - qs[0.10]
        width_90 = qs[0.95] - qs[0.05]
        width_98 = qs[0.99] - qs[0.01]

        # tail probs at 0 (often meaningful for returns)
        p_le_0 = _cdf_at_x(x_grid, F, 0.0)
        p_ge_0 = 1.0 - p_le_0

        # derive pdf and moments
        f = _pdf_from_cdf(x_grid, F)
        mu, sd, skew, kurt, ent = _moments_from_pdf(x_grid, f)

        # expected shortfall at alpha=5% and 1%
        es_05 = _expected_shortfall(x_grid, f, qs[0.05])
        es_01 = _expected_shortfall(x_grid, f, qs[0.01])

        k_val = k_values[i].item() if k_values.ndim == 1 else k_values[i]
        h_val = None
        if h_used is not None:
            try:
                h_val = float(np.asarray(h_used)[i])
            except Exception:
                h_val = None

        rows.append({
            "file": path,
            "ticker": ticker,
            "snapshot_id": snapshot_id,
            "forecast_origin_date": str(origin),
            "forecast_time": str(ftime),
            "k": int(k_val) if np.isfinite(k_val) else "",
            "bandwidth_h": f"{h_val:.6g}" if (h_val is not None and np.isfinite(h_val)) else "",

            "dx": dx,
            "grid_min": float(x_grid[0]),
            "grid_max": float(x_grid[-1]),

            "q01": qs[0.01],
            "q05": qs[0.05],
            "q10": qs[0.10],
            "q25": qs[0.25],
            "q50": qs[0.50],
            "q75": qs[0.75],
            "q90": qs[0.90],
            "q95": qs[0.95],
            "q99": qs[0.99],

            "width_50": width_50,
            "width_80": width_80,
            "width_90": width_90,
            "width_98": width_98,

            "p_le_0": p_le_0,
            "p_ge_0": p_ge_0,

            "mean": mu,
            "std": sd,
            "skew": skew,
            "kurtosis": kurt,
            "entropy": ent,

            "es_05": es_05,
            "es_01": es_01,
        })

    return rows


def main():
    import argparse

    ap = argparse.ArgumentParser(
        description="Extract statistics from NW multi-k conditional CDF outputs and write a CSV summary."
    )
    ap.add_argument(
        "--glob",
        required=True,
        help='Pattern for outputs, e.g. "data/outputs/snapshot_nw_multik/**/*.npz"',
    )
    ap.add_argument(
        "--out_csv",
        required=True,
        help='Where to write CSV, e.g. "data/outputs/snapshot_nw_multik/cdf_stats.csv"',
    )
    ap.add_argument(
        "--recursive",
        action="store_true",
        help="Use recursive glob (needed when using ** in pattern).",
    )
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob, recursive=args.recursive))
    if not paths:
        raise FileNotFoundError(f"No files match: {args.glob}")

    out_dir = os.path.dirname(args.out_csv) or "."
    os.makedirs(out_dir, exist_ok=True)

    bad_log = os.path.join(out_dir, "cdf_stats_bad_files.log")

    all_rows = []
    n_bad = 0

    for p in paths:
        try:
            rows = compute_stats_for_file(p)
            all_rows.extend(rows)
        except Exception as e:
            n_bad += 1
            with open(bad_log, "a") as f:
                f.write(f"{p} :: {type(e).__name__}: {e}\n")

    if not all_rows:
        raise RuntimeError("No rows produced (all files failed or empty cond_cdf).")

    # stable column order
    fieldnames = list(all_rows[0].keys())

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    print(f"Wrote: {args.out_csv}")
    print(f"Rows: {len(all_rows)} | bad files: {n_bad}")
    if n_bad:
        print(f"Bad log: {bad_log}")
