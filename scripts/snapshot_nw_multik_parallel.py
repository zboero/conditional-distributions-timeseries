#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 08:42:02 2025

@author: ezequiel

    # Run it as follow...
    python code/snapshot_cdf_nw_multik_parallel.py \
      "data/minute_data/return_snapshots/*/*/*.npz" \
      data/minute_data/out_snapshot_nw_multik \
      --k_list "1,2,5,10,20" \
      --grid_size 600 \
      --cv_points 150 \
      --max_workers 6
      
"""

import os
import json
import glob
import numpy as np
import zipfile
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed

RET_KEY = "signal"
DATE_KEY = "dates"

# ------------------ IO helpers ------------------

def load_snapshot(path: str):
    """
    Robust loader. Raises RuntimeError on corrupted/unreadable files.
    """
    try:
        z = np.load(path, allow_pickle=True)
    except (OSError, zipfile.BadZipFile, zlib.error, ValueError) as e:
        raise RuntimeError(f"Cannot load npz (possibly corrupted): {path} :: {type(e).__name__}: {e}")

    if RET_KEY not in z or DATE_KEY not in z:
        raise RuntimeError(f"{path}: needs keys '{RET_KEY}' and '{DATE_KEY}'. keys={list(z.keys())}")

    try:
        x = np.asarray(z[RET_KEY], dtype=float)
        dates = np.asarray(z[DATE_KEY])
    except Exception as e:
        raise RuntimeError(f"{path}: failed reading arrays :: {type(e).__name__}: {e}")

    meta_in = {}
    for k in z.files:
        if k in (RET_KEY, DATE_KEY):
            continue
        try:
            arr = np.asarray(z[k])
            meta_in[k] = arr.item() if arr.shape == () else arr
        except Exception:
            meta_in[k] = "UNREADABLE_META"

    return x, dates, meta_in


def infer_ticker_from_path(path: str) -> str:
    parts = os.path.normpath(path).split(os.sep)
    return parts[-3] if len(parts) >= 3 else os.path.splitext(os.path.basename(path))[0]


def infer_forecast_time(meta_in: dict, last_date):
    try:
        resample_min = meta_in.get("resample_min", None)
        if resample_min is None:
            return None
        resample_min = int(np.asarray(resample_min).item())
        return last_date + np.timedelta64(resample_min, "m")
    except Exception:
        return None


def safe_jsonable(v):
    try:
        json.dumps(v, default=str)
        return v
    except Exception:
        return str(v)


# ------------------ estimator core ------------------

def make_lagged_pairs(x: np.ndarray, k: int):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if k < 1 or k >= n:
        raise ValueError("k must satisfy 1 <= k < len(x).")
    Y = np.zeros((n - k, k), dtype=float)
    y_next = x[k:].copy()
    for i in range(n - k):
        Y[i, :] = x[i + k - 1 : i - 1 : -1] if i > 0 else x[k - 1 :: -1]
    return Y, y_next


def robust_standardize_matrix(Y: np.ndarray, eps: float = 1e-12):
    med = np.median(Y, axis=0)
    mad = np.median(np.abs(Y - med), axis=0)
    scale = 1.4826 * mad
    scale = np.maximum(scale, eps)
    Z = (Y - med) / scale
    return Z, med, scale


def gaussian_kernel(u2: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * u2)


def nw_weights(Yz: np.ndarray, qz: np.ndarray, h: float, min_wsum: float = 1e-12):
    d2 = np.sum((Yz - qz[None, :]) ** 2, axis=1)
    h2 = max(h, 1e-12) ** 2
    w = gaussian_kernel(d2 / h2)
    wsum = float(np.sum(w))
    if wsum < min_wsum:
        w = np.ones_like(w) / len(w)
    else:
        w = w / wsum
    return w


def weighted_cdf_at_grid(x_next: np.ndarray, w: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    order = np.argsort(x_next)
    xs = x_next[order]
    ws = w[order]
    cws = np.cumsum(ws)
    idx = np.searchsorted(xs, x_grid, side="right") - 1
    F = np.where(idx >= 0, cws[np.clip(idx, 0, len(cws)-1)], 0.0)
    return F


def crps_from_cdf(x_grid: np.ndarray, F: np.ndarray, y: float) -> float:
    dx = float(x_grid[1] - x_grid[0])
    H = (x_grid >= y).astype(float)
    return float(np.sum((F - H) ** 2) * dx)


def choose_bandwidth_crps_snapshot(x: np.ndarray, k: int, x_grid: np.ndarray, h_grid: np.ndarray,
                                  cv_points: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(x)
    if n <= k + 60:
        return float(h_grid[len(h_grid)//2])

    Y, y_next = make_lagged_pairs(x, k)
    Yz, med, scale = robust_standardize_matrix(Y)

    candidates = np.arange(k + 10, n)
    if len(candidates) > cv_points:
        eval_t = np.sort(rng.choice(candidates, size=cv_points, replace=False))
    else:
        eval_t = candidates

    best_h, best = None, np.inf
    for h in h_grid:
        scores = []
        for t in eval_t:
            i_max = (t - 1) - k
            if i_max < 30:
                continue
            Yz_train = Yz[: i_max + 1]
            y_train = y_next[: i_max + 1]

            q = x[t-1:t-k-1:-1]
            qz = (q - med) / scale
            w = nw_weights(Yz_train, qz, h)
            F = weighted_cdf_at_grid(y_train, w, x_grid)
            scores.append(crps_from_cdf(x_grid, F, x[t]))
        if scores:
            m = float(np.mean(scores))
            if m < best:
                best = m
                best_h = float(h)
    return best_h if best_h is not None else float(h_grid[len(h_grid)//2])


def compute_one_snapshot_multik_nw(
    x: np.ndarray,
    meta_in: dict,
    k_values,
    grid_size: int,
    grid_quantiles=(0.001, 0.999),
    cv_points: int = 200,
):
    lo, hi = np.quantile(x, grid_quantiles[0]), np.quantile(x, grid_quantiles[1])
    x_grid = np.linspace(lo, hi, grid_size)

    h_grid = np.array([0.5, 0.7, 0.9, 1.1, 1.4, 1.8], dtype=float)

    cdf_stack = []
    h_used = []

    for k in k_values:
        k = int(k)
        if len(x) <= k + 20:
            cdf_stack.append(np.full_like(x_grid, np.nan))
            h_used.append(np.nan)
            continue

        h = choose_bandwidth_crps_snapshot(x, k, x_grid, h_grid, cv_points=cv_points, seed=0)

        Y, y_next = make_lagged_pairs(x, k)
        Yz, med, scale = robust_standardize_matrix(Y)

        # forecast X_{T+1} given last k values including X_T
        q = x[-1:-k-1:-1]
        qz = (q - med) / scale

        w = nw_weights(Yz, qz, h)
        F = weighted_cdf_at_grid(y_next, w, x_grid)

        cdf_stack.append(F)
        h_used.append(h)

    return x_grid, np.vstack(cdf_stack), np.asarray(h_used, dtype=float)


# ------------------ worker ------------------

def process_one_file(args):
    """
    Worker-safe function. Returns a dict with status + info.
    """
    p, out_dir, k_values, grid_size, cv_points = args
    try:
        x, dates, meta_in = load_snapshot(p)
    except RuntimeError as e:
        return {"status": "bad", "path": p, "error": str(e)}

    if len(x) < (max(k_values) + 30):
        return {"status": "bad", "path": p, "error": f"Too short for k_max={max(k_values)} (n={len(x)})"}

    ticker = infer_ticker_from_path(p)
    snap_id = os.path.splitext(os.path.basename(p))[0]

    x_grid, cdf_stack, h_used = compute_one_snapshot_multik_nw(
        x=x,
        meta_in=meta_in,
        k_values=k_values,
        grid_size=grid_size,
        cv_points=cv_points,
    )

    last_date = dates[-1]
    forecast_time = infer_forecast_time(meta_in, last_date)

    meta = {
        "method": "snapshot_nw_conditional_cdf_forecast_beyond_multik",
        "ticker": ticker,
        "snapshot_id": snap_id,
        "k_values": list(map(int, k_values)),
        "grid_size": grid_size,
        "cv_points": cv_points,
        "forecast_horizon_steps": 1,
        "forecast_is_beyond_snapshot": True,
        "forecast_origin_date": str(last_date),
        "forecast_time": str(forecast_time) if forecast_time is not None else None,
        "input_file": p,
    }
    for kk, vv in meta_in.items():
        meta[f"src_{kk}"] = safe_jsonable(vv)

    out_path = os.path.join(out_dir, f"{ticker}__{snap_id}__nw_cdf_next_multik.npz")
    np.savez_compressed(
        out_path,
        ticker=np.array(ticker),
        snapshot_id=np.array(snap_id),
        forecast_origin_date=np.array(last_date),
        forecast_time=np.array(forecast_time) if forecast_time is not None else np.array(""),
        k_values=np.array(k_values, dtype=int),
        x_grid=x_grid,
        cond_cdf=cdf_stack,
        bandwidth_h=h_used,
        meta_json=np.array(json.dumps(meta, default=str)),
    )

    return {"status": "ok", "path": p, "out_path": out_path}


# ------------------ main ------------------

def main(input_glob: str, out_dir: str, k_list: str,
         grid_size: int = 600, cv_points: int = 200,
         max_workers: int = None, chunksize: int = 1):
    os.makedirs(out_dir, exist_ok=True)
    paths = sorted(glob.glob(input_glob))
    if not paths:
        raise FileNotFoundError(f"No files match: {input_glob}")

    k_values = [int(s.strip()) for s in k_list.split(",") if s.strip()]
    if not k_values:
        raise ValueError("k_list must be like '1,2,5,10'")

    bad_log = os.path.join(out_dir, "bad_snapshots.log")

    tasks = [(p, out_dir, k_values, grid_size, cv_points) for p in paths]

    n_ok = 0
    n_bad = 0

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(process_one_file, t) for t in tasks]
        for fut in as_completed(futures):
            res = fut.result()
            if res["status"] == "ok":
                n_ok += 1
                print(f"[OK] {res['out_path']}")
            else:
                n_bad += 1
                msg = f"{res['path']} :: {res['error']}"
                with open(bad_log, "a") as f:
                    f.write(msg + "\n")
                print(f"[SKIP] {msg}")

    print(f"Done. ok={n_ok} bad={n_bad} total={len(paths)}")
    print(f"Bad log: {bad_log}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Parallel per-snapshot NW conditional CDF (multi-k) + bad file logging.")
    ap.add_argument("input_glob")
    ap.add_argument("out_dir")
    ap.add_argument("--k_list", type=str, default="1,2,5,10,20")
    ap.add_argument("--grid_size", type=int, default=600)
    ap.add_argument("--cv_points", type=int, default=200)
    ap.add_argument("--max_workers", type=int, default=None, help="Defaults to #cores. Try 4, 6, 8...")
    args = ap.parse_args()

    main(
        input_glob=args.input_glob,
        out_dir=args.out_dir,
        k_list=args.k_list,
        grid_size=args.grid_size,
        cv_points=args.cv_points,
        max_workers=args.max_workers,
    )

