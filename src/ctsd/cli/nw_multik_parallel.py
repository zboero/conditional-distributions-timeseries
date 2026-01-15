import os
import json
import glob
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from ctsd.io.npz_snapshot import load_snapshot, infer_ticker_from_path, infer_forecast_time
from ctsd.estimators.nw_multik import compute_snapshot_multik_nw, safe_jsonable

def _process_one_file(args):
    p, out_dir, k_values, grid_size, cv_points = args
    try:
        x, dates, meta_in = load_snapshot(p)
    except RuntimeError as e:
        return {"status": "bad", "path": p, "error": str(e)}

    if len(x) < (max(k_values) + 30):
        return {"status": "bad", "path": p, "error": f"Too short for k_max={max(k_values)} (n={len(x)})"}

    ticker = infer_ticker_from_path(p)
    snap_id = os.path.splitext(os.path.basename(p))[0]

    x_grid, cdf_stack, h_used = compute_snapshot_multik_nw(
        x=x,
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
    return {"status": "ok", "out_path": out_path}

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Parallel per-snapshot NW conditional CDF (multi-k) + bad file logging.")
    ap.add_argument("--glob", dest="pattern", required=True, help='e.g. "data/return_snapshots/*/*/*.npz"')
    ap.add_argument("--out_dir", required=True, help='e.g. "data/outputs/snapshot_nw_multik"')
    ap.add_argument("--k_list", type=str, default="1,2,5,10,20")
    ap.add_argument("--grid_size", type=int, default=600)
    ap.add_argument("--cv_points", type=int, default=150)
    ap.add_argument("--max_workers", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    paths = sorted(glob.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"No files match: {args.pattern}")

    k_values = [int(s.strip()) for s in args.k_list.split(",") if s.strip()]
    if not k_values:
        raise ValueError("k_list must be like '1,2,5,10'")

    bad_log = os.path.join(args.out_dir, "bad_snapshots.log")

    tasks = [(p, args.out_dir, k_values, args.grid_size, args.cv_points) for p in paths]

    n_ok = 0
    n_bad = 0

    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        futures = [ex.submit(_process_one_file, t) for t in tasks]
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
