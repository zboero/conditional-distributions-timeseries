#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 08:23:31 2025

@author: ezequiel

    # NW output
    python code/plot_snapshot_multik_cdf.py \
      data/minute_data/out_snapshot_nw_multik/WYNN__20240516_20250519__nw_cdf_next_multik.npz \
      --snapshot_npz return_snapshots/WYNN/3month/20240516_20250519.npz
      
"""

import json
import numpy as np
import matplotlib.pyplot as plt

def load_multik_output(path: str):
    z = np.load(path, allow_pickle=True)
    out = {
        "ticker": z["ticker"].item() if "ticker" in z else "UNKNOWN",
        "snapshot_id": z["snapshot_id"].item() if "snapshot_id" in z else "",
        "forecast_origin_date": z["forecast_origin_date"],
        "forecast_time": z["forecast_time"] if "forecast_time" in z else None,
        "k_values": np.asarray(z["k_values"], dtype=int),
        "x_grid": np.asarray(z["x_grid"], dtype=float),
        "cond_cdf": np.asarray(z["cond_cdf"], dtype=float),
        "bandwidth_h": np.asarray(z["bandwidth_h"], dtype=float) if "bandwidth_h" in z else None,
        "meta": {},
    }
    if "meta_json" in z:
        try:
            out["meta"] = json.loads(z["meta_json"].item())
        except Exception:
            out["meta"] = {}
    return out

def load_snapshot_tail(snapshot_npz: str, k_max: int):
    z = np.load(snapshot_npz, allow_pickle=True)
    if "signal" not in z or "dates" not in z:
        return None, None
    x = np.asarray(z["signal"], dtype=float)
    dates = np.asarray(z["dates"])
    if len(x) < k_max:
        k_max = len(x)
    return dates[-k_max:], x[-k_max:]

def plot_multik_cdfs(output_npz: str, snapshot_npz: str = None, show_quantiles=True):
    out = load_multik_output(output_npz)

    ticker = out["ticker"]
    snap_id = out["snapshot_id"]
    origin = out["forecast_origin_date"]
    ftime = out["forecast_time"]
    k_values = out["k_values"]
    x_grid = out["x_grid"]
    C = out["cond_cdf"]   # (K,G)
    h_used = out["bandwidth_h"]

    # basic checks
    if C.ndim != 2 or C.shape[0] != len(k_values):
        raise ValueError("cond_cdf must have shape (len(k_values), grid_size)")

    k_max = int(np.max(k_values))

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [1.3, 1.0]})
    ax = axes[0]
    ax2 = axes[1]

    # ---- panel 1: overlay CDFs ----
    for i, k in enumerate(k_values):
        F = C[i]
        if np.any(np.isnan(F)):
            continue
        label = f"k={k}"
        if h_used is not None and np.isfinite(h_used[i]):
            label += f" (h={h_used[i]:.2f})"
        ax.plot(x_grid, F, lw=2, label=label)

        if show_quantiles:
            # mark Q05/Q50/Q95 for each k (small ticks)
            q05 = np.interp(0.05, F, x_grid)
            q50 = np.interp(0.50, F, x_grid)
            q95 = np.interp(0.95, F, x_grid)
            ax.plot([q50], [0.50], marker="o")
            ax.plot([q05], [0.05], marker=".")
            ax.plot([q95], [0.95], marker=".")

    ax.set_title(f"{ticker} | {snap_id}\nConditional CDF for next step beyond snapshot (multi-k)")
    ax.set_xlabel("x (next value)")
    ax.set_ylabel("F(x | last k)")
    ax.grid(True)
    ax.legend(fontsize=9)

    # footer info
    footer = f"origin: {origin}"
    if ftime is not None and str(ftime) != "":
        footer += f"  → forecast time: {ftime}"
    fig.text(0.5, 0.01, footer, ha="center", fontsize=9)

    # ---- panel 2: show last k_max observed points ----
    if snapshot_npz is not None:
        d_tail, x_tail = load_snapshot_tail(snapshot_npz, k_max)
        if x_tail is not None:
            rel = np.arange(-len(x_tail) + 1, 1)  # ends at 0
            ax2.plot(rel, x_tail, marker="o")
            ax2.axvline(0, ls="--")
            ax2.set_title(f"Last {len(x_tail)} observed values (used to condition k up to {k_max})")
            ax2.set_xlabel("relative index (0 = last observed X_T)")
            ax2.set_ylabel("signal")
            ax2.grid(True)
        else:
            ax2.text(0.5, 0.5, "Could not load snapshot context", ha="center", va="center")
    else:
        ax2.text(0.5, 0.5, "No snapshot file provided\n(context not shown)", ha="center", va="center")

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.show()

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Overlay multi-k conditional CDFs from snapshot_cdf_nw_multik.py output.")
    ap.add_argument("output_npz", help="Output .npz produced by snapshot_cdf_nw_multik.py")
    ap.add_argument("--snapshot_npz", default=None, help="Original snapshot .npz to show last-k context")
    ap.add_argument("--no_quantiles", action="store_true", help="Disable Q05/Q50/Q95 markers")
    args = ap.parse_args()

    plot_multik_cdfs(
        output_npz=args.output_npz,
        snapshot_npz=args.snapshot_npz,
        show_quantiles=(not args.no_quantiles),
    )

