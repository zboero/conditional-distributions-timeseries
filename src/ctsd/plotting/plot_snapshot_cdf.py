#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 08:23:31 2025

@author: ezequiel

    # NW output
    python plot_snapshot_cdf.py \
      out_snapshot_nw/WYNN__20240516_20250519__nw_cdf_next_k5.npz \
      --snapshot_npz return_snapshots/WYNN/3month/20240516_20250519.npz \
      --k 5

    # Bootstrap output 
    python plot_snapshot_cdf.py \
      out_snapshot_boot/WYNN__20240516_20250519__boot_cdf_next_k5.npz \
      --snapshot_npz return_snapshots/WYNN/3month/20240516_20250519.npz \
      --k 5
      
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

# -------- helpers --------

def load_output_npz(path):
    z = np.load(path, allow_pickle=True)

    # Required
    x_grid = z["x_grid"]
    F = z["cond_cdf"]

    # Metadata
    ticker = z["ticker"].item() if "ticker" in z else "UNKNOWN"
    snapshot_id = z["snapshot_id"].item() if "snapshot_id" in z else ""
    origin_date = z["forecast_origin_date"]
    forecast_time = z["forecast_time"] if "forecast_time" in z else None

    meta = {}
    if "meta_json" in z:
        try:
            meta = json.loads(z["meta_json"].item())
        except Exception:
            meta = {}

    return {
        "x_grid": x_grid,
        "cdf": F,
        "ticker": ticker,
        "snapshot_id": snapshot_id,
        "origin_date": origin_date,
        "forecast_time": forecast_time,
        "meta": meta,
    }


def load_snapshot_context(snapshot_path, k):
    """
    Load the original snapshot to extract the last k observed points.
    """
    z = np.load(snapshot_path, allow_pickle=True)
    if "signal" not in z or "dates" not in z:
        return None, None

    x = np.asarray(z["signal"], dtype=float)
    dates = np.asarray(z["dates"])

    if len(x) < k:
        return None, None

    return dates[-k:], x[-k:]


# -------- plotting --------

def plot_snapshot_cdf(output_npz, snapshot_npz=None, k=5):
    out = load_output_npz(output_npz)

    x_grid = out["x_grid"]
    F = out["cdf"]
    ticker = out["ticker"]
    snap_id = out["snapshot_id"]
    origin_date = out["origin_date"]
    forecast_time = out["forecast_time"]
    meta = out["meta"]

    fig, axes = plt.subplots(
        2, 1, figsize=(9, 7), gridspec_kw={"height_ratios": [1.2, 1.0]}
    )
    ax_cdf, ax_ctx = axes

    # ---- Panel 1: Conditional CDF ----
    ax_cdf.plot(x_grid, F, lw=2)
    ax_cdf.set_xlabel("x (next value)")
    ax_cdf.set_ylabel("F(x | last k)")
    ax_cdf.set_title(
        f"{ticker}  |  snapshot {snap_id}\n"
        f"Conditional CDF for next step beyond snapshot"
    )
    ax_cdf.grid(True)

    # Mark median and quantiles
    q50 = np.interp(0.5, F, x_grid)
    q05 = np.interp(0.05, F, x_grid)
    q95 = np.interp(0.95, F, x_grid)

    ax_cdf.axvline(q50, ls="--", lw=1)
    ax_cdf.axvline(q05, ls=":", lw=1)
    ax_cdf.axvline(q95, ls=":", lw=1)

    ax_cdf.text(
        0.02,
        0.95,
        f"Q05={q05:+.4g}\nQ50={q50:+.4g}\nQ95={q95:+.4g}",
        transform=ax_cdf.transAxes,
        va="top",
        family="monospace",
    )

    # ---- Panel 2: last-k context ----
    if snapshot_npz is not None:
        ctx_dates, ctx_vals = load_snapshot_context(snapshot_npz, k)
        if ctx_vals is not None:
            ax_ctx.plot(range(-k + 1, 1), ctx_vals, marker="o")
            ax_ctx.axvline(0, ls="--")
            ax_ctx.set_xlabel("relative time (0 = last observed)")
            ax_ctx.set_ylabel("signal")
            ax_ctx.set_title(f"Last {k} observed points used for conditioning")
            ax_ctx.grid(True)
        else:
            ax_ctx.text(0.5, 0.5, "Could not load snapshot context",
                        ha="center", va="center")
    else:
        ax_ctx.text(
            0.5,
            0.5,
            "No snapshot file provided\n(context not shown)",
            ha="center",
            va="center",
        )

    # ---- footer info ----
    footer = f"origin: {origin_date}"
    if forecast_time is not None and str(forecast_time) != "":
        footer += f"  → forecast time: {forecast_time}"

    if "method" in meta:
        footer += f"\nmethod: {meta.get('method')}"

    fig.text(0.5, 0.01, footer, ha="center", fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.show()


# -------- CLI --------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Plot one snapshot conditional CDF.")
    ap.add_argument("output_npz", help="Output file produced by snapshot_cdf_*.py")
    ap.add_argument("--snapshot_npz", help="Original snapshot file (optional)")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    plot_snapshot_cdf(
        output_npz=args.output_npz,
        snapshot_npz=args.snapshot_npz,
        k=args.k,
    )
