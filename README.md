# conditional-timeseries-distributions (ctsd)

Nonparametric estimation of conditional distribution functions in time series snapshots.

This project is designed for datasets stored as `.npz` snapshots containing:
- `signal`: 1D time series (e.g., returns)
- `dates`: aligned timestamps
- optional metadata like `resample_min`

## Install

From the repo root:

```bash
python -m pip install -U pip
python -m pip install -e .
```

## Integrity check

```bash
ctsd-check-npz --glob "data/return_snapshots/*/*/*.npz"
```

## Run parallel NW multi-k

```bash
ctsd-nw-multik-par \
  --glob "data/return_snapshots/*/*/*.npz" \
  --out_dir "data/outputs/snapshot_nw_multik" \
  --k_list "1,2,5,10,20" \
  --grid_size 600 \
  --cv_points 150 \
  --max_workers 6
```

## Plot output only

```bash
ctsd-plot-multik data/outputs/snapshot_nw_multik/WYNN__20240516_20250519__nw_cdf_next_multik.npz
```

## Plot output + show conditioning context from snapshot

```bash
ctsd-plot-multik \
  data/outputs/snapshot_nw_multik/WYNN__20240516_20250519__nw_cdf_next_multik.npz \
  --snapshot_npz data/return_snapshots/WYNN/3month/20240516_20250519.npz
