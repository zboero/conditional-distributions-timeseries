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
