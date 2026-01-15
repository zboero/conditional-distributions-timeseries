import os
import numpy as np
import zipfile
import zlib

RET_KEY = "signal"
DATE_KEY = "dates"

def load_snapshot(path: str, ret_key: str = RET_KEY, date_key: str = DATE_KEY):
    """
    Robust loader for snapshot npz.

    Expected:
      - ret_key ('signal'): 1D numeric array
      - date_key ('dates'): aligned timestamps

    Returns:
      x (float ndarray), dates (ndarray), meta_in (dict)
    Raises:
      RuntimeError if the file is corrupted/unreadable or missing required keys.
    """
    try:
        z = np.load(path, allow_pickle=True)
    except (OSError, zipfile.BadZipFile, zlib.error, ValueError) as e:
        raise RuntimeError(f"Cannot load npz (possibly corrupted): {path} :: {type(e).__name__}: {e}")

    if ret_key not in z or date_key not in z:
        raise RuntimeError(f"{path}: needs keys '{ret_key}' and '{date_key}'. keys={list(z.keys())}")

    try:
        x = np.asarray(z[ret_key], dtype=float)
        dates = np.asarray(z[date_key])
    except Exception as e:
        raise RuntimeError(f"{path}: failed reading arrays :: {type(e).__name__}: {e}")

    meta_in = {}
    for k in z.files:
        if k in (ret_key, date_key):
            continue
        try:
            arr = np.asarray(z[k])
            meta_in[k] = arr.item() if arr.shape == () else arr
        except Exception:
            meta_in[k] = "UNREADABLE_META"

    return x, dates, meta_in


def infer_ticker_from_path(path: str) -> str:
    """
    Expects .../TICKER/WINDOW/file.npz (as in return_snapshots).
    """
    parts = os.path.normpath(path).split(os.sep)
    return parts[-3] if len(parts) >= 3 else os.path.splitext(os.path.basename(path))[0]


def infer_forecast_time(meta_in: dict, last_date):
    """
    If meta_in contains resample_min, compute forecast_time = last_date + resample_min minutes.
    """
    try:
        resample_min = meta_in.get("resample_min", None)
        if resample_min is None:
            return None
        resample_min = int(np.asarray(resample_min).item())
        return last_date + np.timedelta64(resample_min, "m")
    except Exception:
        return None
