import os
import shutil
import glob
import numpy as np


def _safe_ticker_from_npz(path: str) -> str:
    """
    Read ticker from output npz file.
    Falls back to filename prefix if missing.
    """
    try:
        z = np.load(path, allow_pickle=True)
        if "ticker" in z:
            t = z["ticker"]
            if isinstance(t, np.ndarray):
                return str(t.item())
            return str(t)
    except Exception:
        pass

    # fallback: filename prefix before first '__'
    base = os.path.basename(path)
    return base.split("__")[0]


def organize_by_ticker(
    root_dir: str,
    pattern: str = "*.npz",
    dry_run: bool = False,
):
    """
    Move output files into subfolders named by ticker.
    """
    files = sorted(glob.glob(os.path.join(root_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"No files found in {root_dir} matching {pattern}")

    log_path = os.path.join(root_dir, "move_by_ticker.log")

    n_moved = 0
    n_skipped = 0

    for p in files:
        fname = os.path.basename(p)

        # Skip logs or non-output artifacts
        if fname.endswith(".log"):
            n_skipped += 1
            continue

        ticker = _safe_ticker_from_npz(p)
        if not ticker:
            n_skipped += 1
            continue

        target_dir = os.path.join(root_dir, ticker)
        target_path = os.path.join(target_dir, fname)

        if not dry_run:
            os.makedirs(target_dir, exist_ok=True)
            shutil.move(p, target_path)

        with open(log_path, "a") as f:
            f.write(f"{p} -> {target_path}\n")

        n_moved += 1

    return n_moved, n_skipped, log_path


def main():
    import argparse

    ap = argparse.ArgumentParser(
        description="Organize NW multi-k output files into per-ticker folders."
    )
    ap.add_argument(
        "root_dir",
        help="Directory containing output .npz files (e.g. data/outputs/snapshot_nw_multik)",
    )
    ap.add_argument(
        "--pattern",
        default="*.npz",
        help="Glob pattern (default: *.npz)",
    )
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="Print actions without moving files.",
    )
    args = ap.parse_args()

    n_moved, n_skipped, log_path = organize_by_ticker(
        root_dir=args.root_dir,
        pattern=args.pattern,
        dry_run=args.dry_run,
    )

    print(f"Moved: {n_moved}")
    print(f"Skipped: {n_skipped}")
    print(f"Log: {log_path}")
