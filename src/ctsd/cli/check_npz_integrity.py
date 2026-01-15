import glob
import numpy as np

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Check integrity of snapshot npz files.")
    ap.add_argument("--glob", dest="pattern", required=True, help='e.g. "data/return_snapshots/*/*/*.npz"')
    args = ap.parse_args()

    bad = []
    for p in glob.glob(args.pattern):
        try:
            z = np.load(p, allow_pickle=True)
            _ = z["dates"]
            _ = z["signal"]
        except Exception as e:
            bad.append((p, type(e).__name__, str(e)))

    print("bad files:", len(bad))
    for row in bad[:20]:
        print(row[0], "=>", row[1], row[2])
    if len(bad) > 20:
        print(f"... and {len(bad)-20} more")
