from ctsd.plotting.snapshot_multik import plot_multik_cdfs

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Plot multi-k conditional CDF output file.")
    ap.add_argument("output_npz", help="Output .npz produced by ctsd-nw-multik-par (multi-k).")
    ap.add_argument("--snapshot_npz", default=None, help="Original snapshot .npz to show conditioning context.")
    ap.add_argument("--no_quantiles", action="store_true", help="Disable Q05/Q50/Q95 markers.")
    ap.add_argument("--title", default=None, help="Override plot title.")
    args = ap.parse_args()

    plot_multik_cdfs(
        output_npz=args.output_npz,
        snapshot_npz=args.snapshot_npz,
        show_quantiles=(not args.no_quantiles),
        title=args.title,
    )
