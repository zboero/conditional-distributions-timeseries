from ctsd.features.cdf_features import extract_features_from_outputs


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Extract fast feature sets from NW multi-k conditional CDF outputs.")
    ap.add_argument("--glob", dest="pattern", required=True, help='e.g. "data/outputs/snapshot_nw_multik/**/*.npz"')
    ap.add_argument("--recursive", action="store_true", help="Use recursive glob (needed for ** patterns).")
    ap.add_argument("--out_csv", required=True, help='e.g. "data/outputs/snapshot_nw_multik/cdf_features.csv"')

    ap.add_argument("--features", default="exhaustive_fast", choices=["core", "exhaustive_fast"])

    ap.add_argument(
        "--quantiles",
        default="",
        help="Optional override quantiles: comma-separated in (0,1). If empty, preset is used.",
    )
    ap.add_argument(
        "--tail_thresholds",
        default="",
        help="Optional override tail thresholds: comma-separated thresholds (e.g. -0.02,-0.01,0,0.01,0.02).",
    )
    ap.add_argument("--bad_log", default="", help="Optional path to log failures.")

    args = ap.parse_args()

    quantiles_override = None
    if args.quantiles.strip():
        quantiles_override = tuple(float(s) for s in args.quantiles.split(",") if s.strip())

    tail_thresholds_override = None
    if args.tail_thresholds.strip():
        tail_thresholds_override = tuple(float(s) for s in args.tail_thresholds.split(",") if s.strip())

    out = extract_features_from_outputs(
        pattern=args.pattern,
        out_csv=args.out_csv,
        recursive=args.recursive,
        features=args.features,
        quantiles_override=quantiles_override,
        tail_thresholds_override=tail_thresholds_override,
        bad_log=(args.bad_log.strip() or None),
    )

    print(f"Wrote: {out}")
