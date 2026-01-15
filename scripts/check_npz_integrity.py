#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 14:11:26 2025

@author: ezequiel

    # Run it as follow...
    python check_npz_integrity.py
    
"""

import glob
import numpy as np
import zipfile
import zlib

def main():
    bad = []
    for p in glob.glob("data/minute_data/return_snapshots/*/*/*.npz"):
        try:
            z = np.load(p, allow_pickle=True)
            # force decompression of key arrays
            _ = z["dates"]
            _ = z["signal"]
        except Exception as e:
            bad.append((p, type(e).__name__, str(e)))

    print("bad files:", len(bad))
    for row in bad[:20]:
        print(row[0], "=>", row[1], row[2])

    if len(bad) > 20:
        print(f"... and {len(bad) - 20} more")

if __name__ == "__main__":
    main()


