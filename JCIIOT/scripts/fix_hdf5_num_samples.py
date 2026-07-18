#!/usr/bin/env python3
"""Backfill robomimic-required attrs/datasets on an existing grasp HDF5."""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def fix_file(path: Path) -> None:
    with h5py.File(path, "a") as f:
        data = f["data"]
        for demo in data.keys():
            g = data[demo]
            n = int(g["actions"].shape[0])
            g.attrs["num_samples"] = n
            if "rewards" not in g:
                g.create_dataset("rewards", data=np.zeros((n, 1), dtype=np.float32))
            if "dones" not in g:
                dones = np.zeros((n, 1), dtype=np.float32)
                if n > 0:
                    dones[-1] = 1.0
                g.create_dataset("dones", data=dones)
            print(f"{demo}: num_samples={n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hdf5", type=Path)
    args = ap.parse_args()
    fix_file(args.hdf5)
    print("FIXED", args.hdf5)


if __name__ == "__main__":
    main()
