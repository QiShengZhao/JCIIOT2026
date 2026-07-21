#!/usr/bin/env python3
"""Concatenate demos from multiple robomimic HDF5 files into one dataset.

All inputs must already be in the same convention (same obs keys, same image
orientation). Used to aggregate DAgger correction demos with the original
scripted demos before retraining. Demos are renumbered demo_1..demo_N and
top-level attrs are copied from the first input (env metadata must match).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def _copy_demo(src_grp, dst_data, new_index):
    dst = dst_data.create_group(f"demo_{new_index}")
    for attr_k, attr_v in src_grp.attrs.items():
        dst.attrs[attr_k] = attr_v
    for k in src_grp.keys():
        if k == "obs":
            obs = dst.create_group("obs")
            for ok in src_grp["obs"].keys():
                arr = src_grp["obs"][ok][...]
                if ok.endswith("_image"):
                    obs.create_dataset(ok, data=arr, compression="gzip", compression_opts=4)
                else:
                    obs.create_dataset(ok, data=arr)
        else:
            dst.create_dataset(k, data=src_grp[k][...])
    return int(src_grp["actions"].shape[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="HDF5 files to merge, in order")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    inputs = [Path(p).resolve() for p in args.inputs]
    for p in inputs:
        if not p.exists():
            raise SystemExit(f"missing input: {p}")
    out = Path(args.output).resolve()
    if out.exists():
        out.unlink()

    total = 0
    obs_key_ref = None
    with h5py.File(out, "w") as fout:
        data = fout.create_group("data")
        first_attrs = None
        for src_i, src_path in enumerate(inputs):
            with h5py.File(src_path, "r") as fin:
                src_data = fin["data"]
                if first_attrs is None:
                    first_attrs = dict(src_data.attrs)
                demo_names = sorted(src_data.keys(), key=lambda s: int(s.split("_")[1]))
                for name in demo_names:
                    grp = src_data[name]
                    keys = set(grp["obs"].keys())
                    if obs_key_ref is None:
                        obs_key_ref = keys
                    elif keys != obs_key_ref:
                        raise SystemExit(
                            f"obs key mismatch in {src_path}/{name}: {keys} != {obs_key_ref}"
                        )
                    total += 1
                    _copy_demo(grp, data, total)
                print(f"[{src_i+1}/{len(inputs)}] {src_path.name}: +{len(demo_names)} demos "
                      f"(running total {total})")

        # top-level attrs from the first input (env metadata identical across sets)
        for k, v in (first_attrs or {}).items():
            data.attrs[k] = v
        data.attrs["num_successful_demos"] = total
        data.attrs["num_demos"] = total
        # record provenance
        try:
            merged = data.attrs.get("policy_info", "{}")
            info = json.loads(merged) if isinstance(merged, str) else {}
        except Exception:
            info = {}
        info["aggregated_from"] = [p.name for p in inputs]
        info["aggregated_total"] = total
        data.attrs["policy_info"] = json.dumps(info)

    print(f"\nMerged {total} demos -> {out}")


if __name__ == "__main__":
    main()
