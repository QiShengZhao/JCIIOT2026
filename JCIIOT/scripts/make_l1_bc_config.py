#!/usr/bin/env python3
"""Build a robomimic BC config matching L1 Tiago grasp HDF5 obs keys."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "robomimic" / "exps" / "templates" / "bc.json"

LOW_DIM_KEYS = [
    "robot0_left_eef_pos",
    "robot0_left_eef_quat",
    "robot0_left_gripper_qpos",
    "robot0_right_eef_pos",
    "robot0_right_eef_quat",
    "robot0_right_gripper_qpos",
]
RGB_KEYS = ["robot0_robotview_image"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--name", default="l1_grasp_bc")
    ap.add_argument("--output-dir", default=str(ROOT / "trained_models" / "l1_grasp_bc"))
    ap.add_argument("--num-epochs", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument(
        "--cache-mode",
        default="all",
        choices=["all", "low_dim", "none"],
        help="all=cache RGB in RAM (fastest GPU feed on this box)",
    )
    ap.add_argument(
        "--epoch-steps",
        type=int,
        default=0,
        help="0 = full pass over dataset each epoch (better GPU saturation)",
    )
    args = ap.parse_args()

    if not args.dataset.exists():
        raise SystemExit(f"Dataset missing: {args.dataset}")

    with h5py.File(args.dataset, "r") as f:
        demo0 = next(iter(f["data"].keys()))
        obs_keys = sorted(f[f"data/{demo0}/obs"].keys())
        n_demos = len(f["data"].keys())
    print(f"Demos: {n_demos}, obs keys: {obs_keys}")

    cfg = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    cfg["experiment"]["name"] = args.name
    cfg["experiment"]["validate"] = False
    cfg["experiment"]["rollout"]["enabled"] = False  # offline train first; eval via factory script
    cfg["experiment"]["render_video"] = False
    cfg["experiment"]["save"]["every_n_epochs"] = 50
    cfg["experiment"]["save"]["on_best_rollout_success_rate"] = False
    cfg["experiment"]["save"]["on_best_validation"] = False
    cfg["experiment"]["save"]["epochs"] = [args.num_epochs]
    # None => run_epoch uses len(dataloader); keeps GPU busy for a full pass
    cfg["experiment"]["epoch_every_n_steps"] = None if args.epoch_steps <= 0 else args.epoch_steps

    cfg["train"]["data"] = str(args.dataset.resolve())
    cfg["train"]["output_dir"] = str(Path(args.output_dir).resolve())
    cfg["train"]["num_epochs"] = args.num_epochs
    cfg["train"]["batch_size"] = args.batch_size
    cfg["train"]["num_data_workers"] = args.num_workers
    cfg["train"]["hdf5_cache_mode"] = args.cache_mode
    cfg["train"]["cuda"] = True
    print(
        f"Train throughput: batch={args.batch_size}, workers={args.num_workers}, "
        f"cache={args.cache_mode}, epoch_steps={cfg['experiment']['epoch_every_n_steps']}"
    )

    low = [k for k in LOW_DIM_KEYS if k in obs_keys]
    rgb = [k for k in RGB_KEYS if k in obs_keys]
    if not low:
        raise SystemExit(f"No expected low-dim keys in dataset. Found: {obs_keys}")
    cfg["observation"]["modalities"]["obs"]["low_dim"] = low
    cfg["observation"]["modalities"]["obs"]["rgb"] = rgb
    if rgb:
        cfg["observation"]["encoder"]["rgb"]["core_kwargs"] = {
            "feature_dimension": 64,
            "backbone_class": "ResNet18Conv",
            "backbone_kwargs": {"pretrained": False, "input_coord_conv": False},
            "pool_class": "SpatialSoftmax",
            "pool_kwargs": {"num_kp": 32, "learnable_temperature": False, "temperature": 1.0, "noise_std": 0.0},
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"Wrote config: {args.output}")


if __name__ == "__main__":
    main()
