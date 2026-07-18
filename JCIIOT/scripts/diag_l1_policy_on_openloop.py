#!/usr/bin/env python3
"""While open-looping demo actions, measure policy(action|obs) vs demo action."""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path("/workspace/collj/JCIIOT")
sys.path[:0] = [str(ROOT), str(ROOT / "robosuite"), str(ROOT / "src")]

from robosuite.environments.factory_sorting.load_factory_sorting_evalization import (
    load_policy_and_config,
    make_eval_env,
    parse_args,
)


def main() -> None:
    ckpt = (ROOT / "trained_models/l1_grasp_bc/latest_checkpoint.txt").read_text().strip()
    demo_path = ROOT / (
        "trained_models/l1_grasp_bc/demos/l1_perturbed_202607180255/"
        "l1_grasp_perturbed_202607180255.hdf5"
    )
    # Prefer a demo whose base is close to nominal if attrs exist; else demo_60 mid-set
    demo_key = "demo_60"

    old = sys.argv
    sys.argv = [
        "diag", "--checkpoint", ckpt,
        "--factory-scene", "factory_sorting_1_3fo3erfhisem",
        "--num-rollouts", "1", "--no-render", "--device", "cuda",
    ]
    try:
        args = parse_args()
    finally:
        sys.argv = old

    policy, config, ckpt_dict = load_policy_and_config(args)
    env = make_eval_env(args, config=config, ckpt_dict=ckpt_dict, render=False)

    with h5py.File(demo_path, "r") as f:
        demo = f[f"data/{demo_key}"]
        actions = np.asarray(demo["actions"])
        print(f"demo={demo_key} T={len(actions)}")

        policy.start_episode()
        obs = env.reset()
        obs = env.reset_to(env.get_state())

        errs = []
        for t, a in enumerate(actions):
            a_pol = np.asarray(policy(ob=obs)).reshape(-1)
            a_demo = a.reshape(-1)
            err = float(np.linalg.norm(a_pol - a_demo))
            errs.append(err)
            if t % 40 == 0 or t == len(actions) - 1:
                print(f"t={t:3d} action_L2={err:.4f} pol_norm={np.linalg.norm(a_pol):.3f} demo_norm={np.linalg.norm(a_demo):.3f}")
            obs, _, _, _ = env.step(a_demo)

        errs = np.asarray(errs)
        print(f"action_L2 mean/p50/p90/max={errs.mean():.4f}/{np.median(errs):.4f}/{np.quantile(errs,0.9):.4f}/{errs.max():.4f}")


if __name__ == "__main__":
    main()
