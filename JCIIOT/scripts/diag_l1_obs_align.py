#!/usr/bin/env python3
"""Compare demo vs eval obs/actions with flip_visual_obs=False."""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path("/workspace/collj/JCIIOT")
sys.path[:0] = [str(ROOT), str(ROOT / "robosuite"), str(ROOT / "src")]

from robosuite.environments.factory_sorting.load_factory_sorting_evalization import (
    load_policy_and_config,
    make_eval_env,
    parse_args,
)


def main() -> None:
    ckpt = (ROOT / "trained_models/l1_grasp_bc/latest_checkpoint.txt").read_text().strip()
    demo_path = next(
        (ROOT / "trained_models/l1_grasp_bc/demos").rglob("l1_grasp_perturbed_*.hdf5")
    )
    print(f"ckpt={ckpt}")
    print(f"demo={demo_path}")

    argv = [
        "--checkpoint",
        ckpt,
        "--factory-scene",
        "factory_sorting_1_3fo3erfhisem",
        "--num-rollouts",
        "1",
        "--no-render",
        "--device",
        "cuda",
    ]
    old = sys.argv
    sys.argv = ["diag"] + argv
    try:
        args = parse_args()
    finally:
        sys.argv = old

    policy, config, ckpt_dict = load_policy_and_config(args)
    env = make_eval_env(args, config=config, ckpt_dict=ckpt_dict, render=False)
    print(f"flip_visual_obs={getattr(env, 'flip_visual_obs', getattr(getattr(env, 'env', None), 'flip_visual_obs', 'n/a'))}")

    # unwrap FrameStack if present
    base = env
    while hasattr(base, "env"):
        if hasattr(base, "flip_visual_obs"):
            print(f"found flip_visual_obs={base.flip_visual_obs} on {type(base)}")
        base = base.env
    print(f"raw env type={type(base)}")

    obs = env.reset()
    with h5py.File(demo_path, "r") as f:
        demo = f["data/demo_0"]
        demo_obs_keys = [k for k in demo["obs"].keys()]
        print(f"demo obs keys sample: {demo_obs_keys[:8]}")
        img_key = "robot0_robotview_image"
        d_img = demo["obs"][img_key][0]
        d_act = demo["actions"][0]
        e_img = np.asarray(obs[img_key])
        print(f"demo img shape/dtype/mean={d_img.shape}/{d_img.dtype}/{d_img.mean():.2f}")
        print(f"env  img shape/dtype/mean={e_img.shape}/{e_img.dtype}/{e_img.mean():.2f}")
        print(f"img L2(env,demo0)={np.linalg.norm(e_img.astype(np.float32)-d_img.astype(np.float32)):.1f}")
        print(f"img L2(env,flip demo)={np.linalg.norm(e_img.astype(np.float32)-d_img[::-1].astype(np.float32)):.1f}")

        # low-dim comparison for keys used by policy
        for k in [
            "robot0_left_eef_pos",
            "robot0_right_eef_pos",
            "robot0_left_eef_quat",
            "robot0_right_eef_quat",
            "robot0_left_gripper_qpos",
            "robot0_right_gripper_qpos",
        ]:
            dv = np.asarray(demo["obs"][k][0], dtype=np.float64)
            ev = np.asarray(obs[k], dtype=np.float64).reshape(-1)[: dv.size]
            print(f"{k}: demo={dv} env={ev} L2={np.linalg.norm(ev-dv):.4f}")

        # actions: raw policy vs demo
        # Build obs dict matching policy expectations (may be stacked)
        act = policy(ob=obs)
        act = np.asarray(act).reshape(-1)
        print(f"demo act[:8]={d_act[:8]}")
        print(f"pol  act[:8]={act[:8]}")
        print(f"action L2 vs demo0={np.linalg.norm(act.astype(np.float64)-d_act.astype(np.float64)):.4f}")

        # force-feed demo first frame through policy (exact demo obs)
        demo_ob = {k: np.asarray(demo["obs"][k][0]) for k in demo["obs"].keys() if k in obs}
        # if frame stack, need to expand
        if isinstance(obs[img_key], np.ndarray) and obs[img_key].ndim == e_img.ndim + 1:
            print(f"frame stack dim detected: env img ndim={obs[img_key].ndim}")
        act2 = policy(ob=demo_ob)
        act2 = np.asarray(act2).reshape(-1)
        print(f"policy(demo_ob) L2 vs demo act={np.linalg.norm(act2.astype(np.float64)-d_act.astype(np.float64)):.4f}")

    env.close()


if __name__ == "__main__":
    main()
