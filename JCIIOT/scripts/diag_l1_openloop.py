#!/usr/bin/env python3
"""Open-loop replay of demo actions in eval env (flip_visual_obs=False path)."""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path("/workspace/collj/JCIIOT")
sys.path[:0] = [str(ROOT), str(ROOT / "robosuite"), str(ROOT / "src")]

from robosuite.environments.factory_sorting.load_factory_sorting_evalization import (
    base_robosuite_env,
    load_policy_and_config,
    make_eval_env,
    parse_args,
    print_grasp_debug_info,
    print_reset_debug_info,
)


def main() -> None:
    ckpt = (ROOT / "trained_models/l1_grasp_bc/latest_checkpoint.txt").read_text().strip()
    demo_path = ROOT / (
        "trained_models/l1_grasp_bc/demos/l1_perturbed_202607180255/"
        "l1_grasp_perturbed_202607180255.hdf5"
    )
    demo_key = "demo_1"

    old = sys.argv
    sys.argv = [
        "diag",
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
    try:
        args = parse_args()
    finally:
        sys.argv = old

    policy, config, ckpt_dict = load_policy_and_config(args)
    env = make_eval_env(args, config=config, ckpt_dict=ckpt_dict, render=False)

    with h5py.File(demo_path, "r") as f:
        demo = f[f"data/{demo_key}"]
        actions = np.asarray(demo["actions"])
        states = np.asarray(demo["states"]) if "states" in demo else None
        print(f"demo={demo_key} T={len(actions)} states={'yes' if states is not None else 'no'}")

        # --- A: open-loop actions from reset (no state restore) ---
        policy.start_episode()
        obs = env.reset()
        state_dict = env.get_state()
        obs = env.reset_to(state_dict)
        raw = base_robosuite_env(env)
        object_name = args.object_name or "line_5_container_h01_near"
        goals = print_reset_debug_info(raw, object_name, args)
        for a in actions:
            obs, _, _, _ = env.step(a)
        _, grasps = print_grasp_debug_info(
            env=raw, robot=raw.robots[0], object_name=object_name,
            goal_targets=goals, label="OpenLoop from reset",
        )
        print(f"OpenLoop-from-reset success={all(grasps.values())} grasps={grasps}")

        # --- B: reset_to first demo state then open-loop ---
        if states is not None:
            policy.start_episode()
            obs = env.reset()
            # robomimic-style: set full state
            try:
                obs = env.reset_to({"states": states[0]})
            except Exception as exc:
                print(f"reset_to states[0] failed: {exc}")
                obs = env.reset()
            raw = base_robosuite_env(env)
            goals = print_reset_debug_info(raw, object_name, args)
            for a in actions:
                obs, _, _, _ = env.step(a)
            _, grasps = print_grasp_debug_info(
                env=raw, robot=raw.robots[0], object_name=object_name,
                goal_targets=goals, label="OpenLoop from demo state0",
            )
            print(f"OpenLoop-from-demo-state0 success={all(grasps.values())} grasps={grasps}")

        # --- C: closed-loop BC from reset ---
        policy.start_episode()
        obs = env.reset()
        state_dict = env.get_state()
        obs = env.reset_to(state_dict)
        raw = base_robosuite_env(env)
        goals = print_reset_debug_info(raw, object_name, args)
        last = None
        for t in range(len(actions)):
            last = policy(ob=obs)
            obs, _, _, _ = env.step(last)
        for _ in range(10):
            obs, _, _, _ = env.step(last)
        _, grasps = print_grasp_debug_info(
            env=raw, robot=raw.robots[0], object_name=object_name,
            goal_targets=goals, label="ClosedLoop BC",
        )
        print(f"ClosedLoop success={all(grasps.values())} grasps={grasps}")

        # first-step action match at reset
        policy.start_episode()
        obs = env.reset()
        obs = env.reset_to(env.get_state())
        a_pol = np.asarray(policy(ob=obs)).reshape(-1)
        a_demo = actions[0]
        print(f"first-step action L2={np.linalg.norm(a_pol-a_demo):.4f}")
        demo_ob = {k: np.asarray(demo["obs"][k][0]) for k in demo["obs"].keys()}
        # pad frame stack if needed
        img = obs.get("robot0_robotview_image")
        if img is not None and np.asarray(img).ndim == 4:
            for k, v in list(demo_ob.items()):
                vv = np.asarray(v)
                if vv.ndim >= 1 and k in obs and np.asarray(obs[k]).ndim == vv.ndim + 1:
                    demo_ob[k] = np.stack([vv] * np.asarray(obs[k]).shape[0], axis=0)
        a_pol2 = np.asarray(policy(ob=demo_ob)).reshape(-1)
        print(f"policy(demo0) L2={np.linalg.norm(a_pol2-a_demo):.4f}")

    env.close()


if __name__ == "__main__":
    main()
