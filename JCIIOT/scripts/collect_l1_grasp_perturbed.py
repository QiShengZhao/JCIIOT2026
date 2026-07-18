#!/usr/bin/env python3
"""Collect L1 grasp demos with base-pose perturbation for BC robustness."""
from __future__ import annotations

import argparse
import datetime
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROBOSUITE_ROOT = ROOT / "robosuite"
if str(ROBOSUITE_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOSUITE_ROOT))

import robosuite as suite  # noqa: E402
from robosuite.environments.factory_sorting.factory_sorting_1_3fo3erfhisem import (  # noqa: E402,F401
    FactorySorting1_3FO3ERFHISEM,
)
from robosuite.environments.factory_sorting.load_factory_sorting_1_3fo3erfhisem_collect import (  # noqa: E402
    DEFAULT_OBJECT_NAME,
    DEFAULT_ROBOT_BASE_ORI,
    DEFAULT_ROBOT_BASE_POS,
    gather_successful_demonstrations_as_hdf5,
    make_env_kwargs,
    parse_args as _base_parse_args,
    rollout_once,
)
from robosuite.environments.factory_sorting.turn_to_station import (  # noqa: E402
    set_base_world_yaw_direct,
    set_base_xy_direct,
)
from robosuite.wrappers import DataCollectionWrapper  # noqa: E402


def parse_args():
    # Reuse collector CLI, then add perturbation knobs.
    sys_argv_backup = sys.argv[:]
    # Inject --no-render default for container unless user overrides.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--xy-noise", type=float, default=0.03)
    parser.add_argument("--yaw-noise", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    known, remaining = parser.parse_known_args()
    sys.argv = [sys_argv_backup[0]] + remaining
    args = _base_parse_args()
    sys.argv = sys_argv_backup
    args.xy_noise = known.xy_noise
    args.yaw_noise = known.yaw_noise
    if known.seed is not None:
        args.seed = known.seed
    return args


def apply_base_perturbation(env, args, rng: np.random.Generator):
    base_env = env.unwrapped
    robot = base_env.robots[0]
    nom_xy = np.asarray(args.robot_base_pos[:2], dtype=float)
    nom_yaw = float(args.robot_base_ori[2])
    dxy = rng.uniform(-args.xy_noise, args.xy_noise, size=2)
    dyaw = rng.uniform(-args.yaw_noise, args.yaw_noise)
    target_xy = nom_xy + dxy
    target_yaw = nom_yaw + dyaw
    set_base_xy_direct(base_env, robot, target_xy)
    set_base_world_yaw_direct(base_env, robot, target_yaw)
    print(
        f"Perturbed base xy=({target_xy[0]:.4f},{target_xy[1]:.4f}) "
        f"yaw={target_yaw:.4f} (dxy={dxy}, dyaw={dyaw:.4f})"
    )


def rollout_once_perturbed(env, render, args, rng):
    # Capture original reset path: call rollout pieces with post-reset teleport.
    # We monkey-patch by wrapping env.reset temporarily.
    original_reset = env.reset

    def reset_with_noise(*a, **k):
        out = original_reset(*a, **k)
        apply_base_perturbation(env, args, rng)
        return out

    env.reset = reset_with_noise
    try:
        return rollout_once(env, render=render, args=args)
    finally:
        env.reset = original_reset


def main():
    args = parse_args()
    if args.object_name is None:
        args.object_name = DEFAULT_OBJECT_NAME
    if args.robot_base_pos is None:
        args.robot_base_pos = list(DEFAULT_ROBOT_BASE_POS)
    if args.robot_base_ori is None:
        args.robot_base_ori = list(DEFAULT_ROBOT_BASE_ORI)

    rng = np.random.default_rng(args.seed)
    render = not args.no_render
    env_name = "FactorySorting1_3FO3ERFHISEM"
    env_kwargs = make_env_kwargs(args, render=render)
    dataset_env_kwargs = dict(env_kwargs)
    dataset_env_kwargs["has_renderer"] = False

    raw_env = suite.make(env_name=env_name, **env_kwargs)
    tmp_directory = tempfile.mkdtemp(prefix="l1_grasp_perturbed_raw_")
    env = DataCollectionWrapper(raw_env, tmp_directory, collect_freq=1, flush_freq=1000)

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
    out_dir = os.path.join(args.directory, f"l1_perturbed_{timestamp}")
    hdf5_name = f"l1_grasp_perturbed_{timestamp}.hdf5"
    os.makedirs(out_dir, exist_ok=True)

    successes = 0
    obs_cache = {}
    for rollout_idx in range(args.num_rollouts):
        print(f"\nRollout {rollout_idx + 1}/{args.num_rollouts}")
        success, reason, ep_directory, obs_buffer = rollout_once_perturbed(
            env, render=render, args=args, rng=rng
        )
        successes += int(success)
        if success:
            obs_cache[os.path.normpath(ep_directory)] = obs_buffer
        print(f"Result: {reason}")

    env.close()
    hdf5_path, num_saved = gather_successful_demonstrations_as_hdf5(
        tmp_directory,
        out_dir,
        hdf5_name=hdf5_name,
        env_name=env_name,
        env_kwargs=dataset_env_kwargs,
        policy_info={
            **vars(args),
            "xy_noise": args.xy_noise,
            "yaw_noise": args.yaw_noise,
            "collect_script": "collect_l1_grasp_perturbed.py",
        },
        obs_cache=obs_cache,
    )
    print(f"\nAttempts: {args.num_rollouts}, successes: {successes}, saved demos: {num_saved}")
    print(f"HDF5 saved to: {hdf5_path}")
    if num_saved == 0:
        raise SystemExit("No successful demos saved.")
    # Write a pointer file for train script.
    pointer = ROOT / "trained_models" / "l1_grasp_bc" / "latest_dataset.txt"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(str(hdf5_path) + "\n", encoding="utf-8")
    print(f"Dataset pointer: {pointer}")


if __name__ == "__main__":
    main()
