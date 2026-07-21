#!/usr/bin/env python3
"""DAgger correction collection for L1 grasp BC.

Root problem this addresses: pure-BC closed-loop drifts off the demo manifold
(covariate shift), so a policy that is near-perfect on demo states fails in the
closed loop. DAgger fixes this by collecting expert labels *on the state
distribution the current policy actually visits*.

The scripted side-grasp is a state->action oracle: at any robot state the
correct action is "move each arm's EEF toward the current phase target"
(safe-Z lift -> XY approach -> descend below grasp sites -> close). We roll out
with a beta-mix of policy and expert actions (so the robot drifts the way the
policy would), and at every visited observation we record the *expert* action
as the training label.

Output: an HDF5 in the same schema as the scripted-collect dataset, plus a
pointer file. Aggregate with the original demos before retraining.
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "robosuite", ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import robosuite as suite  # noqa: E402
from robosuite.environments.factory_sorting.factory_sorting_1_3fo3erfhisem import (  # noqa: E402,F401
    FactorySorting1_3FO3ERFHISEM,
)
from robosuite.environments.factory_sorting import (  # noqa: E402
    load_factory_sorting_1_3fo3erfhisem_collect as C,
)
from robosuite.environments.factory_sorting.load_factory_sorting_1_3fo3erfhisem_collect import (  # noqa: E402
    DEFAULT_OBJECT_NAME,
    DEFAULT_ROBOT_BASE_ORI,
    DEFAULT_ROBOT_BASE_POS,
    gather_successful_demonstrations_as_hdf5,
    make_env_kwargs,
    parse_args as _base_parse_args,
)
from robosuite.environments.factory_sorting.turn_to_station import (  # noqa: E402
    set_base_world_yaw_direct,
    set_base_xy_direct,
)
from robosuite.wrappers import DataCollectionWrapper  # noqa: E402


# ── Expert oracle ────────────────────────────────────────────────────────────

class ScriptedExpert:
    """State->action oracle mirroring rollout_once's phased side grasp.

    Phases (per arm target, in order):
        0 safe   : lift straight up to safe_z (keep current xy)
        1 xy     : move in xy above the grasp site (at safe_z)
        2 below  : descend to the below-site grasp target
        3 close  : hold position, close gripper
    Advancement is by proximity: once BOTH arms are within `tol` of the current
    phase target, advance. This makes the oracle valid from *any* state — if the
    policy drifted, the oracle still points back toward the correct next target.
    """

    def __init__(self, base_env, robot, object_name, args):
        self.base_env = base_env
        self.robot = robot
        self.object_name = object_name
        self.args = args
        self.tol = float(args.arrival_tolerance)

        below, _ = C.get_target_positions(base_env, object_name, args.site_below_offset)
        starts = {arm: C.get_eef_pos(base_env, robot, arm) for arm in C.ARMS}
        site_pos = {
            arm: below[arm] + np.array([0.0, 0.0, args.site_below_offset]) for arm in C.ARMS
        }
        safe_z = max(
            args.safe_z,
            max(starts[arm][2] for arm in C.ARMS),
            max(site_pos[arm][2] + args.site_above_clearance for arm in C.ARMS),
        )
        self.below = below
        self.safe_targets = {
            arm: np.array([starts[arm][0], starts[arm][1], safe_z]) for arm in C.ARMS
        }
        self.xy_targets = {
            arm: np.array([site_pos[arm][0], site_pos[arm][1], safe_z]) for arm in C.ARMS
        }
        self.phase = 0

    def _phase_targets(self):
        if self.phase == 0:
            return self.safe_targets
        if self.phase == 1:
            return self.xy_targets
        return self.below  # phases 2 and 3 both aim at the grasp target

    def _maybe_advance(self):
        targets = self._phase_targets()
        dist = {
            arm: np.linalg.norm(C.get_eef_pos(self.base_env, self.robot, arm) - targets[arm])
            for arm in C.ARMS
        }
        if all(d <= self.tol for d in dist.values()) and self.phase < 3:
            self.phase += 1

    def action(self) -> np.ndarray:
        """Expert action for the current state (records nothing, no stepping)."""
        self._maybe_advance()
        robot = self.robot
        robot.composite_controller.update_state()
        gripper_value = 1.0 if self.phase == 3 else -1.0
        arm_actions = {}
        if self.phase < 3:
            targets = self._phase_targets()
            for arm in C.ARMS:
                world_delta = targets[arm] - C.get_eef_pos(self.base_env, robot, arm)
                controller_delta = C.world_delta_to_controller_frame(robot, arm, world_delta)
                arm_actions[arm] = C.arm_delta_to_normalized_action(
                    robot=robot, arm=arm, delta_pos=controller_delta,
                    max_action=self.args.max_action,
                )
        return C.build_action(self.base_env, robot, arm_actions, gripper_value=gripper_value)


# ── Perturbation ─────────────────────────────────────────────────────────────

def apply_base_perturbation(base_env, args, rng):
    robot = base_env.robots[0]
    nom_xy = np.asarray(args.robot_base_pos[:2], dtype=float)
    nom_yaw = float(args.robot_base_ori[2])
    dxy = rng.uniform(-args.xy_noise, args.xy_noise, size=2)
    dyaw = rng.uniform(-args.yaw_noise, args.yaw_noise)
    set_base_xy_direct(base_env, robot, nom_xy + dxy)
    set_base_world_yaw_direct(base_env, robot, nom_yaw + dyaw)
    print(f"Perturbed base dxy={np.round(dxy,4)} dyaw={dyaw:+.4f}")


# ── DAgger rollout ───────────────────────────────────────────────────────────

def dagger_rollout(env, policy, args, rng, beta: float):
    """One rollout: beta-mix policy+expert actions, label every obs with expert.

    Returns (success, reason, ep_directory, obs_buffer). obs_buffer holds the
    visited observations; the recorded actions (env-stepped) are the *expert*
    labels, so the saved demo teaches expert recovery on visited states.
    """
    base_env = env.unwrapped
    env.reset()
    apply_base_perturbation(base_env, args, rng)
    base_env.sim.forward()

    robot = base_env.robots[0]
    setattr(robot, C.CAMERA_HOLD_TARGET_ATTR, C.capture_camera_hold_targets(robot))
    object_name = args.object_name or C.default_object_name(base_env)
    C.configure_object_site_markers(
        base_env, object_name=object_name,
        visible=False, site_size=args.object_site_size,
    )

    expert = ScriptedExpert(base_env, robot, object_name, args)
    obs_buffer = C.make_obs_buffer()
    expert_labels: list = []  # per-visited-obs expert action, aligned 1:1

    if policy is not None:
        policy.start_episode()

    horizon = int(args.horizon)
    for _ in range(horizon):
        a_expert = expert.action()  # label = expert action at THIS state

        # Choose executed action: expert w.p. beta, else policy (induces drift).
        if policy is not None and rng.random() > beta:
            try:
                obs = base_env._get_observations(force_update=True)
                # Policy trained on vertically-flipped RGB (EnvRobosuite default);
                # raw mujoco obs is unflipped, so flip images to match so the
                # policy drifts the way it does at real eval time.
                pol_obs = dict(obs)
                for k in list(pol_obs.keys()):
                    if k.endswith("_image"):
                        pol_obs[k] = np.ascontiguousarray(np.asarray(pol_obs[k])[::-1, ...])
                a_exec = np.asarray(policy(ob=pol_obs)).reshape(-1)
                if a_exec.shape != a_expert.shape:
                    a_exec = a_expert
            except Exception:
                a_exec = a_expert
        else:
            a_exec = a_expert

        # DAgger core: append the CURRENT obs, record the EXPERT action as this
        # obs's label, then STEP the executed (possibly policy) action so the
        # robot drifts the way the policy would. The DataCollectionWrapper logs
        # a_exec as its action; we overwrite it with the expert label in a
        # post-process pass (relabel_actions_to_expert) after saving, keyed by
        # obs<->action count. This is what makes it DAgger and not re-collection.
        C.append_current_obs(base_env, obs_buffer)
        expert_labels.append(np.asarray(a_expert, dtype=np.float32))
        env.step(a_exec)

        if expert.phase == 3 and C.grippers_grasp_object(base_env, robot, object_name):
            for _ in range(args.grasp_steps):  # settle-hold, all labeled close
                C.append_current_obs(base_env, obs_buffer)
                expert_labels.append(np.asarray(a_expert, dtype=np.float32))
                env.step(a_expert)
            break

    success = C.grippers_grasp_object(base_env, robot, object_name)
    reason = "grasp established" if success else "no grasp at horizon"
    env.successful = bool(success)
    return success, reason, env.ep_directory, obs_buffer, expert_labels


def relabel_actions_to_expert(hdf5_path, expert_label_by_len):
    """Overwrite each demo's `actions` with the cached expert labels.

    The DataCollectionWrapper stored the *executed* (drifted) actions; DAgger
    needs the *expert* action as the label at each visited state. We match each
    saved demo to its expert-label array. Demos and expert arrays are produced
    in the same order for successful rollouts, but HDF5 group order is not
    guaranteed, so we match by exact action-count and consume greedily.
    """
    import h5py

    pools = list(expert_label_by_len)
    with h5py.File(hdf5_path, "r+") as f:
        data = f["data"]
        demo_names = sorted(data.keys(), key=lambda s: int(s.split("_")[1]))
        for name in demo_names:
            grp = data[name]
            n = int(grp["actions"].shape[0])
            # find an unused expert-label array of matching length
            match_idx = next(
                (j for j, arr in enumerate(pools)
                 if arr is not None and arr.shape[0] == n),
                None,
            )
            if match_idx is None:
                print(f"[relabel] WARN {name}: no expert labels of len {n}; leaving executed actions")
                continue
            labels = pools[match_idx]
            pools[match_idx] = None
            if labels.shape[1] != grp["actions"].shape[1]:
                print(f"[relabel] WARN {name}: action dim mismatch "
                      f"{labels.shape[1]} vs {grp['actions'].shape[1]}; skipping")
                continue
            del grp["actions"]
            grp.create_dataset("actions", data=labels)
            print(f"[relabel] {name}: actions <- expert labels ({n} steps)")
    print("[relabel] done")


def parse_args():
    backup = sys.argv[:]
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="BC policy checkpoint to drift with; omit for expert-only (beta=1).")
    parser.add_argument("--beta", type=float, default=0.5,
                        help="P(execute expert) each step; lower = more policy drift.")
    parser.add_argument("--horizon", type=int, default=360)
    parser.add_argument("--xy-noise", type=float, default=0.08)
    parser.add_argument("--yaw-noise", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    known, remaining = parser.parse_known_args()
    sys.argv = [backup[0]] + remaining
    args = _base_parse_args()
    sys.argv = backup
    for k in ("checkpoint", "beta", "horizon", "xy_noise", "yaw_noise"):
        setattr(args, k, getattr(known, k))
    if known.seed is not None:
        args.seed = known.seed
    return args


def maybe_load_policy(args):
    """Load the BC policy directly via robomimic FileUtils (no eval env).

    We deliberately avoid the eval module's parse_args/make_eval_env plumbing:
    it builds extra env machinery and clashes with the collector's own argv
    handling. The direct load is ~14s and gives a callable policy(ob=...).
    """
    if not args.checkpoint:
        print("No checkpoint -> expert-only DAgger (beta forced to 1.0)")
        args.beta = 1.0
        return None
    import torch
    import robomimic.utils.file_utils as FileUtils

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, _ = FileUtils.policy_from_checkpoint(
        ckpt_path=str(args.checkpoint), device=device, verbose=False,
    )
    print(f"Loaded policy for drift: {args.checkpoint} (device={device})")
    return policy


def main():
    args = parse_args()
    if args.object_name is None:
        args.object_name = DEFAULT_OBJECT_NAME
    if args.robot_base_pos is None:
        args.robot_base_pos = list(DEFAULT_ROBOT_BASE_POS)
    if args.robot_base_ori is None:
        args.robot_base_ori = list(DEFAULT_ROBOT_BASE_ORI)

    rng = np.random.default_rng(args.seed)
    policy = maybe_load_policy(args)

    env_name = "FactorySorting1_3FO3ERFHISEM"
    env_kwargs = make_env_kwargs(args, render=False)
    dataset_env_kwargs = dict(env_kwargs)
    dataset_env_kwargs["has_renderer"] = False

    raw_env = suite.make(env_name=env_name, **env_kwargs)
    tmp_dir = tempfile.mkdtemp(prefix="l1_dagger_raw_")
    env = DataCollectionWrapper(raw_env, tmp_dir, collect_freq=1, flush_freq=1000)

    ts = datetime.datetime.now().strftime("%Y%m%d%H%M")
    out_dir = os.path.join(args.directory, f"l1_dagger_{ts}")
    hdf5_name = f"l1_grasp_dagger_{ts}.hdf5"
    os.makedirs(out_dir, exist_ok=True)

    successes = 0
    obs_cache = {}
    expert_label_by_len: list = []  # ordered list of expert-label arrays, per saved demo
    for i in range(args.num_rollouts):
        print(f"\n=== DAgger rollout {i+1}/{args.num_rollouts} (beta={args.beta}) ===")
        success, reason, ep_dir, obs_buffer, expert_labels = dagger_rollout(
            env, policy, args, rng, args.beta
        )
        successes += int(success)
        if success:
            obs_cache[os.path.normpath(ep_dir)] = obs_buffer
            expert_label_by_len.append(np.asarray(expert_labels, dtype=np.float32))
        print(f"Result: {reason}")

    env.close()

    hdf5_path, num_saved = gather_successful_demonstrations_as_hdf5(
        tmp_dir, out_dir, hdf5_name=hdf5_name,
        env_name=env_name, env_kwargs=dataset_env_kwargs,
        policy_info={
            **{k: v for k, v in vars(args).items() if k != "checkpoint"},
            "collect_script": "collect_l1_dagger.py",
            "beta": args.beta, "xy_noise": args.xy_noise, "yaw_noise": args.yaw_noise,
            "checkpoint": args.checkpoint or "expert_only",
        },
        obs_cache=obs_cache,
    )
    print(f"\nRollouts: {args.num_rollouts}, successes: {successes}, saved: {num_saved}")
    print(f"HDF5: {hdf5_path}")
    if num_saved == 0:
        raise SystemExit("No successful DAgger demos saved.")

    # ── Relabel wrapper-recorded (executed) actions with EXPERT labels ──
    relabel_actions_to_expert(hdf5_path, expert_label_by_len)
    pointer = ROOT / "trained_models" / "l1_grasp_bc" / "latest_dagger_dataset.txt"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(str(hdf5_path) + "\n", encoding="utf-8")
    print(f"DAgger dataset pointer: {pointer}")


if __name__ == "__main__":
    main()
