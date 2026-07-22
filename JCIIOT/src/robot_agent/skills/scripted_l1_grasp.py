"""Scripted L1 dual-arm side grasp (collect motion primitives, no BC).

Drop-in replacement for ``run_factory_sorting_grasp_in_wrapped_env`` so
``RobosuiteBackend.grasp_object_physics`` can still handle lift + sync.
"""
from __future__ import annotations

import argparse
import logging
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


def run_scripted_grasp_in_wrapped_env(
    env,
    policy=None,  # unused; kept for signature compatibility
    eval_steps=360,
    debug_policy=False,
    debug_every=25,
    object_name=None,
    site_below_offset=0.035,
    post_hold_steps=5,
    initial_view_steps=5,
    render=True,
    render_sleep=0.0,
    show_object_sites=False,
    object_site_size=0.04,
    camera="robot0_robotview",
    render_callback: Callable[[], None] | None = None,
) -> dict[str, Any]:
    from robosuite.environments.factory_sorting.load_factory_sorting_evalization import (
        base_robosuite_env,
    )
    from robosuite.environments.factory_sorting import (
        load_factory_sorting_1_3fo3erfhisem_collect as C,
    )

    raw_env = base_robosuite_env(env)
    robot = raw_env.robots[0]
    object_name = object_name or C.default_object_name(raw_env)
    setattr(robot, C.CAMERA_HOLD_TARGET_ATTR, C.capture_camera_hold_targets(robot))

    args = argparse.Namespace(
        object_name=object_name,
        up_steps=C.DEFAULT_UP_STEPS,
        xy_steps=C.DEFAULT_XY_STEPS,
        down_steps=C.DEFAULT_DOWN_STEPS,
        safe_z=C.DEFAULT_SAFE_Z,
        site_above_clearance=C.DEFAULT_SITE_ABOVE_CLEARANCE,
        site_below_offset=float(site_below_offset),
        arrival_tolerance=C.DEFAULT_ARRIVAL_TOLERANCE,
        gripper_end_arrival_tolerance=C.DEFAULT_GRIPPER_END_ARRIVAL_TOLERANCE,
        settle_steps=C.DEFAULT_SETTLE_STEPS,
        grasp_steps=C.DEFAULT_GRASP_STEPS,
        post_success_hold_steps=max(int(post_hold_steps), C.DEFAULT_POST_SUCCESS_HOLD_STEPS),
        max_action=C.DEFAULT_MAX_ACTION,
        initial_view_steps=int(initial_view_steps),
        render_sleep=float(render_sleep),
        camera=camera,
        show_object_sites=bool(show_object_sites),
        object_site_size=float(object_site_size),
    )

    # Prefer backend recording callback over viewer.render when provided.
    _orig_render = C.render_frame

    def _render_frame(e, do_render, a):
        if render_callback is not None:
            try:
                render_callback()
            except Exception:
                pass
            return
        _orig_render(e, do_render, a)

    C.render_frame = _render_frame
    obs_buffer = C.make_obs_buffer()

    try:
        try:
            C.configure_object_site_markers(
                raw_env,
                object_name=object_name,
                visible=args.show_object_sites,
                site_size=args.object_site_size,
            )
        except Exception as exc:
            logger.debug("site markers skipped: %s", exc)

        below_site_targets, site_names = C.get_target_positions(
            raw_env, object_name, args.site_below_offset
        )
        starts = {arm: C.get_eef_pos(raw_env, robot, arm) for arm in C.ARMS}

        # Assign each arm to the grasp site nearest its current end-effector.
        # For y-split container sites (L1/L4) with the base facing -x this is
        # the identity (right→right). For x-split tote sites (L2/L3/L5) with the
        # base facing -y the arms are laid out along x and the nearest sites are
        # swapped, so this rebinds right→left-site / left→right-site — without a
        # swap the arms would reach across the object and collide on approach.
        _r, _l = "right", "left"
        # Classify by how the two grasp sites are laid out (stable, geometry-only
        # — do NOT infer from eef distances, which vary with run-time arm pose):
        #   x-split → tote (arms must spread along x, base faces ±y)
        #   y-split → container (arms spread along y, base faces ±x, L1/L4)
        _site_delta = below_site_targets[_r][:2] - below_site_targets[_l][:2]
        _is_tote = abs(_site_delta[0]) > abs(_site_delta[1])
        # For x-split totes the base faces ±y, so the right arm ends up nearer the
        # left site (and vice-versa); rebind each arm to its nearest site so the
        # arms don't cross over the object.
        _keep = (
            np.linalg.norm(starts[_r][:2] - below_site_targets[_r][:2])
            + np.linalg.norm(starts[_l][:2] - below_site_targets[_l][:2])
        )
        _swap = (
            np.linalg.norm(starts[_r][:2] - below_site_targets[_l][:2])
            + np.linalg.norm(starts[_l][:2] - below_site_targets[_r][:2])
        )
        if _swap < _keep:
            below_site_targets = {_r: below_site_targets[_l], _l: below_site_targets[_r]}
            site_names = {_r: site_names[_l], _l: site_names[_r]}
            print(f"[scripted_grasp] arm↔site swapped (x-split tote): "
                  f"keep={_keep:.2f} swap={_swap:.2f}")
        site_positions = {
            arm: below_site_targets[arm] + np.array([0.0, 0.0, args.site_below_offset])
            for arm in C.ARMS
        }
        safe_z = max(
            args.safe_z,
            max(starts[arm][2] for arm in C.ARMS),
            max(site_positions[arm][2] + args.site_above_clearance for arm in C.ARMS),
        )
        # x-split totes: the caller (run_l1_fixed_plan.py) stands the base at a
        # bbox-edge-relative standoff (0.55m from the object's collision
        # bounding box, on the side the grasp sites face) so BOTH default
        # end-effectors already clear the object before any motion — verified
        # via the actual grasp-env creation path (robot_base_pos/ori applied
        # atomically at reset). No horizontal retract is needed: a retract
        # shim here previously pulled the arms in the wrong direction (the
        # axis/sign heuristic doesn't generalise) and re-introduced the exact
        # collision it was meant to avoid. Keep safe_z low and just above the
        # sites — the grasp start point already clears the object.
        if _is_tote:
            safe_z = max(
                max(starts[arm][2] for arm in C.ARMS),
                max(site_positions[arm][2] for arm in C.ARMS) + 0.03,
            )
        safe_targets = {
            arm: np.array([starts[arm][0], starts[arm][1], safe_z]) for arm in C.ARMS
        }
        xy_targets = {
            arm: np.array([site_positions[arm][0], site_positions[arm][1], safe_z])
            for arm in C.ARMS
        }

        print(f"[scripted_grasp] object={object_name} sites={site_names}")
        print(f"[scripted_grasp] below_site_targets={below_site_targets}")

        for _ in range(args.initial_view_steps):
            C.render_frame(env, render, args)

        failed = False
        failure_reason = ""

        ok, reason = C.move_along_linear_segment(
            env=env,
            base_env=raw_env,
            robot=robot,
            object_name=object_name,
            goal_targets=safe_targets,
            num_steps=args.up_steps,
            gripper_value=-1.0,
            render=render,
            args=args,
            obs_buffer=obs_buffer,
            reject_object_contact=True,
            label="safe vertical lift",
        )
        if not ok:
            failed, failure_reason = True, reason

        if not failed:
            ok, reason = C.move_along_linear_segment(
                env=env,
                base_env=raw_env,
                robot=robot,
                object_name=object_name,
                goal_targets=xy_targets,
                num_steps=args.xy_steps,
                gripper_value=-1.0,
                render=render,
                args=args,
                obs_buffer=obs_buffer,
                reject_object_contact=True,
                label="XY approach",
            )
            if not ok:
                failed, failure_reason = True, reason

        if not failed:
            ok, reason = C.move_vertically_below_sites(
                env=env,
                base_env=raw_env,
                robot=robot,
                goal_targets=below_site_targets,
                site_positions=site_positions,
                num_steps=args.down_steps,
                gripper_value=-1.0,
                render=render,
                args=args,
                obs_buffer=obs_buffer,
                label="vertical descent below sites",
            )
            if not ok:
                failed, failure_reason = True, reason

        if not failed:
            ok, reason = C.settle_gripper_end_centers_at_targets(
                env=env,
                base_env=raw_env,
                robot=robot,
                goal_targets=below_site_targets,
                gripper_value=-1.0,
                render=render,
                args=args,
                obs_buffer=obs_buffer,
                label="gripper end center arrival",
            )
            if not ok:
                failed, failure_reason = True, reason

        if not failed:
            C.print_grasp_debug_info(
                env=raw_env,
                robot=robot,
                object_name=object_name,
                goal_targets=below_site_targets,
                label="Before grasp close",
            )
            for _ in range(args.grasp_steps):
                action = C.build_action(raw_env, robot, {}, gripper_value=1.0)
                C.step_with_record(env, raw_env, action, obs_buffer, render, args)

            post_contact, post_grasp = C.print_grasp_debug_info(
                env=raw_env,
                robot=robot,
                object_name=object_name,
                goal_targets=below_site_targets,
                label="After grasp close",
            )
            if not all(post_grasp.values()):
                failed = True
                failure_reason = (
                    "both grippers did not establish a grasp on the object: "
                    f"grasp_status={post_grasp}, fingerpad_contact_status={post_contact}"
                )

        if not failed:
            for _ in range(args.post_success_hold_steps):
                action = C.build_action(raw_env, robot, {}, gripper_value=1.0)
                C.step_with_record(env, raw_env, action, obs_buffer, render, args)

        success = not failed
        if failed:
            print(f"[scripted_grasp] FAIL: {failure_reason}")
        else:
            print("[scripted_grasp] SUCCESS")

        return {
            "success": success,
            "successes": int(success),
            "num_rollouts": 1,
            "return": 0.0,
            "failure_reason": failure_reason,
        }
    finally:
        C.render_frame = _orig_render


def install_scripted_grasp_patch() -> Callable[[], None]:
    """Monkey-patch evalization grasp runner. Returns restore callable."""
    import robosuite.environments.factory_sorting.load_factory_sorting_evalization as ev

    original = ev.run_factory_sorting_grasp_in_wrapped_env
    ev.run_factory_sorting_grasp_in_wrapped_env = run_scripted_grasp_in_wrapped_env

    def restore() -> None:
        ev.run_factory_sorting_grasp_in_wrapped_env = original

    return restore
