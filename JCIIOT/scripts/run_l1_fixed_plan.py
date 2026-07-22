#!/usr/bin/env python3
"""Deterministically run L1 move->pick->place with a fixed plan (no LLM).

Bypasses the planner entirely by building the RobotAgent and feeding it a
hand-written plan, so we can verify the full transport pipeline scores under
the official rule (object moved >1m and lands <0.8m of the target) using the
reliable scripted grasp. Saves a scored-compatible trajectory JSON.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT, ROOT / "robomimic", ROOT / "robosuite" / "robosuite"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

os.environ.setdefault("GATE_OLLAMA", "true")
os.environ.setdefault("GATE_STEP_TIMEOUT", "false")
os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "egl")
os.environ["PYOPENGL_PLATFORM"] = os.environ.get("PYOPENGL_PLATFORM", "egl")


def main() -> int:
    import argparse

    from robot_agent.core.agent import RobotAgent
    from robot_agent.core.map_loader import load_map_files
    from robot_agent.core.scene_context import SceneContext
    from robot_agent.environments import RobosuiteBackend

    ap = argparse.ArgumentParser()
    ap.add_argument("--task-index", type=int, default=0,
                    help="0-based task index (0=L1 .. 4=L5)")
    ap.add_argument("--object", default=None,
                    help="override object name (e.g. L5 has 3 totes, pick the "
                         "most accessible one)")
    a = ap.parse_args()
    ti = int(a.task_index)

    # ── Task parameters (from knowledge/task_config.json) ──
    task_cfg = json.loads((ROOT / "knowledge" / "task_config.json").read_text())
    t = task_cfg["tasks"][ti]
    level = t.get("level", f"L{ti + 1}")
    env_name = t["env_name"]
    source = t["source"]
    target = t["target"]
    obj = a.object or t["object"]
    prefix = t["scene_prefix"]
    print(f"{level}: env={env_name} pick={source} obj={obj} place={target}")

    map_dir = (ROOT / "robosuite" / "robosuite" / "environments"
               / "factory_sorting" / "generated_maps")
    semantic = map_dir / f"{prefix}_scene_regenerated_semantic_map.json"
    grid_file = map_dir / f"{prefix}_scene_regenerated_occupancy_grid.npy"
    scene, grid = load_map_files(semantic, grid_file)
    scene_ctx = SceneContext.from_semantic_map(scene)

    backend = RobosuiteBackend(env_name=env_name, camera="birdview",
                               drive_mode="direct", headless=True)
    backend._scene_context = scene_ctx
    backend.reset()

    # object map: source -> object (mirror task_subprocess_runner logic)
    obj_map = {source: obj}
    raw_meta = getattr(backend.env, "material_metadata", {}) or {}
    for oname, info in raw_meta.items():
        if isinstance(info, dict) and info.get("port_name"):
            obj_map[str(info["port_name"])] = oname
    backend.set_physics_grasp_config(device="cpu", object_map=obj_map)
    backend.reset()

    scene_metadata = {
        "task_index": 0, "env_name": env_name, "map_prefix": prefix,
        "input_object_map": obj_map,
    }
    backend._scene_metadata = scene_metadata

    agent = RobotAgent(
        backend=backend, scene_context=scene_ctx, grid=grid,
        scene_metadata=scene_metadata, knowledge_enabled=False,
    )

    # ── Grasp base pose ──
    # The scripted side grasp needs the robot standing a fixed offset in front
    # of the object (L1 proven: object at x=7.06, base at x=8.0 → +0.94m along
    # the approach axis, base yaw facing the object). The config table's
    # grasp_poses are unreliable for some levels (e.g. input_6 lists (6.0,4.8)
    # while the object sits at x≈11.9, ~5.9m away → arm can't reach). So we
    # derive the base pose from the object's ACTUAL position: stand OFFSET
    # metres toward +x/-x depending on where the object is, keeping the yaw
    # that faces it. This generalises across levels without trusting the table.
    import numpy as _np

    def _object_xy(be, name):
        raw = be.env
        for sfx in ("_joint0", "_free"):
            try:
                q = raw.sim.data.get_joint_qpos(f"{name}{sfx}")
                return _np.array([float(q[0]), float(q[1])])
            except Exception:
                continue
        return None

    def _grasp_sites_xy(be, name):
        raw = be.env
        out = {}
        for arm in ("right", "left"):
            try:
                sid = raw.sim.model.site_name2id(f"{name}_{arm}_grasp_site")
                out[arm] = _np.array(raw.sim.data.site_xpos[sid][:2], dtype=float)
            except Exception:
                return None
        return out

    def _object_bbox_xy(be, name):
        """World-xy bounding box of the object's *_col_* collision geoms."""
        raw = be.env
        m, d = raw.sim.model, raw.sim.data
        xs, ys = [], []
        for i in range(m.ngeom):
            gname = m.geom_id2name(i)
            if gname and name in gname and "col" in gname:
                c = d.geom_xpos[i]
                s = m.geom_size[i]
                xs += [c[0] - s[0], c[0] + s[0]]
                ys += [c[1] - s[1], c[1] + s[1]]
        if not xs:
            return None
        return {"xlo": min(xs), "xhi": max(xs), "ylo": min(ys), "yhi": max(ys)}

    OFFSET = 0.94  # L1-proven stand-off distance from object to base
    obj_xy = _object_xy(backend, obj)
    sites = _grasp_sites_xy(backend, obj)
    bbox = _object_bbox_xy(backend, obj)
    _gp = task_cfg.get("grasp_poses", {}).get(source, {})
    grasp_pose = None
    if obj_xy is not None and sites is not None:
        # Choose approach from how the two grasp sites are laid out:
        #  - y-split (container, e.g. L1/L4): sites differ mainly in y. The
        #    dual arms spread along y when the base faces ±x, so stand on the
        #    +x side and face -x (yaw≈±pi). L1-proven.
        #  - x-split (tote, e.g. L2/L3/L5): sites differ mainly in x. The arms
        #    spread along x when the base faces ±y, so stand on the +y side and
        #    face -y (yaw=-pi/2). The scripted grasp assigns each arm to its
        #    nearest site, so the left/right swap is handled there.
        site_c = 0.5 * (sites["right"] + sites["left"])
        d = sites["right"] - sites["left"]
        if abs(d[1]) >= abs(d[0]):   # y-split → container approach (face -x)
            base_xy = [float(obj_xy[0]) + OFFSET, float(site_c[1])]
            yaw = _gp.get("yaw", -3.139453) if _gp else -3.139453
            kind = "container/y-split face -x"
        else:                         # x-split → tote approach along y
            # Standoff (L2-proven: 0.55m from the object's collision bbox edge
            # nearest the sites, facing the object) clears the gripper's own
            # geometry — it extends ~0.10m past the eef reference point along
            # the approach axis, so a standoff that only clears the eef point
            # still leaves the fingers inside the object.
            #
            # L3/L5 remain UNRESOLVED at any standoff — this is a scene-geometry
            # conflict, not a fixable offset. This scene's `scene_aabb_proxy_
            # input_*` collision proxy (used for official collision scoring)
            # covers the entire ground-level footprint of the pick station.
            # Standing close enough to reach the grasp-site HEIGHT (~1.52m)
            # keeps the wheels inside that proxy, and the resulting contact
            # continuously pushes the whole robot body (confirmed: a zero-delta
            # arm command drifts identically to a real lift command, and
            # `judge_collision_detected: robot_geom=wheel` fires every step),
            # suppressing the arm's max reachable height to ~1.20m — well short
            # of the site. Standing far enough for the wheels to clear the
            # proxy (measured ~1.5m further) restores full height reach but
            # puts the sites ~1.2m out of horizontal arm reach. No standoff
            # satisfies both constraints simultaneously for this tote's
            # geometry in this scene. (`move_along_linear_segment` in
            # load_factory_sorting_1_3fo3erfhisem_collect.py now re-anchors the
            # base every step during grasp primitives via `lock_base_xy`, the
            # same pattern `turn_to_face_xy` already used for the place
            # sequence — this measurably improves precision, e.g. L2 arrival
            # dropped from 0.40m to 0.04m, but does not fix the L3/L5 reach
            # conflict since re-anchoring position doesn't restore the arm's
            # suppressed vertical range.)
            #
            # Verified via the actual grasp env creation path (robot_base_pos/
            # ori applied atomically at reset), which is not subject to the
            # qpos-teleport ordering pitfall of set_base_xy_direct/
            # set_base_world_yaw_direct (those two must be called yaw-before-xy
            # or the xy silently drifts — see robosuite_backend.py
            # grasp_object_physics for the same fix, needed there too).
            TOTE_OFFSET = 0.55
            if bbox is not None:
                near_site_is_lo = abs(site_c[1] - bbox["ylo"]) < abs(site_c[1] - bbox["yhi"])
                if near_site_is_lo:
                    base_y = bbox["ylo"] - TOTE_OFFSET
                    yaw = 1.570796   # face +y, toward the object
                else:
                    base_y = bbox["yhi"] + TOTE_OFFSET
                    yaw = -1.570796  # face -y, toward the object
            else:
                side = 1.0 if site_c[1] < obj_xy[1] else -1.0
                base_y = float(site_c[1]) - side * 0.55
                yaw = 1.570796 if side > 0 else -1.570796
            base_xy = [float(site_c[0]), base_y]
            kind = f"tote/x-split bbox-edge standoff={TOTE_OFFSET}"
        grasp_pose = {"xy": base_xy, "yaw": yaw}
        print(f"grasp pose [{kind}] obj=({obj_xy[0]:.2f},{obj_xy[1]:.2f}) "
              f"sites R={sites['right'].round(2).tolist()} L={sites['left'].round(2).tolist()} "
              f"-> base ({base_xy[0]:.2f},{base_xy[1]:.2f}) yaw={yaw:.3f}")
    elif obj_xy is not None:
        base_xy = [float(obj_xy[0]) + OFFSET, float(obj_xy[1])]
        yaw = _gp.get("yaw", -3.139453) if _gp else -3.139453
        grasp_pose = {"xy": base_xy, "yaw": yaw}
        print(f"grasp pose (no sites, fallback) -> base {base_xy} yaw={yaw:.3f}")
    elif _gp:
        grasp_pose = {"xy": _gp["pos"][:2], "yaw": _gp["yaw"]}
        print(f"grasp pose from config for {source}: {grasp_pose}")

    # ── Fixed plan: move -> pick -> move -> place -> record ──
    pick_inputs = {"target": source, "object_name": obj}
    if grasp_pose is not None:
        pick_inputs["grasp_initial_base_pose"] = grasp_pose
    plan = [
        {"skill_name": "move", "inputs": {"target": source},
         "description": f"navigate to pick station {source}"},
        {"skill_name": "pick_up", "inputs": pick_inputs,
         "description": f"grasp {obj}"},
        {"skill_name": "move", "inputs": {"target": target},
         "description": f"navigate to place station {target}"},
        {"skill_name": "place_down", "inputs": {"target": target},
         "description": f"place {obj} at {target}"},
    ]

    from robot_agent.core.planner import PlanDecision  # type: ignore
    decision = PlanDecision(
        skill_name="composed", response=f"fixed {level} plan", raw="", raw_llm_text="",
        details={"version": "2.0", "understanding": f"{level} fixed plan",
                 "reason": "deterministic verification", "plan": plan,
                 "explanation": "", "warnings": []},
    )

    rec_dir = ROOT / "recordings" / env_name
    rec_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    backend.start_recording()
    t0 = time.perf_counter()
    output = agent._run_multi_step(f"{level} fixed plan", decision, plan, decision.details)
    try:
        backend._record_trajectory_frame()
    except Exception:
        pass
    elapsed = time.perf_counter() - t0

    status = "OK" if output.success else "FAIL"
    traj_path = backend.save_trajectory(rec_dir / f"trajectory_{ts}_{status}.json")
    print(f"\n=== {level} done: success={output.success} elapsed={elapsed:.1f}s ===")
    for s in output.steps:
        print(f"  step {s.skill}: success={s.success} msg={s.message}")
    print(f"trajectory: {traj_path}")

    # ── Score against official rule directly from the trajectory ──
    max_score = t.get("max_score", 10)
    score_l1_official(traj_path, scene_ctx, source, target, obj, level, max_score)
    backend.close()
    return 0 if output.success else 1


def score_l1_official(traj_path, scene_ctx, source, target, obj, level="L1", max_score=10):
    import numpy as np
    traj = json.loads(Path(traj_path).read_text())
    frames = traj.get("frames", [])
    if not frames:
        print("SCORE: no frames"); return

    tgt = scene_ctx.output_ports.get(target)
    tgt_xy = np.array(tgt.center[:2], dtype=float)

    def obj_xy(positions):
        for n, p in positions.items():
            if obj in n or n in obj:
                return np.array([p[0], p[1]], dtype=float), float(p[2])
        return None, None

    spawn = None
    for f in frames:
        xy, _ = obj_xy(f.get("object_positions", {}))
        if xy is not None:
            spawn = xy; break
    final_xy, final_z = obj_xy(frames[-1].get("object_positions", {}))
    if spawn is None or final_xy is None:
        print(f"SCORE: object '{obj}' not found in trajectory"); return

    dx, dy = abs(final_xy[0] - spawn[0]), abs(final_xy[1] - spawn[1])
    dist_tgt = float(np.linalg.norm(final_xy - tgt_xy))
    departure = dx > 1.0 or dy > 1.0
    arrival = dist_tgt < 0.80
    collision = any(fr.get("has_collision") for fr in frames)

    # official split: departure = half, arrival = the rest (matches app.py)
    w_leave = max(1, max_score // 2)
    w_place = max_score - w_leave
    score = (w_leave if departure else 0) + (w_place if arrival else 0) - (5 if collision else 0)
    score = max(0, score)
    print(f"\n===== OFFICIAL-RULE SCORE ({level}, max {max_score}) =====")
    print(f"  spawn=({spawn[0]:.2f},{spawn[1]:.2f}) final=({final_xy[0]:.2f},{final_xy[1]:.2f},z={final_z:.2f})")
    print(f"  target=({tgt_xy[0]:.2f},{tgt_xy[1]:.2f})")
    print(f"  departure: dx={dx:.2f} dy={dy:.2f}  -> {'PASS (+%d)' % w_leave if departure else 'FAIL (0)'}")
    print(f"  arrival:   dist={dist_tgt:.2f}m     -> {'PASS (+%d)' % w_place if arrival else 'FAIL (0)'}")
    print(f"  collision: {collision}  {'(-5)' if collision else ''}")
    print(f"  TOTAL: {score}/{max_score}")


if __name__ == "__main__":
    raise SystemExit(main())
