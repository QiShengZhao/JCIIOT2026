# L1 BC Retrain Design (Approach A)

## Goal

Replace the official demo BC with a competition-grade L1 grasp policy:

1. End-to-end L1 score 10 (`move → pick(BC) → move → place`)
2. Robust under slight base pose perturbations at grasp approach
3. Navigation upgrade deferred (keep A* for now; isolate in `skills/move.py`)

## Pipeline

```text
perturbed scripted collect → HDF5 → robomimic BC train → robot_params.checkpoint_path → L1 eval
```

## Collection

- Script: wrap/extend `load_factory_sorting_1_3fo3erfhisem_collect.py`
- Object: `line_5_container_h01_near`
- Nominal pose: pos `(8.0, 4.6)`, yaw `±π` as used by collect defaults
- Perturbation per rollout (uniform):
  - `xy ± 0.03 m`
  - `yaw ± 0.05 rad`
- Target: ≥80 successful demos (attempt ~120 with `--no-render`)
- Obs: low-dim EEF/gripper + `robot0_robotview` 128×128 (match eval)

## Training

- Algo: robomimic BC (`exps/templates/bc.json` adapted)
- Output: `JCIIOT/trained_models/l1_grasp_bc/`
- Epochs: start 500, save best; pick final `model_epoch_*.pth`
- Device: CUDA in `cvpr-dev`

## Integration

- Set `knowledge/robot_params.json`:
  - `grasp_policy.checkpoint_path` → trained pth
  - keep fallback path for safety
- No changes to forbidden `core/` / `app.py` / `task_config.json`

## Eval

1. Policy-only eval via `load_factory_sorting_evalization.py` (nominal + perturbed)
2. Full agent L1 via `task_subprocess_runner` with fixed plan (bypass LLM if needed)

## Out of scope (phase 2)

- Stronger navigation algorithm (interface only: do not couple into BC train)
