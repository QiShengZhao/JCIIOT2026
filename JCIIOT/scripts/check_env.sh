#!/bin/bash
set -e
cd /workspace/collj/JCIIOT
PY=/workspace/.venv/bin/python
echo "python: $($PY --version)"
$PY - <<'PY'
import importlib
mods = ["torch", "numpy", "h5py", "mujoco", "glfw"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        ver = getattr(mod, "__version__", "?")
        extra = ""
        if m == "torch":
            import torch
            extra = f" cuda={torch.cuda.is_available()}"
        print(f"OK {m} {ver}{extra}")
    except Exception as e:
        print(f"MISSING {m}: {e}")
PY
ls -la robosuite/setup.py pyproject.toml requirements.txt 2>&1 | head
