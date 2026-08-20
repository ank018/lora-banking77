"""Seed of src/runner.py - environment capture.

A lockfile in the repo records what you intended to install. This records
what actually executed, on the machine that executed it, and it goes into
every run's output JSON. Kaggle hands out T4s and P100s interchangeably;
if that isn't recorded per run, a hardware confound can sit inside the
scaling curve undetected.
"""

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone

# Libraries whose version can move an accuracy number. Everything else is
# noise in a 500-line pip freeze and belongs in the full dump, not here.
DECISIVE = [
    "torch", "transformers", "peft", "trl", "bitsandbytes",
    "accelerate", "tokenizers", "datasets", "numpy",
]


def _versions():
    out = {}
    for name in DECISIVE:
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            out[name] = None
    return out


def _gpu():
    try:
        import torch
        if not torch.cuda.is_available():
            return {"available": False}
        return {
            "available": True,
            "name": torch.cuda.get_device_name(0),
            "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "count": torch.cuda.device_count(),
            "cuda": torch.version.cuda,
            "bf16_supported": torch.cuda.is_bf16_supported(),
            "vram_gb": round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
            ),
        }
    except Exception as e:  # noqa: BLE001 - never let this kill a run
        return {"available": False, "error": str(e)}


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001
        return None


def _pip_freeze():
    try:
        return subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True
        )
    except Exception:  # noqa: BLE001
        return ""


def env_manifest():
    """Decision-relevant environment, small enough to sit in every run JSON."""
    freeze = _pip_freeze()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_sha": _git_sha(),
        "versions": _versions(),
        "gpu": _gpu(),
        # Full freeze is too long to embed; the hash proves two runs shared
        # an environment, and the text is written beside the run.
        "pip_freeze_sha256": hashlib.sha256(freeze.encode()).hexdigest()[:16],
    }


def write_env(run_dir):
    """Call once at the top of every stage. Writes env.json + env.txt."""
    from pathlib import Path

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = env_manifest()
    (run_dir / "env.json").write_text(json.dumps(manifest, indent=2))
    (run_dir / "env.txt").write_text(_pip_freeze())

    gpu = manifest["gpu"]
    if gpu.get("available"):
        print(f"  GPU  {gpu['name']}  sm{gpu['capability'].replace('.', '')}  "
              f"{gpu['vram_gb']}GB  cuda {gpu['cuda']}  bf16={gpu['bf16_supported']}")
        if not gpu["bf16_supported"]:
            print("  NOTE bf16 unsupported on this card - fp16 path required")
    else:
        print("  GPU  none")
    return manifest


if __name__ == "__main__":
    print(json.dumps(env_manifest(), indent=2))
