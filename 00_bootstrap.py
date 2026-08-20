"""
Stage 0b - scaffold the repo.

Run once from an empty directory. Idempotent: existing files are left alone
and reported as SKIP, so re-running after you have written real code is safe.

    python 00_bootstrap.py
"""

from pathlib import Path

ROOT = Path.cwd()

DIRS = [
    "app",
    "src",
    "configs",
    "eval/splits",
    "eval/audit",
    "reports/runs",
    "notebooks",
    "docs",
    "tests",
    "artifacts/adapters",
    "artifacts/predictions",
]

# Numbered files are pipeline stages, unnumbered are libraries - same
# convention as project 2 so the two repos read alike.
STUBS = {
    "src/01_build_dataset.py": "build frozen train/dev/test splits + dedup report",
    "src/02_audit_sample.py": "draw the stratified hand-audit sample",
    "src/03_baselines.py": "zero-shot, few-shot, retrieval-prompted, encoder",
    "src/04_train_lora.py": "one training run, every knob a CLI arg",
    "src/05_noise_floor.py": "same config, k seeds, measure sigma_seed",
    "src/06_ablations.py": "rank / lr / dataset-size grid",
    "src/07_taxonomy.py": "error classification over the dev split",
    "src/08_cost_latency.py": "tokens, wall clock, $/1k predictions",
    "src/dataset.py": "loading, splitting, subsampling, hashing",
    "src/evaluator.py": "verdicts; tested in both directions",
    "src/prompts.py": "every prompt template, versioned",
    "src/inference.py": "generation wrapper (transformers / vLLM)",
    "src/runner.py": "the shared run loop every stage uses",
    "tests/test_evaluator.py": "evaluator self-tests",
    "tests/test_dataset.py": "split integrity: no leakage, stable hashes",
}

REQ_LOCAL = """\
# Local Windows box: authoring, splits, analysis, plots. CPU only.
# Do NOT install GPU torch here - training happens on Kaggle.
numpy>=1.26
pandas>=2.2
scikit-learn>=1.5
datasets>=2.19
transformers>=4.44
matplotlib>=3.8
pyarrow>=16.0
pytest>=8.0
python-dotenv>=1.0
"""

REQ_GPU = """\
# Kaggle T4 image. Pin these in the notebook's first cell.
# Turing (sm75) has no bf16 and no flash-attn-2: fp16 only.
torch  # supplied by the Kaggle image - do not reinstall
transformers>=4.44
peft>=0.12
trl>=0.9
bitsandbytes>=0.43
accelerate>=0.33
datasets>=2.19
"""

GITIGNORE = """\
.venv/
__pycache__/
*.pyc
.env
artifacts/adapters/*
!artifacts/adapters/.gitkeep
*.safetensors
*.bin
.ipynb_checkpoints/
"""

README = """\
# LoRA Fine-Tune with a Real Experimental Design

Placeholder. The README gets written when there are numbers in it.

Rules carried over from project 2:
- the evaluation exists before the thing it evaluates
- the noise floor is measured before any delta is believed
- what a change breaks is reported alongside what it fixes
- no intervention is tuned against observed failures
- outcomes are predicted in writing before the run
"""


def write(rel, body):
    p = ROOT / rel
    if p.exists():
        print(f"  SKIP  {rel}")
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    print(f"  WRITE {rel}")


def main():
    print(f"Scaffolding into {ROOT}\n")

    print("directories")
    for d in DIRS:
        path = ROOT / d
        existed = path.exists()
        path.mkdir(parents=True, exist_ok=True)
        (path / ".gitkeep").touch(exist_ok=True)
        print(f"  {'SKIP ' if existed else 'MKDIR'} {d}/")

    print("\nstubs")
    for rel, purpose in STUBS.items():
        write(rel, f'"""{purpose}"""\n')

    print("\nproject files")
    write("requirements.txt", REQ_LOCAL)
    write("requirements-gpu.txt", REQ_GPU)
    write(".gitignore", GITIGNORE)
    write("README.md", README)

    print("\nenvironment check")
    import sys
    print(f"  python {sys.version.split()[0]}  ({sys.executable})")
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    print(f"  virtualenv active: {in_venv}")
    if not in_venv:
        print("  !! not in a venv - stop and create one before pip installing")

    for mod in ["numpy", "pandas", "sklearn", "datasets", "transformers"]:
        try:
            m = __import__(mod)
            print(f"  {mod:14s} {getattr(m, '__version__', '?')}")
        except ImportError:
            print(f"  {mod:14s} MISSING")

    print("\nnext: pip install -r requirements.txt, then python 01_dataset_probe.py")


if __name__ == "__main__":
    main()
