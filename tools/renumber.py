"""
One-off: delete unused stubs and align src/ numbering with docs/ stages.

src/ was numbered by pipeline order early on; docs/ is numbered by stage.
The two drifted, so `03_baselines.py` produced `docs/04_baselines.md` and
`04_train_lora.py` produced `docs/06_lora.md`. This makes the leading
number mean the same thing in both places.

Two scripts load others BY FILENAME rather than importing them, because
module names cannot start with a digit:

    05_ablations.py   -> 04_train_lora.py
    inspect_prompts.py -> 03_baselines.py

Renaming without fixing those raises FileNotFoundError at runtime, not at
import, so it would survive a syntax check and fail during a run. They are
rewritten here along with every reference in docs/ and README.md.

Run from the repo root. Dry run by default.

    python tools/renumber.py            # show what would change
    python tools/renumber.py --apply    # do it
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

# Unused stubs: docstring only, never imported, never run.
DELETE = [
    "src/02_audit_sample.py",      # audit sampling was done inline
    "src/05_noise_floor.py",       # seed noise measured inside the trainers
    "src/06_ablations.py",         # superseded by 05_ablations.py
    "src/dataset.py",              # loading stayed inline in each stage
    "src/file_list.csv",           # not source
    "src/.gitkeep",                # src/ has real files now
]

# old -> new. Leading number now matches the docs/ stage number.
RENAME = {
    "src/00_sizing.py":             "src/00a_sizing.py",
    "src/00b_smoke_gpu.py":         "src/00b_smoke_gpu.py",
    "src/01a_probe_dataset.py":     "src/01a_probe_dataset.py",
    "src/01_build_dataset.py":      "src/01b_build_dataset.py",
    "src/03a_check_inference.py":   "src/03a_check_inference.py",
    "src/03_baselines.py":          "src/04a_baselines.py",
    "src/03c_knn_baseline.py":      "src/04b_knn_baseline.py",
    "src/03d_prior_probe.py":       "src/04c_prior_probe.py",
    "src/03b_encoder_baseline.py":  "src/05a_encoder_baseline.py",
    "src/03e_diagnose_encoder.py":  "src/05b_diagnose_encoder.py",
    "src/03f_overfit_test.py":      "src/05c_overfit_test.py",
    "src/03g_isolate_training.py":  "src/05d_isolate_training.py",
    "src/03h_model_vs_pipeline.py": "src/05e_model_vs_pipeline.py",
    "src/04_train_lora.py":         "src/06a_train_lora.py",
    "src/05_ablations.py":          "src/07a_ablations.py",
    "src/07_taxonomy.py":           "src/08a_taxonomy.py",
    "src/08_cost_latency.py":       "src/09a_cost_latency.py",
}

# Files that may mention any old script name.
SCAN = ["docs", "src", "tests", "README.md"]


def git(*args, apply=True):
    if not apply:
        print(f"    would run: git {' '.join(args)}")
        return
    r = subprocess.run(["git", *args], capture_output=True, text=True)
    if r.returncode:
        print(f"    git {' '.join(args)} FAILED: {r.stderr.strip()[:200]}")
    return r


def targets():
    out = []
    for base in SCAN:
        p = Path(base)
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            out.extend(f for f in p.rglob("*")
                       if f.suffix in {".py", ".md"} and f.is_file())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    apply = args.apply

    if not Path("src").is_dir():
        sys.exit("run this from the repo root")

    print("1. delete unused stubs")
    for rel in DELETE:
        if Path(rel).exists():
            print(f"    {rel}")
            git("rm", "-q", "--", rel, apply=apply)
        else:
            print(f"    {rel}  (already gone)")

    print("\n2. rename so src/NN matches docs/NN")
    renames = {o: n for o, n in RENAME.items() if o != n}
    for old, new in renames.items():
        if not Path(old).exists():
            print(f"    {old}  (missing, skipped)")
            continue
        print(f"    {old:32s} -> {new}")
        git("mv", old, new, apply=apply)

    print("\n3. rewrite references")
    # Longest first, so 03b_encoder is not partly matched by 03_baselines.
    pairs = sorted(((Path(o).name, Path(n).name) for o, n in renames.items()),
                   key=lambda p: -len(p[0]))
    changed = 0
    for f in targets():
        if not f.exists():
            continue
        text = original = f.read_text(encoding="utf-8")
        for old_name, new_name in pairs:
            text = re.sub(re.escape(old_name), new_name, text)
        if text != original:
            changed += 1
            hits = sum(1 for o, _ in pairs if o in original)
            print(f"    {f}  ({hits} name(s))")
            if apply:
                f.write_text(text, encoding="utf-8", newline="\n")

    print(f"\n  {changed} file(s) reference renamed scripts")
    print("\n  Load-bearing: 07a_ablations.py loads 06a_train_lora.py and")
    print("  inspect_prompts.py loads 04a_baselines.py, both by filename.")
    print("  Those fail at RUN time, not import time, so verify with:")
    print("    python src/08a_taxonomy.py")
    print("    python src/inspect_prompts.py --configs few_shot_k20")

    if not apply:
        print("\n  DRY RUN. Rerun with --apply to make these changes.")


if __name__ == "__main__":
    main()
