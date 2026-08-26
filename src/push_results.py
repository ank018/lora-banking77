"""
Persist run artefacts from a Kaggle session back to the repo.

/kaggle/working survives a kernel restart but is discarded when an
interactive session ends. A multi-hour pass that vanishes at session end
is worse than one never started, so results are pushed as soon as each
configuration completes rather than collected at the end of the day.

Chain it onto the run so persistence is not a step anyone can forget:

    !python src/04a_baselines.py --runs-dir /kaggle/working/runs \\
        --configs few_shot_k77 \\
      && python src/push_results.py --runs-dir /kaggle/working/runs \\
        --message "results: few_shot_k77"

Auth: create a fine-grained GitHub token with Contents read/write on this
repository only, then add it in the Kaggle notebook sidebar under
Add-ons -> Secrets with the name GITHUB_TOKEN. The token is read into
memory and never printed, never written to disk, and never committed - the
remote URL containing it is reset before the script exits.

If no token is available the script falls back to writing a timestamped
zip into /kaggle/working for manual download, and says so.
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_RUNS = Path("reports/runs")
DEFAULT_REMOTE = "https://github.com/ank018/lora-banking77.git"


def sh(cmd, check=True, quiet=False):
    """Run a git command. Never echoes the command, which may hold a token."""
    r = subprocess.run(cmd, capture_output=True, text=True)
    if not quiet and r.stdout.strip():
        print("   " + r.stdout.strip().replace("\n", "\n   "))
    if r.returncode and check:
        err = r.stderr.strip()
        raise RuntimeError(f"git failed ({r.returncode}): {err[:400]}")
    return r


def get_token():
    import os
    if os.environ.get("GITHUB_TOKEN"):
        return os.environ["GITHUB_TOKEN"], "environment"
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("GITHUB_TOKEN"), "kaggle secret"
    except Exception:  # noqa: BLE001
        return None, None


def dir_size_mb(p):
    return sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file()) / 1e6


def stage_results(runs_dir):
    """Copy run artefacts into the repo, reporting what moved."""
    src = Path(runs_dir)
    if not src.exists():
        raise FileNotFoundError(f"{src} does not exist - nothing to push")
    REPO_RUNS.mkdir(parents=True, exist_ok=True)

    moved = []
    for d in sorted(src.iterdir()):
        if not d.is_dir():
            continue
        dest = REPO_RUNS / d.name
        shutil.copytree(d, dest, dirs_exist_ok=True)
        pred = dest / "predictions.jsonl"
        rows = sum(1 for _ in open(pred, encoding="utf-8")) if pred.exists() else 0
        moved.append((d.name, rows, dir_size_mb(dest)))
    return moved


def zip_fallback(runs_dir):
    """Timestamped archive beside the runs directory.

    Kaggle surfaces anything under /kaggle/working in the sidebar, so that
    is preferred when it exists; otherwise the archive lands next to the
    results, which is what you want on a laptop.
    """
    stamp = time.strftime("%Y%m%d-%H%M")
    kaggle = Path("/kaggle/working")
    dest = kaggle if kaggle.is_dir() else Path(runs_dir).resolve().parent
    archive = shutil.make_archive(str(dest / f"runs_{stamp}"), "zip", runs_dir)
    print(f"\n  no GITHUB_TOKEN available - wrote {archive}")
    if dest == kaggle:
        print("  download it from the notebook sidebar: Data -> Output")
    return archive


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="/kaggle/working/runs")
    ap.add_argument("--message", default="results: baseline run")
    ap.add_argument("--remote", default=DEFAULT_REMOTE)
    ap.add_argument("--name", default="ank018")
    ap.add_argument("--email", default="ank018@users.noreply.github.com")
    ap.add_argument("--zip-only", action="store_true")
    args = ap.parse_args()

    print(f"staging from {args.runs_dir}")
    moved = stage_results(args.runs_dir)
    if not moved:
        print("  nothing to stage")
        return
    for name, rows, mb in moved:
        print(f"  {name:34s} {rows:6,d} rows   {mb:6.2f} MB")
    print(f"  reports/runs total: {dir_size_mb(REPO_RUNS):.2f} MB")

    if args.zip_only:
        zip_fallback(args.runs_dir)
        return

    token, source = get_token()
    if not token:
        zip_fallback(args.runs_dir)
        return
    print(f"  token from {source}")

    sh(["git", "config", "user.name", args.name])
    sh(["git", "config", "user.email", args.email])

    authed = args.remote.replace("https://", f"https://x-access-token:{token}@")
    try:
        sh(["git", "remote", "set-url", "origin", authed], quiet=True)

        sh(["git", "add", "reports/runs"])
        staged = sh(["git", "diff", "--cached", "--name-only"], quiet=True)
        if not staged.stdout.strip():
            print("  no changes to commit - results already pushed")
            return
        n_files = len(staged.stdout.strip().splitlines())
        print(f"  committing {n_files} file(s)")

        sh(["git", "commit", "-m", args.message])
        # Someone may have pushed from the laptop meanwhile.
        sh(["git", "pull", "--rebase", "origin", "main"], check=False)
        sh(["git", "push", "origin", "HEAD:main"], quiet=True)
        print(f"  pushed: {args.message}")
    finally:
        # Leave no token in .git/config, even if the push failed.
        sh(["git", "remote", "set-url", "origin", args.remote], check=False,
           quiet=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        print(f"\n  FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        print("  results are still in --runs-dir; rerun with --zip-only "
              "to download them manually", file=sys.stderr)
        sys.exit(1)
