#!/usr/bin/env python3
"""
Safely mirror clean publishable files from the workspace directory to the
GitHub Desktop repository directory, strictly excluding secrets (.env),
raw scraped caches (data/raw/), Python bytecode, and internal status notes.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

WORKSPACE_DIR = Path(r"D:\Documents\antigravity-workspace\kaoyan-english-sprint")
DEFAULT_REPO_DIR = Path(r"D:\Documents\GitHub\kaoyan")

EXCLUDE_TOP = {".git", ".env", ".env.local", "__pycache__", ".venv", "tmp", "status.md", "AGENTS.md"}
EXCLUDE_FILENAMES = {"CONTENT_SPEC.md"}
EXCLUDE_REL_PREFIXES = ("data/raw", "scripts/__pycache__")
STALE_FILES_TO_REMOVE = (".github/workflows/pages.yml",)


def should_exclude(rel: Path) -> bool:
    parts = rel.parts
    if not parts:
        return True
    if parts[0] in EXCLUDE_TOP:
        return True
    if rel.name in EXCLUDE_FILENAMES:
        return True
    if any(p == "__pycache__" or p.endswith(".pyc") for p in parts):
        return True
    rel_posix = rel.as_posix()
    for prefix in EXCLUDE_REL_PREFIXES:
        if rel_posix == prefix or rel_posix.startswith(prefix + "/"):
            return True
    return False


def sync(src: Path = WORKSPACE_DIR, dst: Path = DEFAULT_REPO_DIR) -> int:
    if not dst.exists():
        raise FileNotFoundError(f"Target GitHub repo directory does not exist: {dst}")

    copied = 0
    # Copy .gitignore first so git immediately ignores local untracked patterns
    gitignore = src / ".gitignore"
    if gitignore.exists():
        shutil.copy2(gitignore, dst / ".gitignore")
        copied += 1

    for path in sorted(src.rglob("*")):
        rel = path.relative_to(src)
        if rel.as_posix() == ".gitignore" or should_exclude(rel):
            continue
        target = dst / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            copied += 1

    # Remove obsolete GitHub Actions Pages workflow so pure branch deploy runs cleanly
    for stale_rel in STALE_FILES_TO_REMOVE:
        stale_path = dst / stale_rel
        if stale_path.exists():
            stale_path.unlink()
            print(f"[CLEAN] Removed obsolete file in target repo: {stale_rel}")
            parent = stale_path.parent
            while parent != dst and parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
                parent = parent.parent

    # Double-check forbidden files do NOT exist in destination
    for forbidden in (dst / ".env", dst / "data" / "raw", dst / ".venv", dst / "tmp", dst / "status.md"):
        if forbidden.exists():
            raise RuntimeError(f"SECURITY GUARD: Forbidden path exists in target repo: {forbidden}")

    print(f"[OK] Synced {copied} clean files from {src} -> {dst}")
    print("[OK] Verified .env, data/raw, .venv, tmp, and status.md are NOT present in target repo.")
    return copied


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync clean workspace files to GitHub repo.")
    parser.add_argument("--dst", type=Path, default=DEFAULT_REPO_DIR)
    args = parser.parse_args()
    sync(dst=args.dst)
