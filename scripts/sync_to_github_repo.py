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
TEXT_SUFFIXES = {
    ".json", ".js", ".css", ".html", ".md", ".py", ".txt", ".yml", ".yaml",
    ".svg", ".xml", ".csv", ".tsv", ".sh", ".ps1",
}
TEXT_NAMES = {".gitignore", ".gitattributes", ".nojekyll", ".env.example"}


def copy_normalized(src_file: Path, dst_file: Path) -> None:
    """Copy file to target repo, normalizing CRLF/CR to LF for text files to match .gitattributes."""
    if src_file.suffix.lower() in TEXT_SUFFIXES or src_file.name in TEXT_NAMES:
        raw = src_file.read_bytes()
        norm = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        dst_file.write_bytes(norm)
    else:
        shutil.copy2(src_file, dst_file)


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
    # Copy .gitignore and .gitattributes first so git immediately applies ignore & LF rules
    for meta_name in (".gitignore", ".gitattributes"):
        meta_file = src / meta_name
        if meta_file.exists():
            copy_normalized(meta_file, dst / meta_name)
            copied += 1

    for path in sorted(src.rglob("*")):
        rel = path.relative_to(src)
        if rel.as_posix() in {".gitignore", ".gitattributes"} or should_exclude(rel):
            continue
        target = dst / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            copy_normalized(path, target)
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
