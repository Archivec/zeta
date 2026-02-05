#!/usr/bin/env python3

import argparse
import hashlib
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Set


EXCLUDE_DIRS = {
    ".git",
    "node_modules",
    "dist",
    "build",
    "coverage",
    "assets",
    "fonts",
    "icons",
    "migrations",
    "sql",
    "generated",
}

INCLUDE_EXTS = {".ts", ".tsx", ".js", ".jsx", ".prisma"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract seed excerpts from repositories.")
    parser.add_argument("--api-repo", required=True, help="Path or git URL for yard-api")
    parser.add_argument("--app-repo", required=True, help="Path or git URL for yard-app")
    parser.add_argument("--out", default="seeds", help="Output seeds directory")
    parser.add_argument("--api-count", type=int, default=100, help="Number of seeds from api repo")
    parser.add_argument("--app-count", type=int, default=100, help="Number of seeds from app repo")
    parser.add_argument("--min-lines", type=int, default=8, help="Minimum lines per excerpt")
    parser.add_argument("--max-lines", type=int, default=16, help="Maximum lines per excerpt")
    parser.add_argument("--seed", default="default", help="Seed for deterministic sampling")
    parser.add_argument("--clone", action="store_true", help="Clone repos if paths are missing and inputs are URLs")
    parser.add_argument("--clean", action="store_true", help="Remove output directory before writing")
    return parser.parse_args()


def is_git_url(value: str) -> bool:
    return value.startswith("git@") or value.startswith("https://") or value.startswith("ssh://")


def repo_name_from(value: str) -> str:
    cleaned = re.sub(r"\.git$", "", value)
    if "/" in cleaned:
        return cleaned.split("/")[-1]
    if ":" in cleaned:
        return cleaned.split(":")[-1]
    return cleaned


def resolve_repo_path(value: str, clone: bool, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if path.exists():
        return path.resolve()

    if not is_git_url(value):
        raise ValueError(f"Repo path not found: {value}")

    if not clone:
        raise ValueError(f"Repo not found locally: {value}. Use --clone to clone it.")

    base_dir.mkdir(parents=True, exist_ok=True)
    target = base_dir / repo_name_from(value)
    if target.exists():
        return target.resolve()

    print(f"Cloning {value} into {target}...")
    subprocess.run(["git", "clone", value, str(target)], check=True)
    return target.resolve()


class XorShift32:
    def __init__(self, seed: str) -> None:
        digest = hashlib.sha256(seed.encode("utf-8")).digest()
        self.state = int.from_bytes(digest[:4], "big") or 1

    def next(self) -> float:
        x = self.state & 0xFFFFFFFF
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= (x >> 17) & 0xFFFFFFFF
        x ^= (x << 5) & 0xFFFFFFFF
        self.state = x & 0xFFFFFFFF
        return self.state / 2**32

    def int_between(self, min_value: int, max_value: int) -> int:
        if max_value < min_value:
            return min_value
        return min_value + int(self.next() * (max_value - min_value + 1))


def walk_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for current_root, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for filename in filenames:
            files.append(Path(current_root) / filename)
    return files


def should_include_file(path: Path) -> bool:
    if path.suffix not in INCLUDE_EXTS:
        return False
    if path.name.endswith(".d.ts"):
        return False
    return True


def categorize_api(rel_path: str) -> str:
    rel_path = rel_path.replace("\\", "/")
    if "/controllers/" in rel_path:
        return "controllers"
    if "/services/" in rel_path:
        return "services"
    if "/schemas/" in rel_path:
        return "schemas"
    if "/mappers/" in rel_path:
        return "mappers"
    if "/data/" in rel_path:
        return "data"
    if "/integrations/" in rel_path:
        return "integrations"
    if "/modules/" in rel_path:
        return "modules"
    if "/shared/" in rel_path:
        return "shared"
    if "/prisma/" in rel_path:
        return "prisma"
    if rel_path.startswith("scripts/"):
        return "scripts"
    return "other"


def categorize_app(rel_path: str) -> str:
    rel_path = rel_path.replace("\\", "/")
    if "/components/" in rel_path:
        return "components"
    if "/hooks/" in rel_path:
        return "hooks"
    if "/shared/" in rel_path:
        return "shared"
    if "/modules/" in rel_path:
        return "modules"
    if rel_path.startswith("src/app/"):
        return "screens"
    if "/stores/" in rel_path:
        return "stores"
    if "/utils/" in rel_path:
        return "utils"
    if "/api/" in rel_path:
        return "api"
    return "other"


def select_files_by_category(repo_root: Path, repo_type: str) -> Dict[str, List[Path]]:
    files_by_category: Dict[str, List[Path]] = {}
    for file_path in walk_files(repo_root):
        if not should_include_file(file_path):
            continue
        rel_path = str(file_path.relative_to(repo_root))
        category = categorize_api(rel_path) if repo_type == "api" else categorize_app(rel_path)
        files_by_category.setdefault(category, []).append(file_path)
    return files_by_category


def extract_excerpt(
    file_path: Path,
    min_lines: int,
    max_lines: int,
    rng: XorShift32,
    used_slices: Set[str],
) -> str | None:
    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    if "\x00" in text:
        return None

    lines = text.splitlines()
    if not lines:
        return None

    target_len = min(len(lines), rng.int_between(min_lines, max_lines))
    attempts = 0
    while attempts < 10:
        start = 0 if len(lines) == target_len else rng.int_between(0, len(lines) - target_len)
        key = f"{file_path}:{start}:{target_len}"
        if key in used_slices:
            attempts += 1
            continue

        slice_lines = lines[start : start + target_len]
        while slice_lines and slice_lines[0].strip() == "":
            slice_lines.pop(0)
        while slice_lines and slice_lines[-1].strip() == "":
            slice_lines.pop()
        if not slice_lines:
            attempts += 1
            continue

        used_slices.add(key)
        return "\n".join(slice_lines)

    return None


def write_seed(out_dir: Path, repo_name: str, category: str, index: int, rel_path: str, excerpt: str) -> None:
    target_dir = out_dir / repo_name / category
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = f"seed-{index:04d}.md"
    normalized_rel = rel_path.replace(os.sep, "/")
    content = f"path: {repo_name}/{normalized_rel}\n---\n{excerpt}\n"
    (target_dir / filename).write_text(content, encoding="utf-8")


def sample_seeds(
    repo_root: Path,
    repo_name: str,
    repo_type: str,
    count: int,
    min_lines: int,
    max_lines: int,
    rng: XorShift32,
    out_dir: Path,
) -> int:
    files_by_category = select_files_by_category(repo_root, repo_type)
    categories = [cat for cat, files in files_by_category.items() if files]
    if not categories:
        raise ValueError(f"No source files found in {repo_root}")

    used_slices: Set[str] = set()
    index = 1
    category_index = 0
    guard = 0

    while index <= count and guard < count * 20:
        category = categories[category_index % len(categories)]
        files = files_by_category[category]
        file_path = files[rng.int_between(0, len(files) - 1)]
        excerpt = extract_excerpt(file_path, min_lines, max_lines, rng, used_slices)
        if excerpt:
            rel_path = str(file_path.relative_to(repo_root))
            write_seed(out_dir, repo_name, category, index, rel_path, excerpt)
            index += 1
        category_index += 1
        guard += 1

    if index <= count:
        print(f"Warning: only generated {index - 1} seeds for {repo_name} (requested {count}).")

    return index - 1


def main() -> None:
    args = parse_args()

    if args.min_lines <= 0 or args.max_lines < args.min_lines:
        raise ValueError("--min-lines and --max-lines must be positive and max >= min")
    if args.api_count <= 0 or args.app_count <= 0:
        raise ValueError("--api-count and --app-count must be positive numbers")

    out_dir = Path(args.out)
    if args.clean and out_dir.exists():
        print(f"Cleaning output directory: {out_dir}")
        for root, dirnames, filenames in os.walk(out_dir, topdown=False):
            for filename in filenames:
                Path(root, filename).unlink(missing_ok=True)
            for dirname in dirnames:
                Path(root, dirname).rmdir()
        out_dir.rmdir()

    base_clone_dir = Path("repos")
    print("Resolving repositories...")
    api_repo = resolve_repo_path(args.api_repo, args.clone, base_clone_dir)
    app_repo = resolve_repo_path(args.app_repo, args.clone, base_clone_dir)

    api_name = repo_name_from(args.api_repo)
    app_name = repo_name_from(args.app_repo)

    out_dir.mkdir(parents=True, exist_ok=True)

    rng_api = XorShift32(f"{args.seed}|api")
    rng_app = XorShift32(f"{args.seed}|app")

    print(f"Sampling seeds from {api_name} ({api_repo})")
    api_count = sample_seeds(
        api_repo,
        api_name,
        "api",
        args.api_count,
        args.min_lines,
        args.max_lines,
        rng_api,
        out_dir,
    )
    print(f"Sampling seeds from {app_name} ({app_repo})")
    app_count = sample_seeds(
        app_repo,
        app_name,
        "app",
        args.app_count,
        args.min_lines,
        args.max_lines,
        rng_app,
        out_dir,
    )

    print(f"Generated {api_count} seeds for {api_name} and {app_count} seeds for {app_name} in {out_dir}/")


if __name__ == "__main__":
    main()
