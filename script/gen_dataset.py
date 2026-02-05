#!/usr/bin/env python3

import json
import re
import subprocess
import sys
from pathlib import Path


def get_files(dir_path: Path) -> list[Path]:
    files: list[Path] = []
    for path in dir_path.rglob("*"):
        if path.is_file() and path.suffix == ".md":
            files.append(path)
    return files


def get_tag_content(content: str, tag: str) -> str:
    match = re.search(rf"<{tag}>([\s\S]*?)</{tag}>", content)
    return match.group(1).strip() if match else ""


def process_directories(directories: list[Path], output_file: Path) -> None:
    sources = ", ".join(f"{directory}/" for directory in directories)
    print(f"Generating {output_file} from {sources}")
    output_file.write_text("", encoding="utf-8")
    count = 0
    for directory in directories:
        if not directory.exists():
            print(f"Skipping {directory}/ (not found)")
            continue
        for file_path in get_files(directory):
            content = file_path.read_text(encoding="utf-8")
            events = get_tag_content(content, "events")
            input_text = get_tag_content(content, "input")
            output_text = get_tag_content(content, "output")
            assertions = get_tag_content(content, "assertions")
            labels = get_tag_content(content, "labels")
            json_line = json.dumps(
                {
                    "events": events,
                    "input": input_text,
                    "output": output_text,
                    "assertions": assertions,
                    "labels": labels,
                }
            )
            with output_file.open("a", encoding="utf-8") as handle:
                handle.write(json_line + "\n")
            count += 1
    print(f"Wrote {count} examples to {output_file}")


def main() -> None:
    check_script = Path(__file__).parent / "check_format.py"
    print("Running format check...")
    subprocess.run([sys.executable, str(check_script)], check=True)
    print("Format check passed.")

    train_dir = Path("train")
    train_original_dir = Path("train_original")
    eval_dir = Path("eval")
    if train_dir.exists() or train_original_dir.exists():
        process_directories(
            [train_dir, train_original_dir],
            Path("train.jsonl"),
        )
    else:
        print("Skipping train.jsonl (train/ and train_original/ not found)")
    if eval_dir.exists():
        process_directories([eval_dir], Path("eval.jsonl"))
    else:
        print("Skipping eval.jsonl (eval/ not found)")


if __name__ == "__main__":
    main()
