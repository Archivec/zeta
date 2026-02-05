#!/usr/bin/env python3

import re
from pathlib import Path


def process_file(file_path: Path) -> list[str]:
    content = file_path.read_text(encoding="utf-8")

    input_match = re.search(r"<input>([\s\S]*?)</input>", content)
    output_match = re.search(r"<output>([\s\S]*?)</output>", content)
    events_match = re.search(r"<events>([\s\S]*?)</events>", content)
    assertions_match = re.search(r"<assertions>([\s\S]*?)</assertions>", content)
    labels_match = re.search(r"<labels>([\s\S]*?)</labels>", content)

    missing = []
    if not input_match:
        missing.append("input")
    if not output_match:
        missing.append("output")
    if not events_match:
        missing.append("events")
    # assertions are optional
    if not labels_match:
        missing.append("labels")
    if missing:
        return [f"missing tags: {', '.join(missing)}"]

    input_content = input_match.group(1)
    output_content = output_match.group(1)
    assertions_content = assertions_match.group(1) if assertions_match else ""
    labels_content = labels_match.group(1)

    issues = []
    if not input_content.strip():
        issues.append("input empty")
    if not output_content.strip():
        issues.append("output empty")
    # assertions are optional
    if not labels_content.strip():
        issues.append("labels empty")

    editable_start_count = len(re.findall(r"<\|editable_region_start\|>", input_content))
    editable_end_count = len(re.findall(r"<\|editable_region_end\|>", input_content))
    cursor_count = len(re.findall(r"<\|user_cursor_is_here\|>", input_content))

    if editable_start_count != 1 or editable_end_count != 1 or cursor_count != 1:
        issues.append(
            f"input markers: start={editable_start_count}, end={editable_end_count}, cursor={cursor_count}"
        )

    output_editable_start_count = len(re.findall(r"<\|editable_region_start\|>", output_content))
    output_editable_end_count = len(re.findall(r"<\|editable_region_end\|>", output_content))
    output_cursor_count = len(re.findall(r"<\|user_cursor_is_here\|>", output_content))

    if output_editable_start_count != 1 or output_editable_end_count != 1 or output_cursor_count != 0:
        issues.append(
            f"output markers: start={output_editable_start_count}, end={output_editable_end_count}, cursor={output_cursor_count}"
        )

    if editable_start_count == 1 and output_editable_start_count == 1:
        input_pre = input_content.split("<|editable_region_start|>")[0]
        output_pre = output_content.split("<|editable_region_start|>")[0]
        if input_pre != output_pre:
            issues.append("mismatch before editable region")

    if editable_end_count == 1 and output_editable_end_count == 1:
        input_post = input_content.split("<|editable_region_end|>")[1]
        output_post = output_content.split("<|editable_region_end|>")[1]
        if input_post != output_post:
            issues.append("mismatch after editable region")

    return issues


def process_directory(dir_path: Path) -> tuple[int, list[tuple[Path, list[str]]]]:
    count = 0
    problems: list[tuple[Path, list[str]]] = []
    for file_path in dir_path.iterdir():
        if file_path.is_file() and file_path.suffix == ".md":
            issues = process_file(file_path)
            if issues:
                problems.append((file_path, issues))
            count += 1
    print(f"Checked {count} file(s) in {dir_path}/")
    return count, problems


def main() -> None:
    all_problems: list[tuple[Path, list[str]]] = []
    train_path = Path("train")
    if train_path.exists():
        print("Validating train/ ...")
        _, problems = process_directory(train_path)
        all_problems.extend(problems)

    train_original_path = Path("train_original")
    if train_original_path.exists():
        print("Validating train_original/ ...")
        _, problems = process_directory(train_original_path)
        all_problems.extend(problems)

    eval_path = Path("eval")
    if eval_path.exists():
        print("Validating eval/ ...")
        _, problems = process_directory(eval_path)
        all_problems.extend(problems)

    if all_problems:
        print("\nValidation issues:")
        for path, issues in all_problems:
            rel = path.as_posix()
            issue_text = "; ".join(issues)
            print(f"- {rel}: {issue_text}")
        raise SystemExit(1)
    else:
        print("\nAll files passed validation.")


if __name__ == "__main__":
    main()
