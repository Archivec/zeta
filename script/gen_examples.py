#!/usr/bin/env python3

import argparse
import hashlib
import os
import re
import time
from pathlib import Path

from typing import List

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionUserMessageParam

EXAMPLE_DELIMITER = "<<<EXAMPLE>>>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dataset examples using OpenAI.")
    parser.add_argument("--seeds", default="seeds", help="Seeds directory")
    parser.add_argument("--out", default=".", help="Output root directory")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Train split ratio")
    parser.add_argument("--seed", default="default", help="Seed for deterministic split")
    parser.add_argument("--model", default="gpt-5.2", help="Model name")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max completion tokens")
    parser.add_argument("--max-completion-tokens", type=int, help="Max completion tokens (alias)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--sleep-ms", type=int, default=0, help="Delay between requests in ms")
    parser.add_argument("--limit", type=int, help="Limit number of seeds processed")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    return parser.parse_args()


def list_seed_files(root: Path) -> list[Path]:
    if not root.exists():
        raise ValueError(f"Seeds directory not found: {root}")
    return sorted([p for p in root.rglob("*") if p.is_file()])


def parse_seed(content: str, path: Path) -> tuple[str, str]:
    lines = content.splitlines()
    if len(lines) < 3:
        raise ValueError(f"Seed file too short: {path}")

    first_line = lines[0]
    if not first_line.startswith("path:"):
        raise ValueError(f"Missing 'path:' header in {path}")

    target_path = first_line[len("path:") :].strip()
    if not target_path:
        raise ValueError(f"Empty path header in {path}")

    delimiter_index = next((i for i, line in enumerate(lines[1:], start=1) if line.strip() == "---"), -1)
    if delimiter_index == -1:
        raise ValueError(f"Missing '---' delimiter in {path}")

    excerpt = "\n".join(lines[delimiter_index + 1 :]).rstrip()
    if not excerpt.strip():
        raise ValueError(f"Empty excerpt in {path}")

    return target_path, excerpt


def deterministic_split(key: str, seed: str, train_ratio: float) -> str:
    digest = hashlib.sha256(f"{seed}|{key}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:4], "big") / 2**32
    return "train" if value < train_ratio else "eval"


def split_examples(text: str) -> list[str]:
    parts = re.split(r"\n\s*" + re.escape(EXAMPLE_DELIMITER) + r"\s*\n", text)
    parts = [part.strip() for part in parts if part.strip()]
    return parts if parts else [text.strip()]


def has_required_tags(text: str) -> bool:
    return (
        re.search(r"<events>[\s\S]*?</events>", text) is not None
        and re.search(r"<input>[\s\S]*?</input>", text) is not None
        and re.search(r"<output>[\s\S]*?</output>", text) is not None
        and re.search(r"<assertions>[\s\S]*?</assertions>", text) is not None
        and re.search(r"<labels>[\s\S]*?</labels>", text) is not None
    )


def next_available_index(directory: Path, index: int, overwrite: bool) -> int:
    if overwrite:
        return index
    while (directory / f"{index:04d}.md").exists():
        index += 1
    return index


def main() -> None:
    args = parse_args()

    if args.train_ratio <= 0 or args.train_ratio >= 1:
        raise ValueError("--train-ratio must be between 0 and 1")

    max_tokens = args.max_completion_tokens or args.max_tokens
    if max_tokens <= 0:
        raise ValueError("--max-tokens must be a positive number")

    if args.temperature < 0:
        raise ValueError("--temperature must be non-negative")

    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not set. Add it to your environment or .env file.")

    project_id = os.getenv("OPENAI_PROJECT_ID") or os.getenv("OEPNAI_PROJECT_ID")
    if os.getenv("OEPNAI_PROJECT_ID") and not os.getenv("OPENAI_PROJECT_ID"):
        print("Warning: using OEPNAI_PROJECT_ID (typo). Consider renaming to OPENAI_PROJECT_ID.")

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        organization=os.getenv("OPENAI_ORGANIZATION_ID"),
        project=project_id,
    )

    template_path = Path(__file__).parent / "prompt-template.md"
    print(f"Loading template: {template_path}")
    template = template_path.read_text(encoding="utf-8")

    seed_files = list_seed_files(Path(args.seeds))
    if not seed_files:
        raise ValueError(f"No seed files found under {args.seeds}")

    if args.limit:
        seed_files = seed_files[: args.limit]
    print(f"Processing {len(seed_files)} seed file(s) with model={args.model}")

    out_root = Path(args.out)
    out_train = out_root / "train"
    out_eval = out_root / "eval"
    out_train.mkdir(parents=True, exist_ok=True)
    out_eval.mkdir(parents=True, exist_ok=True)

    train_index = 1
    eval_index = 1
    processed = 0

    for idx, seed_file in enumerate(seed_files, start=1):
        print(f"[{idx}/{len(seed_files)}] Generating from {seed_file}")
        content = seed_file.read_text(encoding="utf-8")
        target_path, excerpt = parse_seed(content, seed_file)

        split = deterministic_split(str(seed_file), args.seed, args.train_ratio)
        if split == "train":
            train_index = next_available_index(out_train, train_index, args.overwrite)
            output_base = f"train/{train_index:04d}.md"
        else:
            eval_index = next_available_index(out_eval, eval_index, args.overwrite)
            output_base = f"eval/{eval_index:04d}.md"

        prompt = (
            template.replace("{{output_path}}", output_base)
            .replace("{{path}}", target_path)
            .replace("{{excerpt}}", excerpt)
        )

        messages: List[ChatCompletionUserMessageParam] = [
            {"role": "user", "content": prompt}
        ]
        response = client.chat.completions.create(
            model=args.model,
            messages=messages,
            max_completion_tokens=max_tokens,
            temperature=args.temperature,
        )

        output_text = (response.choices[0].message.content or "").strip()
        if not output_text:
            print(f"Empty response for {seed_file}, skipping.")
            continue

        examples = split_examples(output_text)
        written = 0

        for example in examples:
            if not has_required_tags(example):
                print(f"Skipping invalid example for {seed_file} (missing tags).")
                continue

            if split == "train":
                train_index = next_available_index(out_train, train_index, args.overwrite)
                output_path = out_train / f"{train_index:04d}.md"
                train_index += 1
            else:
                eval_index = next_available_index(out_eval, eval_index, args.overwrite)
                output_path = out_eval / f"{eval_index:04d}.md"
                eval_index += 1

            output_path.write_text(example.strip() + "\n", encoding="utf-8")
            processed += 1
            written += 1

        if written == 0:
            print(f"No valid examples written for {seed_file}.")
        else:
            print(f"Wrote {written} example(s) for {seed_file}.")

        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)

    print(f"Generated {processed} example(s).")


if __name__ == "__main__":
    main()
