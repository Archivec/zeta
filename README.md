# Zeta Dataset (Python Generator)

This repository provides a Python + uv workflow for generating an edit‑prediction dataset. You supply short code excerpts ("seeds") and generate training examples with OpenAI GPT‑5.2. The resulting dataset keeps Zeta‑style markers and is suitable for fine‑tuning models such as `qwen3-coder-30b-a3b-instruct`.

## Dataset Format

Each example is a Markdown file containing these tags, in this order:

- `<events>`: A short description of the user edit, including a unified diff.
- `<input>`: The original excerpt with editable region markers and one cursor marker.
- `<output>`: The edited excerpt with the same editable region markers and **no** cursor marker.
- `<assertions>` (optional): One short sentence validating the change.
- `<labels>`: Two comma-separated labels (edit location + intent).

Marker rules:
- `<input>` must contain exactly one `<|editable_region_start|>`, one `<|editable_region_end|>`, and one `<|user_cursor_is_here|>`.
- `<output>` must contain exactly one `<|editable_region_start|>` and one `<|editable_region_end|>`, and **no** `<|user_cursor_is_here|>`.
- Content outside the editable region must be identical between `<input>` and `<output>`.
  
Label format:
- Location: `no-op` | `local-edit` | `non-local-edit`
- Intent: `add-imports` | `complete-implementation` | `complete-pattern` | `infer-intent` | `infer-refactor` | `unknown`

## Workflow

0. Install dependencies (uv)

```
uv sync
```

1. Extract seeds automatically

```
uv run python script/extract_seeds.py \\
  --api-repo /path/to/yard-api \\
  --app-repo /path/to/yard-app \\
  --out seeds \\
  --api-count 100 \\
  --app-count 100 \\
  --min-lines 8 \\
  --max-lines 16 \\
  --clean
```

If you want to clone over SSH instead of providing local paths:

```
uv run python script/extract_seeds.py \\
  --api-repo git@github.com:Hyperduality/yard-api.git \\
  --app-repo git@github.com:Hyperduality/yard-app.git \\
  --out seeds \\
  --api-count 100 \\
  --app-count 100 \\
  --min-lines 8 \\
  --max-lines 16 \\
  --clean \\
  --clone
```

When `--clone` is used, repos are cloned into `repos/` (ignored by git).

This samples a variety of files from both repos (controllers, services, schemas, components, hooks, shared, etc.) into `seeds/`.

2. Generate examples with OpenAI

Set your API key (from `.env` or environment):

```
OPENAI_API_KEY=your_key_here
```

Optional:
- `OPENAI_ORGANIZATION_ID`
- `OPENAI_PROJECT_ID`

Run:

```
uv run python script/gen_examples.py --seeds seeds --out . --model gpt-5.2
```

This writes examples into `train/` and `eval/`. Each model response can include **multiple examples** separated by a line containing only `<<<EXAMPLE>>>`.

3. Validate formatting

```
uv run python script/check_format.py
```

4. Generate JSONL datasets

```
uv run python script/gen_dataset.py
```

Outputs:
- `train.jsonl`
- `eval.jsonl`

## Manual Seeds (Optional)

If you want to add manual seeds, use this format:

```
path: relative/path/to/file.ts
---
<raw excerpt here>
```

You can mix manual seed files with auto‑generated ones under `seeds/`.

## Customizing Prompts

Edit `script/prompt-template.md` to change how examples are generated. Placeholders:
- `{{output_path}}`
- `{{path}}`
- `{{excerpt}}`

## Notes

- Prompts, seeds, and generated datasets are ignored by default via `.gitignore`.
- This repo no longer includes Snowflake, DPO, or labeling pipelines.
