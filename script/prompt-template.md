# Dataset Example Generation

You are generating multiple training examples for a code-edit prediction dataset.

Target output path (base): {{output_path}}

File path: {{path}}

Excerpt:

{{excerpt}}

Requirements:
- Output ONLY dataset examples. No commentary.
- Generate 2 to 4 examples from this excerpt. Each example should reflect a different plausible cursor stop and completion intent.
- Separate each example with a line containing only: `<<<EXAMPLE>>>`
- Each example MUST use exactly these tags in this order: `<events>`, `<input>`, `<output>`, `<assertions>`, `<labels>`.
- `<events>` must describe a plausible user edit to `{{path}}` and include a small unified diff in a fenced `diff` block.
- `<input>` must include a short excerpt (8–16 lines) wrapped in a fenced code block labeled with `{{path}}`.
- Insert `<|editable_region_start|>` and `<|editable_region_end|>` inside the excerpt to bound the edit region.
- Insert exactly one `<|user_cursor_is_here|>` inside the editable region.
- `<output>` must contain the edited version of the excerpt with the same code fence label and the same editable region markers.
- `<output>` MUST NOT include `<|user_cursor_is_here|>`.
- Content outside the editable region must be identical between `<input>` and `<output>`.
- `<assertions>` must contain one short sentence that validates the change.
- `<labels>` must be two comma-separated labels: one from {`no-op`, `local-edit`, `non-local-edit`} and one from {`add-imports`, `complete-implementation`, `complete-pattern`, `infer-intent`, `infer-refactor`, `unknown`}.

Return the examples now.
