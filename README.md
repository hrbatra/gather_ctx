# gather_ctx

`gather_ctx` bundles source files into a prompt-ready block for LLM workflows.
It copies to clipboard by default, or prints/saves when requested.

It now supports hybrid, budget-aware context selection so you can mix:
- full files,
- deterministic line slices,
- optional symbol-level slices (basic Python/TS/JS extraction),
while preserving provenance in both output and a manifest.

## Install

```bash
# From this repo
uv tool install --from . gather_ctx

# or
pipx install .
```

For direct script use:

```bash
python3 gather_ctx.py --help
```

## Quick Usage

```bash
# Full mode (default): files -> clipboard
gather_ctx src/main.py src/utils.py -q "Explain this code"

# Directory -> recursive expansion
gather_ctx src/ -q "Refactor for clarity"

# Globs
gather_ctx "*.py" -q "Find bugs"

# Print output instead of clipboard
gather_ctx src/main.py --print

# Save output to .ctx/context_YYYYMMDD_HHMMSS.txt
gather_ctx src/ -q "Analyze" --save
```

## Hybrid Selection

```bash
# Budget-aware hybrid selection with inclusion planning only
gather_ctx src/ \
  -q "Trace cityscan internals and API routes" \
  --selection-mode hybrid \
  --budget-tokens 50000 \
  --plan-only

# Write final context + manifest
gather_ctx src/ \
  -q "Focus on route handlers and type contracts" \
  --selection-mode hybrid \
  --budget-tokens 45000 \
  --manifest .ctx/hybrid_manifest.json \
  --save
```

## Interactive Refinement (REPL)

```bash
gather_ctx src/ -q "Auth flow" --selection-mode interactive --budget-tokens 45000
```

Available commands:
- `show`
- `drop <item>` (index, unit id, or path)
- `full <path>`
- `slice <path> <start>-<end>`
- `budget <n>`
- `finalize`

Token stats are recomputed after each command.

## Useful Flags

- `--selection-mode full|hybrid|interactive`
  - `full` is default and preserves existing behavior.
- `--budget-tokens <int>`
  - Used by `hybrid` and `interactive` modes.
  - Defaults to `50000` in those modes if omitted.
- `--plan-only`
  - Show inclusion/exclusion plan and token budget status without producing final output.
- `--manifest <path>`
  - Writes machine-readable selection metadata and provenance.
- `--no-tree`
  - Omit file tree section.
- `--full-tree`
  - Show full directory tree (legacy mode; focused tree is default).
- `--stats` / `--tokens`
  - Show chars, bytes, and token estimate/count.
- `--token-encoding`
  - Set token encoding (default: `cl100k_base`).

## Output Shape

````xml
<context>
<file_tree>...</file_tree>
<files>
<file path="/abs/path/to/file.py">
```python
# full file content
```
</file>
<file path="/abs/path/to/routes.py" kind="slice" line_start="117" line_end="260">
```python
# sliced content
```
</file>
<file path="/abs/path/to/api.ts" kind="symbol" symbol="searchCityScan" line_start="93" line_end="170">
```typescript
// symbol slice
```
</file>
</files>
</context>
<query>Your prompt</query>
````

## Manifest Notes

A manifest includes:
- truncation limits used,
- deterministic sampling/selection strategy,
- filtering criteria,
- explicit unit schema (`full`/`slice`/`symbol`),
- provenance for each included unit (`path`, `line_start`, `line_end`, `symbol` when available),
- excluded items and reasons.

## Notes

- Hidden paths and common large folders are skipped when expanding directories (`.git`, `node_modules`, `__pycache__`, `venv`, `.venv`).
- Clipboard copy supports macOS (`pbcopy`) and Linux (`xclip`).
- If clipboard is unavailable, output is saved to `.ctx/`.
