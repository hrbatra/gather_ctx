# gather_ctx

`gather_ctx` is a small CLI that bundles files into a prompt-ready block for LLMs.
It copies to clipboard by default, or prints/saves when requested.

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
# Files -> clipboard (default)
gather_ctx src/main.py src/utils.py -q "Explain this code"

# Directory -> recursive file expansion
gather_ctx src/ -q "Refactor for clarity"

# Globs
gather_ctx "*.py" -q "Find bugs"

# Print output instead of clipboard
gather_ctx src/main.py --print

# Save output to .ctx/context_YYYYMMDD_HHMMSS.txt
gather_ctx src/ -q "Analyze" --save
```

## Useful Flags

- `--no-tree`: omit file tree section
- `--full-tree`: show full directory tree (default is focused tree of selected files)
- `--stats` / `--tokens`: show chars, bytes, and token estimate/count
- `--token-encoding`: set tiktoken encoding (default: `cl100k_base`)

## Output Shape

````xml
<context>
<file_tree>...</file_tree>
<files>
<file path="/abs/path/to/file.py">
```python
# file contents
```
</file>
</files>
</context>
<query>Your prompt</query>
````

## Notes

- Hidden paths and common large folders are skipped when expanding directories (`.git`, `node_modules`, `__pycache__`, `venv`, `.venv`).
- Clipboard copy currently supports macOS (`pbcopy`) and Linux (`xclip`).
- If clipboard is unavailable, output is saved to `.ctx/`.
