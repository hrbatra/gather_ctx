#!/usr/bin/env python3
"""
gather_ctx - Gather context files into a prompt-ready format.

Usage:
    gather_ctx path1.py path2.py -q "What does this code do?"
    gather_ctx src/ -q "Refactor for clarity" --no-tree
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def count_tokens(text: str, encoding: str = "cl100k_base") -> tuple[int, str]:
    """
    Return (token_count, source).

    - If `tiktoken` is installed, uses the requested encoding (default: cl100k_base).
    - Otherwise falls back to a rough heuristic (~4 UTF-8 bytes per token).
    """
    try:
        import tiktoken  # type: ignore
    except Exception:
        byte_count = len(text.encode("utf-8"))
        return max(0, math.ceil(byte_count / 4)), "approx"

    try:
        enc = tiktoken.get_encoding(encoding)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
        encoding = enc.name

    return len(enc.encode(text)), encoding


def format_stats_details(*, text: str, token_encoding: str) -> str:
    char_count = len(text)
    byte_count = len(text.encode("utf-8"))
    token_count, token_source = count_tokens(text, encoding=token_encoding)

    token_part = (
        f"~{token_count:,} tokens (approx)"
        if token_source == "approx"
        else f"{token_count:,} tokens ({token_source})"
    )
    return f"{char_count:,} chars, {byte_count:,} bytes, {token_part}"


def format_default_copy_details(*, text: str, token_encoding: str) -> str:
    """
    Default concise stats for clipboard/save output.

    Always includes chars + tokens so users get quick context-size feedback
    without needing --stats.
    """
    char_count = len(text)
    token_count, token_source = count_tokens(text, encoding=token_encoding)

    if token_source == "approx":
        return f"{char_count:,} chars, ~{token_count:,} tokens"

    return f"{char_count:,} chars, {token_count:,} tokens ({token_source})"


def get_file_tree_via_tree_cmd(directory: Path) -> str | None:
    """Try to get tree output using the tree command."""
    try:
        result = subprocess.run(
            ["tree", "--gitignore", "-I", "node_modules|__pycache__|.venv|venv", str(directory)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def get_common_base(paths: list[Path]) -> Path:
    """Get the common directory base for a list of files."""
    if not paths:
        return Path.cwd()
    cwd = Path.cwd().resolve()
    in_cwd = [p for p in paths if p.is_relative_to(cwd)]
    if in_cwd and len(in_cwd) >= max(2, math.ceil(len(paths) * 0.6)):
        return cwd
    return Path(os.path.commonpath([str(p) for p in paths]))


def build_focused_tree_lines(paths: list[Path], base: Path) -> list[str]:
    """
    Build a compact tree that contains only selected files and their parent directories.
    """
    tree: dict[str, dict | None] = {}
    external_paths: list[Path] = []

    for path in sorted(paths):
        try:
            rel = path.relative_to(base)
        except ValueError:
            external_paths.append(path)
            continue
        parts = rel.parts
        if not parts:
            continue

        node = tree
        for part in parts[:-1]:
            child = node.get(part)
            if child is None:
                child = {}
                node[part] = child
            node = child  # type: ignore[assignment]
        node[parts[-1]] = None

    if external_paths:
        external_node: dict[str, dict | None] = {}
        for ext_path in sorted(external_paths):
            external_node[str(ext_path)] = None
        tree["[external]"] = external_node

    lines = [str(base)]

    def render(node: dict[str, dict | None], prefix: str = "") -> None:
        items = sorted(
            node.items(),
            key=lambda kv: (kv[1] is None, kv[0] == "[external]", kv[0].lower()),
        )
        for idx, (name, child) in enumerate(items):
            last = idx == len(items) - 1
            connector = "└── " if last else "├── "
            lines.append(f"{prefix}{connector}{name}")
            if isinstance(child, dict):
                extension = "    " if last else "│   "
                render(child, prefix + extension)

    render(tree)

    return lines


def get_file_tree(
    paths: list[Path],
    source_dir: Path | None = None,
    full_tree: bool = False,
) -> str:
    """Generate a file tree, focused by default with optional full-tree mode."""
    if not paths:
        return ""

    if full_tree:
        # Legacy mode: show a complete tree for the selected root.
        tree_dir = source_dir if source_dir and source_dir.is_dir() else Path.cwd()
        tree_output = get_file_tree_via_tree_cmd(tree_dir)
        if tree_output:
            return f"<file_tree>\n{tree_output}\n</file_tree>"

    # Default mode: show only selected files and their parent directories.
    base = get_common_base(paths)
    focused_lines = build_focused_tree_lines(paths, base)
    return f"<file_tree>\n" + "\n".join(focused_lines) + "\n</file_tree>"


def read_file_content(path: Path) -> str:
    """Read file content, handling errors gracefully."""
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        return f"[Error reading file: {e}]"


def get_language_hint(path: Path) -> str:
    """Get language hint for code fence."""
    ext_map = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".jsx": "jsx",
        ".rs": "rust",
        ".go": "go",
        ".md": "markdown",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".json": "json",
        ".toml": "toml",
        ".sh": "bash",
        ".fish": "fish",
        ".sql": "sql",
        ".html": "html",
        ".css": "css",
    }
    return ext_map.get(path.suffix.lower(), "")


def expand_paths(paths: list[str]) -> tuple[list[Path], Path | None]:
    """Expand directories and globs into file list. Returns (files, source_dir)."""
    result = []
    source_dirs = []
    for p_str in paths:
        p = Path(p_str).expanduser().resolve()
        if p.is_dir():
            source_dirs.append(p)
            # Recursively get all files, skip hidden and common excludes
            for f in p.rglob("*"):
                if f.is_file() and not any(
                    part.startswith(".") or part in ("node_modules", "__pycache__", ".git", "venv", ".venv")
                    for part in f.parts
                ):
                    result.append(f)
        elif p.is_file():
            result.append(p)
        elif "*" in p_str:
            # Handle glob
            parent = Path(p_str).parent
            pattern = Path(p_str).name
            for f in parent.glob(pattern):
                if f.is_file():
                    result.append(f.resolve())
    
    # Use source_dir for tree if exactly one directory was passed
    source_dir = source_dirs[0] if len(source_dirs) == 1 else None
    return sorted(set(result)), source_dir


def format_context(
    paths: list[Path],
    query: str,
    include_tree: bool = True,
    source_dir: Path | None = None,
    full_tree: bool = False,
) -> str:
    """Format files into prompt-ready context."""
    sections = []
    
    # Header
    sections.append("<context>")
    
    # File tree
    if include_tree and paths:
        sections.append(get_file_tree(paths, source_dir=source_dir, full_tree=full_tree))
    
    # File contents
    sections.append("<files>")
    for path in paths:
        lang = get_language_hint(path)
        content = read_file_content(path)
        sections.append(f'<file path="{path}">')
        sections.append(f"```{lang}")
        sections.append(content)
        sections.append("```")
        sections.append("</file>")
    sections.append("</files>")
    
    sections.append("</context>")
    
    # Query
    if query:
        sections.append("")
        sections.append(f"<query>{query}</query>")
    
    return "\n".join(sections)


def copy_to_clipboard(text: str) -> bool:
    """Copy text to clipboard. Returns True on success."""
    try:
        # macOS
        proc = subprocess.Popen(
            ["pbcopy"], stdin=subprocess.PIPE, env={"LANG": "en_US.UTF-8"}
        )
        proc.communicate(text.encode("utf-8"))
        return proc.returncode == 0
    except FileNotFoundError:
        pass
    
    try:
        # Linux with xclip
        proc = subprocess.Popen(
            ["xclip", "-selection", "clipboard"], stdin=subprocess.PIPE
        )
        proc.communicate(text.encode("utf-8"))
        return proc.returncode == 0
    except FileNotFoundError:
        pass
    
    return False


def save_to_ctx_dir(text: str, base_dir: Path) -> Path:
    """Save context to .ctx/ directory."""
    ctx_dir = base_dir / ".ctx"
    ctx_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = ctx_dir / f"context_{timestamp}.txt"
    out_path.write_text(text, encoding="utf-8")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Gather context files into a prompt-ready format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  gather_ctx src/main.py src/utils.py -q "Explain this code"
  gather_ctx ./src -q "Refactor for clarity" --no-tree
  gather_ctx *.py -q "Find bugs" --save
        """,
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Files or directories to include",
    )
    parser.add_argument(
        "-q", "--query",
        default="",
        help="Query/instruction to append",
    )
    parser.add_argument(
        "--no-tree",
        action="store_true",
        help="Omit file tree from output",
    )
    parser.add_argument(
        "--full-tree",
        action="store_true",
        help="Use complete directory tree output (legacy behavior). Default is focused tree of selected files.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save to .ctx/ instead of clipboard",
    )
    parser.add_argument(
        "-p", "--print",
        action="store_true",
        dest="print_output",
        help="Print to stdout instead of clipboard",
    )
    parser.add_argument(
        "-s", "--stats",
        "--tokens",
        action="store_true",
        dest="print_stats",
        help="Print output stats (chars/bytes/tokens). For --print, stats go to stderr.",
    )
    parser.add_argument(
        "--token-encoding",
        default="cl100k_base",
        help="Token encoding for --stats when using tiktoken (default: cl100k_base).",
    )
    
    args = parser.parse_args()
    
    # Expand paths
    paths, source_dir = expand_paths(args.paths)
    if not paths:
        print("Error: No files found matching the given paths.", file=sys.stderr)
        sys.exit(1)
    
    # Format context
    context = format_context(
        paths=paths,
        query=args.query,
        include_tree=not args.no_tree,
        source_dir=source_dir,
        full_tree=args.full_tree,
    )

    stats_details = None
    if args.print_stats:
        stats_details = format_stats_details(
            text=context,
            token_encoding=args.token_encoding,
        )
    
    # Output
    default_details = format_default_copy_details(
        text=context,
        token_encoding=args.token_encoding,
    )

    if args.print_output:
        print(context)
        if stats_details:
            print(f"Stats: {len(paths)} file(s), {stats_details}", file=sys.stderr)
    elif args.save:
        out_path = save_to_ctx_dir(context, Path.cwd())
        print(f"Saved to: {out_path}")
        if stats_details:
            print(f"Stats: {len(paths)} file(s), {stats_details}")
        else:
            print(f"Stats: {len(paths)} file(s), {default_details}")
    else:
        if copy_to_clipboard(context):
            if stats_details:
                print(f"✓ Copied {len(paths)} file(s) to clipboard ({stats_details})")
            else:
                print(f"✓ Copied {len(paths)} file(s) to clipboard ({default_details})")
        else:
            # Fallback to save
            out_path = save_to_ctx_dir(context, Path.cwd())
            print(f"Clipboard unavailable. Saved to: {out_path}")
            if stats_details:
                print(f"Stats: {len(paths)} file(s), {stats_details}")
            else:
                print(f"Stats: {len(paths)} file(s), {default_details}")


if __name__ == "__main__":
    main()
