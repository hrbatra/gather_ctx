#!/usr/bin/env python3
"""
gather_ctx - Gather context files into a prompt-ready format.

Usage:
    gather_ctx path1.py path2.py -q "What does this code do?"
    gather_ctx src/ -q "Refactor for clarity" --no-tree
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_TOKEN_ENCODING = "cl100k_base"
DEFAULT_HYBRID_BUDGET = 50_000
DEFAULT_EXCLUDED_DIRS = ("node_modules", "__pycache__", ".git", "venv", ".venv")
DEFAULT_HYBRID_SETTINGS: dict[str, int | float | bool] = {
    "small_file_token_threshold": 1_200,
    "large_file_head_lines": 240,
    "query_window_lines": 35,
    "max_slices_per_file": 4,
    "max_symbol_slices_per_file": 2,
    "full_file_budget_fraction_cap": 0.20,
    "symbol_fallback_lines": 80,
    "enable_symbol_slices": True,
}
QUERY_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "what",
    "when",
    "where",
    "which",
    "would",
    "about",
    "there",
    "have",
    "using",
    "your",
    "code",
    "file",
    "files",
    "class",
    "function",
}


@dataclass(frozen=True)
class SymbolSpan:
    name: str
    kind: str
    start_line: int
    end_line: int


@dataclass
class FileCandidate:
    path: Path
    content: str
    lines: list[str]
    full_tokens: int
    path_hits: int
    content_hits: int
    relevance_score: float
    symbols: list[SymbolSpan]


@dataclass
class ContextUnit:
    path: Path
    unit_type: str  # full | slice | symbol
    content: str
    reason: str
    score: float
    estimated_tokens: int
    line_start: int | None = None
    line_end: int | None = None
    symbol: str | None = None

    @property
    def unit_id(self) -> str:
        if self.unit_type == "full":
            return f"full:{self.path}"
        if self.symbol:
            return f"{self.unit_type}:{self.path}:{self.symbol}:{self.line_start}-{self.line_end}"
        return f"{self.unit_type}:{self.path}:{self.line_start}-{self.line_end}"


@dataclass
class SelectionResult:
    units: list[ContextUnit]
    excluded_files: list[dict[str, Any]]
    budget_tokens: int | None
    total_tokens: int
    settings: dict[str, int | float | bool]


_TOKENIZER_STATE = "unknown"  # unknown | ready | disabled
_TOKENIZER_MOD: Any | None = None
_TOKENIZER_CACHE: dict[str, Any] = {}


def approx_token_count(text: str) -> tuple[int, str]:
    byte_count = len(text.encode("utf-8"))
    return max(0, math.ceil(byte_count / 4)), "approx"


def get_tokenizer_encoder(encoding: str) -> tuple[Any | None, str]:
    global _TOKENIZER_STATE
    global _TOKENIZER_MOD

    if _TOKENIZER_STATE == "disabled":
        return None, "approx"

    if _TOKENIZER_MOD is None:
        try:
            import tiktoken  # type: ignore
        except BaseException:
            _TOKENIZER_STATE = "disabled"
            return None, "approx"
        _TOKENIZER_MOD = tiktoken

    cached = _TOKENIZER_CACHE.get(encoding)
    if cached is not None:
        return cached, encoding

    try:
        enc = _TOKENIZER_MOD.get_encoding(encoding)
        _TOKENIZER_CACHE[encoding] = enc
        _TOKENIZER_STATE = "ready"
        return enc, enc.name
    except BaseException:
        try:
            fallback = _TOKENIZER_CACHE.get(DEFAULT_TOKEN_ENCODING)
            if fallback is None:
                fallback = _TOKENIZER_MOD.get_encoding(DEFAULT_TOKEN_ENCODING)
                _TOKENIZER_CACHE[DEFAULT_TOKEN_ENCODING] = fallback
            _TOKENIZER_STATE = "ready"
            return fallback, fallback.name
        except BaseException:
            _TOKENIZER_STATE = "disabled"
            _TOKENIZER_CACHE.clear()
            return None, "approx"


def count_tokens(text: str, encoding: str = DEFAULT_TOKEN_ENCODING) -> tuple[int, str]:
    """
    Return (token_count, source).

    - If `tiktoken` is installed, uses the requested encoding.
    - Otherwise falls back to a rough heuristic (~4 UTF-8 bytes per token).
    """
    enc, source = get_tokenizer_encoder(encoding)
    if enc is None:
        return approx_token_count(text)

    try:
        return len(enc.encode(text)), source
    except BaseException:
        global _TOKENIZER_STATE
        _TOKENIZER_STATE = "disabled"
        _TOKENIZER_CACHE.clear()
        return approx_token_count(text)


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
    tree: dict[str, dict[str, Any] | None] = {}
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

        node: dict[str, dict[str, Any] | None] = tree
        for part in parts[:-1]:
            child = node.get(part)
            if child is None:
                child = {}
                node[part] = child
            node = child
        node[parts[-1]] = None

    if external_paths:
        external_node: dict[str, dict[str, Any] | None] = {}
        for ext_path in sorted(external_paths):
            external_node[str(ext_path)] = None
        tree["[external]"] = external_node

    lines = [str(base)]

    def render(node: dict[str, dict[str, Any] | None], prefix: str = "") -> None:
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
    return "<file_tree>\n" + "\n".join(focused_lines) + "\n</file_tree>"


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


def should_skip_expanded_path(path: Path) -> bool:
    return any(part.startswith(".") or part in DEFAULT_EXCLUDED_DIRS for part in path.parts)


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
                if f.is_file() and not should_skip_expanded_path(f):
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
    """Format full files into prompt-ready context (legacy behavior)."""
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


def format_context_from_units(
    units: list[ContextUnit],
    query: str,
    include_tree: bool = True,
    source_dir: Path | None = None,
    full_tree: bool = False,
) -> str:
    """Format context from explicit units (full files, line slices, symbol slices)."""
    sections = ["<context>"]

    if include_tree and units:
        tree_paths = sorted({unit.path for unit in units})
        sections.append(get_file_tree(tree_paths, source_dir=source_dir, full_tree=full_tree))

    sections.append("<files>")
    for unit in units:
        lang = get_language_hint(unit.path)
        attrs = [f'path="{unit.path}"']
        if unit.unit_type != "full":
            attrs.append(f'kind="{unit.unit_type}"')
            if unit.line_start is not None:
                attrs.append(f'line_start="{unit.line_start}"')
            if unit.line_end is not None:
                attrs.append(f'line_end="{unit.line_end}"')
            if unit.symbol:
                attrs.append(f'symbol="{unit.symbol}"')
        sections.append(f"<file {' '.join(attrs)}>")
        sections.append(f"```{lang}")
        sections.append(unit.content)
        sections.append("```")
        sections.append("</file>")
    sections.append("</files>")
    sections.append("</context>")

    if query:
        sections.append("")
        sections.append(f"<query>{query}</query>")

    return "\n".join(sections)


def extract_query_terms(query: str) -> list[str]:
    terms = []
    seen = set()
    for raw in re.findall(r"[A-Za-z_][A-Za-z0-9_/-]*", query.lower()):
        if len(raw) < 3 or raw in QUERY_STOPWORDS:
            continue
        if raw not in seen:
            seen.add(raw)
            terms.append(raw)
    return terms


def leading_indent_width(text: str) -> int:
    expanded = text.replace("\t", "    ")
    return len(expanded) - len(expanded.lstrip(" "))


def find_python_symbol_header_end(lines: list[str], start_idx: int) -> int:
    paren_balance = 0
    for idx in range(start_idx, len(lines)):
        line = lines[idx]
        stripped = line.strip()
        if not stripped:
            continue
        paren_balance += line.count("(") - line.count(")")
        if stripped.endswith(":") and paren_balance <= 0:
            return idx
    return start_idx


def find_python_symbol_end(lines: list[str], start_idx: int, indent: int) -> int:
    header_end_idx = find_python_symbol_header_end(lines, start_idx)
    for idx in range(header_end_idx + 1, len(lines)):
        line = lines[idx]
        stripped = line.strip()
        if not stripped:
            continue
        current_indent = leading_indent_width(line)
        if current_indent <= indent and not stripped.startswith("#"):
            return idx
    return len(lines)


def extract_python_symbols(lines: list[str]) -> list[SymbolSpan]:
    pattern = re.compile(r"^(\s*)(def|class)\s+([A-Za-z_]\w*)")
    symbols = []
    for idx, line in enumerate(lines):
        match = pattern.match(line)
        if not match:
            continue
        indent = leading_indent_width(match.group(1))
        end_line = find_python_symbol_end(lines, idx, indent)
        symbols.append(
            SymbolSpan(
                name=match.group(3),
                kind=match.group(2),
                start_line=idx + 1,
                end_line=max(idx + 1, end_line),
            )
        )
    return symbols


def find_brace_symbol_end(lines: list[str], start_idx: int, fallback_lines: int) -> int:
    brace_balance = 0
    seen_open = False
    for idx in range(start_idx, len(lines)):
        line = lines[idx]
        opens = line.count("{")
        closes = line.count("}")
        if opens > 0:
            seen_open = True
        if seen_open:
            brace_balance += opens - closes
            if idx > start_idx and brace_balance <= 0:
                return idx + 1
    if seen_open:
        return len(lines)
    return min(len(lines), start_idx + fallback_lines)


def extract_ts_symbols(lines: list[str], fallback_lines: int) -> list[SymbolSpan]:
    patterns: list[tuple[re.Pattern[str], str]] = [
        (re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_]\w*)\s*\("), "function"),
        (re.compile(r"^\s*(?:export\s+)?class\s+([A-Za-z_]\w*)\b"), "class"),
        (
            re.compile(
                r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_]\w*)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_]\w*)\s*=>"
            ),
            "function",
        ),
    ]
    symbols = []
    for idx, line in enumerate(lines):
        for pattern, kind in patterns:
            match = pattern.match(line)
            if not match:
                continue
            end_line = find_brace_symbol_end(lines, idx, fallback_lines)
            symbols.append(
                SymbolSpan(
                    name=match.group(1),
                    kind=kind,
                    start_line=idx + 1,
                    end_line=max(idx + 1, end_line),
                )
            )
            break
    return symbols


def extract_symbols(
    path: Path,
    lines: list[str],
    settings: dict[str, int | float | bool],
) -> list[SymbolSpan]:
    suffix = path.suffix.lower()
    fallback_lines = int(settings["symbol_fallback_lines"])
    if suffix == ".py":
        return extract_python_symbols(lines)
    if suffix in {".ts", ".tsx", ".js", ".jsx"}:
        return extract_ts_symbols(lines, fallback_lines=fallback_lines)
    return []


def score_file_relevance(path: Path, content: str, query_terms: list[str]) -> tuple[int, int, float]:
    if not query_terms:
        return 0, 0, 0.0

    path_lower = str(path).lower()
    content_lower = content.lower()

    path_hits = sum(1 for term in query_terms if term in path_lower)
    content_hits = 0
    for term in query_terms:
        content_hits += content_lower.count(term)
        if content_hits > 250:
            break

    size_penalty = math.log(max(len(content), 10), 10)
    score = (path_hits * 8.0) + (min(content_hits, 80) * 0.35) - (size_penalty * 0.2)
    return path_hits, content_hits, score


def build_file_candidates(
    paths: list[Path],
    query_terms: list[str],
    token_encoding: str,
    settings: dict[str, int | float | bool] | None = None,
) -> list[FileCandidate]:
    active_settings = dict(DEFAULT_HYBRID_SETTINGS)
    if settings:
        active_settings.update(settings)

    candidates = []
    for path in paths:
        content = read_file_content(path)
        lines = content.splitlines()
        full_tokens, _ = count_tokens(content, encoding=token_encoding)
        path_hits, content_hits, relevance_score = score_file_relevance(path, content, query_terms)
        symbols = extract_symbols(path, lines, settings=active_settings)
        candidates.append(
            FileCandidate(
                path=path,
                content=content,
                lines=lines,
                full_tokens=full_tokens,
                path_hits=path_hits,
                content_hits=content_hits,
                relevance_score=relevance_score,
                symbols=symbols,
            )
        )

    return sorted(candidates, key=lambda c: (-c.relevance_score, c.full_tokens, str(c.path)))


def build_full_units(paths: list[Path], token_encoding: str, reason: str = "full_mode") -> list[ContextUnit]:
    units = []
    for path in paths:
        content = read_file_content(path)
        tokens, _ = count_tokens(content, encoding=token_encoding)
        units.append(
            ContextUnit(
                path=path,
                unit_type="full",
                content=content,
                reason=reason,
                score=0.0,
                estimated_tokens=tokens,
            )
        )
    return units


def clamp_line_range(start: int, end: int, max_line: int) -> tuple[int, int]:
    start = max(1, start)
    end = min(max_line, end)
    if end < start:
        start, end = end, start
    start = max(1, start)
    end = max(start, end)
    return start, end


def slice_content(lines: list[str], start: int, end: int) -> str:
    if not lines:
        return ""
    start, end = clamp_line_range(start, end, len(lines))
    if start > len(lines):
        return ""
    return "\n".join(lines[start - 1 : end])


def make_full_unit(candidate: FileCandidate, reason: str, score: float) -> ContextUnit:
    return ContextUnit(
        path=candidate.path,
        unit_type="full",
        content=candidate.content,
        reason=reason,
        score=score,
        estimated_tokens=candidate.full_tokens,
    )


def make_slice_unit(
    candidate: FileCandidate,
    *,
    start_line: int,
    end_line: int,
    reason: str,
    score: float,
    token_encoding: str,
    unit_type: str = "slice",
    symbol: str | None = None,
) -> ContextUnit | None:
    if not candidate.lines:
        return None
    start_line, end_line = clamp_line_range(start_line, end_line, len(candidate.lines))
    content = slice_content(candidate.lines, start_line, end_line)
    if not content:
        return None
    tokens, _ = count_tokens(content, encoding=token_encoding)
    return ContextUnit(
        path=candidate.path,
        unit_type=unit_type,
        content=content,
        reason=reason,
        score=score,
        estimated_tokens=tokens,
        line_start=start_line,
        line_end=end_line,
        symbol=symbol,
    )


def merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not ranges:
        return []
    merged = []
    for start, end in sorted((min(s, e), max(s, e)) for s, e in ranges):
        if not merged:
            merged.append((start, end))
            continue
        prev_start, prev_end = merged[-1]
        if start <= prev_end + 1:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def build_query_ranges(lines: list[str], query_terms: list[str], window: int) -> list[tuple[int, int, float]]:
    if not lines or not query_terms:
        return []

    raw_ranges: list[tuple[int, int, int]] = []
    lower_lines = [line.lower() for line in lines]
    total_lines = len(lines)

    for idx, line in enumerate(lower_lines, start=1):
        hit_count = sum(1 for term in query_terms if term in line)
        if hit_count == 0:
            continue
        raw_ranges.append((max(1, idx - window), min(total_lines, idx + window), hit_count))

    if not raw_ranges:
        return []

    merged: list[list[float]] = []
    for start, end, hit_count in sorted(raw_ranges):
        if not merged or start > int(merged[-1][1]) + 1:
            merged.append([float(start), float(end), float(hit_count)])
            continue
        merged[-1][1] = max(merged[-1][1], float(end))
        merged[-1][2] += float(hit_count)

    return [(int(start), int(end), strength) for start, end, strength in merged]


def build_ranked_slice_units(
    candidate: FileCandidate,
    query_terms: list[str],
    token_encoding: str,
    settings: dict[str, int | float | bool],
) -> list[ContextUnit]:
    if not candidate.lines:
        return []

    units: list[ContextUnit] = []
    total_lines = len(candidate.lines)

    head_lines = int(settings["large_file_head_lines"])
    if head_lines > 0:
        head_end = min(total_lines, head_lines)
        head_unit = make_slice_unit(
            candidate,
            start_line=1,
            end_line=head_end,
            reason="header_slice",
            score=2.0 + candidate.path_hits * 0.3,
            token_encoding=token_encoding,
        )
        if head_unit:
            units.append(head_unit)

    window = int(settings["query_window_lines"])
    for start, end, hit_strength in build_query_ranges(candidate.lines, query_terms, window=window):
        query_unit = make_slice_unit(
            candidate,
            start_line=start,
            end_line=end,
            reason="query_match",
            score=4.0 + min(hit_strength, 25.0) * 0.4,
            token_encoding=token_encoding,
        )
        if query_unit:
            units.append(query_unit)

    if bool(settings["enable_symbol_slices"]) and query_terms and candidate.symbols:
        for symbol in candidate.symbols:
            symbol_hits = sum(1 for term in query_terms if term in symbol.name.lower())
            if symbol_hits == 0:
                continue
            symbol_unit = make_slice_unit(
                candidate,
                start_line=symbol.start_line,
                end_line=symbol.end_line,
                reason=f"symbol_match:{symbol.name}",
                score=7.0 + symbol_hits * 2.0,
                token_encoding=token_encoding,
                unit_type="symbol",
                symbol=symbol.name,
            )
            if symbol_unit:
                units.append(symbol_unit)

    deduped: dict[str, ContextUnit] = {}
    for unit in units:
        existing = deduped.get(unit.unit_id)
        if existing is None or unit.score > existing.score:
            deduped[unit.unit_id] = unit

    return sorted(
        deduped.values(),
        key=lambda unit: (
            -unit.score,
            unit.line_start if unit.line_start is not None else 0,
            unit.line_end if unit.line_end is not None else 0,
            unit.symbol or "",
        ),
    )


def ranges_overlap(first: ContextUnit, second: ContextUnit) -> bool:
    if first.path != second.path:
        return False
    if first.line_start is None or first.line_end is None:
        return False
    if second.line_start is None or second.line_end is None:
        return False
    return not (first.line_end < second.line_start or second.line_end < first.line_start)


def compute_context_tokens_for_units(
    units: list[ContextUnit],
    *,
    query: str,
    include_tree: bool,
    source_dir: Path | None,
    full_tree: bool,
    token_encoding: str,
) -> int:
    context_text = format_context_from_units(
        units=units,
        query=query,
        include_tree=include_tree,
        source_dir=source_dir,
        full_tree=full_tree,
    )
    token_count, _ = count_tokens(context_text, encoding=token_encoding)
    return token_count


def select_hybrid_units(
    *,
    candidates: list[FileCandidate],
    query: str,
    budget_tokens: int,
    token_encoding: str,
    include_tree: bool,
    source_dir: Path | None,
    full_tree: bool,
    settings: dict[str, int | float | bool] | None = None,
) -> SelectionResult:
    active_settings = dict(DEFAULT_HYBRID_SETTINGS)
    if settings:
        active_settings.update(settings)

    query_terms = extract_query_terms(query)

    units: list[ContextUnit] = []
    excluded_files: list[dict[str, Any]] = []
    token_cache: dict[tuple[str, ...], int] = {}

    def token_count_for(candidate_units: list[ContextUnit]) -> int:
        cache_key = tuple(unit.unit_id for unit in candidate_units)
        cached = token_cache.get(cache_key)
        if cached is not None:
            return cached
        value = compute_context_tokens_for_units(
            candidate_units,
            query=query,
            include_tree=include_tree,
            source_dir=source_dir,
            full_tree=full_tree,
            token_encoding=token_encoding,
        )
        token_cache[cache_key] = value
        return value

    def can_add(unit: ContextUnit) -> bool:
        if budget_tokens <= 0:
            return True
        return token_count_for(units + [unit]) <= budget_tokens

    ranked_candidates = sorted(
        candidates,
        key=lambda c: (-c.relevance_score, c.full_tokens, str(c.path)),
    )

    small_threshold = int(active_settings["small_file_token_threshold"])
    max_slices_per_file = int(active_settings["max_slices_per_file"])
    max_symbol_slices_per_file = int(active_settings["max_symbol_slices_per_file"])
    full_cap_tokens = max(small_threshold, int(budget_tokens * float(active_settings["full_file_budget_fraction_cap"])))

    for candidate in ranked_candidates:
        if candidate.full_tokens <= small_threshold:
            full_unit = make_full_unit(
                candidate,
                reason="small_file_full",
                score=6.0 + candidate.relevance_score,
            )
            if can_add(full_unit):
                units.append(full_unit)
            else:
                excluded_files.append(
                    {
                        "path": str(candidate.path),
                        "reason": "over_budget_small_file",
                        "file_tokens": candidate.full_tokens,
                    }
                )
            continue

        full_unit = make_full_unit(
            candidate,
            reason="large_file_full",
            score=4.0 + candidate.relevance_score,
        )

        if candidate.full_tokens <= full_cap_tokens and can_add(full_unit):
            units.append(full_unit)
            continue

        added_for_file: list[ContextUnit] = []
        symbol_count = 0
        ranked_slices = build_ranked_slice_units(
            candidate,
            query_terms=query_terms,
            token_encoding=token_encoding,
            settings=active_settings,
        )

        for slice_unit in ranked_slices:
            if len(added_for_file) >= max_slices_per_file:
                break
            if slice_unit.unit_type == "symbol" and symbol_count >= max_symbol_slices_per_file:
                continue
            if any(ranges_overlap(slice_unit, existing) for existing in added_for_file):
                continue
            if can_add(slice_unit):
                units.append(slice_unit)
                added_for_file.append(slice_unit)
                if slice_unit.unit_type == "symbol":
                    symbol_count += 1

        if not added_for_file:
            excluded_files.append(
                {
                    "path": str(candidate.path),
                    "reason": "over_budget_no_slice_fit",
                    "file_tokens": candidate.full_tokens,
                }
            )

    total_tokens = token_count_for(units)
    while units and total_tokens > budget_tokens:
        drop_index = min(
            range(len(units)),
            key=lambda idx: (
                units[idx].score,
                units[idx].estimated_tokens,
                units[idx].unit_id,
            ),
        )
        removed = units.pop(drop_index)
        excluded_files.append(
            {
                "path": str(removed.path),
                "reason": "trimmed_for_budget",
                "unit_id": removed.unit_id,
                "unit_tokens": removed.estimated_tokens,
            }
        )
        total_tokens = token_count_for(units)

    return SelectionResult(
        units=units,
        excluded_files=excluded_files,
        budget_tokens=budget_tokens,
        total_tokens=total_tokens,
        settings=active_settings,
    )


def parse_line_range_spec(spec: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"(\d+)-(\d+)", spec.strip())
    if not match:
        return None
    start, end = int(match.group(1)), int(match.group(2))
    if start <= 0 or end <= 0:
        return None
    return start, end


def unit_location_label(unit: ContextUnit) -> str:
    if unit.unit_type == "full":
        return str(unit.path)
    return f"{unit.path}:{unit.line_start}-{unit.line_end}"


def print_units_table(units: list[ContextUnit]) -> None:
    if not units:
        print("No units selected.")
        return
    for idx, unit in enumerate(units):
        print(
            f"[{idx}] {unit.unit_type:<6} {unit_location_label(unit)} "
            f"({unit.estimated_tokens:,} tok) reason={unit.reason}"
        )


def print_selection_plan(
    *,
    selection_mode: str,
    units: list[ContextUnit],
    excluded_files: list[dict[str, Any]],
    budget_tokens: int | None,
    total_tokens: int,
    token_source: str,
) -> None:
    file_count = len({unit.path for unit in units})
    print(f"Selection mode: {selection_mode}")
    if budget_tokens is not None:
        print(f"Budget tokens: {budget_tokens:,}")
    token_text = f"~{total_tokens:,} (approx)" if token_source == "approx" else f"{total_tokens:,} ({token_source})"
    print(f"Planned output tokens: {token_text}")
    print(f"Included units: {len(units)} across {file_count} file(s)")
    print_units_table(units)

    if excluded_files:
        print("Excluded candidates:")
        for item in excluded_files:
            path = item.get("path", "")
            reason = item.get("reason", "")
            file_tokens = item.get("file_tokens")
            if file_tokens is None:
                print(f"- {path} ({reason})")
            else:
                print(f"- {path} ({reason}, {file_tokens} tok)")


def resolve_candidate_path(path_token: str, candidates_by_path: dict[Path, FileCandidate]) -> Path:
    direct = Path(path_token).expanduser()
    if not direct.is_absolute():
        direct = (Path.cwd() / direct).resolve()
    else:
        direct = direct.resolve()

    if direct in candidates_by_path:
        return direct

    matches = [candidate_path for candidate_path in candidates_by_path if str(candidate_path).endswith(path_token)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous path: {path_token}")
    raise ValueError(f"Path not found in candidate set: {path_token}")


def drop_units(units: list[ContextUnit], target: str) -> tuple[list[ContextUnit], list[ContextUnit]]:
    if target.isdigit():
        idx = int(target)
        if idx < 0 or idx >= len(units):
            return units, []
        removed = units[idx]
        return units[:idx] + units[idx + 1 :], [removed]

    kept = []
    removed = []
    for unit in units:
        if target in {unit.unit_id, str(unit.path)} or str(unit.path).endswith(target):
            removed.append(unit)
        else:
            kept.append(unit)
    return kept, removed


def sort_units(units: list[ContextUnit]) -> list[ContextUnit]:
    unit_order = {"full": 0, "symbol": 1, "slice": 2}
    deduped: dict[str, ContextUnit] = {}
    for unit in units:
        deduped[unit.unit_id] = unit
    return sorted(
        deduped.values(),
        key=lambda unit: (
            str(unit.path),
            unit_order.get(unit.unit_type, 9),
            unit.line_start if unit.line_start is not None else 0,
            unit.line_end if unit.line_end is not None else 0,
            unit.symbol or "",
            unit.unit_id,
        ),
    )


def run_interactive_refinement(
    *,
    units: list[ContextUnit],
    candidates_by_path: dict[Path, FileCandidate],
    query: str,
    budget_tokens: int,
    token_encoding: str,
    include_tree: bool,
    source_dir: Path | None,
    full_tree: bool,
    excluded_files: list[dict[str, Any]],
) -> tuple[list[ContextUnit], int, list[dict[str, Any]], int]:
    active_units = sort_units(units)
    active_budget = budget_tokens

    print("Interactive mode commands: show | drop <item> | full <path> | slice <path> <start>-<end> | budget <n> | finalize")

    def current_token_count() -> int:
        return compute_context_tokens_for_units(
            active_units,
            query=query,
            include_tree=include_tree,
            source_dir=source_dir,
            full_tree=full_tree,
            token_encoding=token_encoding,
        )

    while True:
        total_tokens = current_token_count()
        budget_status = f"{total_tokens:,}/{active_budget:,}" if active_budget else f"{total_tokens:,}"
        print(f"Current tokens: {budget_status}")
        if active_budget and total_tokens > active_budget:
            print(f"Over budget by {total_tokens - active_budget:,} tokens.")

        try:
            raw = input("gather_ctx> ").strip()
        except EOFError:
            print()
            break

        if not raw:
            continue

        try:
            parts = shlex.split(raw)
        except ValueError as exc:
            print(f"Invalid command syntax: {exc}")
            continue

        command = parts[0].lower()

        if command == "finalize":
            break

        if command == "show":
            print_units_table(active_units)
            continue

        if command == "drop":
            if len(parts) != 2:
                print("Usage: drop <index|unit_id|path>")
                continue
            active_units, removed = drop_units(active_units, parts[1])
            if not removed:
                print("No matching unit found.")
                continue
            for unit in removed:
                excluded_files.append(
                    {
                        "path": str(unit.path),
                        "reason": "manual_drop",
                        "unit_id": unit.unit_id,
                        "unit_tokens": unit.estimated_tokens,
                    }
                )
            active_units = sort_units(active_units)
            continue

        if command == "full":
            if len(parts) != 2:
                print("Usage: full <path>")
                continue
            try:
                path = resolve_candidate_path(parts[1], candidates_by_path)
            except ValueError as exc:
                print(str(exc))
                continue
            candidate = candidates_by_path[path]
            active_units = [unit for unit in active_units if unit.path != path]
            active_units.append(
                make_full_unit(candidate, reason="manual_full_override", score=999.0)
            )
            active_units = sort_units(active_units)
            continue

        if command == "slice":
            if len(parts) != 3:
                print("Usage: slice <path> <start>-<end>")
                continue
            try:
                path = resolve_candidate_path(parts[1], candidates_by_path)
            except ValueError as exc:
                print(str(exc))
                continue
            line_range = parse_line_range_spec(parts[2])
            if line_range is None:
                print("Invalid range. Use <start>-<end> with positive integers.")
                continue
            candidate = candidates_by_path[path]
            new_slice = make_slice_unit(
                candidate,
                start_line=line_range[0],
                end_line=line_range[1],
                reason="manual_slice_override",
                score=999.0,
                token_encoding=token_encoding,
                unit_type="slice",
            )
            if not new_slice:
                print("Could not create slice for that range.")
                continue
            active_units = [
                unit
                for unit in active_units
                if not (unit.path == path and unit.unit_type == "full")
            ]
            active_units.append(new_slice)
            active_units = sort_units(active_units)
            continue

        if command == "budget":
            if len(parts) != 2:
                print("Usage: budget <n>")
                continue
            try:
                value = int(parts[1])
            except ValueError:
                print("Budget must be an integer.")
                continue
            if value <= 0:
                print("Budget must be > 0.")
                continue
            active_budget = value
            continue

        print("Unknown command. Use: show, drop, full, slice, budget, finalize")

    final_tokens = current_token_count()
    return active_units, active_budget, excluded_files, final_tokens


def build_manifest(
    *,
    selection_mode: str,
    query: str,
    token_encoding: str,
    budget_tokens: int | None,
    include_tree: bool,
    full_tree: bool,
    settings: dict[str, int | float | bool],
    units: list[ContextUnit],
    excluded_files: list[dict[str, Any]],
    context_text: str,
    output_token_count: int,
    output_token_source: str,
) -> dict[str, Any]:
    return {
        "manifest_version": 1,
        "generated_at": datetime.now().isoformat(),
        "selection_mode": selection_mode,
        "query": query,
        "budget_tokens": budget_tokens,
        "token_encoding": token_encoding,
        "tree_mode": "none" if not include_tree else ("full" if full_tree else "focused"),
        "output_stats": {
            "chars": len(context_text),
            "bytes": len(context_text.encode("utf-8")),
            "tokens": output_token_count,
            "token_source": output_token_source,
            "included_units": len(units),
            "included_files": len({unit.path for unit in units}),
        },
        "truncation_limits_used": {
            "small_file_token_threshold": settings.get("small_file_token_threshold"),
            "large_file_head_lines": settings.get("large_file_head_lines"),
            "query_window_lines": settings.get("query_window_lines"),
            "max_slices_per_file": settings.get("max_slices_per_file"),
            "max_symbol_slices_per_file": settings.get("max_symbol_slices_per_file"),
            "full_file_budget_fraction_cap": settings.get("full_file_budget_fraction_cap"),
        },
        "sampling_selection_strategy": (
            "Deterministic greedy selection: full-file preference for small files, "
            "ranked line/symbol slices for larger files, then strict budget trim."
        ),
        "filtering_criteria": {
            "path_expansion_excludes": [*DEFAULT_EXCLUDED_DIRS, "hidden_paths_prefix_dot"],
            "query_terms": extract_query_terms(query),
            "symbol_slices_enabled": bool(settings.get("enable_symbol_slices", False)),
        },
        "included_unit_schema": {
            "types": ["full", "slice", "symbol"],
            "fields": {
                "id": "unique unit id",
                "type": "full|slice|symbol",
                "path": "absolute file path",
                "line_start": "1-based inclusive start or null",
                "line_end": "1-based inclusive end or null",
                "symbol": "symbol name when available",
                "reason": "selection rationale",
                "estimated_tokens": "tokens in this unit body",
            },
        },
        "included_units": [
            {
                "id": unit.unit_id,
                "type": unit.unit_type,
                "path": str(unit.path),
                "line_start": unit.line_start,
                "line_end": unit.line_end,
                "symbol": unit.symbol,
                "reason": unit.reason,
                "estimated_tokens": unit.estimated_tokens,
            }
            for unit in units
        ],
        "excluded_files": excluded_files,
    }


def write_manifest(manifest: dict[str, Any], manifest_path: str) -> Path:
    out_path = Path(manifest_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return out_path


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gather context files into a prompt-ready format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  gather_ctx src/main.py src/utils.py -q "Explain this code"
  gather_ctx ./src -q "Refactor for clarity" --no-tree
  gather_ctx ./src -q "Focus on auth flow" --selection-mode hybrid --budget-tokens 50000 --plan-only
  gather_ctx ./src -q "Find cityscan internals" --selection-mode interactive --budget-tokens 45000
        """,
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Files or directories to include",
    )
    parser.add_argument(
        "-q",
        "--query",
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
        "--selection-mode",
        choices=["full", "hybrid", "interactive"],
        default="full",
        help="Selection strategy. Default 'full' preserves current behavior.",
    )
    parser.add_argument(
        "--budget-tokens",
        type=int,
        help="Target token budget for hybrid/interactive selection (default: 50000 in those modes).",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Show inclusion plan and token stats without copying/printing context.",
    )
    parser.add_argument(
        "--manifest",
        help="Write JSON manifest with selection/provenance details.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save to .ctx/ instead of clipboard",
    )
    parser.add_argument(
        "-p",
        "--print",
        action="store_true",
        dest="print_output",
        help="Print to stdout instead of clipboard",
    )
    parser.add_argument(
        "-s",
        "--stats",
        "--tokens",
        action="store_true",
        dest="print_stats",
        help="Print output stats (chars/bytes/tokens). For --print, stats go to stderr.",
    )
    parser.add_argument(
        "--token-encoding",
        default=DEFAULT_TOKEN_ENCODING,
        help=f"Token encoding for --stats when using tiktoken (default: {DEFAULT_TOKEN_ENCODING}).",
    )

    args = parser.parse_args()

    if args.budget_tokens is not None and args.budget_tokens <= 0:
        print("Error: --budget-tokens must be > 0.", file=sys.stderr)
        sys.exit(1)

    paths, source_dir = expand_paths(args.paths)
    if not paths:
        print("Error: No files found matching the given paths.", file=sys.stderr)
        sys.exit(1)

    include_tree = not args.no_tree
    selection_mode = args.selection_mode

    units: list[ContextUnit]
    excluded_files: list[dict[str, Any]]
    selection_settings: dict[str, int | float | bool]
    selection_budget: int | None

    if selection_mode == "full":
        context = format_context(
            paths=paths,
            query=args.query,
            include_tree=include_tree,
            source_dir=source_dir,
            full_tree=args.full_tree,
        )
        units = build_full_units(paths, token_encoding=args.token_encoding)
        excluded_files = []
        selection_settings = {
            "small_file_token_threshold": 0,
            "large_file_head_lines": 0,
            "query_window_lines": 0,
            "max_slices_per_file": 0,
            "max_symbol_slices_per_file": 0,
            "full_file_budget_fraction_cap": 1.0,
            "enable_symbol_slices": False,
        }
        selection_budget = args.budget_tokens
    else:
        selection_budget = args.budget_tokens or DEFAULT_HYBRID_BUDGET
        selection_settings = dict(DEFAULT_HYBRID_SETTINGS)
        query_terms = extract_query_terms(args.query)
        candidates = build_file_candidates(
            paths,
            query_terms=query_terms,
            token_encoding=args.token_encoding,
            settings=selection_settings,
        )
        selection = select_hybrid_units(
            candidates=candidates,
            query=args.query,
            budget_tokens=selection_budget,
            token_encoding=args.token_encoding,
            include_tree=include_tree,
            source_dir=source_dir,
            full_tree=args.full_tree,
            settings=selection_settings,
        )
        units = selection.units
        excluded_files = selection.excluded_files
        selection_settings = selection.settings

        if selection_mode == "interactive":
            if not sys.stdin.isatty():
                print("Error: --selection-mode interactive requires a TTY.", file=sys.stderr)
                sys.exit(2)
            candidate_map = {candidate.path: candidate for candidate in candidates}
            units, selection_budget, excluded_files, _ = run_interactive_refinement(
                units=units,
                candidates_by_path=candidate_map,
                query=args.query,
                budget_tokens=selection_budget,
                token_encoding=args.token_encoding,
                include_tree=include_tree,
                source_dir=source_dir,
                full_tree=args.full_tree,
                excluded_files=excluded_files,
            )

        context = format_context_from_units(
            units=units,
            query=args.query,
            include_tree=include_tree,
            source_dir=source_dir,
            full_tree=args.full_tree,
        )

    token_count, token_source = count_tokens(context, encoding=args.token_encoding)

    if args.plan_only:
        print_selection_plan(
            selection_mode=selection_mode,
            units=units,
            excluded_files=excluded_files,
            budget_tokens=selection_budget,
            total_tokens=token_count,
            token_source=token_source,
        )
        if args.manifest:
            manifest = build_manifest(
                selection_mode=selection_mode,
                query=args.query,
                token_encoding=args.token_encoding,
                budget_tokens=selection_budget,
                include_tree=include_tree,
                full_tree=args.full_tree,
                settings=selection_settings,
                units=units,
                excluded_files=excluded_files,
                context_text=context,
                output_token_count=token_count,
                output_token_source=token_source,
            )
            manifest_path = write_manifest(manifest, args.manifest)
            print(f"Manifest written to: {manifest_path}")
        return

    stats_details = None
    if args.print_stats:
        stats_details = format_stats_details(
            text=context,
            token_encoding=args.token_encoding,
        )

    if args.manifest:
        manifest = build_manifest(
            selection_mode=selection_mode,
            query=args.query,
            token_encoding=args.token_encoding,
            budget_tokens=selection_budget,
            include_tree=include_tree,
            full_tree=args.full_tree,
            settings=selection_settings,
            units=units,
            excluded_files=excluded_files,
            context_text=context,
            output_token_count=token_count,
            output_token_source=token_source,
        )
        manifest_path = write_manifest(manifest, args.manifest)
    else:
        manifest_path = None

    default_details = format_default_copy_details(
        text=context,
        token_encoding=args.token_encoding,
    )

    if selection_mode == "full":
        output_count_text = f"{len(paths)} file(s)"
    else:
        unique_files = len({unit.path for unit in units})
        output_count_text = f"{len(units)} unit(s) from {unique_files} file(s)"

    if args.print_output:
        print(context)
        if stats_details:
            print(f"Stats: {output_count_text}, {stats_details}", file=sys.stderr)
        if manifest_path:
            print(f"Manifest: {manifest_path}", file=sys.stderr)
    elif args.save:
        out_path = save_to_ctx_dir(context, Path.cwd())
        print(f"Saved to: {out_path}")
        if stats_details:
            print(f"Stats: {output_count_text}, {stats_details}")
        else:
            print(f"Stats: {output_count_text}, {default_details}")
        if manifest_path:
            print(f"Manifest: {manifest_path}")
    else:
        if copy_to_clipboard(context):
            if stats_details:
                print(f"✓ Copied {output_count_text} to clipboard ({stats_details})")
            else:
                print(f"✓ Copied {output_count_text} to clipboard ({default_details})")
            if manifest_path:
                print(f"Manifest: {manifest_path}")
        else:
            out_path = save_to_ctx_dir(context, Path.cwd())
            print(f"Clipboard unavailable. Saved to: {out_path}")
            if stats_details:
                print(f"Stats: {output_count_text}, {stats_details}")
            else:
                print(f"Stats: {output_count_text}, {default_details}")
            if manifest_path:
                print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
