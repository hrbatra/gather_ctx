from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gather_ctx


class GatherCtxHybridTests(unittest.TestCase):
    def setUp(self) -> None:
        gather_ctx._TOKENIZER_STATE = "unknown"
        gather_ctx._TOKENIZER_MOD = None
        gather_ctx._TOKENIZER_CACHE.clear()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write_file(self, rel_path: str, content: str) -> Path:
        path = self.root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def test_hybrid_selection_is_deterministic(self) -> None:
        large_lines = []
        for i in range(1, 900):
            if i in {110, 320, 640}:
                large_lines.append(f"def cityscan_target_{i}():")
                large_lines.append("    return 'cityscan'")
            else:
                large_lines.append(f"line_{i} = 'filler {i}'")
        large = self.write_file("src/routes.py", "\n".join(large_lines))
        small = self.write_file("src/types.py", "type_hint = 'cityscan'\n")

        query = "trace cityscan route handlers"
        terms = gather_ctx.extract_query_terms(query)
        candidates = gather_ctx.build_file_candidates(
            [large, small],
            query_terms=terms,
            token_encoding=gather_ctx.DEFAULT_TOKEN_ENCODING,
            settings=gather_ctx.DEFAULT_HYBRID_SETTINGS,
        )

        first = gather_ctx.select_hybrid_units(
            candidates=candidates,
            query=query,
            budget_tokens=1_400,
            token_encoding=gather_ctx.DEFAULT_TOKEN_ENCODING,
            include_tree=False,
            source_dir=None,
            full_tree=False,
            settings=gather_ctx.DEFAULT_HYBRID_SETTINGS,
        )
        second = gather_ctx.select_hybrid_units(
            candidates=candidates,
            query=query,
            budget_tokens=1_400,
            token_encoding=gather_ctx.DEFAULT_TOKEN_ENCODING,
            include_tree=False,
            source_dir=None,
            full_tree=False,
            settings=gather_ctx.DEFAULT_HYBRID_SETTINGS,
        )

        self.assertEqual([unit.unit_id for unit in first.units], [unit.unit_id for unit in second.units])

    def test_hybrid_budget_compliance(self) -> None:
        body = "\n".join(f"const value_{i} = 'x{i}';" for i in range(1, 2500))
        first = self.write_file("web/page.tsx", body)
        second = self.write_file("web/api.ts", body)

        query = "api value filtering"
        terms = gather_ctx.extract_query_terms(query)
        candidates = gather_ctx.build_file_candidates(
            [first, second],
            query_terms=terms,
            token_encoding=gather_ctx.DEFAULT_TOKEN_ENCODING,
            settings=gather_ctx.DEFAULT_HYBRID_SETTINGS,
        )

        selection = gather_ctx.select_hybrid_units(
            candidates=candidates,
            query=query,
            budget_tokens=450,
            token_encoding=gather_ctx.DEFAULT_TOKEN_ENCODING,
            include_tree=False,
            source_dir=None,
            full_tree=False,
            settings=gather_ctx.DEFAULT_HYBRID_SETTINGS,
        )

        context = gather_ctx.format_context_from_units(
            selection.units,
            query=query,
            include_tree=False,
            source_dir=None,
            full_tree=False,
        )
        tokens, _ = gather_ctx.count_tokens(context, encoding=gather_ctx.DEFAULT_TOKEN_ENCODING)
        self.assertLessEqual(tokens, 450)

    def test_manifest_has_required_hybrid_metadata(self) -> None:
        lines = []
        for i in range(1, 600):
            if i in {100, 260, 470}:
                lines.append(f"def cityscan_core_{i}():")
                lines.append("    return 'cityscan-core'")
            else:
                lines.append(f"placeholder_{i} = {i}")
        target = self.write_file("services/cityscan.py", "\n".join(lines))

        query = "cityscan core flow"
        terms = gather_ctx.extract_query_terms(query)
        candidates = gather_ctx.build_file_candidates(
            [target],
            query_terms=terms,
            token_encoding=gather_ctx.DEFAULT_TOKEN_ENCODING,
            settings=gather_ctx.DEFAULT_HYBRID_SETTINGS,
        )
        selection = gather_ctx.select_hybrid_units(
            candidates=candidates,
            query=query,
            budget_tokens=320,
            token_encoding=gather_ctx.DEFAULT_TOKEN_ENCODING,
            include_tree=False,
            source_dir=None,
            full_tree=False,
            settings=gather_ctx.DEFAULT_HYBRID_SETTINGS,
        )

        context = gather_ctx.format_context_from_units(
            selection.units,
            query=query,
            include_tree=False,
            source_dir=None,
            full_tree=False,
        )
        token_count, token_source = gather_ctx.count_tokens(context, encoding=gather_ctx.DEFAULT_TOKEN_ENCODING)

        manifest = gather_ctx.build_manifest(
            selection_mode="hybrid",
            query=query,
            token_encoding=gather_ctx.DEFAULT_TOKEN_ENCODING,
            budget_tokens=320,
            include_tree=False,
            full_tree=False,
            settings=selection.settings,
            units=selection.units,
            excluded_files=selection.excluded_files,
            context_text=context,
            output_token_count=token_count,
            output_token_source=token_source,
        )

        self.assertIn("truncation_limits_used", manifest)
        self.assertIn("sampling_selection_strategy", manifest)
        self.assertIn("filtering_criteria", manifest)
        self.assertIn("included_unit_schema", manifest)

        sliced = [unit for unit in manifest["included_units"] if unit["type"] in {"slice", "symbol"}]
        self.assertTrue(sliced)
        for unit in sliced:
            self.assertIsNotNone(unit["line_start"])
            self.assertIsNotNone(unit["line_end"])
            self.assertGreaterEqual(unit["line_end"], unit["line_start"])

    def test_python_symbol_slice_handles_multiline_signature(self) -> None:
        file_path = self.write_file(
            "service/multi.py",
            "\n".join(
                [
                    "def target_symbol(",
                    "    first: int,",
                    "    second: int,",
                    ") -> int:",
                    "    value = first + second",
                    "    return value",
                    "",
                    "def later_symbol() -> int:",
                    "    return 1",
                ]
            ),
        )

        terms = gather_ctx.extract_query_terms("target_symbol")
        settings = dict(gather_ctx.DEFAULT_HYBRID_SETTINGS)
        settings["small_file_token_threshold"] = 1
        settings["full_file_budget_fraction_cap"] = 0.0
        candidates = gather_ctx.build_file_candidates(
            [file_path],
            query_terms=terms,
            token_encoding=gather_ctx.DEFAULT_TOKEN_ENCODING,
            settings=settings,
        )
        selection = gather_ctx.select_hybrid_units(
            candidates=candidates,
            query="target_symbol",
            budget_tokens=300,
            token_encoding=gather_ctx.DEFAULT_TOKEN_ENCODING,
            include_tree=False,
            source_dir=None,
            full_tree=False,
            settings=settings,
        )

        symbol_units = [unit for unit in selection.units if unit.unit_type == "symbol" and unit.symbol == "target_symbol"]
        self.assertTrue(symbol_units)
        unit = symbol_units[0]
        self.assertIsNotNone(unit.line_start)
        self.assertIsNotNone(unit.line_end)
        self.assertGreaterEqual(unit.line_end, (unit.line_start or 0) + 4)
        self.assertIn("return value", unit.content)

    def test_legacy_full_format_stays_compatible(self) -> None:
        file_path = self.write_file("legacy/main.py", "print('ok')\n")
        context = gather_ctx.format_context(
            [file_path],
            query="Explain",
            include_tree=False,
            source_dir=None,
            full_tree=False,
        )

        self.assertIn(f'<file path="{file_path}">', context)
        self.assertNotIn('kind="', context)
        self.assertIn("<query>Explain</query>", context)

    def test_count_tokens_falls_back_when_tiktoken_runtime_breaks(self) -> None:
        class _BrokenEnc:
            def __init__(self, name: str) -> None:
                self.name = name

            def encode(self, _text: str) -> list[int]:
                raise TypeError("broken encoder")

        class _BrokenTiktoken:
            @staticmethod
            def get_encoding(name: str) -> _BrokenEnc:
                return _BrokenEnc(name)

        with mock.patch.dict("sys.modules", {"tiktoken": _BrokenTiktoken()}):
            count, source = gather_ctx.count_tokens("hello world", encoding="cl100k_base")

        self.assertEqual(source, "approx")
        self.assertGreater(count, 0)

    def test_count_tokens_falls_back_on_base_exception(self) -> None:
        class _Panic(BaseException):
            pass

        class _PanickingTiktoken:
            call_count = 0

            @staticmethod
            def get_encoding(_name: str) -> object:
                _PanickingTiktoken.call_count += 1
                raise _Panic("ffi panic")

        with mock.patch.dict("sys.modules", {"tiktoken": _PanickingTiktoken()}):
            count, source = gather_ctx.count_tokens("hello panic", encoding="cl100k_base")
            first_calls = _PanickingTiktoken.call_count
            second_count, second_source = gather_ctx.count_tokens("hello panic again", encoding="cl100k_base")

        self.assertEqual(source, "approx")
        self.assertGreater(count, 0)
        self.assertEqual(second_source, "approx")
        self.assertGreater(second_count, 0)
        self.assertEqual(first_calls, 2)
        self.assertEqual(_PanickingTiktoken.call_count, first_calls)


if __name__ == "__main__":
    unittest.main()
