"""Tests for core.utils — shared utility functions."""

import os
import pytest

from core.utils import (
    get_latest_file,
    load_models,
    discover_openbench_csv,
    normalize_reasoning_flag,
    extract_json_payload,
)


# ── get_latest_file ─────────────────────────────────────────────────────

class TestGetLatestFile:
    def test_finds_highest_date(self, tmp_path):
        (tmp_path / "data_20240101.csv").write_text("old")
        (tmp_path / "data_20240615.csv").write_text("mid")
        (tmp_path / "data_20250101.csv").write_text("new")
        result = get_latest_file(str(tmp_path / "data_*.csv"))
        assert result.endswith("data_20250101.csv")

    def test_handles_hyphenated_dates(self, tmp_path):
        (tmp_path / "data_2024-01-01.csv").write_text("old")
        (tmp_path / "data_2025-06-15.csv").write_text("new")
        result = get_latest_file(str(tmp_path / "data_*.csv"))
        assert result.endswith("data_2025-06-15.csv")

    def test_falls_back_when_no_date(self, tmp_path):
        (tmp_path / "data_foo.csv").write_text("no date")
        result = get_latest_file(str(tmp_path / "data_*.csv"))
        assert result.endswith("data_foo.csv")

    def test_raises_on_empty(self, tmp_path):
        with pytest.raises(ValueError, match="No files found"):
            get_latest_file(str(tmp_path / "nonexistent_*.csv"))


# ── load_models ──────────────────────────────────────────────────────────

class TestLoadModels:
    def test_parses_csv(self, sample_openbench_csv):
        models = load_models(sample_openbench_csv)
        assert len(models) == 3  # 4th row has missing openbench_id
        names = {m["name"] for m in models}
        assert "GPT-4o" in names
        assert "Claude 3.5 Sonnet" in names

    def test_drops_nan_openbench_id(self, sample_openbench_csv):
        models = load_models(sample_openbench_csv)
        ids = {m["id"] for m in models}
        assert "" not in ids
        assert None not in ids

    def test_normalizes_reasoning_column(self, sample_openbench_csv_lowercase_reasoning):
        models = load_models(sample_openbench_csv_lowercase_reasoning)
        # Should have 'Reasoning' key (capitalized) even from lowercase source
        for m in models:
            assert "Reasoning" in m

    def test_raises_on_missing_columns(self, tmp_path):
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("Name,ID\nfoo,bar\n")
        with pytest.raises(ValueError, match="must contain"):
            load_models(str(bad_csv))


# ── discover_openbench_csv ───────────────────────────────────────────────

class TestDiscoverOpenbenchCSV:
    def test_finds_in_benchmark_combiner(self, tmp_path):
        # Simulate: script_dir/../benchmark_combiner/benchmarks/openbench_20250101.csv
        benchmarks_dir = tmp_path / "benchmark_combiner" / "benchmarks"
        benchmarks_dir.mkdir(parents=True)
        (benchmarks_dir / "openbench_20250101.csv").write_text("Model,openbench_id\n")

        script_dir = tmp_path / "soothsayer_eq"
        script_dir.mkdir()

        result = discover_openbench_csv(str(script_dir))
        assert "openbench_20250101.csv" in result

    def test_raises_when_not_found(self, tmp_path):
        with pytest.raises(ValueError, match="No openbench CSV found"):
            discover_openbench_csv(str(tmp_path))


# ── normalize_reasoning_flag ──────────────────────────────────────────────

class TestNormalizeReasoningFlag:
    @pytest.mark.parametrize("value,expected", [
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("false", False),
        ("False", False),
        ("1", True),
        ("0", False),
        ("yes", True),
        ("no", False),
        (True, True),
        (False, False),
        (1, True),
        (0, False),
    ])
    def test_various_inputs(self, value, expected):
        assert normalize_reasoning_flag(value) == expected


# ── extract_json_payload ──────────────────────────────────────────────────

class TestExtractJsonPayload:
    def test_bare_json(self):
        result = extract_json_payload('{"winner": "A", "score": 5}')
        assert result == {"winner": "A", "score": 5}

    def test_fenced_json(self):
        result = extract_json_payload('```json\n{"winner": "B"}\n```')
        assert result == {"winner": "B"}

    def test_json_with_surrounding_text(self):
        result = extract_json_payload('Here is the result: {"winner": "A"} and that is all.')
        assert result == {"winner": "A"}

    def test_nested_braces(self):
        result = extract_json_payload('{"data": {"inner": 1}}')
        assert result == {"data": {"inner": 1}}

    def test_invalid_json(self):
        result = extract_json_payload('{"broken": }')
        assert result is None

    def test_no_json(self):
        result = extract_json_payload("just some text with no json")
        assert result is None

    def test_empty_string(self):
        assert extract_json_payload("") is None

    def test_none_input(self):
        assert extract_json_payload(None) is None
