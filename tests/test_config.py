"""Tests for core.config — central configuration."""

from core.config import BenchmarkConfig, BENCHMARK_DEFAULTS


class TestBenchmarkConfig:
    def test_defaults(self):
        cfg = BenchmarkConfig()
        assert cfg.judge_model == ""
        assert cfg.max_workers == 10
        assert cfg.max_retries == 3
        assert cfg.retry_delay == 5.0
        assert cfg.max_battles == 100
        assert cfg.draw_probability == 0.03

    def test_custom_values(self):
        cfg = BenchmarkConfig(judge_model="TestJudge", max_workers=5)
        assert cfg.judge_model == "TestJudge"
        assert cfg.max_workers == 5


class TestBenchmarkDefaults:
    def test_all_benchmarks_present(self):
        expected = {"eq", "writing", "logic", "style"}
        assert set(BENCHMARK_DEFAULTS.keys()) == expected

    def test_eq_config(self):
        cfg = BENCHMARK_DEFAULTS["eq"]
        assert cfg.judge_model == "Gemini 3.0 Flash Preview (2025-12-17)"
        assert cfg.max_workers == 40
        assert cfg.draw_probability == 0.03

    def test_writing_config(self):
        cfg = BENCHMARK_DEFAULTS["writing"]
        assert cfg.judge_model == "Grok 4 Fast"
        assert cfg.max_workers == 10
        assert cfg.draw_probability == 0.015

    def test_logic_config(self):
        cfg = BENCHMARK_DEFAULTS["logic"]
        assert cfg.max_workers == 20

    def test_style_workers_positive(self):
        cfg = BENCHMARK_DEFAULTS["style"]
        assert cfg.max_workers >= 1
