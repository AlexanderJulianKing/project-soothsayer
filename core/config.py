"""Central configuration for all benchmarks."""
import os
from dataclasses import dataclass, field


@dataclass
class BenchmarkConfig:
    """Per-benchmark configuration."""
    judge_model: str = ""
    max_workers: int = 10
    max_retries: int = 3
    retry_delay: float = 5.0
    max_battles: int = 20
    draw_probability: float = 0.03


# Defaults per benchmark (overridable via CLI args)
BENCHMARK_DEFAULTS = {
    "eq": BenchmarkConfig(
        judge_model="Gemini 3.0 Flash Preview (2025-12-17)",
        max_workers=40,
        draw_probability=0.03,
    ),
    "writing": BenchmarkConfig(
        judge_model="Grok 4 Fast",
        max_workers=10,
        draw_probability=0.015,
    ),
    "logic": BenchmarkConfig(
        judge_model="Gemini 3.0 Flash Preview (2025-12-17)",
        max_workers=20,
    ),
    "style": BenchmarkConfig(
        max_workers=os.cpu_count() or 4,
    ),
}
