"""EQBench benchmark adapter for the unified benchmark interface."""

import os
from typing import List, Set

from core.benchmark import Benchmark

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))


class EQBenchmark(Benchmark):

    @property
    def name(self) -> str:
        return "eq"

    @property
    def bench_dir(self) -> str:
        return _BENCH_DIR

    @property
    def stages(self) -> List[str]:
        return ["generate", "judge"]

    def stage_commands(self) -> List[str]:
        return ["python3 main.py", "python3 super_bench.py"]

    def get_completed_models(self) -> Set[str]:
        base = os.path.join(_BENCH_DIR, "generated_responses")
        if not os.path.isdir(base):
            return set()
        return {s.strip() for s in os.listdir(base)}
