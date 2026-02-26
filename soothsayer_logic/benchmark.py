"""SimpleBench benchmark adapter for the unified benchmark interface."""

import os
from typing import List, Set

import pandas as pd

from core.benchmark import Benchmark

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))


class LogicBenchmark(Benchmark):

    @property
    def name(self) -> str:
        return "logic"

    @property
    def bench_dir(self) -> str:
        return _BENCH_DIR

    @property
    def stages(self) -> List[str]:
        return ["assess", "score"]

    def stage_commands(self) -> List[str]:
        return ["python3 collect_and_grade.py", "python3 score.py"]

    def get_completed_models(self) -> Set[str]:
        path = os.path.join(_BENCH_DIR, "benchmark_results_multi_run.csv")
        if not os.path.exists(path):
            return set()
        return {str(s).strip() for s in pd.read_csv(path)["model_name"].dropna().unique()}
