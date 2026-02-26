"""StyleBench benchmark adapter for the unified benchmark interface."""

import os
from typing import List, Set

import pandas as pd

from core.benchmark import Benchmark

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))

N_RUNS = 3
N_QUESTIONS = 9


class StyleBenchmark(Benchmark):

    @property
    def name(self) -> str:
        return "style"

    @property
    def bench_dir(self) -> str:
        return _BENCH_DIR

    @property
    def stages(self) -> List[str]:
        return ["collect", "judge", "score"]

    def stage_commands(self) -> List[str]:
        return [
            "python3 collect.py",
            "python3 super_bench.py",
            "python3 score.py",
        ]

    def get_completed_models(self) -> Set[str]:
        path = os.path.join(_BENCH_DIR, "responses.csv")
        if not os.path.exists(path):
            return set()
        try:
            df = pd.read_csv(path)
            # Only count models with all unique (question_id, run_number) pairs completed
            ok_df = df[df['status'] == 'ok']
            unique_counts = ok_df.groupby('model_name')[['question_id', 'run_number']].apply(
                lambda g: g.drop_duplicates().shape[0]
            )
            required = N_RUNS * N_QUESTIONS
            return {str(m).strip() for m, c in unique_counts.items() if c >= required}
        except (FileNotFoundError, pd.errors.EmptyDataError, KeyError):
            return set()
