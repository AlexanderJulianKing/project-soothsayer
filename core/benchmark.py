"""Abstract benchmark interface.

All 4 benchmarks implement this so the orchestrator (cli.py) can treat
them uniformly: discover completed models, run stages, collect results.
"""

import os
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Set

from core.config import BenchmarkConfig


@dataclass
class BenchmarkResult:
    """Standard result from a benchmark stage run."""
    stage: str
    exit_code: int
    log_output: str = ""


class Benchmark(ABC):
    """Base class for all benchmarks."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short name: 'eq', 'writing', 'logic', 'style'"""

    @property
    @abstractmethod
    def bench_dir(self) -> str:
        """Absolute path to the benchmark's working directory."""

    @property
    @abstractmethod
    def stages(self) -> List[str]:
        """Ordered list of stage names, e.g. ['generate', 'judge']"""

    @abstractmethod
    def stage_commands(self) -> List[str]:
        """Return the shell command for each stage (same order as stages)."""

    @abstractmethod
    def get_completed_models(self) -> Set[str]:
        """Return set of model names already fully evaluated."""

    def run_stage(self, stage: str, config: BenchmarkConfig = None) -> BenchmarkResult:
        """Run a single stage of the benchmark pipeline via subprocess."""
        try:
            idx = self.stages.index(stage)
        except ValueError:
            return BenchmarkResult(stage=stage, exit_code=1, log_output=f"Unknown stage: {stage}")

        cmd = self.stage_commands()[idx]
        result = subprocess.run(
            cmd, shell=True, cwd=self.bench_dir,
            capture_output=True, text=True,
        )
        output = result.stdout + result.stderr
        return BenchmarkResult(stage=stage, exit_code=result.returncode, log_output=output)

    def run_all(self, config: BenchmarkConfig = None) -> List[BenchmarkResult]:
        """Run all stages sequentially, stopping on first failure."""
        results = []
        for stage in self.stages:
            result = self.run_stage(stage, config)
            results.append(result)
            if result.exit_code != 0:
                break
        return results
