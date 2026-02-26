"""Shared fixtures for the Soothsayer test suite."""

import os
import tempfile

import pytest


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory that cleans up automatically."""
    return tmp_path


@pytest.fixture
def sample_openbench_csv(tmp_path):
    """Create a sample openbench CSV for testing."""
    csv_content = (
        "Model,openbench_id,Reasoning\n"
        "GPT-4o,openai/gpt-4o,true\n"
        "Claude 3.5 Sonnet,anthropic/claude-3.5-sonnet,false\n"
        "Gemini Pro,google/gemini-pro,true\n"
        "Bad Model,,true\n"  # Missing openbench_id — should be dropped
    )
    csv_path = tmp_path / "openbench_20250101.csv"
    csv_path.write_text(csv_content)
    return str(csv_path)


@pytest.fixture
def sample_openbench_csv_lowercase_reasoning(tmp_path):
    """Create a sample openbench CSV with lowercase 'reasoning' column (soothsayer_style variant)."""
    csv_content = (
        "Model,openbench_id,reasoning\n"
        "GPT-4o,openai/gpt-4o,true\n"
        "Claude 3.5 Sonnet,anthropic/claude-3.5-sonnet,false\n"
    )
    csv_path = tmp_path / "openbench_20250201.csv"
    csv_path.write_text(csv_content)
    return str(csv_path)
