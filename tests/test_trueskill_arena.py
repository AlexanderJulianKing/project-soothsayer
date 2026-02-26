"""Tests for core.trueskill_arena — shared TrueSkill engine."""

import os
import pandas as pd
import pytest

from core.trueskill_arena import TrueSkillArena, ArenaConfig


@pytest.fixture
def eqbench_arena(tmp_path):
    """Create an arena configured like EQBench."""
    config = ArenaConfig(
        draw_probability=0.03,
        paired_mode=True,
        bench_prefix="eq_",
        results_dir=str(tmp_path / "results"),
        item_id_col="scenario_id",
        model_a_col="response_a_model",
        model_b_col="response_b_model",
        item_type="scenario",
        winner_label_a="A0493",
        winner_label_b="A0488",
    )
    return TrueSkillArena(config)


@pytest.fixture
def writingbench_arena(tmp_path):
    """Create an arena configured like WritingBench."""
    config = ArenaConfig(
        draw_probability=0.015,
        paired_mode=True,
        bench_prefix="writing_",
        results_dir=str(tmp_path / "results"),
        item_id_col="prompt_id",
        model_a_col="story_a_model",
        model_b_col="story_b_model",
        item_type="prompt",
        winner_label_a="A",
        winner_label_b="B",
    )
    return TrueSkillArena(config)


class TestArenaConfig:
    def test_defaults(self):
        cfg = ArenaConfig()
        assert cfg.draw_probability == 0.03
        assert cfg.paired_mode is True
        assert cfg.item_id_col == "item_id"
        assert cfg.item_type == "item"

    def test_battle_history_csv_derived(self, tmp_path):
        cfg = ArenaConfig(results_dir=str(tmp_path / "results"))
        assert cfg.battle_history_csv == os.path.join(str(tmp_path / "results"), "battle_history.csv")

    def test_history_columns_include_item_col(self):
        cfg = ArenaConfig(item_id_col="scenario_id", model_a_col="response_a_model")
        assert "scenario_id" in cfg.history_columns
        assert "response_a_model" in cfg.history_columns


class TestBuildBattleKey:
    def test_eqbench_format(self, eqbench_arena):
        key = eqbench_arena.build_battle_key("ModelA", "ModelB", "1", "Judge")
        assert key == "ModelA__A_vs_B__ModelB__scenario_1__Judge"

    def test_writingbench_format(self, writingbench_arena):
        key = writingbench_arena.build_battle_key("ModelA", "ModelB", "0", "Judge")
        assert key == "ModelA__A_vs_B__ModelB__prompt_0__Judge"


class TestBuildOrientationKey:
    def test_returns_tuple(self, eqbench_arena):
        key = eqbench_arena.build_orientation_key("1", "ModelA", "ModelB", "Judge")
        assert key == ("1", "ModelA", "ModelB", "Judge")


class TestComputeMatchInfoGain:
    def test_positive(self, eqbench_arena):
        r1 = eqbench_arena._create_rating()
        r2 = eqbench_arena._create_rating()
        gain = eqbench_arena.compute_match_info_gain(r1, r2)
        assert gain > 0

    def test_equal_ratings_highest_gain(self, eqbench_arena):
        r_equal = eqbench_arena._create_rating()
        import trueskill
        r_strong = trueskill.Rating(mu=40, sigma=1)
        r_weak = trueskill.Rating(mu=10, sigma=1)
        gain_equal = eqbench_arena.compute_match_info_gain(r_equal, r_equal)
        gain_mismatch = eqbench_arena.compute_match_info_gain(r_strong, r_weak)
        assert gain_equal > gain_mismatch


class TestLoadBattleHistory:
    def test_empty_when_no_file(self, eqbench_arena):
        df, keys = eqbench_arena.load_battle_history()
        assert df.empty
        assert len(keys) == 0
        assert "scenario_id" in df.columns
        assert "response_a_model" in df.columns

    def test_loads_existing_csv(self, eqbench_arena, tmp_path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        csv_path = results_dir / "battle_history.csv"
        csv_path.write_text(
            "battle_key,timestamp_utc,scenario_id,judge_model,judge_model_id,"
            "response_a_model,response_b_model,winner_model,loser_model,"
            "winner_label,criteria_json,raw_response\n"
            "key1,2025-01-01,1,Judge,j1,MA,MB,MA,MB,A0493,{},resp\n"
        )
        df, keys = eqbench_arena.load_battle_history()
        assert len(df) == 1
        assert "key1" in keys


class TestExtractExistingOrientations:
    def test_empty(self, eqbench_arena):
        df = pd.DataFrame()
        assert eqbench_arena.extract_existing_orientations(df) == set()

    def test_extracts_from_dataframe(self, eqbench_arena):
        df = pd.DataFrame({
            "scenario_id": ["1", "1"],
            "response_a_model": ["MA", "MB"],
            "response_b_model": ["MB", "MA"],
            "judge_model": ["Judge", "Judge"],
        })
        orientations = eqbench_arena.extract_existing_orientations(df)
        assert ("1", "MA", "MB", "Judge") in orientations
        assert ("1", "MB", "MA", "Judge") in orientations


class TestComputeBattleCounts:
    def test_empty(self, eqbench_arena):
        df = pd.DataFrame()
        assert eqbench_arena.compute_battle_counts(df) == {}

    def test_counts_models(self, eqbench_arena):
        df = pd.DataFrame({
            "winner_model": ["MA", "MA", "MB"],
            "loser_model": ["MB", "MC", "MA"],
        })
        counts = eqbench_arena.compute_battle_counts(df)
        assert counts["MA"] == 3  # 2 wins + 1 loss
        assert counts["MB"] == 2  # 1 loser + 1 winner
        assert counts["MC"] == 1  # 1 loser only


class TestComputeTrueSkillRatings:
    def test_empty(self, eqbench_arena):
        df = pd.DataFrame()
        assert eqbench_arena.compute_trueskill_ratings(df) == {}

    def test_winner_gets_higher_mu(self, eqbench_arena):
        df = pd.DataFrame({
            "timestamp_utc": ["2025-01-01"] * 5,
            "model_1": ["ModelA"] * 5,
            "model_2": ["ModelB"] * 5,
            "model_1_margin": [10] * 5,
            "model_2_margin": [2] * 5,
            "result": ["model_1"] * 5,
        })
        ratings = eqbench_arena.compute_trueskill_ratings(df)
        assert ratings["ModelA"].mu > ratings["ModelB"].mu

    def test_draw_handling(self, eqbench_arena):
        df = pd.DataFrame({
            "timestamp_utc": ["2025-01-01"],
            "model_1": ["ModelA"],
            "model_2": ["ModelB"],
            "model_1_margin": [5],
            "model_2_margin": [5],
            "result": ["draw"],
        })
        ratings = eqbench_arena.compute_trueskill_ratings(df)
        assert "ModelA" in ratings
        assert "ModelB" in ratings
        # After one draw, both should have similar mu
        assert abs(ratings["ModelA"].mu - ratings["ModelB"].mu) < 0.01


class TestSelectMatchesByInfoGain:
    def test_prioritizes_pair_completion(self, eqbench_arena):
        pending = [("1", "A", "B"), ("1", "B", "A"), ("2", "C", "D")]
        priority = {("1", "B", "A")}
        ratings = {}
        selected = eqbench_arena.select_matches_by_info_gain(pending, priority, ratings, 2)
        # Priority match should be first
        assert ("1", "B", "A") in selected

    def test_respects_max_battles(self, eqbench_arena):
        pending = [("1", "A", "B"), ("2", "C", "D"), ("3", "E", "F")]
        selected = eqbench_arena.select_matches_by_info_gain(pending, set(), {}, 1)
        assert len(selected) == 1


class TestListPendingMatches:
    def test_finds_common_items(self, eqbench_arena):
        content_index = {
            "ModelA": {"1": "data", "2": "data"},
            "ModelB": {"1": "data", "3": "data"},
        }
        pending, priority = eqbench_arena.list_pending_matches(
            content_index, set(), "Judge"
        )
        # Only item "1" is common, both orientations
        assert len(pending) == 2
        assert ("1", "ModelA", "ModelB") in pending
        assert ("1", "ModelB", "ModelA") in pending

    def test_skips_completed(self, eqbench_arena):
        content_index = {
            "ModelA": {"1": "data"},
            "ModelB": {"1": "data"},
        }
        existing = {("1", "ModelA", "ModelB", "Judge")}
        pending, priority = eqbench_arena.list_pending_matches(
            content_index, existing, "Judge"
        )
        # Only the reverse orientation should be pending
        assert len(pending) == 1
        assert ("1", "ModelB", "ModelA") in pending
        # And it should be priority (completing the pair)
        assert ("1", "ModelB", "ModelA") in priority

    def test_item_filter(self, eqbench_arena):
        content_index = {
            "ModelA": {"1": "data", "2": "data"},
            "ModelB": {"1": "data", "2": "data"},
        }
        pending, _ = eqbench_arena.list_pending_matches(
            content_index, set(), "Judge", item_filter={"1"}
        )
        item_ids = {p[0] for p in pending}
        assert "1" in item_ids
        assert "2" not in item_ids


class TestPositionBiasSummary:
    def test_eqbench_labels(self, eqbench_arena, capsys):
        df = pd.DataFrame({
            "winner_label": ["A0493", "A0493", "A0488"],
            "judge_model": ["Judge", "Judge", "Judge"],
        })
        eqbench_arena.summarize_position_bias(df, "Judge")
        output = capsys.readouterr().out
        assert "Position bias" in output
        assert "3 battles" in output

    def test_writingbench_labels(self, writingbench_arena, capsys):
        df = pd.DataFrame({
            "winner_label": ["A", "B", "A"],
            "judge_model": ["Judge", "Judge", "Judge"],
        })
        writingbench_arena.summarize_position_bias(df, "Judge")
        output = capsys.readouterr().out
        assert "Position bias" in output
        assert "A=67%" in output
