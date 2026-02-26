"""Shared TrueSkill pairwise comparison engine.

Used by eqbench and writingbench (and future benchmarks) for
head-to-head LLM evaluation with information-gain-based match selection.

The engine uses generic column names configured via ArenaConfig so the
same code handles EQBench's ``scenario_id / response_a_model`` columns
and WritingBench's ``prompt_id / story_a_model`` columns.
"""

import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

import pandas as pd
import trueskill


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ArenaConfig:
    """Configuration for a TrueSkill arena."""
    draw_probability: float = 0.03
    max_parse_attempts: int = 3
    paired_mode: bool = True
    bench_prefix: str = "bench_"
    results_dir: str = "results"
    battle_history_csv: str = ""  # set from results_dir if empty

    # Column mapping — adapters set these to match their CSV schema
    item_id_col: str = "item_id"        # "scenario_id" or "prompt_id"
    model_a_col: str = "model_a"        # "response_a_model" or "story_a_model"
    model_b_col: str = "model_b"        # "response_b_model" or "story_b_model"

    # Short label used in battle key formatting (e.g. "scenario" or "prompt")
    item_type: str = "item"

    # Winner labels used in the raw response (for position-bias reporting)
    winner_label_a: str = "A"
    winner_label_b: str = "B"

    # History CSV columns (derived from the above)
    history_columns: list = field(default_factory=list)

    def __post_init__(self):
        if not self.battle_history_csv:
            self.battle_history_csv = os.path.join(self.results_dir, "battle_history.csv")
        if not self.history_columns:
            self.history_columns = [
                "battle_key",
                "timestamp_utc",
                self.item_id_col,
                "judge_model",
                "judge_model_id",
                self.model_a_col,
                self.model_b_col,
                "winner_model",
                "loser_model",
                "winner_label",
                "criteria_json",
                "raw_response",
            ]


# ---------------------------------------------------------------------------
# TrueSkill Arena
# ---------------------------------------------------------------------------

class TrueSkillArena:
    """Shared TrueSkill pairwise comparison engine."""

    def __init__(self, config: ArenaConfig):
        self.cfg = config
        self.ts_env = trueskill.TrueSkill(draw_probability=config.draw_probability)

    # --- helpers -----------------------------------------------------------

    def _create_rating(self) -> trueskill.Rating:
        return self.ts_env.create_rating()

    def ensure_player(self, ratings: Dict[str, trueskill.Rating], player: str):
        if player not in ratings:
            ratings[player] = self._create_rating()

    # --- key builders (use column mapping) ---------------------------------

    def build_battle_key(self, model_a: str, model_b: str, item_id: str, judge_name: str) -> str:
        return f"{model_a}__A_vs_B__{model_b}__{self.cfg.item_type}_{item_id}__{judge_name}"

    def build_orientation_key(self, item_id: str, model_a: str, model_b: str, judge_name: str) -> Tuple[str, str, str, str]:
        return (str(item_id), model_a, model_b, judge_name)

    # --- info gain ---------------------------------------------------------

    def compute_match_info_gain(self, rating_a: trueskill.Rating, rating_b: trueskill.Rating) -> float:
        quality = trueskill.quality_1vs1(rating_a, rating_b)
        combined_sigma = rating_a.sigma + rating_b.sigma
        return combined_sigma * quality

    # --- history I/O -------------------------------------------------------

    def load_battle_history(self) -> Tuple[pd.DataFrame, set]:
        if os.path.exists(self.cfg.battle_history_csv):
            history_df = pd.read_csv(self.cfg.battle_history_csv)
        else:
            history_df = pd.DataFrame(columns=self.cfg.history_columns)
        for col in self.cfg.history_columns:
            if col not in history_df.columns:
                history_df[col] = pd.Series(dtype="object")
        known_keys = set(history_df["battle_key"].dropna().tolist())
        return history_df, known_keys

    def extract_existing_orientations(self, history_df: pd.DataFrame) -> set:
        orientations = set()
        if history_df.empty:
            return orientations
        item_col = self.cfg.item_id_col
        ma_col = self.cfg.model_a_col
        mb_col = self.cfg.model_b_col
        for _, row in history_df.iterrows():
            item_id = row.get(item_col)
            model_a = row.get(ma_col)
            model_b = row.get(mb_col)
            judge_model = row.get("judge_model")
            if not all(isinstance(val, str) for val in (str(item_id), model_a, model_b)):
                continue
            if not isinstance(judge_model, str):
                continue
            orientations.add(self.build_orientation_key(str(item_id), model_a, model_b, judge_model))
        return orientations

    # --- battle counts / stats ---------------------------------------------

    def compute_battle_counts(self, history_df: pd.DataFrame) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        if history_df.empty:
            return counts
        for _, row in history_df.iterrows():
            for col in ("winner_model", "loser_model"):
                model = row.get(col)
                if isinstance(model, str):
                    counts[model] = counts.get(model, 0) + 1
        return counts

    # --- TrueSkill ratings -------------------------------------------------

    def compute_trueskill_ratings(self, pairs_df: pd.DataFrame) -> Dict[str, trueskill.Rating]:
        ratings: Dict[str, trueskill.Rating] = {}
        if pairs_df.empty:
            return ratings

        pairs_df = pairs_df.copy()
        pairs_df["timestamp_utc"] = pd.to_datetime(pairs_df["timestamp_utc"])
        pairs_df.sort_values("timestamp_utc", inplace=True)

        for _, row in pairs_df.iterrows():
            model_1 = row.get("model_1")
            model_2 = row.get("model_2")
            result = row.get("result")
            if not isinstance(model_1, str) or not isinstance(model_2, str):
                continue

            self.ensure_player(ratings, model_1)
            self.ensure_player(ratings, model_2)

            m1_margin = row.get("model_1_margin")
            m2_margin = row.get("model_2_margin")

            if isinstance(m1_margin, (int, float)) and isinstance(m2_margin, (int, float)):
                if m1_margin > m2_margin:
                    winner, loser = model_1, model_2
                elif m2_margin > m1_margin:
                    winner, loser = model_2, model_1
                else:
                    new_r1, new_r2 = self.ts_env.rate_1vs1(ratings[model_1], ratings[model_2], drawn=True)
                    ratings[model_1], ratings[model_2] = new_r1, new_r2
                    continue
            else:
                if result == "model_1":
                    winner, loser = model_1, model_2
                elif result == "model_2":
                    winner, loser = model_2, model_1
                else:
                    new_r1, new_r2 = self.ts_env.rate_1vs1(ratings[model_1], ratings[model_2], drawn=True)
                    ratings[model_1], ratings[model_2] = new_r1, new_r2
                    continue

            new_winner, new_loser = self.ts_env.rate_1vs1(ratings[winner], ratings[loser])
            ratings[winner], ratings[loser] = new_winner, new_loser

        return ratings

    # --- match selection ---------------------------------------------------

    def select_matches_by_info_gain(
        self,
        pending_matches: List[Tuple[str, str, str]],
        priority_matches: Set[Tuple[str, str, str]],
        ratings: Dict[str, trueskill.Rating],
        max_battles: int,
    ) -> List[Tuple[str, str, str]]:
        selection = []

        # FIRST: Complete pairs
        priority_list = [m for m in pending_matches if m in priority_matches]
        for match in priority_list:
            if len(selection) >= max_battles:
                break
            selection.append(match)

        if len(selection) >= max_battles:
            return selection

        # SECOND: Score remaining by information gain
        non_priority = [m for m in pending_matches if m not in priority_matches]

        def get_rating(model: str) -> trueskill.Rating:
            return ratings.get(model, self._create_rating())

        pair_to_matches: Dict[tuple, list] = {}
        for match in non_priority:
            _, model_a, model_b = match
            pair_key = tuple(sorted([model_a, model_b]))
            pair_to_matches.setdefault(pair_key, []).append(match)

        scored_matches = []
        for pair_key, matches in pair_to_matches.items():
            match = random.choice(matches)
            info_gain = self.compute_match_info_gain(get_rating(match[1]), get_rating(match[2]))
            scored_matches.append((info_gain, match))

        scored_matches.sort(key=lambda x: x[0], reverse=True)

        for info_gain, match in scored_matches:
            if len(selection) >= max_battles:
                break
            selection.append(match)

        return selection

    # --- pending matches (generic) -----------------------------------------

    def list_pending_matches(
        self,
        content_index: Dict[str, Dict[str, Any]],
        existing_orientations: set,
        judge_name: str,
        item_filter: Optional[Set[str]] = None,
        paired_mode: bool = True,
    ) -> Tuple[List[Tuple[str, str, str]], Set[Tuple[str, str, str]]]:
        """List pending matches for models in content_index.

        content_index maps model_name -> item_id -> content.
        """
        models = sorted(content_index.keys())
        pending = []
        priority = set()

        for i, model_a in enumerate(models):
            for model_b in models[i + 1:]:
                items_a = set(content_index[model_a].keys())
                items_b = set(content_index[model_b].keys())
                common = items_a & items_b
                if item_filter:
                    common = common & item_filter

                for item_id in common:
                    key_ab = self.build_orientation_key(item_id, model_a, model_b, judge_name)
                    key_ba = self.build_orientation_key(item_id, model_b, model_a, judge_name)

                    if key_ab not in existing_orientations:
                        pending.append((item_id, model_a, model_b))
                        if paired_mode and key_ba in existing_orientations:
                            priority.add((item_id, model_a, model_b))

                    if key_ba not in existing_orientations:
                        pending.append((item_id, model_b, model_a))
                        if paired_mode and key_ab in existing_orientations:
                            priority.add((item_id, model_b, model_a))

        return pending, priority

    # --- battle execution --------------------------------------------------

    def run_battles_batch(
        self,
        matches: List[Tuple[str, str, str]],
        content_index: Dict[str, Dict[str, Any]],
        judge: Dict[str, str],
        run_single_battle_fn: Callable,
        workers: int,
        **extra_kwargs,
    ) -> List[Dict[str, str]]:
        """Run a batch of battles in parallel.

        run_single_battle_fn signature:
            (item_id, model_a, model_b, content_index, judge, battle_key, **extra) -> dict
        """
        records = []
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            future_map = {}
            for item_id, model_a, model_b in matches:
                battle_key = self.build_battle_key(model_a, model_b, item_id, judge["name"])
                future = executor.submit(
                    run_single_battle_fn,
                    item_id, model_a, model_b,
                    content_index, judge, battle_key,
                    **extra_kwargs,
                )
                future_map[future] = battle_key

            for future in as_completed(future_map):
                battle_key = future_map[future]
                try:
                    record = future.result()
                    records.append(record)
                    print(f"\u2713 {battle_key} -> winner: {record['winner_model']}")
                except Exception as exc:
                    print(f"\u2717 {battle_key}: {exc}")

        return records

    def run_info_gain_loop(
        self,
        content_index: Dict[str, Dict[str, Any]],
        judge: Dict[str, str],
        run_single_battle_fn: Callable,
        build_paired_results_fn: Callable,
        max_battles: int,
        workers: int,
        item_filter: Optional[Set[str]] = None,
        list_pending_matches_fn: Optional[Callable] = None,
        **extra_kwargs,
    ) -> Tuple[pd.DataFrame, List[Dict[str, str]]]:
        """Run the info-gain batch loop.

        Args:
            list_pending_matches_fn: Optional callback with signature
                (content_index, existing_orientations, judge_name, item_filter, paired_mode)
                that returns (pending_matches, priority_matches). If None, uses
                the arena's built-in list_pending_matches method.

        Returns (updated history_df, all_new_records).
        """
        history_df, _ = self.load_battle_history()
        existing_orientations = self.extract_existing_orientations(history_df)

        all_new_records: List[Dict[str, str]] = []
        remaining_limit = max_battles

        pair_df = build_paired_results_fn(history_df, judge["name"])
        ratings = self.compute_trueskill_ratings(pair_df)

        print(f"\n=== INFO-GAIN MATCH SELECTION ===")

        batch_num = 0
        while remaining_limit > 0:
            batch_num += 1

            if list_pending_matches_fn:
                pending_matches, priority_matches = list_pending_matches_fn(
                    content_index, existing_orientations, judge["name"],
                    item_filter, paired_mode=self.cfg.paired_mode,
                )
            else:
                pending_matches, priority_matches = self.list_pending_matches(
                    content_index, existing_orientations, judge["name"],
                    item_filter, paired_mode=self.cfg.paired_mode,
                )

            if not pending_matches:
                print(f"\nNo pending matches remaining.")
                break

            batch_size = min(workers * 2, remaining_limit)
            batch = self.select_matches_by_info_gain(
                pending_matches, priority_matches, ratings, batch_size
            )

            if not batch:
                print(f"\nNo matches selected.")
                break

            priority_count = sum(1 for m in batch if m in priority_matches)
            print(f"\nBatch {batch_num}: Running {len(batch)} battles "
                  f"({priority_count} completing pairs, {len(priority_matches)} incomplete pairs)")

            records = self.run_battles_batch(
                batch, content_index, judge,
                run_single_battle_fn, workers,
                **extra_kwargs,
            )

            if records:
                history_df = pd.concat([history_df, pd.DataFrame(records)], ignore_index=True)
                existing_orientations = self.extract_existing_orientations(history_df)
                all_new_records.extend(records)
                remaining_limit -= len(records)

                pair_df = build_paired_results_fn(history_df, judge["name"])
                ratings = self.compute_trueskill_ratings(pair_df)

                print(f"  Completed {len(records)} battles, {remaining_limit} remaining in budget")
            else:
                print(f"  No battles completed")
                break

        return history_df, all_new_records

    # --- summary / reporting -----------------------------------------------

    def summarize_position_bias(self, history_df: pd.DataFrame, judge_name: Optional[str] = None) -> None:
        if history_df.empty or "winner_label" not in history_df.columns:
            print("\nNo battle history available to analyze A/B bias.")
            return

        if judge_name and "judge_model" in history_df.columns:
            history_df = history_df[history_df["judge_model"] == judge_name]
            if history_df.empty:
                print(f"\nNo battle history for judge '{judge_name}' to analyze A/B bias.")
                return

        label_series = history_df["winner_label"].dropna()
        total = len(label_series)
        if total == 0:
            print("\nNo battle history available to analyze A/B bias.")
            return

        counts = label_series.value_counts()
        wins_a = counts.get(self.cfg.winner_label_a, 0)
        wins_b = counts.get(self.cfg.winner_label_b, 0)

        judge_label = f" ({judge_name})" if judge_name else ""
        bias_pct = abs(wins_a / total - 0.5) * 100 if total else 0
        bias_note = f" (bias: {bias_pct:.0f}%)" if bias_pct > 10 else ""
        print(f"Position bias{judge_label}: {total} battles, A={wins_a / total:.0%} B={wins_b / total:.0%}{bias_note}")

    def summarize_win_loss_stats(self, history_df: pd.DataFrame, judge_name: Optional[str] = None) -> None:
        if history_df.empty or "winner_model" not in history_df.columns:
            print("\nNo battle history available to summarize W/L stats.")
            return

        if judge_name and "judge_model" in history_df.columns:
            history_df = history_df[history_df["judge_model"] == judge_name]
            if history_df.empty:
                print(f"\nNo battle history for judge '{judge_name}' to summarize W/L stats.")
                return

        wins = history_df["winner_model"].value_counts()
        losses = history_df["loser_model"].value_counts() if "loser_model" in history_df.columns else pd.Series(dtype=int)

        all_models = set(wins.index) | set(losses.index)
        if not all_models:
            print("\nNo W/L data available.")
            return

        stats = []
        for model in all_models:
            w = wins.get(model, 0)
            l = losses.get(model, 0)
            total = w + l
            win_rate = w / total if total > 0 else 0.0
            stats.append((model, w, l, total, win_rate))

        stats.sort(key=lambda x: (x[4], x[3]), reverse=True)

        judge_label = f" ({judge_name})" if judge_name else ""
        top3 = stats[:3]
        bot3 = stats[-3:] if len(stats) > 3 else []
        top_str = ", ".join(f"{m} ({wr:.0%})" for m, w, l, t, wr in top3)
        bot_str = ", ".join(f"{m} ({wr:.0%})" for m, w, l, t, wr in bot3)
        print(f"W/L stats{judge_label}: {len(stats)} models. Top: {top_str}" + (f"  Bot: {bot_str}" if bot_str else ""))
