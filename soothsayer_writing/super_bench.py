"""
WritingBench - Pairwise comparison and TrueSkill rating system.
Head-to-head creative writing evaluation with info-gain match selection.

This module is a thin adapter that delegates shared TrueSkill logic
to core.trueskill_arena and keeps only WritingBench-specific functions:
criteria, prompt building, validation, run_battle, paired results,
result augmentation, and WritingBench-specific reporting.
"""

import argparse
import datetime as dt
import itertools
import json
import os
import re
import sys
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from core.utils import (
    get_latest_file,
    load_models,
    normalize_reasoning_flag,
    extract_json_payload,
    discover_openbench_csv,
)
from core.trueskill_arena import TrueSkillArena, ArenaConfig

from llm_client import get_llm_response

STORIES_DIR = "generated_stories"
RESULTS_DIR = "results"
JUDGING_PROMPT_FILE = "judging_prompt.txt"
BATTLE_PAIRS_CSV = os.path.join(RESULTS_DIR, "battle_pairs.csv")
SUPER_BENCH_PREFIX = "writing_"
DEFAULT_JUDGE_NAME = "Grok 4 Fast"
MAX_PARSE_ATTEMPTS = 3
DEFAULT_BATTLES_TO_RUN = 125

# Arena configuration
ARENA_CONFIG = ArenaConfig(
    draw_probability=0.015,
    paired_mode=True,
    bench_prefix=SUPER_BENCH_PREFIX,
    results_dir=RESULTS_DIR,
    item_id_col="prompt_id",
    model_a_col="story_a_model",
    model_b_col="story_b_model",
    item_type="prompt",
    winner_label_a="A",
    winner_label_b="B",
)
arena = TrueSkillArena(ARENA_CONFIG)

CRITERIA = [
    "Character authenticity and insight",
    "Interesting and original",
    "Writing quality",
    "Coherence in plot, character choices, metaphor",
    "Instruction following (followed the prompt)",
    "World and atmosphere",
    "Avoids cliches in characters, dialogue & plot",
    "Avoids flowery verbosity & show-offy vocab maxxing",
    "Avoids gratuitous metaphor or poetic overload",
]

SYSTEM_PROMPT = (
    "You are an impartial literary tournament judge. "
    "You must follow the provided rubric, compare the two anonymous stories, pick a winner, "
    "and respond with valid JSON only."
)


def resolve_judge_model(judge_name: str) -> Dict[str, str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_csv = discover_openbench_csv(script_dir)
    for model in load_models(model_csv):
        if model["name"] == judge_name:
            if not model.get("id"):
                raise ValueError(f"Judge '{judge_name}' is missing an openbench_id.")
            return {
                "name": model["name"],
                "id": model["id"],
                "Reasoning": normalize_reasoning_flag(model.get("Reasoning", False)),
            }
    raise ValueError(f"Judge '{judge_name}' not found in {model_csv}.")


def load_story_index() -> Dict[str, Dict[str, str]]:
    story_index: Dict[str, Dict[str, str]] = {}
    if not os.path.isdir(STORIES_DIR):
        return story_index
    for writer_dir in os.listdir(STORIES_DIR):
        abs_writer_dir = os.path.join(STORIES_DIR, writer_dir)
        if not os.path.isdir(abs_writer_dir):
            continue
        prompt_map: Dict[str, str] = {}
        for fname in os.listdir(abs_writer_dir):
            if not fname.startswith("prompt_") or not fname.endswith(".txt"):
                continue
            prompt_id = fname[len("prompt_") : -4]
            fpath = os.path.join(abs_writer_dir, fname)
            with open(fpath, "r", encoding="utf-8") as story_file:
                prompt_map[prompt_id] = story_file.read().strip()
        if prompt_map:
            story_index[writer_dir] = prompt_map
    return story_index


# --- WritingBench-specific pending match logic ---
# (different from the arena's built-in: always returns ALL matches including
# completed ones, supporting multiple independent judgements per pairing)

def extract_orientation_counts(history_df: pd.DataFrame, judge_name: str) -> Dict[Tuple[str, str, str], int]:
    counts: Dict[Tuple[str, str, str], int] = {}
    if history_df.empty or "judge_model" not in history_df.columns:
        return counts

    filtered = history_df[
        (history_df["judge_model"] == judge_name)
        & history_df["story_a_model"].notna()
        & history_df["story_b_model"].notna()
        & history_df["prompt_id"].notna()
    ]

    for _, row in filtered.iterrows():
        prompt_id = str(row["prompt_id"])
        story_a = row["story_a_model"]
        story_b = row["story_b_model"]
        if not isinstance(story_a, str) or not isinstance(story_b, str):
            continue
        key = (prompt_id, story_a, story_b)
        counts[key] = counts.get(key, 0) + 1

    return counts


def build_pair_orientation_map(existing_orientations: set, judge_name: str) -> Dict[Tuple[str, Tuple[str, str]], Set[Tuple[str, str]]]:
    orientation_map: Dict[Tuple[str, Tuple[str, str]], Set[Tuple[str, str]]] = {}
    for prompt_id, story_a, story_b, judge in existing_orientations:
        if judge != judge_name:
            continue
        key = (prompt_id, tuple(sorted((story_a, story_b))))
        orientation_map.setdefault(key, set()).add((story_a, story_b))
    return orientation_map


def list_pending_matches(
    story_index: Dict[str, Dict[str, str]],
    existing_orientations: set,
    judge_name: str,
    prompt_filter: Optional[set] = None,
    paired_mode: bool = True,
) -> Tuple[List[Tuple[str, str, str]], Set[Tuple[str, str, str]]]:
    prompt_ids = sorted({str(pid) for prompts in story_index.values() for pid in prompts})
    if prompt_filter is not None:
        prompt_filter = {str(pid) for pid in prompt_filter}

    orientation_map = build_pair_orientation_map(existing_orientations, judge_name)
    matches: List[Tuple[str, str, str]] = []
    priority_orientations: Set[Tuple[str, str, str]] = set()

    for prompt_id in prompt_ids:
        if prompt_filter and prompt_id not in prompt_filter:
            continue
        contenders = sorted([model for model, prompts in story_index.items() if prompt_id in prompts])
        if len(contenders) < 2:
            continue
        for model_a, model_b in itertools.combinations(contenders, 2):
            pair_key = (prompt_id, tuple(sorted((model_a, model_b))))
            seen_orientations = orientation_map.get(pair_key, set())

            # Prioritize completing the missing reverse orientation when exactly one exists.
            if len(seen_orientations) == 1:
                seen_a, seen_b = next(iter(seen_orientations))
                priority_orientations.add((prompt_id, seen_b, seen_a))

            if paired_mode:
                matches.append((prompt_id, model_a, model_b))
                matches.append((prompt_id, model_b, model_a))
            else:
                matches.append((prompt_id, model_a, model_b))
    return matches, priority_orientations


# --- Battle logic ---

def margin_totals_for_battle(row: pd.Series) -> Optional[Dict[str, int]]:
    criteria_json = row.get("criteria_json")
    model_a = row.get("story_a_model")
    model_b = row.get("story_b_model")
    if not isinstance(criteria_json, str) or not isinstance(model_a, str) or not isinstance(model_b, str):
        return None
    try:
        criteria = json.loads(criteria_json)
    except Exception:
        return None

    totals = {model_a: 0, model_b: 0}
    for item in criteria:
        winner_model = item.get("winner")
        if winner_model not in totals:
            continue
        margin_score = item.get("margin_score")
        if not isinstance(margin_score, (int, float)):
            margin_score = str(item.get("margin", "")).count("+")
        totals[winner_model] += int(margin_score)
    return totals


def build_comparison_prompt(instructions: str, prompt_id: str, model_a: str, story_a: str, model_b: str, story_b: str) -> str:
    criteria_requirements = "\n".join([f"- {item}" for item in CRITERIA])
    criteria_schema = ",\n".join([f'    {{"name": "{item}"}}' for item in CRITERIA])
    return f"""{instructions.strip()}

You must score BOTH stories head-to-head for the criteria above. Each criterion needs a winner (no draws) and a strength indicator using + through +++++.

Story A (prompt {prompt_id}):
{story_a.strip()}

---

Story B (prompt {prompt_id}):
{story_b.strip()}

Return STRICT JSON that matches this schema exactly:
{{
  "winner": "A" or "B",
  "criteria": [
{criteria_schema}
  ],
  "notes": "2 concise sentences explaining the decision, refer to 'Story A' or 'Story B' only"
}}

For every entry inside criteria include both keys "winner" ("A" or "B") and "margin" ("+" | "++" | "+++" | "++++" | "+++++").
"""


def validate_payload(payload: dict) -> bool:
    if not isinstance(payload, dict):
        return False
    winner = payload.get("winner")
    if winner not in {"A", "B"}:
        return False
    criteria = payload.get("criteria")
    if not isinstance(criteria, list) or len(criteria) != len(CRITERIA):
        return False
    seen = []
    for row in criteria:
        if not isinstance(row, dict):
            return False
        if row.get("name") not in CRITERIA:
            return False
        if row.get("winner") not in {"A", "B"}:
            return False
        if row.get("margin") not in {"+", "++", "+++", "++++", "+++++"}:
            return False
        seen.append(row["name"])
    return set(seen) == set(CRITERIA)


def run_battle(
    prompt_id: str,
    model_a: str,
    model_b: str,
    story_index: Dict[str, Dict[str, str]],
    judge: Dict[str, str],
    battle_key: str,
    instructions: str = "",
) -> Dict[str, str]:
    """Run a single pairwise comparison battle.

    Signature matches the arena's run_single_battle_fn callback convention:
    (item_id, model_a, model_b, content_index, judge, battle_key, **extra)
    """
    prompt = build_comparison_prompt(
        instructions=instructions,
        prompt_id=prompt_id,
        model_a=model_a,
        story_a=story_index[model_a][prompt_id],
        model_b=model_b,
        story_b=story_index[model_b][prompt_id],
    )

    last_error = None
    for attempt in range(1, MAX_PARSE_ATTEMPTS + 1):
        raw_response = get_llm_response(
            prompt=prompt,
            model=judge["id"],
            name=judge["name"],
            reasoning=judge["Reasoning"],
            system_prompt=SYSTEM_PROMPT,
        )
        payload = extract_json_payload(raw_response)
        if not payload or not validate_payload(payload):
            last_error = ValueError(f"Attempt {attempt}: invalid JSON payload")
            continue

        winner_label = payload["winner"]
        winner_model = model_a if winner_label == "A" else model_b
        loser_model = model_b if winner_model == model_a else model_a

        criteria_records = []
        for row in payload["criteria"]:
            criteria_records.append(
                {
                    "name": row["name"],
                    "winner": model_a if row["winner"] == "A" else model_b,
                    "margin": row["margin"],
                    "margin_score": row["margin"].count("+"),
                }
            )

        return {
            "battle_key": battle_key,
            "timestamp_utc": dt.datetime.utcnow().isoformat(),
            "prompt_id": prompt_id,
            "judge_model": judge["name"],
            "judge_model_id": judge["id"],
            "story_a_model": model_a,
            "story_b_model": model_b,
            "winner_model": winner_model,
            "loser_model": loser_model,
            "winner_label": winner_label,
            "criteria_json": json.dumps(criteria_records),
            "raw_response": raw_response.strip(),
        }

    raise RuntimeError(f"Failed to obtain a valid judgement for {battle_key}: {last_error}")


def build_paired_results(history_df: pd.DataFrame, judge_name: str) -> pd.DataFrame:
    """Build paired results from battle history for TrueSkill calculation.

    WritingBench-specific: handles multiple forward/reverse pairs per
    (prompt_id, model pair) and uses margin_totals_for_battle().
    """
    pair_columns = [
        "pair_id",
        "timestamp_utc",
        "prompt_id",
        "judge_model",
        "model_1",
        "model_2",
        "model_1_margin",
        "model_2_margin",
        "result",
        "battle_key_a",
        "battle_key_b",
    ]

    if history_df.empty or "criteria_json" not in history_df.columns:
        return pd.DataFrame(columns=pair_columns)

    filtered = history_df[
        (history_df["judge_model"] == judge_name)
        & history_df["prompt_id"].notna()
        & history_df["story_a_model"].notna()
        & history_df["story_b_model"].notna()
        & history_df["criteria_json"].notna()
    ].copy()

    if filtered.empty:
        return pd.DataFrame(columns=pair_columns)

    filtered["prompt_id"] = filtered["prompt_id"].astype(str)
    filtered["timestamp_utc"] = pd.to_datetime(filtered["timestamp_utc"])
    filtered.sort_values("timestamp_utc", inplace=True)

    pair_records: List[Dict[str, object]] = []

    for pair_key, group in filtered.groupby(
        filtered.apply(
            lambda r: (r["prompt_id"], tuple(sorted((r["story_a_model"], r["story_b_model"])))),
            axis=1,
        )
    ):
        prompt_id, (model_x, model_y) = pair_key
        orientation_buckets: Dict[Tuple[str, str], List[pd.Series]] = {}
        for _, row in group.iterrows():
            orientation = (row["story_a_model"], row["story_b_model"])
            orientation_buckets.setdefault(orientation, []).append(row)

        forward = (model_x, model_y)
        reverse = (model_y, model_x)
        forward_list = orientation_buckets.get(forward, [])
        reverse_list = orientation_buckets.get(reverse, [])

        pair_count = min(len(forward_list), len(reverse_list))
        for idx in range(pair_count):
            row_a = forward_list[idx]
            row_b = reverse_list[idx]

            margins_a = margin_totals_for_battle(row_a)
            margins_b = margin_totals_for_battle(row_b)
            if not margins_a or not margins_b:
                continue

            total_margins: Dict[str, int] = {
                model_x: margins_a.get(model_x, 0) + margins_b.get(model_x, 0),
                model_y: margins_a.get(model_y, 0) + margins_b.get(model_y, 0),
            }

            if total_margins[model_x] > total_margins[model_y]:
                result = "model_1"
            elif total_margins[model_y] > total_margins[model_x]:
                result = "model_2"
            else:
                result = "draw"

            pair_records.append(
                {
                    "pair_id": f"{prompt_id}__{model_x}__vs__{model_y}__pair_{idx}",
                    "timestamp_utc": max(row_a["timestamp_utc"], row_b["timestamp_utc"]),
                    "prompt_id": prompt_id,
                    "judge_model": judge_name,
                    "model_1": model_x,
                    "model_2": model_y,
                    "model_1_margin": total_margins[model_x],
                    "model_2_margin": total_margins[model_y],
                    "result": result,
                    "battle_key_a": row_a.get("battle_key"),
                    "battle_key_b": row_b.get("battle_key"),
                }
            )

    if not pair_records:
        return pd.DataFrame(columns=pair_columns)

    return pd.DataFrame(pair_records)


def augment_writerbench_columns(writerbench_df: pd.DataFrame, ratings: dict, judge_name: str) -> None:
    """Add TrueSkill rating columns for a single judge to the DataFrame (in-place)."""
    rating_col = f"{judge_name} TrueSkill"
    sigma_col = f"{judge_name} Sigma"

    def model_stats(model: str) -> pd.Series:
        rating = ratings.get(model)
        # Return NA if model has no battles
        if rating is None:
            return pd.Series({rating_col: None, sigma_col: None})
        return pd.Series(
            {
                rating_col: rating.mu,
                sigma_col: rating.sigma,
            }
        )

    writerbench_df[[rating_col, sigma_col]] = writerbench_df["writer_model"].apply(
        model_stats
    )


def save_writerbench(writerbench_df: pd.DataFrame, sort_by_col: Optional[str] = None) -> str:
    """Save the augmented writerbench DataFrame to CSV."""
    if sort_by_col and sort_by_col in writerbench_df.columns:
        writerbench_df.sort_values(by=sort_by_col, ascending=False, inplace=True, na_position='last')
    out_path = os.path.join(
        RESULTS_DIR, f"{SUPER_BENCH_PREFIX}{dt.datetime.now().strftime('%Y%m%d')}.csv"
    )
    writerbench_df.to_csv(out_path, index=False, float_format="%.4f")
    return out_path


# --- WritingBench-specific reporting ---

def summarize_position_bias(history_df: pd.DataFrame, judge_name: Optional[str] = None) -> None:
    """Summarize position bias with per-prompt breakdown (WritingBench-specific)."""
    if history_df.empty or "winner_label" not in history_df.columns:
        print("\nNo battle history available to analyze A/B bias.")
        return

    # Filter by judge if specified
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
    wins_a = counts.get("A", 0)
    wins_b = counts.get("B", 0)

    judge_label = f" ({judge_name})" if judge_name else ""
    print(f"\n--- Position Bias Check{judge_label} ---")
    print(f"Total battles analyzed: {total}")
    print(f"Story A wins: {wins_a} ({wins_a / total:.1%})")
    print(f"Story B wins: {wins_b} ({wins_b / total:.1%})")

    prompt_wins = history_df.dropna(subset=["winner_label"]).copy()
    prompt_wins["prompt_id"] = prompt_wins["prompt_id"].astype(str)
    prompt_counts = (
        prompt_wins.groupby("prompt_id")["winner_label"]
        .value_counts()
        .unstack(fill_value=0)
        .sort_index()
    )

    if not prompt_counts.empty:
        print("\nPer-prompt A/B wins:")
        for prompt_id, row in prompt_counts.iterrows():
            prompt_total = row.sum()
            a_wins = row.get("A", 0)
            b_wins = row.get("B", 0)
            print(
                f"  Prompt {prompt_id}: "
                f"A {a_wins} ({(a_wins / prompt_total):.1%}) | "
                f"B {b_wins} ({(b_wins / prompt_total):.1%}) | "
                f"n={prompt_total}"
            )


def summarize_model_activity(history_df: pd.DataFrame, judge_name: Optional[str] = None) -> None:
    if history_df.empty:
        print("\nNo battle history available to summarize model activity.")
        return

    # Filter by judge if specified
    if judge_name and "judge_model" in history_df.columns:
        history_df = history_df[history_df["judge_model"] == judge_name]
        if history_df.empty:
            print(f"\nNo battle history for judge '{judge_name}' to summarize model activity.")
            return

    counts = arena.compute_battle_counts(history_df)
    if not counts:
        print("\nNo battle history available to summarize model activity.")
        return

    judge_label = f" ({judge_name})" if judge_name else ""
    print(f"\n--- Model Battle Counts{judge_label} ---")
    sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    for model, count in sorted_counts:
        print(f"  {model}: {count} battles")


def summarize_pairing_coverage(history_df: pd.DataFrame, judge_name: str) -> None:
    if history_df.empty or "judge_model" not in history_df.columns:
        print("\nNo battle history available to summarize pairing coverage.")
        return

    orientation_counts = extract_orientation_counts(history_df, judge_name)
    if not orientation_counts:
        print(f"\nNo battle history for judge '{judge_name}' to summarize pairing coverage.")
        return

    pair_counts: Dict[Tuple[str, Tuple[str, str]], Dict[Tuple[str, str], int]] = {}
    for (prompt_id, story_a, story_b), count in orientation_counts.items():
        key = (prompt_id, tuple(sorted((story_a, story_b))))
        pair_counts.setdefault(key, {})
        pair_counts[key][(story_a, story_b)] = count

    total_pairs = len(pair_counts)
    both_orientations = 0
    balanced_pairs = 0
    backlog_needed = 0

    for (prompt_id, (m1, m2)), counts in pair_counts.items():
        forward = counts.get((m1, m2), 0)
        reverse = counts.get((m2, m1), 0)
        if forward > 0 and reverse > 0:
            both_orientations += 1
        if forward == reverse and forward > 0:
            balanced_pairs += 1
        backlog_needed += abs(forward - reverse)

    unpaired_pairs = total_pairs - both_orientations

    print("\n--- Pairing Coverage ---")
    print(f"Battles for judge '{judge_name}': {len(history_df[history_df['judge_model'] == judge_name])}")
    print(f"Prompt/model pairs seen: {total_pairs}")
    print(f"Pairs with both orientations logged: {both_orientations}")
    print(f"Pairs currently balanced (A/B counts equal): {balanced_pairs}")
    print(f"Outstanding reverse battles needed to balance pairs: {backlog_needed}")
    print(f"Pairs missing a reverse orientation entirely: {unpaired_pairs}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run head-to-head judging and export TrueSkill scores.")
    parser.add_argument("--judge", default=DEFAULT_JUDGE_NAME, help="Judge model display name from the OpenBench CSV.")
    parser.add_argument("--max-battles", type=int, default=DEFAULT_BATTLES_TO_RUN, help="Limit number of new battles per run.")
    parser.add_argument(
        "--prompt-ids",
        help="Comma-separated prompt IDs to include (example: '0,3,7'). Defaults to all prompts.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of concurrent judge calls to run.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    judge = resolve_judge_model(args.judge)

    instructions = ""
    if os.path.exists(JUDGING_PROMPT_FILE):
        with open(JUDGING_PROMPT_FILE, "r", encoding="utf-8") as f:
            instructions = f.read()

    story_index = load_story_index()
    if len(story_index) < 2:
        print("Not enough stories to run head-to-head comparisons.")
        return

    if args.prompt_ids is None or args.prompt_ids.strip().lower() == "all":
        prompt_filter = None
    else:
        prompt_filter = {token.strip() for token in args.prompt_ids.split(",") if token.strip()}

    # Run info-gain loop via shared arena, using WritingBench's custom
    # list_pending_matches (which returns ALL matches including completed ones)
    history_df, all_new_records = arena.run_info_gain_loop(
        content_index=story_index,
        judge=judge,
        run_single_battle_fn=run_battle,
        build_paired_results_fn=build_paired_results,
        max_battles=args.max_battles,
        workers=args.workers,
        item_filter=prompt_filter,
        list_pending_matches_fn=list_pending_matches,
        instructions=instructions,
    )

    if all_new_records:
        history_df.to_csv(arena.cfg.battle_history_csv, index=False)
        print(f"Updated battle history at {arena.cfg.battle_history_csv}")
    else:
        print("No new battle records were added this run.")

    # Find all unique judges in the battle history
    if "judge_model" in history_df.columns:
        all_judges = sorted(history_df["judge_model"].dropna().unique())
    else:
        all_judges = []

    if not all_judges:
        print("No judges found in battle history.")
        return

    # Load writerbench once, then add columns for all judges
    writerbench_path = get_latest_file(os.path.join(RESULTS_DIR, "writing_direct_*.csv"))
    writerbench_df = pd.read_csv(writerbench_path)

    # Compute TrueSkill ratings for all judges and augment writerbench
    all_pair_dfs = []
    first_judge_col = None
    for judge_name in all_judges:
        pair_df = build_paired_results(history_df, judge_name)
        if not pair_df.empty:
            all_pair_dfs.append(pair_df)
        ratings = arena.compute_trueskill_ratings(pair_df)
        augment_writerbench_columns(writerbench_df, ratings, judge_name)
        if first_judge_col is None:
            first_judge_col = f"{judge_name} TrueSkill"

    # Save the combined writerbench with all judge columns
    superbench_path = save_writerbench(writerbench_df, sort_by_col=first_judge_col)

    # Save combined paired results
    if all_pair_dfs:
        combined_pairs = pd.concat(all_pair_dfs, ignore_index=True)
        combined_pairs.to_csv(BATTLE_PAIRS_CSV, index=False)
        print(f"Paired battle aggregates written to {BATTLE_PAIRS_CSV}")
    else:
        print("No completed battle pairs yet; paired results CSV not written.")

    print(f"TrueSkill scores for all judges ({len(all_judges)}) exported to {superbench_path}")

    # Show stats for each judge separately
    for judge_name in all_judges:
        summarize_position_bias(history_df, judge_name)
        summarize_model_activity(history_df, judge_name)
        summarize_pairing_coverage(history_df, judge_name)
        arena.summarize_win_loss_stats(history_df, judge_name)


if __name__ == "__main__":
    main()
