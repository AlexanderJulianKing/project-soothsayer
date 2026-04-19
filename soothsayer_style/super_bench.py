"""ToneBench - Pairwise TrueSkill with dual arenas (signal density + conversational confidence).

Replaces style_analysis.py (absolute scoring) with pairwise comparison.
Two independent TrueSkill arenas with combined info-gain scheduling.

Each battle: judge sees two responses to the same question, picks a winner
on each axis with a margin. One API call -> two results -> two arena updates.
"""

import argparse
import datetime as dt
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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

from core.llm_client import get_llm_response

# --- CONFIGURATION ---
SCRIPT_DIR = Path(__file__).parent
RESPONSES_CSV = SCRIPT_DIR / "responses.csv"
QUESTIONS_FILE = SCRIPT_DIR / "questions.txt"
RESULTS_DIR = SCRIPT_DIR / "results"
PAIRWISE_PROMPT_FILE = SCRIPT_DIR / "pairwise_prompt_style.txt"
BATTLE_PAIRS_CSV = RESULTS_DIR / "battle_pairs.csv"
BENCH_PREFIX = "tone_"
DEFAULT_JUDGE_NAME = "Grok 4.1 Fast"
MAX_PARSE_ATTEMPTS = 3
DEFAULT_BATTLES_TO_RUN = 125

AXES = ["signal_density", "conversational_confidence"]

SYSTEM_PROMPT = (
    "You are an impartial conversational quality judge. "
    "Compare the two anonymous AI responses and respond with valid JSON only."
)

# Arena config — used for structural operations (pending matches, battle execution).
# Ratings are managed separately per axis in the dual-arena loop.
ARENA_CONFIG = ArenaConfig(
    draw_probability=0.03,
    paired_mode=True,
    bench_prefix=BENCH_PREFIX,
    results_dir=str(RESULTS_DIR),
    item_id_col="question_id",
    model_a_col="response_a_model",
    model_b_col="response_b_model",
    item_type="question",
    winner_label_a="A",
    winner_label_b="B",
)
arena = TrueSkillArena(ARENA_CONFIG)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_response_index(responses_csv: str) -> Dict[str, Dict[str, str]]:
    """Build {model_name: {question_id: response_text}} from responses.csv.

    Picks the first successful run per (model, question).
    """
    if not os.path.exists(responses_csv):
        return {}

    df = pd.read_csv(responses_csv)
    df = df[df['status'] == 'ok'].copy()
    df = df.drop_duplicates(subset=['model_name', 'question_id'], keep='first')

    index: Dict[str, Dict[str, str]] = {}
    for _, row in df.iterrows():
        model = str(row['model_name'])
        qid = str(row['question_id'])
        response = str(row['response'])
        if model not in index:
            index[model] = {}
        index[model][qid] = response

    return index


def load_questions(questions_file: str) -> Dict[str, str]:
    """Build {question_id: question_text} from questions.txt."""
    questions: Dict[str, str] = {}
    if not os.path.exists(questions_file):
        return questions
    with open(questions_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if line:
                questions[str(i)] = line
    return questions


def load_pairwise_template() -> str:
    """Load the pairwise comparison prompt template."""
    if PAIRWISE_PROMPT_FILE.exists():
        with open(PAIRWISE_PROMPT_FILE, 'r', encoding='utf-8') as f:
            return f.read()
    raise FileNotFoundError(f"Pairwise prompt template not found: {PAIRWISE_PROMPT_FILE}")


def resolve_judge_model(judge_name: str) -> Dict[str, str]:
    """Resolve judge model details from OpenBench CSV."""
    model_csv = discover_openbench_csv(str(SCRIPT_DIR))
    for model in load_models(model_csv):
        if model["name"].strip() == judge_name.strip():
            if not model.get("id"):
                raise ValueError(f"Judge '{judge_name}' is missing an openbench_id.")
            return {
                "name": model["name"],
                "id": model["id"],
                "Reasoning": normalize_reasoning_flag(model.get("Reasoning", False)),
            }
    raise ValueError(f"Judge '{judge_name}' not found in {model_csv}.")


# ---------------------------------------------------------------------------
# Prompt building & validation
# ---------------------------------------------------------------------------

def build_comparison_prompt(
    template: str,
    qid: str,
    q_text: str,
    resp_a: str,
    resp_b: str,
) -> str:
    """Fill pairwise template placeholders."""
    prompt = template.replace("{question_text}", q_text)
    prompt = prompt.replace("{response_a}", resp_a.strip())
    prompt = prompt.replace("{response_b}", resp_b.strip())
    return prompt


def validate_payload(payload: dict) -> bool:
    """Check JSON has both axes with valid winner/margin."""
    if not isinstance(payload, dict):
        return False
    for axis in AXES:
        axis_data = payload.get(axis)
        if not isinstance(axis_data, dict):
            return False
        if axis_data.get("winner") not in ("A", "B"):
            return False
        margin = axis_data.get("margin", "")
        if not isinstance(margin, str) or not re.match(r'^\+{1,5}$', margin):
            return False
    return True


# ---------------------------------------------------------------------------
# Battle callback
# ---------------------------------------------------------------------------

def run_battle(
    qid: str,
    model_a: str,
    model_b: str,
    content_index: Dict[str, Dict[str, str]],
    judge: Dict[str, str],
    battle_key: str,
    template: str = "",
    questions: Dict[str, str] = None,
) -> Dict[str, str]:
    """Run a single pairwise comparison battle.

    Signature matches the arena's run_single_battle_fn callback convention:
    (item_id, model_a, model_b, content_index, judge, battle_key, **extra)
    """
    questions = questions or {}
    q_text = questions.get(qid, f"Question {qid}")
    resp_a = content_index[model_a][qid]
    resp_b = content_index[model_b][qid]

    prompt = build_comparison_prompt(template, qid, q_text, resp_a, resp_b)

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

        # Build per-axis criteria records and compute overall winner
        criteria_records = []
        a_total = 0
        b_total = 0

        for axis in AXES:
            axis_data = payload[axis]
            winner_label = axis_data["winner"]
            margin_str = axis_data["margin"]
            margin_score = len(margin_str)

            winner_model = model_a if winner_label == "A" else model_b

            if winner_label == "A":
                a_total += margin_score
            else:
                b_total += margin_score

            criteria_records.append({
                "name": axis,
                "winner": winner_model,
                "margin": margin_str,
                "margin_score": margin_score,
            })

        # Determine overall winner by total margin across both axes
        if a_total > b_total:
            winner_model = model_a
            loser_model = model_b
            winner_label = "A"
        elif b_total > a_total:
            winner_model = model_b
            loser_model = model_a
            winner_label = "B"
        else:
            # Tie across axes — record as draw (winner_model arbitrary)
            winner_model = model_a
            loser_model = model_b
            winner_label = "draw"

        return {
            "battle_key": battle_key,
            "timestamp_utc": dt.datetime.utcnow().isoformat(),
            "question_id": qid,
            "judge_model": judge["name"],
            "judge_model_id": judge["id"],
            "response_a_model": model_a,
            "response_b_model": model_b,
            "winner_model": winner_model,
            "loser_model": loser_model,
            "winner_label": winner_label,
            "criteria_json": json.dumps(criteria_records),
            "raw_response": raw_response.strip(),
        }

    raise RuntimeError(f"Failed to obtain a valid judgement for {battle_key}: {last_error}")


# ---------------------------------------------------------------------------
# Per-axis paired results
# ---------------------------------------------------------------------------

def _extract_axis_margin(criteria_json: str, axis_name: str, target_model: str) -> int:
    """Extract margin for target_model on a specific axis from criteria_json."""
    try:
        criteria = json.loads(criteria_json)
        for c in criteria:
            if c["name"] == axis_name:
                if c["winner"] == target_model:
                    return c["margin_score"]
                return 0
        return 0
    except (json.JSONDecodeError, KeyError, TypeError):
        return 0


def build_paired_results_for_axis(
    history_df: pd.DataFrame,
    judge_name: str,
    axis_name: str,
) -> pd.DataFrame:
    """Extract single-axis margins from criteria_json, build standard pairs_df."""
    columns = [
        "pair_id", "timestamp_utc", "question_id", "judge_model",
        "model_1", "model_2", "model_1_margin", "model_2_margin",
        "result", "battle_key_a", "battle_key_b",
    ]

    if history_df.empty or "criteria_json" not in history_df.columns:
        return pd.DataFrame(columns=columns)

    filtered = history_df[
        (history_df["judge_model"] == judge_name)
        & history_df["question_id"].notna()
        & history_df["response_a_model"].notna()
        & history_df["response_b_model"].notna()
        & history_df["criteria_json"].notna()
    ].copy()

    if filtered.empty:
        return pd.DataFrame(columns=columns)

    # Group battles by (question_id, sorted model pair)
    pair_records = []
    grouped: Dict[tuple, list] = {}

    for _, row in filtered.iterrows():
        qid = str(row["question_id"])
        model_a = row["response_a_model"]
        model_b = row["response_b_model"]
        pair_key = (qid, tuple(sorted([model_a, model_b])))
        grouped.setdefault(pair_key, []).append(row)

    for (qid, (m1, m2)), battles in grouped.items():
        forward = [b for b in battles if b["response_a_model"] == m1 and b["response_b_model"] == m2]
        reverse = [b for b in battles if b["response_a_model"] == m2 and b["response_b_model"] == m1]

        if not forward or not reverse:
            continue

        fwd = forward[0]
        rev = reverse[0]

        m1_margin = (
            _extract_axis_margin(fwd["criteria_json"], axis_name, m1)
            + _extract_axis_margin(rev["criteria_json"], axis_name, m1)
        )
        m2_margin = (
            _extract_axis_margin(fwd["criteria_json"], axis_name, m2)
            + _extract_axis_margin(rev["criteria_json"], axis_name, m2)
        )

        if m1_margin > m2_margin:
            result = "model_1"
        elif m2_margin > m1_margin:
            result = "model_2"
        else:
            result = "draw"

        pair_records.append({
            "pair_id": f"{qid}_{axis_name}_{m1}_vs_{m2}",
            "timestamp_utc": max(str(fwd["timestamp_utc"]), str(rev["timestamp_utc"])),
            "question_id": qid,
            "judge_model": judge_name,
            "model_1": m1,
            "model_2": m2,
            "model_1_margin": m1_margin,
            "model_2_margin": m2_margin,
            "result": result,
            "battle_key_a": fwd["battle_key"],
            "battle_key_b": rev["battle_key"],
        })

    if not pair_records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(pair_records)


def build_paired_results_combined(history_df: pd.DataFrame, judge_name: str) -> pd.DataFrame:
    """Build combined paired results (both axes summed) for overall reporting."""
    columns = [
        "pair_id", "timestamp_utc", "question_id", "judge_model",
        "model_1", "model_2", "model_1_margin", "model_2_margin",
        "result", "battle_key_a", "battle_key_b",
    ]

    if history_df.empty or "criteria_json" not in history_df.columns:
        return pd.DataFrame(columns=columns)

    filtered = history_df[
        (history_df["judge_model"] == judge_name)
        & history_df["question_id"].notna()
        & history_df["response_a_model"].notna()
        & history_df["response_b_model"].notna()
        & history_df["criteria_json"].notna()
    ].copy()

    if filtered.empty:
        return pd.DataFrame(columns=columns)

    pair_records = []
    grouped: Dict[tuple, list] = {}

    for _, row in filtered.iterrows():
        qid = str(row["question_id"])
        model_a = row["response_a_model"]
        model_b = row["response_b_model"]
        pair_key = (qid, tuple(sorted([model_a, model_b])))
        grouped.setdefault(pair_key, []).append(row)

    for (qid, (m1, m2)), battles in grouped.items():
        forward = [b for b in battles if b["response_a_model"] == m1 and b["response_b_model"] == m2]
        reverse = [b for b in battles if b["response_a_model"] == m2 and b["response_b_model"] == m1]

        if not forward or not reverse:
            continue

        fwd = forward[0]
        rev = reverse[0]

        # Sum margins across all axes
        m1_margin = 0
        m2_margin = 0
        for axis in AXES:
            m1_margin += _extract_axis_margin(fwd["criteria_json"], axis, m1)
            m1_margin += _extract_axis_margin(rev["criteria_json"], axis, m1)
            m2_margin += _extract_axis_margin(fwd["criteria_json"], axis, m2)
            m2_margin += _extract_axis_margin(rev["criteria_json"], axis, m2)

        if m1_margin > m2_margin:
            result = "model_1"
        elif m2_margin > m1_margin:
            result = "model_2"
        else:
            result = "draw"

        pair_records.append({
            "pair_id": f"{qid}_{m1}_vs_{m2}",
            "timestamp_utc": max(str(fwd["timestamp_utc"]), str(rev["timestamp_utc"])),
            "question_id": qid,
            "judge_model": judge_name,
            "model_1": m1,
            "model_2": m2,
            "model_1_margin": m1_margin,
            "model_2_margin": m2_margin,
            "result": result,
            "battle_key_a": fwd["battle_key"],
            "battle_key_b": rev["battle_key"],
        })

    if not pair_records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(pair_records)


# ---------------------------------------------------------------------------
# Combined info-gain match selection (dual-arena)
# ---------------------------------------------------------------------------

def select_matches_combined_info_gain(
    pending_matches: List[Tuple[str, str, str]],
    priority_matches: Set[Tuple[str, str, str]],
    density_ratings: dict,
    confidence_ratings: dict,
    max_battles: int,
) -> List[Tuple[str, str, str]]:
    """Mirror arena.select_matches_by_info_gain() but sum info-gain from both rating dicts."""
    selection = []

    # FIRST: Complete pairs (priority matches)
    priority_list = [m for m in pending_matches if m in priority_matches]
    for match in priority_list:
        if len(selection) >= max_battles:
            break
        selection.append(match)

    if len(selection) >= max_battles:
        return selection

    # SECOND: Score remaining by combined info-gain from both arenas
    non_priority = [m for m in pending_matches if m not in priority_matches]

    def get_rating(ratings_dict: dict, model: str):
        return ratings_dict.get(model, arena._create_rating())

    pair_to_matches: Dict[tuple, list] = {}
    for match in non_priority:
        _, model_a, model_b = match
        pair_key = tuple(sorted([model_a, model_b]))
        pair_to_matches.setdefault(pair_key, []).append(match)

    scored_matches = []
    for pair_key, matches in pair_to_matches.items():
        match = random.choice(matches)
        _, ma, mb = match
        density_gain = arena.compute_match_info_gain(
            get_rating(density_ratings, ma),
            get_rating(density_ratings, mb),
        )
        confidence_gain = arena.compute_match_info_gain(
            get_rating(confidence_ratings, ma),
            get_rating(confidence_ratings, mb),
        )
        combined_gain = density_gain + confidence_gain
        scored_matches.append((combined_gain, match))

    scored_matches.sort(key=lambda x: x[0], reverse=True)

    for _, match in scored_matches:
        if len(selection) >= max_battles:
            break
        selection.append(match)

    return selection


# ---------------------------------------------------------------------------
# Dual-arena loop
# ---------------------------------------------------------------------------

def run_dual_arena_loop(
    content_index: Dict[str, Dict[str, str]],
    judge: Dict[str, str],
    questions: Dict[str, str],
    template: str,
    max_battles: int,
    workers: int,
    item_filter: Optional[Set[str]] = None,
) -> Tuple[pd.DataFrame, List[Dict[str, str]]]:
    """Custom loop: list pending -> combined info-gain selection -> run batch -> recompute both ratings -> repeat."""

    history_df, _ = arena.load_battle_history()
    existing_orientations = arena.extract_existing_orientations(history_df)

    all_new_records: List[Dict[str, str]] = []
    remaining = max_battles

    # Initial per-axis ratings
    density_pairs = build_paired_results_for_axis(history_df, judge["name"], "signal_density")
    confidence_pairs = build_paired_results_for_axis(history_df, judge["name"], "conversational_confidence")
    density_ratings = arena.compute_trueskill_ratings(density_pairs)
    confidence_ratings = arena.compute_trueskill_ratings(confidence_pairs)

    print(f"\n=== DUAL-ARENA INFO-GAIN MATCH SELECTION ===")

    batch_num = 0
    while remaining > 0:
        batch_num += 1

        pending, priority = arena.list_pending_matches(
            content_index, existing_orientations, judge["name"],
            item_filter, paired_mode=True,
        )

        if not pending:
            print(f"\nNo pending matches remaining.")
            break

        batch_size = min(workers * 2, remaining)
        batch = select_matches_combined_info_gain(
            pending, priority,
            density_ratings, confidence_ratings,
            batch_size,
        )

        if not batch:
            print(f"\nNo matches selected.")
            break

        priority_count = sum(1 for m in batch if m in priority)
        print(f"\nBatch {batch_num}: Running {len(batch)} battles "
              f"({priority_count} completing pairs, {len(priority)} incomplete pairs)")

        records = arena.run_battles_batch(
            batch, content_index, judge,
            run_battle, workers,
            template=template, questions=questions,
        )

        if records:
            history_df = pd.concat([history_df, pd.DataFrame(records)], ignore_index=True)
            existing_orientations = arena.extract_existing_orientations(history_df)
            all_new_records.extend(records)
            remaining -= len(records)

            # Recompute per-axis ratings
            density_pairs = build_paired_results_for_axis(history_df, judge["name"], "signal_density")
            confidence_pairs = build_paired_results_for_axis(history_df, judge["name"], "conversational_confidence")
            density_ratings = arena.compute_trueskill_ratings(density_pairs)
            confidence_ratings = arena.compute_trueskill_ratings(confidence_pairs)

            print(f"  Completed {len(records)} battles, {remaining} remaining in budget")
        else:
            print(f"  No battles completed")
            break

    return history_df, all_new_records


# ---------------------------------------------------------------------------
# Results: augment & save
# ---------------------------------------------------------------------------

def augment_tone_columns(
    df: pd.DataFrame,
    density_ratings: dict,
    confidence_ratings: dict,
    judge_name: str,
) -> None:
    """Add per-axis TrueSkill rating columns for a single judge (in-place)."""
    density_col = f"{judge_name} density TrueSkill"
    density_sigma = f"{judge_name} density Sigma"
    confidence_col = f"{judge_name} confidence TrueSkill"
    confidence_sigma = f"{judge_name} confidence Sigma"

    def model_stats(model: str) -> pd.Series:
        d_rating = density_ratings.get(model)
        c_rating = confidence_ratings.get(model)
        return pd.Series({
            density_col: d_rating.mu if d_rating else None,
            density_sigma: d_rating.sigma if d_rating else None,
            confidence_col: c_rating.mu if c_rating else None,
            confidence_sigma: c_rating.sigma if c_rating else None,
        })

    df[[density_col, density_sigma, confidence_col, confidence_sigma]] = (
        df["judged_model"].apply(model_stats)
    )


def save_results(combined_df: pd.DataFrame, sort_by_col: Optional[str] = None) -> str:
    """Save the combined multi-judge TrueSkill ratings to CSV."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if sort_by_col and sort_by_col in combined_df.columns:
        combined_df.sort_values(by=sort_by_col, ascending=False, inplace=True, na_position='last')
    out_path = RESULTS_DIR / f"{BENCH_PREFIX}{dt.datetime.now().strftime('%Y%m%d')}.csv"
    combined_df.to_csv(out_path, index=False, float_format="%.4f")
    return str(out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="ToneBench: pairwise TrueSkill with dual arenas (density + confidence)."
    )
    parser.add_argument("--judge", default=DEFAULT_JUDGE_NAME,
                        help="Judge model display name from OpenBench CSV.")
    parser.add_argument("--max-battles", type=int, default=DEFAULT_BATTLES_TO_RUN,
                        help="Limit number of new battles per run.")
    parser.add_argument("--questions",
                        help="Comma-separated question IDs (e.g., '1,3,7'). Defaults to all.")
    parser.add_argument("--workers", type=int, default=20,
                        help="Number of concurrent judge calls.")
    parser.add_argument("--pilot", type=int, default=None, metavar='N',
                        help="Limit to first N unique models (for pilot runs).")
    parser.add_argument("--models",
                        help="Comma-separated model names to include. Defaults to all.")
    return parser.parse_args()


def main():
    args = parse_args()
    judge = resolve_judge_model(args.judge)

    # Load pairwise template
    template = load_pairwise_template()
    print(f"Using pairwise template: {PAIRWISE_PROMPT_FILE.name}")

    # Load responses
    response_index = load_response_index(str(RESPONSES_CSV))
    if len(response_index) < 2:
        print(f"Not enough responses for pairwise comparison. Found {len(response_index)} models.")
        return

    # Apply model filters
    if args.models:
        requested = {m.strip() for m in args.models.split(",") if m.strip()}
        available = set(response_index.keys())
        missing = requested - available
        if missing:
            print(f"Warning: {len(missing)} requested models not found in responses: {missing}")
        response_index = {m: response_index[m] for m in response_index if m in requested}
        print(f"Model filter: {len(response_index)} models selected")
    elif args.pilot is not None:
        models_sorted = sorted(response_index.keys())[:args.pilot]
        response_index = {m: response_index[m] for m in models_sorted}
        print(f"Pilot mode: limiting to {len(response_index)} models")

    q_counts = [len(qs) for qs in response_index.values()]
    print(f"Loaded responses from {len(response_index)} models "
          f"({min(q_counts)}-{max(q_counts)} questions each)")

    # Load question texts (for prompt building)
    questions = load_questions(str(QUESTIONS_FILE))
    print(f"Loaded {len(questions)} questions from {QUESTIONS_FILE.name}")

    # Parse question filter
    question_filter = None
    if args.questions:
        question_filter = {s.strip() for s in args.questions.split(",") if s.strip()}

    # Run dual-arena loop
    history_df, all_new_records = run_dual_arena_loop(
        content_index=response_index,
        judge=judge,
        questions=questions,
        template=template,
        max_battles=args.max_battles,
        workers=args.workers,
        item_filter=question_filter,
    )

    # --- SAVE RESULTS ---
    if all_new_records:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        history_df.to_csv(arena.cfg.battle_history_csv, index=False)
        print(f"\nBattle history saved to {arena.cfg.battle_history_csv}")
        print(f"Total new battles this run: {len(all_new_records)}")
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

    # Build a base DataFrame with all models that have participated
    all_models = set()
    for _, row in history_df.iterrows():
        for col in ("winner_model", "loser_model"):
            model = row.get(col)
            if isinstance(model, str):
                all_models.add(model)
    combined_df = pd.DataFrame({"judged_model": sorted(all_models)})

    # Compute per-axis TrueSkill ratings for all judges and augment combined DF
    all_pair_dfs = []
    first_judge_col = None
    for judge_name in all_judges:
        density_pairs = build_paired_results_for_axis(history_df, judge_name, "signal_density")
        confidence_pairs = build_paired_results_for_axis(history_df, judge_name, "conversational_confidence")
        density_ratings = arena.compute_trueskill_ratings(density_pairs)
        confidence_ratings = arena.compute_trueskill_ratings(confidence_pairs)

        augment_tone_columns(combined_df, density_ratings, confidence_ratings, judge_name)
        if first_judge_col is None:
            first_judge_col = f"{judge_name} density TrueSkill"

        # Collect combined pairs for battle_pairs.csv
        combined_pairs = build_paired_results_combined(history_df, judge_name)
        if not combined_pairs.empty:
            all_pair_dfs.append(combined_pairs)

    # Save combined results
    out_path = save_results(combined_df, sort_by_col=first_judge_col)

    # Save combined paired results
    if all_pair_dfs:
        pairs_concat = pd.concat(all_pair_dfs, ignore_index=True)
        pairs_concat.to_csv(BATTLE_PAIRS_CSV, index=False)
        print(f"Paired battle aggregates written to {BATTLE_PAIRS_CSV}")

    print(f"TrueSkill scores for all judges ({len(all_judges)}) exported to {out_path}")

    # Show stats for each judge
    for judge_name in all_judges:
        density_pairs = build_paired_results_for_axis(history_df, judge_name, "signal_density")
        confidence_pairs = build_paired_results_for_axis(history_df, judge_name, "conversational_confidence")
        density_ratings = arena.compute_trueskill_ratings(density_pairs)
        confidence_ratings = arena.compute_trueskill_ratings(confidence_pairs)

        print(f"\n--- Signal Density Rankings ({judge_name}) ---")
        if density_ratings:
            for model, rating in sorted(density_ratings.items(), key=lambda x: x[1].mu, reverse=True):
                print(f"  {rating.mu:.1f} (\u03c3={rating.sigma:.1f}) {model}")

        print(f"\n--- Conversational Confidence Rankings ({judge_name}) ---")
        if confidence_ratings:
            for model, rating in sorted(confidence_ratings.items(), key=lambda x: x[1].mu, reverse=True):
                print(f"  {rating.mu:.1f} (\u03c3={rating.sigma:.1f}) {model}")

        arena.summarize_position_bias(history_df, judge_name)
        arena.summarize_win_loss_stats(history_df, judge_name)


if __name__ == "__main__":
    main()
