"""
EQBenchFree - Pairwise comparison and TrueSkill rating system.
Adapted from soothsayer_writing/super_bench.py for emotional intelligence evaluation.

This module is a thin adapter that delegates all shared TrueSkill logic
to core.trueskill_arena and keeps only EQ-Bench-specific functions:
criteria, prompt building, validation, run_battle, result augmentation.
"""

import argparse
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path
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

# --- CONFIGURATION ---
SCRIPT_DIR = Path(__file__).parent
RESPONSES_DIR = SCRIPT_DIR / "generated_responses"
RESULTS_DIR = SCRIPT_DIR / "results"
PAIRWISE_PROMPT_FILE = SCRIPT_DIR / "pairwise_prompt_eqbench3.txt"
BATTLE_PAIRS_CSV = RESULTS_DIR / "battle_pairs.csv"
BENCH_PREFIX = "eq_"
DEFAULT_JUDGE_NAME = "Gemini 3.0 Flash Preview (2025-12-17)"
MAX_PARSE_ATTEMPTS = 3
DEFAULT_BATTLES_TO_RUN = 20

# Arena configuration
ARENA_CONFIG = ArenaConfig(
    draw_probability=0.03,
    paired_mode=True,
    bench_prefix=BENCH_PREFIX,
    results_dir=str(RESULTS_DIR),
    item_id_col="scenario_id",
    model_a_col="response_a_model",
    model_b_col="response_b_model",
    item_type="scenario",
    winner_label_a="A0493",
    winner_label_b="A0488",
)
arena = TrueSkillArena(ARENA_CONFIG)

# Emotional Intelligence criteria (8 dimensions - EQ-Bench 3 style)
CRITERIA = [
    "demonstrated_empathy",
    "pragmatic_ei",
    "depth_of_insight",
    "social_dexterity",
    "emotional_reasoning",
    "appropriate_validating_challenging",
    "message_tailoring",
    "overall_eq",
]

# Map criteria keys to display names
CRITERIA_DISPLAY = {
    "demonstrated_empathy": "Demonstrated Empathy",
    "pragmatic_ei": "Pragmatic EI",
    "depth_of_insight": "Depth of Insight",
    "social_dexterity": "Social Dexterity",
    "emotional_reasoning": "Emotional Reasoning",
    "appropriate_validating_challenging": "Appropriate Validation/Challenging",
    "message_tailoring": "Message Tailoring",
    "overall_eq": "Overall EQ",
}

SYSTEM_PROMPT = (
    "You are an impartial emotional intelligence evaluation judge. "
    "You must follow the provided rubric, compare the two anonymous respondents, "
    "and respond with valid JSON only."
)


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


MAX_TURN_RESPONSE_CHARS = 50_000   # skip scenarios with degenerate (repetitive) output
MIN_TURN_RESPONSE_CHARS = 20       # skip scenarios where a turn clearly failed


def _scenario_has_valid_turns(data: dict) -> bool:
    """Return False if any turn has a degenerate or failed response."""
    for turn in data.get("turns", []):
        raw = turn.get("response", "")
        if len(raw) > MAX_TURN_RESPONSE_CHARS:
            return False
        if len(raw) < MIN_TURN_RESPONSE_CHARS:
            return False
    return True


def load_response_index() -> Dict[str, Dict[str, dict]]:
    """
    Load all generated responses into an index.

    Returns:
        Dict mapping model_name -> scenario_id -> response_data
    """
    response_index: Dict[str, Dict[str, dict]] = {}

    if not RESPONSES_DIR.exists():
        return response_index

    skipped_count = 0

    for model_dir in RESPONSES_DIR.iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        scenario_map: Dict[str, dict] = {}

        for response_file in model_dir.glob("scenario_*.json"):
            try:
                with open(response_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if not data.get("completed", False):
                    continue

                if not _scenario_has_valid_turns(data):
                    skipped_count += 1
                    continue

                # Extract scenario ID from filename
                scenario_id = response_file.stem.replace("scenario_", "")
                scenario_map[scenario_id] = data

            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load {response_file}: {e}")
                continue

        if scenario_map:
            response_index[model_name] = scenario_map

    if skipped_count:
        print(f"Skipped {skipped_count} scenario(s) with failed or degenerate responses.")

    return response_index


# Character limits for the three parsed sections in a message-drafting reply
SECTION_CHAR_LIMITS_MESSAGE_DRAFT = {
    "perspective_taking": 2200,
    "draft_brainstorming": 1600,
    "draft": 1600,
}


def truncate_section(text: str, limit: int) -> str:
    """Truncate text to limit, adding truncation marker if needed."""
    if len(text) > limit:
        return text[:limit] + "\n[...truncated]"
    return text


def format_response_for_judging(response_data: dict) -> str:
    """Format a multi-turn response for judging with section-based truncation."""
    lines = []
    for turn in response_data.get("turns", []):
        lines.append(f"[Turn {turn['turn']}]")
        # Truncate prompt context
        prompt = turn['prompt']
        if len(prompt) > 200:
            prompt = prompt[:200] + "..."
        lines.append(f"Prompt: {prompt}")

        # Check for parsed sections with specific limits
        parsed = turn.get('parsed', {})
        if parsed:
            perspective = parsed.get('perspective_taking', '')
            brainstorm = parsed.get('draft_brainstorming', '')
            draft = parsed.get('draft', '')

            if perspective:
                lines.append(f"# Perspective-taking\n{truncate_section(perspective, SECTION_CHAR_LIMITS_MESSAGE_DRAFT['perspective_taking'])}")
            if brainstorm:
                lines.append(f"# Draft brainstorming\n{truncate_section(brainstorm, SECTION_CHAR_LIMITS_MESSAGE_DRAFT['draft_brainstorming'])}")
            if draft:
                lines.append(f"# Draft\n{truncate_section(draft, SECTION_CHAR_LIMITS_MESSAGE_DRAFT['draft'])}")
        else:
            # Fallback: use raw response with combined limit
            raw = turn.get('response', '')
            total_limit = sum(SECTION_CHAR_LIMITS_MESSAGE_DRAFT.values())
            if len(raw) > total_limit:
                raw = raw[:total_limit] + "\n[...truncated]"
            lines.append(f"Response:\n{raw}")

        lines.append("")
    return "\n".join(lines)


def build_comparison_prompt(
    template: str,
    scenario_id: str,
    response_a: str,
    response_b: str,
    scenario_notes: str = "",
) -> str:
    """Build the pairwise comparison prompt using the EQ-Bench 3 template."""
    # The template uses placeholders that we need to fill
    prompt = template.replace("{conversation_history_A}", response_a.strip())
    prompt = prompt.replace("{conversation_history_B}", response_b.strip())
    prompt = prompt.replace("{debrief_A}", "")  # No separate debrief for now
    prompt = prompt.replace("{debrief_B}", "")
    prompt = prompt.replace("{scenario_notes}", scenario_notes or f"Scenario {scenario_id}")
    return prompt


def validate_payload(payload: dict) -> bool:
    """Validate the EQ-Bench 3 style JSON response."""
    if not isinstance(payload, dict):
        return False

    # Check all required criteria keys exist with valid format
    valid_margin_pattern = re.compile(r'^A0(493|488)\++$')

    for criterion in CRITERIA:
        value = payload.get(criterion)
        if not value or not isinstance(value, str):
            return False
        # Should be like "A0493++" or "A0488+++"
        if not valid_margin_pattern.match(value):
            return False

    return True


def parse_criterion_value(value: str) -> Tuple[str, int]:
    """
    Parse a criterion value like 'A0493++' into (winner_code, margin_score).
    Returns ('A0493', 2) for 'A0493++'.
    """
    match = re.match(r'^(A0\d+)(\++)', value)
    if match:
        code = match.group(1)
        margin = len(match.group(2))
        return code, margin
    return None, 0


def run_battle(
    scenario_id: str,
    model_a: str,
    model_b: str,
    response_index: Dict[str, Dict[str, dict]],
    judge: Dict[str, str],
    battle_key: str,
    template: str = None,
) -> Dict[str, str]:
    """Run a single pairwise comparison battle.

    Signature matches the arena's run_single_battle_fn callback convention:
    (item_id, model_a, model_b, content_index, judge, battle_key, **extra)
    """
    response_a = format_response_for_judging(response_index[model_a][scenario_id])
    response_b = format_response_for_judging(response_index[model_b][scenario_id])

    # A0493 = model_a, A0488 = model_b
    prompt = build_comparison_prompt(
        template=template,
        scenario_id=scenario_id,
        response_a=response_a,
        response_b=response_b,
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

        # Count total margin for each respondent to determine overall winner
        a_total = 0
        b_total = 0
        criteria_records = []

        for criterion in CRITERIA:
            value = payload.get(criterion, "")
            code, margin = parse_criterion_value(value)
            if code == "A0493":
                winner_model_crit = model_a
                a_total += margin
            else:
                winner_model_crit = model_b
                b_total += margin

            criteria_records.append({
                "name": criterion,
                "display_name": CRITERIA_DISPLAY.get(criterion, criterion),
                "winner": winner_model_crit,
                "margin": "+" * margin,
                "margin_score": margin,
            })

        # Determine overall winner based on total margin
        if a_total > b_total:
            winner_model = model_a
            loser_model = model_b
            winner_label = "A0493"
        else:
            winner_model = model_b
            loser_model = model_a
            winner_label = "A0488"

        return {
            "battle_key": battle_key,
            "timestamp_utc": dt.datetime.utcnow().isoformat(),
            "scenario_id": scenario_id,
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


def build_paired_results(history_df: pd.DataFrame, judge_name: str) -> pd.DataFrame:
    """Build paired results from battle history for TrueSkill calculation."""
    columns = [
        "pair_id", "timestamp_utc", "scenario_id", "judge_model",
        "model_1", "model_2", "model_1_margin", "model_2_margin",
        "result", "battle_key_a", "battle_key_b",
    ]

    if history_df.empty or "criteria_json" not in history_df.columns:
        return pd.DataFrame(columns=columns)

    filtered = history_df[
        (history_df["judge_model"] == judge_name)
        & history_df["scenario_id"].notna()
        & history_df["response_a_model"].notna()
        & history_df["response_b_model"].notna()
        & history_df["criteria_json"].notna()
    ].copy()

    if filtered.empty:
        return pd.DataFrame(columns=columns)

    # Group battles by (scenario_id, model pair)
    def compute_margin(criteria_json: str, target_model: str) -> int:
        try:
            criteria = json.loads(criteria_json)
            return sum(c["margin_score"] for c in criteria if c["winner"] == target_model)
        except (json.JSONDecodeError, KeyError, TypeError):
            return 0

    pair_records = []
    grouped = {}

    for _, row in filtered.iterrows():
        scenario_id = str(row["scenario_id"])
        model_a = row["response_a_model"]
        model_b = row["response_b_model"]
        pair_key = (scenario_id, tuple(sorted([model_a, model_b])))

        if pair_key not in grouped:
            grouped[pair_key] = []
        grouped[pair_key].append(row)

    for (scenario_id, (m1, m2)), battles in grouped.items():
        # Look for both orientations
        forward = [b for b in battles if b["response_a_model"] == m1 and b["response_b_model"] == m2]
        reverse = [b for b in battles if b["response_a_model"] == m2 and b["response_b_model"] == m1]

        if not forward or not reverse:
            continue

        fwd = forward[0]
        rev = reverse[0]

        m1_margin = compute_margin(fwd["criteria_json"], m1) + compute_margin(rev["criteria_json"], m1)
        m2_margin = compute_margin(fwd["criteria_json"], m2) + compute_margin(rev["criteria_json"], m2)

        if m1_margin > m2_margin:
            result = "model_1"
        elif m2_margin > m1_margin:
            result = "model_2"
        else:
            result = "draw"

        pair_records.append({
            "pair_id": f"{scenario_id}_{m1}_vs_{m2}",
            "timestamp_utc": max(fwd["timestamp_utc"], rev["timestamp_utc"]),
            "scenario_id": scenario_id,
            "judge_model": judge_name,
            "model_1": m1,
            "model_2": m2,
            "model_1_margin": m1_margin,
            "model_2_margin": m2_margin,
            "result": result,
            "battle_key_a": fwd["battle_key"],
            "battle_key_b": rev["battle_key"],
        })

    return pd.DataFrame(pair_records)


def augment_eqbench_columns(df: pd.DataFrame, ratings: dict, judge_name: str) -> None:
    """Add TrueSkill rating columns for a single judge to the DataFrame (in-place)."""
    rating_col = f"{judge_name} TrueSkill"
    sigma_col = f"{judge_name} Sigma"

    def model_stats(model: str) -> pd.Series:
        rating = ratings.get(model)
        if rating is None:
            return pd.Series({rating_col: None, sigma_col: None})
        return pd.Series({rating_col: rating.mu, sigma_col: rating.sigma})

    df[[rating_col, sigma_col]] = df["model"].apply(model_stats)


def save_results(combined_df: pd.DataFrame, sort_by_col: Optional[str] = None) -> str:
    """Save the combined multi-judge TrueSkill ratings to CSV."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if sort_by_col and sort_by_col in combined_df.columns:
        combined_df.sort_values(by=sort_by_col, ascending=False, inplace=True, na_position='last')
    out_path = RESULTS_DIR / f"{BENCH_PREFIX}{dt.datetime.now().strftime('%Y%m%d')}.csv"
    combined_df.to_csv(out_path, index=False, float_format="%.4f")
    return str(out_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Run EQBenchFree head-to-head judging and export TrueSkill scores.")
    parser.add_argument("--judge", default=DEFAULT_JUDGE_NAME, help="Judge model display name from OpenBench CSV.")
    parser.add_argument("--max-battles", type=int, default=DEFAULT_BATTLES_TO_RUN, help="Limit number of new battles per run.")
    parser.add_argument("--scenarios", help="Comma-separated scenario IDs (e.g., '1,2,6'). Defaults to all.")
    parser.add_argument("--workers", type=int, default=40, help="Number of concurrent judge calls.")
    return parser.parse_args()


def main():
    args = parse_args()
    judge = resolve_judge_model(args.judge)

    # Load pairwise comparison template
    template = load_pairwise_template()
    print(f"Using pairwise template: {PAIRWISE_PROMPT_FILE.name}")

    response_index = load_response_index()
    if len(response_index) < 2:
        print(f"Not enough responses to run head-to-head comparisons. Found {len(response_index)} models.")
        return

    scenario_counts = [len(s) for s in response_index.values()]
    print(f"Loaded responses from {len(response_index)} models ({min(scenario_counts)}-{max(scenario_counts)} scenarios each)")

    scenario_filter = None
    if args.scenarios:
        scenario_filter = {s.strip() for s in args.scenarios.split(",") if s.strip()}

    # Run info-gain loop via shared arena
    history_df, all_new_records = arena.run_info_gain_loop(
        content_index=response_index,
        judge=judge,
        run_single_battle_fn=run_battle,
        build_paired_results_fn=build_paired_results,
        max_battles=args.max_battles,
        workers=args.workers,
        item_filter=scenario_filter,
        template=template,
    )

    # ─────────────────── SAVE RESULTS ───────────────────
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
    combined_df = pd.DataFrame({"model": sorted(all_models)})

    # Compute TrueSkill ratings for all judges and augment combined DF
    all_pair_dfs = []
    first_judge_col = None
    for judge_name in all_judges:
        pair_df = build_paired_results(history_df, judge_name)
        if not pair_df.empty:
            all_pair_dfs.append(pair_df)
        ratings = arena.compute_trueskill_ratings(pair_df)
        augment_eqbench_columns(combined_df, ratings, judge_name)
        if first_judge_col is None:
            first_judge_col = f"{judge_name} TrueSkill"

    # Save combined results
    out_path = save_results(combined_df, sort_by_col=first_judge_col)

    # Save combined paired results
    if all_pair_dfs:
        combined_pairs = pd.concat(all_pair_dfs, ignore_index=True)
        combined_pairs.to_csv(BATTLE_PAIRS_CSV, index=False)
        print(f"Paired battle aggregates written to {BATTLE_PAIRS_CSV}")

    print(f"TrueSkill scores for all judges ({len(all_judges)}) exported to {out_path}")

    # Show stats for each judge
    for judge_name in all_judges:
        # Print rankings
        pair_df = build_paired_results(history_df, judge_name)
        ratings = arena.compute_trueskill_ratings(pair_df)
        if ratings:
            print(f"\n--- Final Rankings ({judge_name}) ---")
            for model, rating in sorted(ratings.items(), key=lambda x: x[1].mu, reverse=True):
                print(f"  {rating.mu:.1f} (\u03c3={rating.sigma:.1f}) {model}")

        arena.summarize_position_bias(history_df, judge_name)
        arena.summarize_win_loss_stats(history_df, judge_name)


if __name__ == "__main__":
    main()
