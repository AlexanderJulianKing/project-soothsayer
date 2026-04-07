"""Re-judge a sample of existing tonebench battles with a second judge model.

Measures inter-judge agreement (concordance) on winner calls per axis.
Does NOT modify any existing results files.

Usage:
    python3 judge_concordance.py --judge "Gemini 3.0 Flash Preview (2025-12-17)" --n 100
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from core.utils import extract_json_payload, discover_openbench_csv, load_models, normalize_reasoning_flag

from core.llm_client import get_llm_response

SCRIPT_DIR = Path(__file__).parent
RESPONSES_CSV = SCRIPT_DIR / "responses.csv"
QUESTIONS_FILE = SCRIPT_DIR / "questions.txt"
RESULTS_DIR = SCRIPT_DIR / "results"
BATTLE_HISTORY_CSV = RESULTS_DIR / "battle_history.csv"
PAIRWISE_PROMPT_FILE = SCRIPT_DIR / "pairwise_prompt_style.txt"

AXES = ["signal_density", "conversational_confidence"]
SYSTEM_PROMPT = (
    "You are an impartial conversational quality judge. "
    "Compare the two anonymous AI responses and respond with valid JSON only."
)
MAX_PARSE_ATTEMPTS = 3


def resolve_judge_model(judge_name: str) -> Dict[str, str]:
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
    raise ValueError(f"Judge '{judge_name}' not found in CSV.")


def load_questions(path: str) -> Dict[str, str]:
    questions = {}
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if line:
                questions[str(i)] = line
    return questions


def load_pairwise_template() -> str:
    with open(PAIRWISE_PROMPT_FILE, 'r', encoding='utf-8') as f:
        return f.read()


def validate_payload(payload: dict) -> bool:
    import re
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


def rejudge_one(row, response_index, questions, template, judge):
    """Re-judge a single battle. Returns (original_row, new_payload) or None."""
    qid = str(row["question_id"])
    model_a = row["response_a_model"]
    model_b = row["response_b_model"]

    # Get responses
    resp_a = response_index.get(model_a, {}).get(qid)
    resp_b = response_index.get(model_b, {}).get(qid)
    if not resp_a or not resp_b:
        return None

    q_text = questions.get(qid, f"Question {qid}")
    prompt = template.replace("{question_text}", q_text)
    prompt = prompt.replace("{response_a}", resp_a.strip())
    prompt = prompt.replace("{response_b}", resp_b.strip())

    for attempt in range(MAX_PARSE_ATTEMPTS):
        raw = get_llm_response(
            prompt=prompt,
            model=judge["id"],
            name=judge["name"],
            reasoning=judge["Reasoning"],
            system_prompt=SYSTEM_PROMPT,
        )
        payload = extract_json_payload(raw)
        if payload and validate_payload(payload):
            return (row, payload)
    return None


def main():
    parser = argparse.ArgumentParser(description="Inter-judge concordance test for ToneBench.")
    parser.add_argument("--judge", required=True, help="Second judge model name.")
    parser.add_argument("--n", type=int, default=100, help="Number of battles to re-judge.")
    parser.add_argument("--workers", type=int, default=10, help="Concurrent API calls.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    args = parser.parse_args()

    judge = resolve_judge_model(args.judge)
    print(f"Second judge: {judge['name']} ({judge['id']})")

    # Load existing battles
    battles = pd.read_csv(BATTLE_HISTORY_CSV)
    print(f"Total existing battles: {len(battles)}")

    # Sample N battles
    sample = battles.sample(n=min(args.n, len(battles)), random_state=args.seed)
    print(f"Sampled {len(sample)} battles for re-judging")

    # Load responses
    responses_df = pd.read_csv(RESPONSES_CSV)
    response_index: Dict[str, Dict[str, str]] = {}
    for _, r in responses_df.iterrows():
        model = r["model_name"]
        qid = str(r["question_id"])
        if r.get("status") == "ok" or pd.isna(r.get("status", "ok")):
            if model not in response_index:
                response_index[model] = {}
            if qid not in response_index[model]:
                response_index[model][qid] = r["response"]

    questions = load_questions(str(QUESTIONS_FILE))
    template = load_pairwise_template()

    # Re-judge in parallel
    results = []
    start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(rejudge_one, row, response_index, questions, template, judge): i
            for i, (_, row) in enumerate(sample.iterrows())
        }
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)
            done = len(results)
            if done % 20 == 0 and done > 0:
                elapsed = time.time() - start
                print(f"  {done}/{len(sample)} re-judged ({elapsed:.0f}s)")

    elapsed = time.time() - start
    print(f"\nRe-judged {len(results)}/{len(sample)} battles in {elapsed:.0f}s")
    print(f"Parse failures: {len(sample) - len(results)}")

    # Compute concordance
    agree_density = 0
    agree_confidence = 0
    agree_both = 0
    margin_diffs_density = []
    margin_diffs_confidence = []

    for orig_row, new_payload in results:
        # Parse original criteria
        orig_criteria = json.loads(orig_row["criteria_json"].replace("'", '"')) \
            if isinstance(orig_row["criteria_json"], str) else orig_row["criteria_json"]
        orig_by_axis = {}
        for c in orig_criteria:
            orig_by_axis[c["name"]] = c

        # Compare signal_density
        orig_sd_winner = "A" if orig_by_axis["signal_density"]["winner"] == orig_row["response_a_model"] else "B"
        new_sd_winner = new_payload["signal_density"]["winner"]
        sd_agree = (orig_sd_winner == new_sd_winner)
        agree_density += sd_agree

        orig_sd_margin = orig_by_axis["signal_density"]["margin_score"]
        new_sd_margin = len(new_payload["signal_density"]["margin"])
        margin_diffs_density.append(abs(orig_sd_margin - new_sd_margin))

        # Compare conversational_confidence
        orig_cc_winner = "A" if orig_by_axis["conversational_confidence"]["winner"] == orig_row["response_a_model"] else "B"
        new_cc_winner = new_payload["conversational_confidence"]["winner"]
        cc_agree = (orig_cc_winner == new_cc_winner)
        agree_confidence += cc_agree

        orig_cc_margin = orig_by_axis["conversational_confidence"]["margin_score"]
        new_cc_margin = len(new_payload["conversational_confidence"]["margin"])
        margin_diffs_confidence.append(abs(orig_cc_margin - new_cc_margin))

        if sd_agree and cc_agree:
            agree_both += 1

    n = len(results)
    print(f"\n{'='*50}")
    print(f"CONCORDANCE RESULTS (Grok 4.1 Fast vs {judge['name']})")
    print(f"{'='*50}")
    print(f"Battles compared:         {n}")
    print(f"Signal density agreement: {agree_density}/{n} ({100*agree_density/n:.1f}%)")
    print(f"Conv. confidence agreement: {agree_confidence}/{n} ({100*agree_confidence/n:.1f}%)")
    print(f"Both axes agree:          {agree_both}/{n} ({100*agree_both/n:.1f}%)")
    print(f"Mean margin diff (density):    {sum(margin_diffs_density)/n:.2f}")
    print(f"Mean margin diff (confidence): {sum(margin_diffs_confidence)/n:.2f}")

    # Save detailed results
    out_rows = []
    for orig_row, new_payload in results:
        orig_criteria = json.loads(orig_row["criteria_json"].replace("'", '"')) \
            if isinstance(orig_row["criteria_json"], str) else orig_row["criteria_json"]
        orig_by_axis = {c["name"]: c for c in orig_criteria}

        orig_sd_winner = "A" if orig_by_axis["signal_density"]["winner"] == orig_row["response_a_model"] else "B"
        new_sd_winner = new_payload["signal_density"]["winner"]
        orig_cc_winner = "A" if orig_by_axis["conversational_confidence"]["winner"] == orig_row["response_a_model"] else "B"
        new_cc_winner = new_payload["conversational_confidence"]["winner"]

        out_rows.append({
            "question_id": orig_row["question_id"],
            "model_a": orig_row["response_a_model"],
            "model_b": orig_row["response_b_model"],
            "grok_sd_winner": orig_sd_winner,
            "new_sd_winner": new_sd_winner,
            "sd_agree": orig_sd_winner == new_sd_winner,
            "grok_sd_margin": orig_by_axis["signal_density"]["margin_score"],
            "new_sd_margin": len(new_payload["signal_density"]["margin"]),
            "grok_cc_winner": orig_cc_winner,
            "new_cc_winner": new_cc_winner,
            "cc_agree": orig_cc_winner == new_cc_winner,
            "grok_cc_margin": orig_by_axis["conversational_confidence"]["margin_score"],
            "new_cc_margin": len(new_payload["conversational_confidence"]["margin"]),
        })

    out_df = pd.DataFrame(out_rows)
    out_path = RESULTS_DIR / f"concordance_{judge['name'].replace(' ', '_')}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    main()
