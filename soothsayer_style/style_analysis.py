"""Stage 2: Per-response LLM-as-judge evaluation for style benchmark.

Reads responses.csv (status=="ok" only), judges each individual response
with multiple replicates across a judge model. Scores two axes:
signal_density and conversational_confidence.

Output: judge_results.csv
Schema: question_id, model_name, model_id, run_number, judge_model_id, judge_model_name, replicate_id, score_density, score_confidence, duration_s
"""

import argparse
import csv
import os
import sys
import time
import concurrent.futures
from typing import Any, Dict, List, Set, Tuple

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from core.utils import load_models, discover_openbench_csv

from core.llm_client import API_KEY, get_llm_response

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESPONSES_CSV = os.path.join(SCRIPT_DIR, 'responses.csv')
JUDGE_RESULTS_CSV = os.path.join(SCRIPT_DIR, 'judge_results.csv')
SYSTEM_PROMPT_FILE = os.path.join(SCRIPT_DIR, 'style_system_prompt3.txt')

MODELS_CSV_FILE = discover_openbench_csv(SCRIPT_DIR)

JUDGE_MODELS = ["Grok 4.1 Fast"]
NUM_REPLICATES = 3
MAX_WORKERS = 10
SAVE_BATCH_SIZE = 10
MAX_PARSE_ATTEMPTS = 5

FIELDNAMES = [
    'question_id', 'model_name', 'model_id', 'run_number',
    'judge_model_id', 'judge_model_name', 'replicate_id',
    'score_density', 'score_confidence', 'duration_s',
]


# ==============================================================================
# --- HELPERS ---
# ==============================================================================
def is_valid_dual_score(s: str) -> Tuple[bool, int, int]:
    """Check if a string is a valid "N,N" dual score. Returns (valid, density, confidence)."""
    try:
        parts = s.strip().split(',')
        if len(parts) != 2:
            return False, 0, 0
        density = int(parts[0].strip())
        confidence = int(parts[1].strip())
        if 1 <= density <= 100 and 1 <= confidence <= 100:
            return True, density, confidence
        return False, 0, 0
    except (ValueError, TypeError, AttributeError):
        return False, 0, 0


def load_existing_judgments(filename: str) -> Set[Tuple[str, str, str, str, str]]:
    """Load 5-tuple resume keys: (question_id, model_name, run_number, judge_model_id, replicate_id).

    Only rows with valid numeric scores count as completed — failed parses will be retried.
    """
    completed = set()
    if not os.path.exists(filename):
        return completed
    try:
        with open(filename, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                density = row.get('score_density', '')
                confidence = row.get('score_confidence', '')
                try:
                    if not (float(density) and float(confidence)):
                        continue
                except (ValueError, TypeError):
                    continue
                key = (
                    str(row.get('question_id', '')),
                    str(row.get('model_name', '')),
                    str(row.get('run_number', '')),
                    str(row.get('judge_model_id', '')),
                    str(row.get('replicate_id', '')),
                )
                completed.add(key)
    except Exception as e:
        print(f"Warning: Could not read existing judgments from {filename}: {e}")
    return completed


def save_results(results: List[Dict[str, Any]], filename: str) -> bool:
    """Append results to CSV with header logic. Returns True on success."""
    if not results:
        return True

    file_exists = os.path.exists(filename)
    write_header = not file_exists or os.path.getsize(filename) == 0

    try:
        if file_exists and not write_header:
            with open(filename, 'rb+') as f:
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b'\n':
                    f.write(b'\n')

        with open(filename, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction='ignore')
            if write_header:
                writer.writeheader()
            writer.writerows(results)

        print(f"Appended {len(results)} judge results to {filename}")
        return True
    except IOError as e:
        print(f"Error writing to {filename}: {e}")
        return False


def build_single_response_prompt(question_text: str, response_text: str, model_name: str) -> str:
    """Build a judge prompt for a single question/response pair."""
    return (
        f"Please analyze the following AI model response based on the criteria "
        f"outlined in the system prompt.\n\n"
        f"--- START RESPONSE DATA ---\n"
        f"Model: {model_name}\n"
        f"Question: {question_text}\n"
        f"Response: {response_text}\n"
        f"--- END RESPONSE DATA ---\n\n"
        f"Provide your analysis score:"
    )


def process_single_judgment(task: Dict[str, Any]) -> Dict[str, Any]:
    """Judge one response with one judge model for one replicate."""
    start_time = time.time()
    question_id = task['question_id']
    model_name = task['model_name']
    model_id = task['model_id']
    run_number = task['run_number']
    judge_model_id = task['judge_model_id']
    judge_model_name = task['judge_model_name']
    replicate_id = task['replicate_id']
    prompt = task['prompt']
    system_prompt = task['system_prompt']

    density_val = None
    confidence_val = None
    for attempt in range(MAX_PARSE_ATTEMPTS):
        try:
            result = get_llm_response(
                prompt=prompt,
                model=judge_model_id,
                name=judge_model_name,
                system_prompt=system_prompt,
                reasoning=True,
            )
            valid, d, c = is_valid_dual_score(result)
            if valid:
                density_val = d
                confidence_val = c
                break
            else:
                print(f"  Bad parse from {judge_model_name} for {model_name} Q{question_id}: '{result[:80]}', retrying ({attempt+1}/{MAX_PARSE_ATTEMPTS})")
        except Exception as e:
            print(f"  Error judging {model_name} Q{question_id} R{run_number} by {judge_model_name} rep{replicate_id}: {e}")
            break

    duration = time.time() - start_time

    return {
        'question_id': question_id,
        'model_name': model_name,
        'model_id': model_id,
        'run_number': str(run_number),
        'judge_model_id': judge_model_id,
        'judge_model_name': judge_model_name,
        'replicate_id': str(replicate_id),
        'score_density': str(density_val) if density_val is not None else '',
        'score_confidence': str(confidence_val) if confidence_val is not None else '',
        'duration_s': round(duration, 2),
    }


# ==============================================================================
# --- MAIN ---
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Style benchmark: LLM-as-judge evaluation (2-axis)")
    parser.add_argument('--pilot', type=int, default=None, metavar='N',
                        help='Limit to first N unique models (for pilot runs)')
    return parser.parse_args()


def main():
    args = parse_args()

    if not API_KEY:
        print("FATAL: API Key is missing. Please set OPENROUTER_API_KEY.")
        return

    # Load system prompt
    try:
        with open(SYSTEM_PROMPT_FILE, 'r', encoding='utf-8') as f:
            system_prompt = f.read()
    except FileNotFoundError:
        print(f"Error: System prompt file not found at {SYSTEM_PROMPT_FILE}")
        return

    # Load responses (status=="ok" only)
    if not os.path.exists(RESPONSES_CSV):
        print(f"Error: {RESPONSES_CSV} not found. Run collect.py first.")
        return

    df = pd.read_csv(RESPONSES_CSV)
    df = df[df['status'] == 'ok'].copy()
    # Dedupe on resume keys to prevent duplicate judgments from duplicate response rows
    df = df.drop_duplicates(subset=['question_id', 'model_name', 'run_number'], keep='last')
    if df.empty:
        print("No valid responses to judge.")
        return

    # Apply --pilot limit
    if args.pilot is not None:
        unique_models = df['model_name'].unique()[:args.pilot]
        df = df[df['model_name'].isin(unique_models)]
        print(f"Pilot mode: limiting to {len(unique_models)} models")

    print(f"Loaded {len(df)} valid responses from {RESPONSES_CSV}")

    # Resolve judge model IDs from openbench CSV
    all_models = load_models(MODELS_CSV_FILE)
    judge_models = [m for m in all_models if m['name'] in JUDGE_MODELS]
    if not judge_models:
        print(f"Error: None of {JUDGE_MODELS} found in model CSV.")
        return
    print(f"Using {len(judge_models)} judge models: {[m['name'] for m in judge_models]}")

    # Load existing judgments for resume
    existing = load_existing_judgments(JUDGE_RESULTS_CSV)
    print(f"Found {len(existing)} existing judgments.")

    # Build task list
    tasks = []
    skipped = 0
    for _, row in df.iterrows():
        question_id = str(row['question_id'])
        model_name = str(row['model_name'])
        model_id = str(row['model_id'])
        run_number = str(row['run_number'])
        response_text = str(row['response'])

        # Use question text stored in responses.csv (matches what the model was actually asked)
        question_text = str(row.get('question_text', ''))

        prompt = build_single_response_prompt(question_text, response_text, model_name)

        for judge in judge_models:
            for rep_id in range(NUM_REPLICATES):
                key = (question_id, model_name, run_number, judge['id'], str(rep_id))
                if key in existing:
                    skipped += 1
                    continue

                tasks.append({
                    'question_id': question_id,
                    'model_name': model_name,
                    'model_id': model_id,
                    'run_number': run_number,
                    'judge_model_id': judge['id'],
                    'judge_model_name': judge['name'],
                    'replicate_id': rep_id,
                    'prompt': prompt,
                    'system_prompt': system_prompt,
                })

    print(f"Tasks to run: {len(tasks)} (skipped {skipped} already completed)")

    if not tasks:
        print("All judgments already completed.")
        return

    # Execute with batched saves
    new_results = []
    total_saved = 0

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_task = {executor.submit(process_single_judgment, t): t for t in tasks}
            for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(tasks), desc="Judging responses"):
                try:
                    result = future.result()
                    new_results.append(result)
                    if len(new_results) >= SAVE_BATCH_SIZE:
                        if save_results(new_results, JUDGE_RESULTS_CSV):
                            total_saved += len(new_results)
                            new_results.clear()
                except Exception as e:
                    task_info = future_to_task[future]
                    print(f"FATAL ERROR: {task_info['model_name']} Q{task_info['question_id']} by {task_info['judge_model_name']}: {e}")
    except KeyboardInterrupt:
        print("\nInterrupted! Saving completed results...")
    finally:
        if new_results:
            if save_results(new_results, JUDGE_RESULTS_CSV):
                total_saved += len(new_results)

    print(f"\nTotal judgments saved this session: {total_saved}")


if __name__ == "__main__":
    main()
