"""Stage 1: Collect model responses for the style benchmark.

Long-format CSV output with multiple runs per (model, question).
Resume-safe via tuple-set pattern. Auto-migrates legacy model_outputs.csv.

Output: responses.csv
Schema: question_id, question_text, model_name, model_id, run_number, response, response_length, status
"""

import csv
import os
import random
import sys
import threading
import time
import concurrent.futures
from typing import Any, Dict, List, Set, Tuple

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from core.utils import load_models, discover_openbench_csv, normalize_reasoning_flag

from llm_client import get_llm_response, APIError

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_CSV_FILE = discover_openbench_csv(SCRIPT_DIR)
QUESTIONS_FILE = os.path.join(SCRIPT_DIR, 'questions.txt')
RESPONSES_CSV = os.path.join(SCRIPT_DIR, 'responses.csv')
LEGACY_OUTPUT_FILE = os.path.join(SCRIPT_DIR, 'model_outputs.csv')

N_RUNS = 3
MAX_PARALLEL_WORKERS = 40
MAX_PER_MODEL = 5          # concurrent requests per model
RATE_LIMIT_RETRIES = 5     # collect-level retries on 429 (on top of llm_client's 5)
SAVE_BATCH_SIZE = 10

FIELDNAMES = [
    'question_id', 'question_text', 'model_name', 'model_id',
    'run_number', 'response', 'response_length', 'status',
]


# ==============================================================================
# --- HELPERS ---
# ==============================================================================
def load_questions(filepath: str) -> List[str]:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(questions)} questions from {filepath}")
        return questions
    except FileNotFoundError:
        print(f"Error: Questions file not found at {filepath}")
        return []


def load_existing_results(filename: str) -> Set[Tuple[str, str, str]]:
    """Load previously completed (question_id, model_name, run_number) tuples.

    Only status="ok" rows count as completed — errors and empties will be retried.
    """
    completed = set()
    if not os.path.exists(filename):
        return completed
    try:
        with open(filename, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('status') != 'ok':
                    continue
                q_id = row.get('question_id')
                m_name = row.get('model_name')
                run_num = row.get('run_number')
                if q_id and m_name and run_num:
                    completed.add((q_id, m_name, run_num))
    except Exception as e:
        print(f"Warning: Could not read existing results from {filename}: {e}")
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

        print(f"Appended {len(results)} results to {filename}")
        return True
    except IOError as e:
        print(f"Error writing to {filename}: {e}")
        return False


def migrate_legacy_data(legacy_file: str, target_file: str, questions: List[str]):
    """Migrate wide-format model_outputs.csv to long-format responses.csv as run_number=1.

    Only runs if target_file doesn't exist yet. Skips empty/NaN responses.
    """
    if os.path.exists(target_file):
        return
    if not os.path.exists(legacy_file):
        return

    print(f"Migrating legacy data from {legacy_file} to {target_file}...")
    try:
        df = pd.read_csv(legacy_file)
    except Exception as e:
        print(f"Warning: Could not read legacy file: {e}")
        return

    q_cols = [col for col in df.columns if col.startswith('Q') and col[1:].isdigit()]
    if not q_cols:
        print("Warning: No question columns found in legacy file.")
        return

    rows = []
    for _, model_row in df.iterrows():
        model_name = str(model_row.get('model', ''))
        model_id = str(model_row.get('code', ''))

        for q_col in q_cols:
            q_num = int(q_col[1:])
            if q_num < 1 or q_num > 9:
                continue

            response = model_row.get(q_col)
            # Determine question text from questions list
            q_text = questions[q_num - 1] if q_num - 1 < len(questions) else ''

            # Determine status
            if pd.isna(response) or str(response).strip() == '':
                status = 'empty'
                response = ''
            elif str(response).startswith('Error:'):
                status = 'error'
            else:
                status = 'ok'
                response = str(response)

            rows.append({
                'question_id': str(q_num),
                'question_text': q_text,
                'model_name': model_name,
                'model_id': model_id,
                'run_number': '1',
                'response': response,
                'response_length': len(str(response)),
                'status': status,
            })

    save_results(rows, target_file)
    print(f"Migrated {len(rows)} rows from legacy data ({len(df)} models x {len(q_cols)} questions).")


# ==============================================================================
# --- PER-MODEL RATE-LIMIT THROTTLING ---
# ==============================================================================
_throttle_lock = threading.Lock()
_model_semaphores: Dict[str, threading.Semaphore] = {}
_model_delays: Dict[str, float] = {}


def _get_gate(model_name: str) -> threading.Semaphore:
    with _throttle_lock:
        if model_name not in _model_semaphores:
            _model_semaphores[model_name] = threading.Semaphore(MAX_PER_MODEL)
            _model_delays[model_name] = 0.0
        return _model_semaphores[model_name]


def _on_rate_limit(model_name: str):
    with _throttle_lock:
        cur = _model_delays.get(model_name, 0.0)
        _model_delays[model_name] = min(max(cur * 2, 4.0), 120.0)
        # Dynamically reduce concurrency for this model
        if model_name in _model_semaphores:
            sem = _model_semaphores[model_name]
            # Try to acquire without blocking to reduce effective concurrency
            if sem._value > 1:
                sem.acquire(blocking=False)


def _on_success(model_name: str):
    with _throttle_lock:
        cur = _model_delays.get(model_name, 0.0)
        if cur > 0:
            _model_delays[model_name] = cur * 0.8


def _get_delay(model_name: str) -> float:
    with _throttle_lock:
        return _model_delays.get(model_name, 0.0)


def process_single_question(task: Dict[str, Any]) -> Dict[str, Any]:
    """Process one (model, question, run) combination.

    Uses per-model semaphore to limit concurrency and retries with backoff
    when rate-limited (429).
    """
    model_name = task['model_name']
    model_id = task['model_id']
    reasoning = task['reasoning']
    question_text = task['question_text']
    question_id = task['question_id']
    run_number = task['run_number']
    gate = _get_gate(model_name)
    last_err = None

    for attempt in range(RATE_LIMIT_RETRIES):
        # Wait for per-model backoff (jittered) before acquiring the gate
        delay = _get_delay(model_name)
        if delay > 0:
            time.sleep(delay + random.uniform(0, delay * 0.5))

        with gate:
            try:
                result = get_llm_response(
                    question_text, model_id, model_name, reasoning, include_usage=True,
                )
                if isinstance(result, tuple):
                    answer, usage = result
                else:
                    answer = result

                if not answer or str(answer).strip() == '':
                    status = 'empty'
                    answer = ''
                else:
                    status = 'ok'
                    answer = str(answer)
                    _on_success(model_name)

                return {
                    'question_id': question_id,
                    'question_text': question_text,
                    'model_name': model_name,
                    'model_id': model_id,
                    'run_number': str(run_number),
                    'response': answer,
                    'response_length': len(answer),
                    'status': status,
                }
            except APIError as e:
                last_err = e
                err_str = str(e)
                if '429' in err_str or 'rate limit' in err_str.lower():
                    _on_rate_limit(model_name)
                    if attempt < RATE_LIMIT_RETRIES - 1:
                        print(
                            f"RATE-LIMITED [{model_name} Q{question_id} R{run_number}] "
                            f"retry {attempt + 1}/{RATE_LIMIT_RETRIES}, "
                            f"backoff {_get_delay(model_name):.0f}s"
                        )
                        continue
                # Non-429 error or final attempt — give up
                break

    # Last resort: if reasoning model failed, retry once with medium effort
    if reasoning:
        print(
            f"FALLBACK [{model_name} Q{question_id} R{run_number}] "
            f"retrying with reasoning_effort=medium"
        )
        with gate:
            try:
                result = get_llm_response(
                    question_text, model_id, model_name, reasoning,
                    include_usage=True, reasoning_effort="medium",
                )
                if isinstance(result, tuple):
                    answer, usage = result
                else:
                    answer = result

                if answer and str(answer).strip():
                    _on_success(model_name)
                    return {
                        'question_id': question_id,
                        'question_text': question_text,
                        'model_name': model_name,
                        'model_id': model_id,
                        'run_number': str(run_number),
                        'response': str(answer),
                        'response_length': len(str(answer)),
                        'status': 'ok',
                    }
            except APIError as e:
                last_err = e

    err_msg = f"Error: {last_err}"
    print(f"ERROR [{model_name} Q{question_id} R{run_number}]: {last_err}")
    return {
        'question_id': question_id,
        'question_text': question_text,
        'model_name': model_name,
        'model_id': model_id,
        'run_number': str(run_number),
        'response': err_msg,
        'response_length': len(err_msg),
        'status': 'error',
    }


# ==============================================================================
# --- MAIN ---
# ==============================================================================
def main():
    questions = load_questions(QUESTIONS_FILE)
    if not questions:
        print("No questions loaded. Exiting.")
        return

    # Use only Q1-Q9 (same as original)
    questions = questions[:9]

    models = load_models(MODELS_CSV_FILE)
    if not models:
        print("No models loaded. Exiting.")
        return
    print(f"Loaded {len(models)} models from {MODELS_CSV_FILE}")

    # Auto-migrate legacy data if needed
    migrate_legacy_data(LEGACY_OUTPUT_FILE, RESPONSES_CSV, questions)

    # Load existing results for resume
    existing = load_existing_results(RESPONSES_CSV)
    print(f"Found {len(existing)} previously completed runs.")

    # Build task list
    tasks = []
    for model in models:
        model_name = model['name']
        model_id = model['id']
        reasoning = normalize_reasoning_flag(model['Reasoning'])

        for q_idx, q_text in enumerate(questions):
            q_id = str(q_idx + 1)
            for run_num in range(1, N_RUNS + 1):
                run_key = (q_id, model_name, str(run_num))
                if run_key not in existing:
                    tasks.append({
                        'question_id': q_id,
                        'question_text': q_text,
                        'model_name': model_name,
                        'model_id': model_id,
                        'reasoning': reasoning,
                        'run_number': run_num,
                    })

    if not tasks:
        print("All runs already completed. Nothing to do.")
        return

    print(f"Tasks to run: {len(tasks)}")

    # Execute with batched saves
    new_results = []
    total_saved = 0

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
            future_to_task = {executor.submit(process_single_question, t): t for t in tasks}
            for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(tasks), desc="Collecting responses"):
                try:
                    result = future.result()
                    new_results.append(result)
                    if len(new_results) >= SAVE_BATCH_SIZE:
                        if save_results(new_results, RESPONSES_CSV):
                            total_saved += len(new_results)
                            new_results.clear()
                except Exception as e:
                    task_info = future_to_task[future]
                    print(f"FATAL ERROR for {task_info['model_name']} Q{task_info['question_id']} R{task_info['run_number']}: {e}")
    except KeyboardInterrupt:
        print("\nInterrupted! Saving completed results...")
    finally:
        if new_results:
            if save_results(new_results, RESPONSES_CSV):
                total_saved += len(new_results)

    print(f"\nTotal results saved this session: {total_saved}")


if __name__ == "__main__":
    main()
