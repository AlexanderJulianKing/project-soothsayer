import json
import csv
import sys
import time
import os
import requests # type: ignore
import random
import concurrent.futures
from typing import List, Dict, Any, Set, Tuple

csv.field_size_limit(sys.maxsize)
from tqdm import tqdm # type: ignore # For a nice progress bar
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from core.utils import get_latest_file, discover_openbench_csv

from llm_client import get_llm_response, APIError, API_KEY



# --- Configuration ---
EVAL_DATA_FILE = 'questions.json'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_CSV_FILE = discover_openbench_csv(SCRIPT_DIR)
OUTPUT_CSV_FILE = 'benchmark_results_multi_run.csv'
JUDGE_MODEL_ID = 'google/gemini-3-flash-preview'
JUDGE_MODEL_NAME = 'Gemini 3.0 Flash Preview (2025-12-17)'
N_RUNS = 4 # Number of times to run each model per question

# --- Parallelism & API Settings ---
# Adjust based on your machine and rate limits. For I/O bound tasks, more workers can be beneficial.
# os.cpu_count() is a good starting point.
MAX_PARALLEL_WORKERS = os.cpu_count() - 1 # type: ignore
MAX_PARALLEL_WORKERS = 20
SAVE_BATCH_SIZE = 10      # Flush partial results after this many completions

# Create a single session object to reuse connections
api_session = requests.Session()
api_session.headers.update({
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost", # Optional, but good practice
    "X-Title": "Parallel-Benchmarker",   # Optional, for OpenRouter analytics
})


def create_judge_prompt(question_prompt: str, reference_answer: str, model_answer: str) -> str:
    """Creates a prompt for the judge LLM to evaluate the model's answer."""
    return f"""**Instruction:** Evaluate if the 'Submitted Answer' correctly answers the 'Question' by conveying the same essential information and conclusion as the 'Reference Answer'. The wording, length, or presence of extra *correct* explanation or reasoning in the Submitted Answer does not matter, as long as the core answer aligns with the Reference Answer and accurately addresses the Question.

Respond with only ONE of the following words based on this comparison:

*   **Correct:** The Submitted Answer accurately provides the key information or conclusion found in the Reference Answer. It correctly answers the Question based on the reference's standard. Extraneous *correct* information or reasoning is permissible.
*   **Incorrect:** The Submitted Answer provides factually incorrect information, fundamentally misunderstands the question, contradicts the Reference Answer's key point, fails to provide the essential information, or contains significant flawed reasoning even if the final conclusion happens to align. 
*   **Partially Correct:** The Submitted Answer contains some correct elements aligned with the Reference Answer but misses crucial details, includes minor inaccuracies, isn't a complete answer, or contains flawed reasoning alongside correct elements.

**Question:**
{question_prompt}

**Reference Answer:**
{reference_answer}

**Submitted Answer:**
{model_answer}

**Judgment (One Word Only):**"""


def process_single_run(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker function to process one question-model-run combination.
    This function is executed in a separate process.
    """
    question_id = task['question_id']
    prompt = task['prompt']
    model_id = task['model_id']
    run_num = task['run_number']
    reasoning = task['reasoning']
    model_name = task['model_name']
    
    # Defaults for usage columns in case calls fail/are skipped
    task['model_output_tokens'] = None
    task['model_reasoning_tokens'] = None
    task['judge_output_tokens'] = None
    task['judge_reasoning_tokens'] = None

    # --- 1. Get Model Response + Usage ---
    try:
        model_response, model_usage = get_llm_response(
            prompt, model_id, model_name, reasoning, include_usage=True
        )
        task['model_output_tokens'] = (model_usage.get("completion_tokens") or 0)
        task['model_reasoning_tokens'] = (model_usage.get("reasoning_tokens") or 0)
    except APIError as e:
        print(f"ERROR [Q:{question_id}, M:{model_id}, R:{run_num}]: Model call failed: {e}")
        model_response = f"Error: {e}"
    
    # --- 2. Get Judge Response + Usage ---
    judge_response = "Skipped: Model failed"
    if not model_response.startswith("Error:"):
        try:
            judge_prompt = create_judge_prompt(prompt, task['reference_answer'], model_response)
            judge_response_raw, judge_usage = get_llm_response(
                judge_prompt, JUDGE_MODEL_ID, JUDGE_MODEL_NAME, True, include_usage=True
            )
            judge_response_raw = judge_response_raw.replace('*', '')

            # Store judge usage
            task['judge_output_tokens'] = (judge_usage.get("completion_tokens") or 0)
            task['judge_reasoning_tokens'] = (judge_usage.get("reasoning_tokens") or 0)

            # Process judge response
            parts = judge_response_raw.split()
            if parts:
                first_word = parts[0].strip().capitalize()
                if first_word.startswith("Correct"):
                    judge_response = "Correct"
                elif first_word.startswith("Incorrect"):
                    judge_response = "Incorrect"
                elif first_word.startswith("Partially"):
                    judge_response = "Partially Correct"
                else:
                    judge_response = f"Unexpected: {judge_response_raw}"
            else:
                judge_response = "Error: Empty judge response"
        except APIError as e:
            print(f"ERROR [Q:{question_id}, M:{model_id}, R:{run_num}]: Judge call failed: {e}")
            judge_response = f"Error: {e}"
            
    # --- 3. Assemble Full Result ---
    task['model_response'] = model_response
    task['judge_model_id'] = JUDGE_MODEL_ID
    task['judge_response'] = judge_response

    return task

def load_data(file_path: str, is_json: bool = False) -> List[Dict[str, Any]]:
    """Loads data from a JSON or CSV file."""
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            if is_json:
                return json.load(f).get('eval_data', [])
            else: # is CSV
                reader = csv.DictReader(f)
                return [row for row in reader]
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

def load_existing_results(filename: str) -> Set[Tuple[str, str, str]]:
    """Loads previously completed (question_id, model_name, run_number) tuples to avoid re-running."""
    completed_runs = set()
    if not os.path.exists(filename):
        return completed_runs

    try:
        with open(filename, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                q_id = row.get('question_id')
                m_id = row.get('model_name')
                run_num = row.get('run_number')
                if q_id and m_id and run_num:
                    completed_runs.add((q_id, m_id, run_num))
    except Exception as e:
        print(f"Warning: Could not properly read existing results from {filename}: {e}. May re-run some tasks.")
    
    return completed_runs


def save_results(results: List[Dict[str, Any]], filename: str):
    """Appends a list of results to the CSV file, writing a header if the file is new."""
    if not results:
        print("\nNo new results to save.")
        return

    fieldnames = [
        'question_id', 'prompt', 'reference_answer', 'model_name', 'model_id',
        'run_number', 'model_response', 'judge_model_id', 'judge_response'
    ]
    
    file_exists = os.path.exists(filename)
    
    try:
        with open(filename, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(results)
        print(f"\nSuccessfully appended {len(results)} new results to {filename}")
    except IOError as e:
        print(f"Error writing to output file {filename}: {e}")



def save_results(results: List[Dict[str, Any]], filename: str):
    """
    Appends a list of results to the CSV file, ensuring a newline exists
    before appending and writing a header if the file is new.
    """
    if not results:
        print("\nNo new results to save.")
        return

    fieldnames = [
        'question_id', 'prompt', 'reference_answer', 'model_name', 'model_id',
        'run_number', 'model_response', 'judge_model_id', 'judge_response',
        # NEW USAGE COLUMNS:
        'model_output_tokens', 'model_reasoning_tokens',
        'judge_output_tokens', 'judge_reasoning_tokens',
    ]

    file_exists = os.path.exists(filename)
    write_header = not file_exists or os.path.getsize(filename) == 0

    try:
        if file_exists and not write_header:
            with open(filename, 'rb+') as f:
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b'\n':
                    f.write(b'\n')

        with open(filename, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            if write_header:
                writer.writeheader()
            writer.writerows(results)

        print(f"\nSuccessfully appended {len(results)} new results to {filename}")
    except IOError as e:
        print(f"Error writing to output file {filename}: {e}")

def main():
    """Main function to orchestrate the benchmarking process."""
    # --- 1. Load all data and existing results (Sequentially) ---
    # print(get_llm_response(prompt = 'say ping', model = 'qwen/qwen3-30b-a3b', name = "Qwen3 30b a3b", reasoning= True))


    eval_data = load_data(EVAL_DATA_FILE, is_json=True)
    models_raw = load_data(MODELS_CSV_FILE)

    if not eval_data or not models_raw:
        print("Failed to load evaluation data or models. Exiting.")
        return

    models_to_test = [{'name': m['Model'], 'id': m['openbench_id'], 'reasoning' : m['Reasoning']} for m in models_raw if m.get('Model') and m.get('openbench_id') and m.get('Reasoning')]
    print(f"Loaded {len(eval_data)} questions and {len(models_to_test)} models.")
    existing_results_set = load_existing_results(OUTPUT_CSV_FILE)
    print(f"Found {len(existing_results_set)} previously completed runs.")

    # --- 2. Prepare the list of tasks to run (Sequentially) ---
    tasks_to_run = []
    for question in eval_data:
        q_id = str(question.get('question_id'))
        if not q_id or not question.get('prompt') or not question.get('answer'):
            continue # Skip invalid questions
            
        for model in models_to_test:
            for run_num in range(1, N_RUNS + 1):
                run_id = (q_id, model['name'], str(run_num))
                if run_id not in existing_results_set:

                    reasoning_enabled = model['reasoning'].lower() in ['true', 'yes', '1']

                    tasks_to_run.append({
                        'question_id': q_id,
                        'prompt': question['prompt'],
                        'reference_answer': question['answer'],
                        'model_name': model['name'],
                        'model_id': model['id'],
                        'reasoning': reasoning_enabled,
                        'run_number': run_num,
                    })

    if not tasks_to_run:
        print("All runs are already completed based on the output file. Nothing to do.")
        return

    print(f"Total runs to perform in this session: {len(tasks_to_run)}")

    # --- 3. Execute tasks in parallel ---
    t0 = time.time()
    new_results = []
    total_saved = 0
    # Using ProcessPoolExecutor to run tasks in parallel processes
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
        # tqdm shows a progress bar
        future_to_task = {executor.submit(process_single_run, task): task for task in tasks_to_run}
        for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(tasks_to_run), desc="Processing runs"):
            try:
                result = future.result()
                if result:
                    new_results.append(result)
                    if len(new_results) >= SAVE_BATCH_SIZE:
                        batch_size = len(new_results)
                        save_results(new_results, OUTPUT_CSV_FILE)
                        total_saved += batch_size
                        new_results.clear()
            except Exception as e:
                task_info = future_to_task[future]
                print(f"FATAL ERROR for task {task_info['question_id']}-{task_info['model_id']}-{task_info['run_number']}: {e}")


    # --- 4. Save all collected results at once ---
    if new_results:
        save_results(new_results, OUTPUT_CSV_FILE)
        total_saved += len(new_results)
    elapsed = time.time() - t0
    rate = total_saved / elapsed * 60 if elapsed > 0 else 0
    print(f"\nTotal results saved: {total_saved} in {elapsed:.0f}s ({rate:.1f} tasks/min)")


if __name__ == "__main__":
    main()
