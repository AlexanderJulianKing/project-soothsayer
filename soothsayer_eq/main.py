"""
EQBenchFree - Main response generation script.
Generates multi-turn responses from models for emotional intelligence scenarios.
"""

import os
import re
import sys
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from core.utils import get_latest_file, load_models, discover_openbench_csv

from core.llm_client import get_llm_response, APIError
from scenario_parser import parse_scenarios, get_initial_scenarios

# --- CONFIGURATION ---
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 5
MAX_CONCURRENT_REQUESTS = 15

# File & Directory Paths
SCRIPT_DIR = Path(__file__).parent
SCENARIOS_FILE = SCRIPT_DIR / "scenario_prompts.txt"
MASTER_PROMPT_FILE = SCRIPT_DIR / "scenario_master_prompt_message_drafting.txt"
RESPONSES_DIR = SCRIPT_DIR / "generated_responses"
RESULTS_DIR = SCRIPT_DIR / "results"


def load_master_prompt_template() -> str:
    """Load the master prompt template for structured responses."""
    if MASTER_PROMPT_FILE.exists():
        with open(MASTER_PROMPT_FILE, 'r', encoding='utf-8') as f:
            return f.read()
    # Fallback if file doesn't exist
    return "{scenario_prompt}"


def format_prompt_with_template(scenario_prompt: str, template: str) -> str:
    """Wrap a scenario prompt with the master template."""
    return template.replace("{scenario_prompt}", scenario_prompt)


def parse_structured_response(response: str) -> Dict[str, str]:
    """
    Parse the structured response into its components.

    Expected format:
    # Perspective-taking
    <content>

    # Draft brainstorming
    <content>

    # Draft
    <content>
    """
    sections = {
        "perspective_taking": "",
        "draft_brainstorming": "",
        "draft": "",
        "raw": response,
    }

    # Try to extract sections
    perspective_match = re.search(
        r'#\s*Perspective-taking\s*\n(.*?)(?=#\s*Draft brainstorming|$)',
        response, re.DOTALL | re.IGNORECASE
    )
    if perspective_match:
        sections["perspective_taking"] = perspective_match.group(1).strip()

    brainstorm_match = re.search(
        r'#\s*Draft brainstorming\s*\n(.*?)(?=#\s*Draft|$)',
        response, re.DOTALL | re.IGNORECASE
    )
    if brainstorm_match:
        sections["draft_brainstorming"] = brainstorm_match.group(1).strip()

    draft_match = re.search(
        r'#\s*Draft\s*\n(.*?)$',
        response, re.DOTALL | re.IGNORECASE
    )
    if draft_match:
        sections["draft"] = draft_match.group(1).strip()

    return sections




def generate_scenario_response(
    model: Dict[str, Any],
    scenario: Dict[str, Any],
    output_path: Path,
    master_template: str,
    force: bool = False,
) -> str:
    """
    Generate multi-turn responses for a scenario.

    Args:
        model: Model info dict with 'name', 'id', 'Reasoning'
        scenario: Scenario dict with 'id', 'category', 'title', 'prompts'
        output_path: Path to save JSON output
        master_template: The master prompt template for structured responses
        force: If True, regenerate even if already completed

    Returns:
        Status message string
    """
    model_name = model['name'].replace('/', '_')
    scenario_id = scenario['id']

    # Skip if already completed (unless force mode)
    if output_path.exists() and not force:
        try:
            with open(output_path, 'r') as f:
                existing = json.load(f)
            if existing.get('completed', False):
                return f"⊘ Already completed: {model_name} scenario {scenario_id}"
        except (json.JSONDecodeError, KeyError):
            pass

    # Build conversation
    conversation = {
        "model": model['name'],
        "model_id": model['id'],
        "scenario_id": scenario_id,
        "category": scenario['category'],
        "title": scenario['title'],
        "turns": [],
        "completed": False,
    }

    messages = []  # Conversation history for context

    for turn_num, prompt_text in enumerate(scenario['prompts'], start=1):
        # Add user message (the scenario prompt)
        messages.append({"role": "user", "content": prompt_text})

        # Get model response
        tries = 0
        response_text = None

        while tries < MAX_RETRIES and response_text is None:
            try:
                # Build full prompt with conversation history and template
                if len(messages) == 1:
                    # First turn - wrap with template
                    full_prompt = format_prompt_with_template(prompt_text, master_template)
                else:
                    # Subsequent turns - include prior context + wrap with template
                    context = ""
                    for msg in messages[:-1]:
                        if msg['role'] == 'user':
                            context += f"[Previous Scenario Context]:\n{msg['content']}\n\n"
                        else:
                            context += f"[Your Previous Response]:\n{msg['content']}\n\n"
                    context += f"[New Development]:\n{prompt_text}"
                    full_prompt = format_prompt_with_template(context, master_template)

                response_text = get_llm_response(
                    prompt=full_prompt,
                    model=model['id'],
                    name=model['name'],
                    reasoning=model.get('Reasoning', False),
                )
            except APIError as e:
                print(f"  Retry {tries+1}/{MAX_RETRIES} for {model_name} turn {turn_num}: {e}")
                time.sleep(INITIAL_RETRY_DELAY * (tries + 1))
                tries += 1

        if response_text is None:
            return f"✗ Failed: {model_name} scenario {scenario_id} turn {turn_num}"

        # Parse structured response
        parsed = parse_structured_response(response_text)

        # Add assistant response to history
        messages.append({"role": "assistant", "content": response_text})

        # Record turn with parsed sections
        conversation["turns"].append({
            "turn": turn_num,
            "prompt": prompt_text,
            "response": response_text,
            "parsed": parsed,
        })

    conversation["completed"] = True

    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(conversation, f, indent=2, ensure_ascii=False)

    return f"✓ Completed: {model_name} scenario {scenario_id} ({len(scenario['prompts'])} turns)"


def generate_response_task(args):
    """Wrapper for parallel execution."""
    model, scenario, output_path, master_template, force = args
    try:
        return generate_scenario_response(model, scenario, output_path, master_template, force)
    except Exception as e:
        model_name = model['name'].replace('/', '_')
        return f"✗ Error: {model_name} scenario {scenario['id']}: {e}"


def run_generation(
    models: List[Dict[str, Any]],
    scenarios: Dict[int, dict],
    max_workers: int = MAX_CONCURRENT_REQUESTS,
    force: bool = False,
):
    """Run response generation for all model-scenario combinations."""
    # Load master prompt template
    master_template = load_master_prompt_template()
    print(f"Using master prompt template: {MASTER_PROMPT_FILE.name}")
    if force:
        print("Force mode enabled - will regenerate existing responses")

    # Build task list
    tasks = []
    for model in models:
        model_dir = RESPONSES_DIR / model['name'].replace('/', '_')
        for scenario_id, scenario in scenarios.items():
            output_path = model_dir / f"scenario_{scenario_id}.json"
            tasks.append((model, scenario, output_path, master_template, force))

    print(f"Generating responses for {len(models)} models x {len(scenarios)} scenarios = {len(tasks)} tasks")

    # Run in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(generate_response_task, task): task for task in tasks}

        for future in tqdm(as_completed(futures), total=len(tasks), desc="Generating"):
            result = future.result()
            if result.startswith("✗"):
                print(f"\n{result}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate EQBenchFree responses")
    parser.add_argument("--models-csv", type=str, help="Path to OpenBench CSV with model list")
    parser.add_argument("--scenarios", type=str, nargs="+", help="Specific scenario IDs to run")
    parser.add_argument("--all-scenarios", action="store_true", help="Run all scenarios (not just initial 5)")
    parser.add_argument("--workers", type=int, default=MAX_CONCURRENT_REQUESTS, help="Parallel workers")
    parser.add_argument("--model-filter", type=str, help="Filter models by name substring")
    parser.add_argument("--force", action="store_true", help="Re-generate even if response already exists")

    args = parser.parse_args()

    # Load models
    if args.models_csv:
        csv_path = args.models_csv
    else:
        try:
            csv_path = discover_openbench_csv(str(SCRIPT_DIR))
        except ValueError:
            print("Error: No model CSV found. Provide --models-csv or place openbench_*.csv in directory.")
            return

    print(f"Loading models from: {csv_path}")
    models = load_models(csv_path)

    if args.model_filter:
        models = [m for m in models if args.model_filter.lower() in m['name'].lower()]
        print(f"Filtered to {len(models)} models matching '{args.model_filter}'")

    if not models:
        print("No models found!")
        return

    print(f"Loaded {len(models)} models")

    # Load scenarios
    if args.scenarios:
        scenario_ids = [int(s) for s in args.scenarios]
        scenarios = parse_scenarios(str(SCENARIOS_FILE), scenario_ids=scenario_ids)
    elif args.all_scenarios:
        scenarios = parse_scenarios(str(SCENARIOS_FILE))
    else:
        scenarios = get_initial_scenarios(str(SCENARIOS_FILE))

    print(f"Using {len(scenarios)} scenarios: {list(scenarios.keys())}")

    # Run generation
    run_generation(models, scenarios, max_workers=args.workers, force=args.force)

    print("\nGeneration complete!")


if __name__ == "__main__":
    main()
