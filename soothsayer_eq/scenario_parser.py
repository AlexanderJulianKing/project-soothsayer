"""
Scenario parser for EQBenchFree.
Parses scenario_prompts.txt into structured scenario data.
"""

import re
from typing import Dict, List, Optional
from pathlib import Path


def parse_scenarios(filepath: str, scenario_ids: Optional[List[int]] = None) -> Dict[int, dict]:
    """
    Parse scenario_prompts.txt and extract scenarios.

    Args:
        filepath: Path to scenario_prompts.txt
        scenario_ids: Optional list of scenario IDs to extract. If None, extracts all.

    Returns:
        Dict mapping scenario ID to scenario data:
        {
            1: {
                "id": 1,
                "category": "Work Dilemma",
                "title": "Lunchroom Theft Scapegoat",
                "prompts": ["Prompt1 content...", "Prompt2 content...", ...]
            },
            ...
        }
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Pattern to match scenario headers: ######## {ID} | {Category} | {Title}
    header_pattern = re.compile(
        r'^######## (\d+)\s*\|\s*([^|]+?)\s*\|\s*(.+?)\s*$',
        re.MULTILINE
    )

    # Find all headers and their positions
    headers = list(header_pattern.finditer(content))

    scenarios = {}

    for i, match in enumerate(headers):
        scenario_id = int(match.group(1))

        # Skip if not in requested IDs
        if scenario_ids is not None and scenario_id not in scenario_ids:
            continue

        category = match.group(2).strip()
        title = match.group(3).strip()

        # Get content between this header and the next (or end of file)
        start = match.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(content)
        scenario_content = content[start:end]

        # Extract prompts (####### Prompt1, ####### Prompt2, etc.)
        prompt_pattern = re.compile(r'^####### Prompt\d+\s*$', re.MULTILINE)
        prompt_matches = list(prompt_pattern.finditer(scenario_content))

        prompts = []
        for j, pmatch in enumerate(prompt_matches):
            pstart = pmatch.end()
            pend = prompt_matches[j + 1].start() if j + 1 < len(prompt_matches) else len(scenario_content)
            prompt_text = scenario_content[pstart:pend].strip()
            if prompt_text:
                prompts.append(prompt_text)

        scenarios[scenario_id] = {
            "id": scenario_id,
            "category": category,
            "title": title,
            "prompts": prompts,
        }

    return scenarios


def get_initial_scenarios(filepath: str) -> Dict[int, dict]:
    """
    Get the initial 5 scenarios for pipeline testing.

    Returns scenarios:
    - 1: Work Dilemma - Lunchroom Theft Scapegoat
    - 2: Romance Issue - Flirtation Accusation
    - 6: Family Issue - Teen Dishes Showdown
    - 101: Mediation Workplace - Clinical Trial Urgency
    - 131: Mediation Family - Teen Privacy Battle
    """
    initial_ids = [1, 2, 6, 101, 131]
    return parse_scenarios(filepath, scenario_ids=initial_ids)


if __name__ == "__main__":
    # Test the parser
    script_dir = Path(__file__).parent
    scenarios_file = script_dir / "scenario_prompts.txt"

    if scenarios_file.exists():
        # Get initial scenarios
        scenarios = get_initial_scenarios(str(scenarios_file))

        print(f"Parsed {len(scenarios)} initial scenarios:\n")
        for sid, data in sorted(scenarios.items()):
            print(f"  [{sid}] {data['category']} - {data['title']}")
            print(f"       {len(data['prompts'])} prompts")

        # Show first prompt of first scenario as sample
        if scenarios:
            first_id = min(scenarios.keys())
            first = scenarios[first_id]
            print(f"\n--- Sample (Scenario {first_id}, Prompt 1) ---")
            print(first['prompts'][0][:500] + "..." if len(first['prompts'][0]) > 500 else first['prompts'][0])
    else:
        print(f"File not found: {scenarios_file}")
