import os
import re
import sys
import time
import random
import datetime  # For dynamic filename
from typing import Any, Dict, List

import pandas as pd
import concurrent.futures
from tqdm import tqdm
import pingouin as pg

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from core.utils import get_latest_file, load_models, discover_openbench_csv

from core.llm_client import get_llm_response, APIError

# --- CONFIGURATION ---

MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 5
MAX_CONCURRENT_REQUESTS = 10 # Number of parallel API requests to make



# File & Directory Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_CSV_PATH = discover_openbench_csv(SCRIPT_DIR)
PROMPTS_DIR = 'prompts'
STORIES_DIR = 'generated_stories'
JUDGEMENTS_DIR = 'judgements'
OUTPUT_DIR = 'results'
FINAL_SCORES_CSV = os.path.join(OUTPUT_DIR, 'all_scores.csv')
SUMMARY_CSV = os.path.join(OUTPUT_DIR, 'summary_by_writer.csv')

JUDGE_MODELS = ['Llama 4 Maverick 17B 128E Instruct', 'Qwen 3 235B A22B', 'GPT-5 (low)', 'Qwen 3 32B']

def parse_prompt_file(filepath: str) -> Dict[str, Any]:
    """Parses a prompt file to extract the template and its elements."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    template_part, elements_part = content.split('sentence.\n\n', 1)
    template = template_part + 'sentence.' + 'Character: {character}\nObject: {object}\nCore Concept: {core_concept}\nAttribute: {attribute}\nAction: {action}\nMethod: {method}\nSetting: {setting}\nTimeframe: {timeframe}\nMotivation: {motivation}\nTone: {tone}'
    
    elements = {}
    for line in elements_part.strip().split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            elements[key.strip().lower().replace(' ', '_')] = value.strip()
            
    prompt_id = os.path.basename(filepath).split('.')[0].replace('prompt_', '')
    return {'id': prompt_id, 'template': template, 'elements': elements}

def extract_story(response_text: str) -> str:
    """Extracts content from between <story></story> tags."""
    match = re.search(r'<story>(.*?)</story>', response_text, re.DOTALL)
    if match:
        story_content = match.group(1).strip()
        cleaned_story = re.sub(r'<words>.*?</words>', '', story_content).strip()
        return cleaned_story
    return ""

def parse_judgement(judgement_text: str) -> Dict[str, int]:
    """Parses the structured judgement output into a dictionary."""
    scores = {}
    matches = re.findall(r'<question>(.*?)</question><grade>(.*?)</grade>', judgement_text, re.DOTALL)
    for q, g in matches:
        try:
            scores[q.strip()] = int(g.strip())
        except (ValueError, TypeError):
            scores[q.strip()] = None
    return scores

# --- PARALLEL TASK FUNCTIONS ---

def generate_story_task(task_args):
    """A single unit of work for generating one story."""
    prompt_info, writer, story_filepath = task_args
    writer_name = writer['name'].replace('/', '_')

    story_prompt = prompt_info['template'].format(**prompt_info['elements'])
    tries = 0

    completed = False
    while completed == False and tries < MAX_RETRIES:

        try:
            story_prompt = prompt_info['template'].format(**prompt_info['elements'])
            raw_response = get_llm_response(story_prompt, writer['id'],writer['name'], writer['Reasoning'])
            story_content = extract_story(raw_response)

            if story_content:
                with open(story_filepath, 'w', encoding='utf-8') as f:
                    f.write(story_content)
                return f"✓ Saved story for {writer_name} on prompt {prompt_info['id']}"
            else:
                print('story prompt:', story_prompt)
                print('response:')
                print(raw_response)
                return f"✗ Failed to extract story for {writer_name} on prompt {prompt_info['id']}"
        except Exception as e:
            print(f"✗ Error for {writer_name} on prompt {prompt_info['id']}: {e}")
            time.sleep(5)
            tries += 1


    return f"✗ Error for {writer_name} on prompt {prompt_info['id']}"

def judge_story_task(task_args):
    """
    A single unit of work for judging one story, with robust retries for format errors.
    """
    prompt_info, writer, judge, judging_prompt_template, judgement_filepath = task_args

    writer_name = writer['name'].replace('/', '_')
    judge_name = judge['name'].replace('/', '_')
    story_filepath = os.path.join(STORIES_DIR, writer_name, f"prompt_{prompt_info['id']}.txt")

    # Define the exact set of keys we expect in the parsed output.
    # This is the most reliable way to validate the structure.
    EXPECTED_KEYS = {
        '1', '2', '3', '4', '5', '6', '7 A', '7 B', '7 C', '7 D', '7 E',
        '7 F', '7 G', '7 H', '7 I', '7 J'
    }

    # Prepare the prompt once, as it's the same for all retries.
    try:
        with open(story_filepath, 'r', encoding='utf-8') as f:
            story_content = f.read()
    except FileNotFoundError:
        return f"✗ Story file not found, cannot judge: {story_filepath}"
        
    judgement_prompt = judging_prompt_template.format(story=story_content, **prompt_info['elements'])

    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            # 1. Get the raw response from the LLM
            raw_judgement = get_llm_response(judgement_prompt, judge['id'], judge['name'], judge['Reasoning'])

            # 2. Parse the response
            parsed_scores = parse_judgement(raw_judgement)
            
            # --- ROBUST VALIDATION ---
            actual_keys = set(parsed_scores.keys())

            # Check 1: Do the parsed keys exactly match our expected structure?
            # This catches mangled tags like '<question>5</grade><question>6'
            if actual_keys != EXPECTED_KEYS:
                error_msg = (f"Invalid judgement format: Mismatched or malformed question keys. "
                             f"Expected {EXPECTED_KEYS}, but got {actual_keys}. "
                             f"Output: '{raw_judgement[:200].replace(os.linesep, ' ')}...'")
                raise ValueError(error_msg)

            # Check 2: Are all grade values valid integers?
            # This catches empty tags like '<grade></grade>'
            if any(score is None for score in parsed_scores.values()):
                error_msg = (f"Invalid judgement format: Found null or non-integer scores. "
                             f"Parsed scores: {parsed_scores}. "
                             f"Output: '{raw_judgement[:200].replace(os.linesep, ' ')}...'")
                raise ValueError(error_msg)

            # 3. If all validation passes, save the file and return success
            with open(judgement_filepath, 'w', encoding='utf-8') as f:
                f.write(raw_judgement)
            return f"✓ Saved judgement by {judge_name} for {writer_name} on prompt {prompt_info['id']}"

        except (APIError, ValueError) as e:
            last_error = e
            print(f"✗ Attempt {attempt + 1}/{MAX_RETRIES} failed for judge {judge_name} on {writer_name}'s story. Error: {e}")
            
            # Don't sleep on the last attempt
            if attempt < MAX_RETRIES - 1:
                # Use exponential backoff for retries
                delay = INITIAL_RETRY_DELAY * (2 ** attempt) + random.uniform(0, 1)
                time.sleep(delay)

    # If the loop completes without returning, all retries have failed.
    return f"✗ FAILED after {MAX_RETRIES} attempts to judge {writer_name}'s story by {judge_name}. Last error: {last_error}"

# --- MAIN EXECUTION LOGIC ---

def main():
    """Main function to run the benchmark pipeline."""
    print("--- Starting LLM Storytelling Benchmark ---")

    # 1. Setup Directories
    os.makedirs(STORIES_DIR, exist_ok=True)
    os.makedirs(JUDGEMENTS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Load Models and Prompts
    all_models = load_models(MODEL_CSV_PATH)
    if not all_models:
        print("No models loaded. Exiting.")
        return
    
    writer_models = all_models
    judge_models = [m for m in all_models if m['name'] in JUDGE_MODELS]

    # Create a mapping for easy lookup, handling file-safe names
    judge_model_map = {m['name']: m['name'].replace('/', '_') for m in judge_models}
    judge_model_map_rev = {v: k for k, v in judge_model_map.items()}

    prompts = [parse_prompt_file(os.path.join(PROMPTS_DIR, f)) for f in os.listdir(PROMPTS_DIR) if f.startswith('prompt_')]

    
    print(f"Loaded {len(writer_models)} writer models, {len(judge_models)} judge models, and {len(prompts)} prompts.")

    # --- PART 1: STORY GENERATION (PARALLEL) ---
    # (This section is unchanged)
    print("\n--- Preparing Story Generation Tasks ---")
    story_tasks = []
    for prompt_info in prompts:
        for writer in writer_models:
            writer_name = writer['name'].replace('/', '_')
            story_dir = os.path.join(STORIES_DIR, writer_name)
            os.makedirs(story_dir, exist_ok=True)
            story_filepath = os.path.join(story_dir, f"prompt_{prompt_info['id']}.txt")
            if not os.path.exists(story_filepath):
                story_tasks.append((prompt_info, writer, story_filepath))
    if story_tasks:
        print(f"Found {len(story_tasks)} stories to generate. Starting...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
            results = list(tqdm(executor.map(generate_story_task, story_tasks), total=len(story_tasks), desc="Generating Stories"))
        for res in results:
            if "✗" in res:
                print(res)
    else:
        print("All stories already generated. Skipping generation.")

    # --- PART 2: STORY JUDGEMENT (PARALLEL) ---
    # (This section is unchanged)
    print("\n--- Preparing Story Judgement Tasks ---")
    judging_prompt_template = """
Story:
{story}

---

For these categories, rate the story above on a scale 0-10. Assign rating 0 (worst) to 10 (best). Just output this rating. You are a very harsh literary critic because this story should be excellent since it's for publishing. There should be no weaknesses.

1. Character Development & Motivation
When assessing the character's portrayal, look for:
- How well does the assigned attribute shape the character's personality or actions?
- Is the character's motivation clear and consistent with their assigned traits?
- Does the character's reaction to the assigned object or concept reveal something about them?
- Are their actions aligned with their established motivation?
- How does the character's personality influence their use of the assigned method?

2. Plot Structure & Coherence
When analyzing the story's construction, examine:
- How effectively does the assigned action drive the plot forward?
- Does the method of accomplishing the action make sense within the story?
- Is there a clear connection between the beginning and end despite the 400-word constraint?
- How well does the assigned concept shape or influence the plot's direction?
- Are cause and effect relationships clear and logical within the story's framework?

3. World & Atmosphere
When evaluating the story's setting and mood, consider:
- How well do the assigned setting and timeframe work together?
- Does the tone complement or intentionally contrast with the concept?
- How effectively does the author use the setting to enhance the action or method?
- Are the object and setting meaningfully connected?
- Does the atmosphere support the character's motivation and the story's concept?

4. Storytelling Impact & Craft
When evaluating the overall storytelling effectiveness, consider:
- Does the story evoke an emotional response or make the reader think?
- Is there effective use of literary devices (metaphor, imagery, symbolism, etc.)?
- How memorable or unique is the story's perspective or approach?
- Is there subtext or deeper meaning beyond the surface narrative?
- Does the ending feel satisfying or purposeful?
- How well does the writer control pacing and tension?
- Is the dialogue (if any) natural and purposeful?
- Does the writing style enhance the story's impact?

5. Authenticity & Originality
Checking for signs of formulaic writing, examine:
- Are there repetitive or unnecessarily flowery phrases that feel machine-like?
- Does the story rely on common tropes or clichés without subverting them?
- Is there inconsistent or generic sensory detail?
- Does the writing feel overly formal or use unnecessarily complex vocabulary?
- Are there sudden narrative jumps or inconsistencies typical of AI writing?
- Is character reasoning overly logical without human nuance?
- Does the story feel like it's following a predictable template?
- Are metaphors and similes fresh and specific, or generic and overused?
- Is there nuanced handling of emotion, or just stated feelings?
- Does the writing show individual style, or could it be from any source?

6. General score
Consider:
- Does the story feel cohesive despite having to incorporate many assigned elements?
- Are the elements used in service of the story rather than the story being built around them?
- Does the brief length feel like a deliberate choice rather than a limitation?
- Is there evidence of creative thinking in how elements are combined and utilized?

7. The story was supposed to use certain elements. Rate each of them on a 0–10 scale of how tightly it fits into the story's logic, tone, and momentum. Consider how each element shapes the narrative arc, enriches the setting, deepens character motives, and supports the theme. If the story would be less cohesive or compelling without that specific element, give it a higher rating. If using this particular element feels forced and another element in the category would fit better, give it a lower rating. All elements must flow and fit together very well.

Rating scale:
0 if it isn't used at all.
1 if it's jarring or drags the story down.
10 if removing or swapping it for a similar element would ruin the story's tension, originality, or emotional impact.
(use any integer number 0 to 10, not just these, and make a precise evalation).

7 A. Character: {character}
7 B. Object: {object}
7 C. Core Concept: {core_concept}
7 D. Attribute: {attribute}
7 E. Action: {action}
7 F. Method: {method}
7 G. Setting: {setting}
7 H. Timeframe: {timeframe}
7 I. Motivation: {motivation}
7 J. Tone: {tone}

Sample output format, you must follow this format and use tags <question></question><grade></grade>:

<question>1</question><grade>4</grade>
<question>2</question><grade>7</grade>
<question>3</question><grade>0</grade>
<question>4</question><grade>10</grade>
<question>5</question><grade>9</grade>
<question>6</question><grade>4</grade>
<question>7 A</question><grade>2</grade>
<question>7 B</question><grade>8</grade>
<question>7 C</question><grade>1</grade>
<question>7 D</question><grade>4</grade>
<question>7 E</question><grade>10</grade>
<question>7 F</question><grade>0</grade>
<question>7 G</question><grade>0</grade>
<question>7 H</question><grade>5</grade>
<question>7 I</question><grade>7</grade>
<question>7 J</question><grade>2</grade>

You may not output any extra comments besides these scores.
"""
    judgement_tasks = []
    for prompt_info in prompts:
        for writer in writer_models:
            writer_name = writer['name'].replace('/', '_')
            story_filepath = os.path.join(STORIES_DIR, writer_name, f"prompt_{prompt_info['id']}.txt")
            if not os.path.exists(story_filepath):
                continue
            for judge in judge_models:
                judge_name = judge['name'].replace('/', '_')
                judgement_dir = os.path.join(JUDGEMENTS_DIR, writer_name)
                os.makedirs(judgement_dir, exist_ok=True)
                judgement_filepath = os.path.join(judgement_dir, f"prompt_{prompt_info['id']}_judged_by_{judge_name}.txt")
                if not os.path.exists(judgement_filepath):
                    judgement_tasks.append((prompt_info, writer, judge, judging_prompt_template, judgement_filepath))
    if judgement_tasks:
        print(f"Found {len(judgement_tasks)} stories to judge. Starting...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
            results = list(tqdm(executor.map(judge_story_task, judgement_tasks), total=len(judgement_tasks), desc="Judging Stories"))
        for res in results:
            if "✗" in res:
                print(res)
    else:
        print("All stories already judged. Skipping judgement.")


    # --- PART 3: AGGREGATION & ANALYSIS (SEQUENTIAL) ---
    print("\n--- Aggregating Scores ---")
    all_scores_data = []
    
    for writer_dir in os.listdir(JUDGEMENTS_DIR):
        writer_name = writer_dir
        judgement_path = os.path.join(JUDGEMENTS_DIR, writer_dir)
        if not os.path.isdir(judgement_path): continue

        for judgement_file in os.listdir(judgement_path):
            match = re.match(r"prompt_(\w+)_judged_by_(.*?)\.txt", judgement_file)
            if not match: continue

            prompt_id, judge_name = match.groups()
            
            filepath = os.path.join(judgement_path, judgement_file)
            with open(filepath, 'r', encoding='utf-8') as f:
                judgement_text = f.read()
            
            parsed_scores = parse_judgement(judgement_text)

            if not parsed_scores:
                print(f"  ! Warning: Could not parse scores from {judgement_file}")
                continue

            for question, score in parsed_scores.items():
                all_scores_data.append({
                    'writer_model': writer_name,
                    'judge_model': judge_name,
                    'prompt_id': prompt_id,
                    'question': question,
                    'score': score
                })

    if not all_scores_data:
        print("No scores were collected. Cannot generate output files. Exiting.")
        return

    df = pd.DataFrame(all_scores_data)
    df.to_csv(FINAL_SCORES_CSV, index=False)
    print(f"\n✓ All collected scores saved to '{FINAL_SCORES_CSV}'")

    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    df.dropna(subset=['score'], inplace=True)




    print("\n--- Applying Z-Score Normalization to Correct for Judge Bias ---")
    df['score_normalized'] = df.groupby('judge_model')['score'].transform(
        lambda x: 100*(x) if (x.max() - x.min()) > 0 else 0
    )

    # =========================================================================
    # --- NEW (REVISED): EXPORT PIVOTED MEAN SCORES BY JUDGE ---
    # =========================================================================
    print("\n--- Generating Pivoted Mean Score Report ---")
    
    # Define the dynamic filename using the current date
    current_date_str = datetime.datetime.now().strftime("%Y%m%d")
    judge_mean_score_filename = f'writing_direct_{current_date_str}.csv'
    judge_mean_score_filepath = os.path.join(OUTPUT_DIR, judge_mean_score_filename)

    # Use pivot_table to create the "wide" format
    # This calculates the mean score for each writer/judge pair and puts it in the desired structure
    pivoted_scores_df = df.pivot_table(
        index='writer_model',   # Each row is a writer model
        columns='judge_model',  # Each column is a judge model
        values='score_normalized',         # The values in the table are the scores
        aggfunc='mean'          # The aggregation function is the average
    )

    # Clean up the column names for clarity (e.g., 'JudgeA' -> 'JudgeA_score')
    pivoted_scores_df.columns = [f'{col}_score' for col in pivoted_scores_df.columns]

    # Move 'writer_model' from the index to a regular column
    pivoted_scores_df.reset_index(inplace=True)
    
    # Remove the name of the column index, which is 'judge_model' after pivoting
    pivoted_scores_df.columns.name = None

    # Save the resulting DataFrame to the new CSV file
    pivoted_scores_df.to_csv(judge_mean_score_filepath, index=False, float_format="%.4f")
    
    print(f"✓ Pivoted mean scores by judge saved to '{judge_mean_score_filepath}'")
    # --- END OF REVISED CODE ---
    


    summary_pivot = df.groupby(['writer_model', 'question'])['score_normalized'].mean().unstack()
    summary_pivot['OVERALL_AVG_NORMALIZED_SCORE'] = df.groupby('writer_model')['score_normalized'].mean()

    summary_pivot.reset_index(inplace=True)
    summary_pivot.sort_values(by='OVERALL_AVG_NORMALIZED_SCORE', ascending=False, inplace=True)

    NORMALIZED_SUMMARY_CSV = os.path.join(OUTPUT_DIR, 'summary_by_writer_normalized.csv')
    summary_pivot.to_csv(NORMALIZED_SUMMARY_CSV, index=False)
    print(f"✓ Normalized summary report saved to '{NORMALIZED_SUMMARY_CSV}'")


    # =========================================================================
    # --- PART 4: MULTI-JUDGE RELIABILITY ANALYSIS ---
    # =========================================================================
    print("\n--- Analyzing Judge Reliability and Agreement ---")

    if len(JUDGE_MODELS) < 2:
        print(f"  ! Warning: Judge analysis requires at least 2 judges. Found {len(JUDGE_MODELS)}. Skipping.")
    else:
        judge_pivot_df = df.pivot_table(
            index=['writer_model', 'prompt_id', 'question'],
            columns='judge_model',
            values='score_normalized'
        )

        judge_score_cols = judge_pivot_df[list(judge_model_map_rev.keys())]
        
        print("\n--- 1. Pairwise Correlation Matrix ---")
        print("Shows the Pearson correlation between each pair of judges. (1.0 is perfect correlation).")
        correlation_matrix = judge_score_cols.corr(numeric_only=True)
        print(correlation_matrix.to_string(float_format="%.3f"))

        print("\n--- 2. Overall Group Reliability (Cronbach's Alpha) ---")
        print("Measures the internal consistency of all judges as a group.")
        print("(> 0.7 is acceptable, > 0.8 is good, > 0.9 is excellent)")
        
        alpha_data = judge_pivot_df.dropna()
        if len(alpha_data) > 1:
            cronbach_alpha_results = pg.cronbach_alpha(data=alpha_data[list(judge_model_map_rev.keys())])
            overall_alpha = cronbach_alpha_results[0]
            print(f"Overall Cronbach's Alpha: {overall_alpha:.4f}")
        else:
            print("Not enough complete data points to calculate Cronbach's Alpha.")


        print("\n--- 3. Individual Judge Statistics ---")
        print("Shows the average score (mean) and variance (std) for each judge.")
        judge_stats = judge_score_cols.describe().T[['mean', 'std', 'min', 'max']]
        print(judge_stats.to_string(float_format="%.2f"))

        print("\n--- 4. Per-Question Reliability and Scores ---")
        per_question_analysis = []
        sorted_questions = sorted(df['question'].unique()) 

        for question in sorted_questions:
            question_df = judge_pivot_df[judge_pivot_df.index.get_level_values('question') == question]
            question_df = question_df.dropna()
            
            if len(question_df) < 2:
                continue

            alpha_val = pg.cronbach_alpha(data=question_df[list(judge_model_map_rev.keys())])[0]
            
            result_row = {'question': question, 'cronbachs_alpha': alpha_val}
            
            mean_scores = question_df[list(judge_model_map_rev.keys())].mean().to_dict()
            for judge, mean_score in mean_scores.items():
                result_row[f"mean_{judge}"] = mean_score
            
            per_question_analysis.append(result_row)

        if per_question_analysis:
            analysis_df = pd.DataFrame(per_question_analysis)
            analysis_df.sort_values(by='cronbachs_alpha', ascending=False, inplace=True)

            print("Analysis of which questions had the most/least judge agreement:")
            print(analysis_df.to_string(index=False, float_format="%.2f"))

            JUDGE_ANALYSIS_CSV = os.path.join(OUTPUT_DIR, 'judge_analysis.csv')
            analysis_df.to_csv(JUDGE_ANALYSIS_CSV, index=False)
            print(f"\n✓ Detailed judge reliability analysis saved to '{JUDGE_ANALYSIS_CSV}'")
        else:
            print("Could not generate per-question analysis. Not enough complete data.")

    
    print("\n--- Benchmark Complete ---")


if __name__ == "__main__":
    main()
