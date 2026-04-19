"""
Benchmark Data Combiner with LLM-Assisted Model Name Mapping.

This module aggregates benchmark data from 13+ sources into a unified dataset,
using OpenBench as the canonical namespace for model names. The key challenge
is that different benchmark sources use different naming conventions for the
same models (e.g., "gpt-4-turbo-2024-04-09" vs "GPT-4 Turbo (April 2024)").

Pipeline Overview:
    1. Load existing model name mappings from JSON files
    2. Load benchmark CSVs from multiple sources
    3. Identify new models that need mapping to OpenBench namespace
    4. Use Gemini LLM to suggest mappings for unmapped models
    5. Apply mappings to create unified model names
    6. Merge all benchmarks via outer join on Unified_Name
    7. Detect and report mapping issues (duplicates, conflicts)

Key Data Structures:
    - Mapping dictionaries: {source_model_name: openbench_name}
    - bench_configs: Configuration for each benchmark source
    - df_merged: Final combined DataFrame with all benchmarks

Dependencies:
    - google.genai: Gemini API for LLM-assisted mapping suggestions
    - pandas: Data manipulation
    - glob: File pattern matching for finding latest benchmark files

Example Usage:
    >>> load_existing_mappings()
    >>> df_combined, configs = combine_benchmarks_with_auto_mapping()
    >>> issues = find_mapping_issues(df_combined)
    >>> df_combined.to_csv('combined_all_benches.csv', index=False)

"""

import pandas as pd
import re
import json
from google import genai
import glob
import os
import math
from io import StringIO

# ==============================================================================
# GLOBAL MAPPING DICTIONARIES
# ==============================================================================
# Each dictionary maps source-specific model names to OpenBench canonical names.
# These are populated from JSON files at startup and updated when new models are found.
simplebench_to_openbench = {}
livebench_to_openbench = {} # New: LiveBench will also be mapped to OpenBench
lmsys_to_openbench = {} # SHARED dictionary for LMSYS and LMArena
lechmazurconfabulations_to_openbench = {}
aiderbench_to_openbench = {}
contextarena_to_openbench = {}
eqbench_to_openbench = {}
arc_to_openbench = {}
aa_to_openbench = {}
aa_evals_to_openbench = {}  # Shared by AAGDPval, AAOmniscience, AACritPt (same AA website model names)
weirdml_to_openbench = {}
yupp_to_openbench = {}
ugi_to_openbench = {}
bench_configs = {}

# ==============================================================================
# MAPPING LOADING
# ==============================================================================

def load_existing_mappings():
    """Load existing model name mappings from JSON files in the mappings/ directory.

    This function populates the global mapping dictionaries with previously
    established model name mappings. Each mapping file corresponds to a specific
    benchmark source and maps its model names to OpenBench canonical names.

    The function is idempotent - calling it multiple times will update the
    dictionaries with the latest file contents.

    Global Variables Modified:
        simplebench_to_openbench: SimpleBench → OpenBench mappings
        livebench_to_openbench: LiveBench → OpenBench mappings
        lmsys_to_openbench: LMSYS/LMArena → OpenBench mappings (shared)
        lechmazurconfabulations_to_openbench: Lechmazur → OpenBench mappings
        aiderbench_to_openbench: AiderBench → OpenBench mappings
        contextarena_to_openbench: ContextArena → OpenBench mappings
        eqbench_to_openbench: EQ-Bench → OpenBench mappings
        arc_to_openbench: ARC → OpenBench mappings
        aa_to_openbench: Artificial Analysis → OpenBench mappings
        weirdml_to_openbench: WeirdML → OpenBench mappings
        yupp_to_openbench: Yupp → OpenBench mappings

    Side Effects:
        - Creates 'mappings/' directory if it doesn't exist
        - Prints warnings for missing or malformed JSON files

    Example:
        >>> load_existing_mappings()
        >>> print(lmsys_to_openbench.get('gpt-4-turbo'))
        'GPT-4 Turbo (2024-04-09)'
    """
    global simplebench_to_openbench, livebench_to_openbench, \
           lmsys_to_openbench, lechmazurconfabulations_to_openbench, aiderbench_to_openbench, \
        contextarena_to_openbench, eqbench_to_openbench, arc_to_openbench, aa_to_openbench, \
           aa_evals_to_openbench, \
           weirdml_to_openbench, yupp_to_openbench, ugi_to_openbench

    # Step 1: Define mapping between filenames and their corresponding dictionaries
    # Note: lmsys_to_openbench is shared by both LMSYS and LMArena benchmarks
    # Note: aa_evals_to_openbench is shared by AAGDPval, AAOmniscience, and AACritPt
    mapping_files = {
        'simplebench_to_openbench': simplebench_to_openbench,
        'livebench_to_openbench': livebench_to_openbench,
        'lmsys_to_openbench': lmsys_to_openbench,
        'lechmazurconfabulations_to_openbench': lechmazurconfabulations_to_openbench,
        'aiderbench_to_openbench': aiderbench_to_openbench,
        'contextarena_to_openbench': contextarena_to_openbench,
        'eqbench_to_openbench': eqbench_to_openbench,
        'arc_to_openbench': arc_to_openbench,
        'aa_to_openbench': aa_to_openbench,
        'aa_evals_to_openbench': aa_evals_to_openbench,
        'weirdml_to_openbench': weirdml_to_openbench,
        'yupp_to_openbench': yupp_to_openbench,
        'ugi_to_openbench': ugi_to_openbench,
    }

    # Step 2: Ensure the mappings directory exists
    os.makedirs('mappings', exist_ok=True)

    # Step 3: Load each mapping file, updating the corresponding dictionary
    # Uses .update() to merge with any existing entries rather than replacing
    for name, dictionary in mapping_files.items():
        try:
            with open(f'mappings/{name}.json', 'r') as f:
                dictionary.update(json.load(f))
        except FileNotFoundError:
            # Not an error - mapping file might not exist yet for new benchmark sources
            print(f"Mapping file mappings/{name}.json not found. Starting with an empty mapping for it.")
        except json.JSONDecodeError:
            # Malformed JSON - warn but continue with empty mapping
            print(f"Error decoding JSON from mappings/{name}.json. Starting with an empty mapping for it.")


# ==============================================================================
# CSV READING UTILITIES
# ==============================================================================

def _read_csv_resilient(path: str, **kwargs) -> pd.DataFrame:
    """Read a CSV file with automatic encoding detection and fallback.

    Benchmark CSV files come from various sources with inconsistent encodings.
    This function tries multiple common encodings in order of likelihood,
    and as a last resort, replaces invalid bytes rather than failing entirely.

    Args:
        path: Path to the CSV file to read.
        **kwargs: Additional arguments passed to pd.read_csv().

    Returns:
        DataFrame containing the CSV data.

    Raises:
        UnicodeDecodeError: If all encoding attempts fail (rare due to fallback).

    Encoding Priority:
        1. UTF-8 (most common)
        2. UTF-8 with BOM (Excel exports)
        3. Latin-1 (Western European)
        4. CP1252 (Windows Western European)
        5. UTF-8 with replacement (last resort - replaces invalid chars with �)

    Example:
        >>> df = _read_csv_resilient('benchmarks/lmsys_20240101.csv')
        >>> df.head()
    """
    # Step 1: Define encoding priority - most common first
    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
    last_err = None

    # Step 2: Try each encoding in order
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError as e:
            last_err = e
            print(f"  ! Decode failed for {path} with encoding {enc}: {e}")

    # Step 3: Last resort - read with replacement characters (replaces bad bytes with �)
    # This ensures we never lose an entire benchmark due to encoding issues
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        data = fh.read()
    try:
        print(f"  ! Using utf-8 with replacement for {path} after decode failures.")
        return pd.read_csv(StringIO(data), **kwargs)
    except Exception as e:  # pragma: no cover
        raise last_err or e


# ==============================================================================
# LLM-ASSISTED MAPPING
# ==============================================================================

def gemini(user_prompt, system_prompt='You are an intelligent assistant.'):
    """Call the Gemini API to generate a response.

    This is a low-level wrapper around the Gemini API used for LLM-assisted
    model name mapping. Uses a moderate temperature (0.5) for balanced
    creativity and consistency.

    Args:
        user_prompt: The user's query or instruction.
        system_prompt: System context/instruction for the model.
            Defaults to a generic assistant prompt.

    Returns:
        The model's text response.

    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is required for LLM-assisted mapping")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=user_prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.5,  # Moderate temperature for balanced responses
        ),
    )
    return response.text


def get_llm_mapping_suggestions(source_model_name, target_models_list, mapping_type, target_benchmark_name="OpenBench"):
    """Use LLM to find the best match for a source model name in the target list.

    This is the core mapping function that leverages Gemini to match model names
    across different benchmark naming conventions. The LLM looks for patterns in:
    - Model family names (GPT, Claude, Llama, etc.)
    - Version numbers and dates
    - Suffixes (instruct, chat, turbo, etc.)

    Args:
        source_model_name: The model name from the source benchmark.
        target_models_list: List of valid model names from the target benchmark.
        mapping_type: Name of the source benchmark (for context in the prompt).
        target_benchmark_name: Name of the target benchmark. Defaults to "OpenBench".

    Returns:
        The matched model name from target_models_list, or empty string if:
        - No good match found
        - LLM confidence < 75%
        - Suggested match not in target list
        - JSON parsing fails

    Algorithm:
        1. Construct a prompt with source name and target list
        2. Ask LLM to return JSON with 'mapping' and 'confidence' keys
        3. Parse response, extracting JSON from markdown code blocks if needed
        4. Validate confidence threshold (>= 75%) and membership in target list
        5. Return mapping or empty string

    Example:
        >>> targets = ['GPT-4 Turbo (2024-04-09)', 'Claude 3 Opus']
        >>> get_llm_mapping_suggestions('gpt-4-turbo', targets, 'LMSYS')
        'GPT-4 Turbo (2024-04-09)'
    """
    system_prompt = f"""
    You are a specialized AI assistant for mapping AI model names between different benchmarking systems.
    Your task is to find the most likely match for a source model name in a target list of {target_benchmark_name} models.
    Look for similar patterns in naming conventions, version numbers, dates, and model families.
    For example, Gemma-3-27B-it will not map to gemma-2-27b-it.

    Return ONLY a JSON object with two keys:
    1. 'mapping' - your best match as a string (or an empty string if no good match from the target list)
    2. 'confidence' - a value from 0-100 indicating your confidence

    If confidence is below 75, return an empty mapping.
    The 'mapping' must be one of the provided target model names or an empty string.
    """
    user_prompt = f"""
    I need to map model names between benchmarking platforms.

    Source model name from {mapping_type}: "{source_model_name}"

    Potential target {target_benchmark_name} models:
    {json.dumps(target_models_list, indent=2)}

    Return a JSON object with keys:
    - 'mapping': your best match from the target list, or an empty string if no good match or confidence < 75.
    - 'confidence': a score (0-100) of your confidence.
    """
    try:
        # For some genai versions, system_prompt needs to be part of the model or contents.
        # Simplified: pass the full context in user_prompt.
        response = gemini(user_prompt, system_prompt) # Old call
        # New approach: combine prompts if system_instruction is not directly supported by generate_content's config
        # full_prompt_for_gemini = f"{system_prompt}\n\n{user_prompt}"
        # response = gemini(full_prompt_for_gemini) # Pass combined prompt

        json_str = None
        # Try to extract JSON from markdown code blocks
        json_match_markdown = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', response, re.DOTALL)
        if json_match_markdown:
            json_str = json_match_markdown.group(1).strip()
        else:
            # Try to extract the first valid-looking JSON object
            json_match_curly = re.search(r'(\{[\s\S]*?\})(?=\s|\Z)', response, re.DOTALL)
            if json_match_curly:
                json_str = json_match_curly.group(1).strip()
            else:
                # If no clear JSON structure, assume the whole response might be it (less reliable)
                json_str = response.strip()

        if not json_str:
            print(f"No JSON found in response for {source_model_name}.")
            return ''

        result = json.loads(json_str)
        mapping = result.get('mapping', '')
        confidence = result.get('confidence', 0)

        if not isinstance(mapping, str): # Ensure mapping is a string
            print(f"LLM returned non-string mapping: {mapping}. Discarding.")
            mapping = ""
        if not isinstance(confidence, (int, float)): # Ensure confidence is a number
            print(f"LLM returned non-numeric confidence: {confidence}. Setting to 0.")
            confidence = 0


        if confidence < 75:
            print(f"Confidence ({confidence}%) for '{mapping}' too low for {source_model_name}. Discarding mapping.")
            return ''
        if mapping and mapping not in target_models_list:
            print(f"Warning: LLM suggested mapping '{mapping}' for {source_model_name}, which is not in the target list. Discarding.")
            return ''
        return mapping
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response for {source_model_name}: {e}\nResponse was: {response}")
        return ''
    except Exception as e:
        print(f"Error getting LLM mapping for {source_model_name}: {e}")
        return ''

def update_mapping_dictionaries(df_simple, df_live, df_open, df_lmsys, df_lech, df_aider, df_context, df_lmarena, df_eqbench, df_arc, df_aa, df_weirdml, df_yupp, df_ugi, df_aa_gdpval, df_aa_omniscience, df_aa_critpt):
    # NOTE: df_aa_gdpval, df_aa_omniscience, df_aa_critpt are passed individually but all share aa_evals_to_openbench
    """Update mapping dictionaries with LLM-suggested mappings to OpenBench.

    For each benchmark source, this function identifies models that don't yet
    have a mapping to OpenBench and uses the LLM to suggest the best match.

    The workflow for each unmapped model:
    1. Query LLM with source model name and list of all OpenBench models
    2. If LLM returns a high-confidence match, add it to the mapping dict
    3. If no good match found, explicitly mark as empty string
    4. Save updated mappings to JSON files

    Args:
        df_simple: SimpleBench DataFrame with 'Model' column.
        df_live: LiveBench DataFrame with 'Model' column.
        df_open: OpenBench DataFrame (target namespace).
        df_lmsys: LMSYS DataFrame with 'Model' column.
        df_lech: Lechmazur Confabulations DataFrame with 'Model' column.
        df_aider: AiderBench DataFrame with 'Model' column.
        df_context: ContextArena DataFrame with 'Model' column.
        df_lmarena: LMArena DataFrame with 'Model' column.
        df_eqbench: EQ-Bench DataFrame with 'model' column.
        df_arc: ARC DataFrame with 'Model' column.
        df_aa: Artificial Analysis DataFrame with 'Model' column.
        df_weirdml: WeirdML DataFrame with 'Model' column.
        df_yupp: Yupp DataFrame with 'Model' column.
        df_ugi: UGI leaderboard DataFrame with 'Model' column.
        df_aa_gdpval: AA GDPval-AA DataFrame with 'Model' column.
        df_aa_omniscience: AA Omniscience DataFrame with 'Model' column.
        df_aa_critpt: AA CritPt DataFrame with 'Model' column.

    Returns:
        Tuple of updated mapping dictionaries (one per source benchmark).

    Side Effects:
        - Updates global mapping dictionaries
        - Writes updated mappings to mappings/*.json files
        - Writes unmapped models to mappings/unmapped_to_openbench_models.txt
        - Writes new mappings to mappings/new_openbench_mappings_for_review.txt

    Note:
        LMSYS and LMArena share the same mapping dictionary since they use
        similar model naming conventions.
    """
    source_bench_data = {
        "SimpleBench": (df_simple, 'Model', simplebench_to_openbench),
        "LiveBench": (df_live, 'Model', livebench_to_openbench),
        "LMSYS": (df_lmsys, 'Model', lmsys_to_openbench),
        "LMArena": (df_lmarena, 'Model', lmsys_to_openbench), # UPDATED: LMArena uses the same dictionary as LMSYS
        "LechmazurConfabulations": (df_lech, 'Model', lechmazurconfabulations_to_openbench),
        "AiderBench": (df_aider, 'Model', aiderbench_to_openbench),
        "ContextArena": (df_context, 'Model', contextarena_to_openbench),
        "EQBench": (df_eqbench, 'model', eqbench_to_openbench),
        "ARC": (df_arc, 'Model', arc_to_openbench),
        "AA": (df_aa, 'Model', aa_to_openbench),
        "AAGDPval": (df_aa_gdpval, 'Model', aa_evals_to_openbench),
        "AAOmniscience": (df_aa_omniscience, 'Model', aa_evals_to_openbench),
        "AACritPt": (df_aa_critpt, 'Model', aa_evals_to_openbench),
        "WeirdML": (df_weirdml, 'Model', weirdml_to_openbench),
        "Yupp": (df_yupp, 'Model', yupp_to_openbench),
        "UGILeaderboard": (df_ugi, 'Model', ugi_to_openbench),
    }

    openbench_target_models = set()
    for model in df_open['Model'].unique():
        if isinstance(model, str):
            openbench_target_models.add(model)

    openbench_target_models_list = sorted(list(openbench_target_models)) # Sorted for consistent LLM prompt

    new_mappings_generated = {}
    models_without_good_mapping = []

    for bench_name, (source_df, model_col, mapping_dict) in source_bench_data.items():
        print(f"\nProcessing mappings for {bench_name} to OpenBench...")
        if model_col not in source_df.columns:
            print(f"  Skipping {bench_name}: missing '{model_col}' column.")
            continue

        source_models = set(source_df[model_col].dropna().unique())
        if not source_models:
            print(f"  Skipping {bench_name}: no source models found.")
            continue
        for model in source_models:
            if model not in mapping_dict: # Check if not mapped or mapped to empty
                suggested_mapping = get_llm_mapping_suggestions(model, openbench_target_models_list, bench_name, target_benchmark_name="OpenBench")
                if suggested_mapping:
                    print(f"New OpenBench mapping: {bench_name}: {model} → {suggested_mapping}")
                    mapping_dict[model] = suggested_mapping
                    new_mappings_generated[f"{bench_name}: {model}"] = suggested_mapping
                else:
                    mapping_dict[model] = '' # Explicitly mark as no good mapping found
                    models_without_good_mapping.append(f"{bench_name}: {model}")
                    print(f"No good OpenBench mapping found for {bench_name} model: {model}")
            elif mapping_dict[model] == '':
                print(f"model not mapped to any OpenBench model: {bench_name}, {model}")


    # Save updated mappings to files
    mappings_to_save = {
        'simplebench_to_openbench': simplebench_to_openbench,
        'livebench_to_openbench': livebench_to_openbench,
        'lmsys_to_openbench': lmsys_to_openbench, # This saves the shared dictionary
        'lechmazurconfabulations_to_openbench': lechmazurconfabulations_to_openbench,
        'aiderbench_to_openbench': aiderbench_to_openbench,
        'contextarena_to_openbench': contextarena_to_openbench,
        'eqbench_to_openbench' : eqbench_to_openbench,
        'arc_to_openbench': arc_to_openbench,
        'aa_to_openbench': aa_to_openbench,
        'aa_evals_to_openbench': aa_evals_to_openbench,  # Shared by AAGDPval, AAOmniscience, AACritPt
        'weirdml_to_openbench': weirdml_to_openbench,
        'yupp_to_openbench': yupp_to_openbench,
        'ugi_to_openbench': ugi_to_openbench,
    }
    for name, dictionary in mappings_to_save.items():
        with open(f'mappings/{name}.json', 'w') as f:
            json.dump(dictionary, f, indent=2)

    if models_without_good_mapping:
        with open('mappings/unmapped_to_openbench_models.txt', 'w') as f:
            f.write("Models with no good mapping to OpenBench:\n")
            f.write("\n".join(sorted(list(set(models_without_good_mapping))))) # Unique and sorted
    else:
        if os.path.exists('mappings/unmapped_to_openbench_models.txt'):
            os.remove('mappings/unmapped_to_openbench_models.txt')


    if new_mappings_generated:
        with open('mappings/new_openbench_mappings_for_review.txt', 'w') as f:
            f.write("New OpenBench mappings generated (please review for accuracy):\n")
            for source, target in sorted(new_mappings_generated.items()):
                f.write(f"{source} → {target}\n")
    else:
        if os.path.exists('mappings/new_openbench_mappings_for_review.txt'):
            os.remove('mappings/new_openbench_mappings_for_review.txt')

    # Return all updated dictionaries
    return (simplebench_to_openbench, livebench_to_openbench,
            lmsys_to_openbench, lechmazurconfabulations_to_openbench,
            aiderbench_to_openbench, contextarena_to_openbench, eqbench_to_openbench, arc_to_openbench, aa_to_openbench,
            aa_evals_to_openbench,
            weirdml_to_openbench, yupp_to_openbench)


def combine_benchmarks_with_auto_mapping():
    """Main orchestrator: load data, update mappings, and combine all benchmarks.

    This is the primary entry point for the benchmark combination pipeline.
    It performs the following steps:

    1. **Load Data**: Find and load the latest CSV file for each benchmark source
       using glob patterns with date extraction.

    2. **Check for New Models**: Identify models in source benchmarks that don't
       yet have mappings to OpenBench.

    3. **Update Mappings**: If new models found, call update_mapping_dictionaries()
       to use LLM for suggesting mappings.

    4. **Apply Mappings**: Create 'Unified_Name' column in each DataFrame by
       applying the appropriate mapping dictionary.

    5. **Merge Datasets**: Perform outer join on 'Unified_Name' to create a
       single combined DataFrame with all benchmark scores.

    6. **Report Results**: Print statistics and write diagnostic files.

    Returns:
        Tuple of (df_merged, bench_configs) where:
        - df_merged: Combined DataFrame with all benchmarks, indexed by Unified_Name
        - bench_configs: Dict of benchmark configurations used

    Side Effects:
        - Writes mappings/final_openbench_mappings_used.txt
        - Prints detailed progress and statistics

    Benchmark Sources:
        - OpenBench (canonical namespace - no mapping needed)
        - SimpleBench, LiveBench, LMSYS, LMArena
        - AiderBench, ContextArena, EQ-Bench, ARC
        - Artificial Analysis, WeirdML, Yupp, UGI
        - StyleBench, SimpleBenchFree, ToneBench, WriterBench (direct names)

    Note:
        Returns empty DataFrame if required source files are missing.
    """
    def get_latest_file(pattern: str) -> str:
        files = glob.glob(pattern)
        if not files: # Handle case where no files are found
            raise ValueError(f"No files found matching pattern: {pattern}")
        
        latest_file = None
        latest_date_str = ""
        
        for f_path in files:
            # Try to extract YYYYMMDD or YYYY-MM-DD
            match = re.search(r'(\d{4}[-/]?\d{2}[-/]?\d{2})', os.path.basename(f_path))
            if match:
                date_str = re.sub(r'[-/]', '', match.group(1)) # Normalize to YYYYMMDD
                if date_str > latest_date_str:
                    latest_date_str = date_str
                    latest_file = f_path
        
        if latest_file is None: # If no date found, pick the first one or handle error
            print(f"Warning: No date pattern found in filenames for {pattern}. Using the first file found: {files[0]}")
            return files[0] # Or raise error: raise ValueError(f"No date pattern in files for {pattern}")
        return latest_file

    # Load data from CSVs
    dfs = {}
    bench_configs = {
        "df1": {"name": "SimpleBench", "pattern": "benchmarks/simplebench_*.csv", "map_dict": simplebench_to_openbench},
        "df2": {"name": "LiveBench", "pattern": "benchmarks/livebench_*.csv", "map_dict": livebench_to_openbench},
        "df3": {"name": "OpenBench", "pattern": "benchmarks/openbench_*.csv", "is_base": True}, # OpenBench is base
        "df5": {"name": "LMSYS", "pattern": "benchmarks/lmsys_*.csv", "map_dict": lmsys_to_openbench,
                "columns": ["Model", "Rank", "Rank Spread", "Score", "95% CI (±)", "Votes", "Organization", "License"]},
        # NOTE: partial correlation analysis (2026-03-16) showed lechmazur hurts
        # per-model (partial_r=+0.212, p=0.019) but removing it worsens pipeline
        # RMSE 19.04→20.53 because the imputer uses it as predictor for other columns.
        "df6": {"name": "LechmazurConfabulations", "pattern": "benchmarks/lechmazur_combined_*.csv", "map_dict": lechmazurconfabulations_to_openbench},
        "df7": {"name": "AiderBench", "pattern": "benchmarks/aiderbench_*.csv", "map_dict": aiderbench_to_openbench},
        "df11": {"name": "ContextArena", "pattern": "benchmarks/contextarena_*.csv", "map_dict": contextarena_to_openbench},
        "df14": {"name": "EQBench", "pattern": "benchmarks/EQ-Bench_combined_*.csv", "map_dict": eqbench_to_openbench},
        "df15": {"name": "ARC", "pattern": "benchmarks/arc_*.csv", "map_dict": arc_to_openbench},
        "df16": {"name": "AA", "pattern": "benchmarks/artificialanalysis_*.csv", "map_dict": aa_to_openbench},
        "df17": {"name": "WeirdML", "pattern": "benchmarks/weirdml_*.csv", "map_dict": weirdml_to_openbench},
        "df18": {"name": "Yupp", "pattern": "benchmarks/yupp_text_coding_scores_*.csv", "map_dict": yupp_to_openbench},
        "df19": {"name": "UGILeaderboard", "pattern": "benchmarks/UGI_Leaderboard_*.csv", "map_dict": ugi_to_openbench},
        "df21": {"name": "AAGDPval", "pattern": "benchmarks/aa_gdpval_*.csv", "map_dict": aa_evals_to_openbench},
        "df22": {"name": "AAOmniscience", "pattern": "benchmarks/aa_omniscience_*.csv", "map_dict": aa_evals_to_openbench},
        "df23": {"name": "AACritPt", "pattern": "benchmarks/aa_critpt_*.csv", "map_dict": aa_evals_to_openbench},
        # UPDATED: LMArena now uses a map_dict to share with LMSYS
        "df13": {"name": "LMArena", "pattern": "benchmarks/lmarena_*.csv", "map_dict": lmsys_to_openbench,
                 "columns": ["Model", "Rank", "Rank Spread", "Score", "95% CI (±)", "Votes", "Organization", "License"]},
        # Benchmarks with direct Unified_Name (no LLM mapping)
        "df8": {"name": "Style", "pattern": "benchmarks/style_[0-9]*.csv", "model_col": "model",
                "columns": ["model", "normalized_length", "log_normalized_length", "normalized_header_count",
                             "normalized_bold_count", "normalized_list_count", "predicted_delta",
                             "cv_length", "cv_header_count", "cv_bold_count", "cv_list_count",
                             "min_length", "min_header_count", "min_bold_count", "min_list_count",
                             "frac_used_length", "frac_used_header_count", "frac_used_bold_count", "frac_used_list_count",
                             "q7_length", "q7_header_count", "q7_bold_count", "q7_list_count",
                             "combined_length", "combined_header_count", "combined_bold_count", "combined_list_count"]},
        "df9": {"name": "Logic", "pattern": "benchmarks/logic_*.csv", "model_col": "model_name",
                "columns": ["model_name", "accuracy", "weighted_accuracy", "physics_acc", "trick_acc",
                             "avg_answer_tokens", "avg_reasoning_tokens", "PC1", "PC2", "PC3", "PC4"]},
        "df10": {"name": "Tone", "pattern": "benchmarks/tone_*.csv", "model_col": "judged_model"},
        "df12": {"name": "Writing", "pattern": "benchmarks/writing_[0-9]*.csv", "model_col": "writer_model"},
        # EQ uses OpenBench canonical names directly
        "df20": {"name": "EQ", "pattern": "benchmarks/eq_[0-9]*.csv", "model_col": "model"},
        # EQ multi-turn behavioral features (draft continuity, back-references, length
        # dynamics, per-criterion winrates) extracted from raw per-turn responses and
        # battle_history.csv — see soothsayer_eq/extract_multiturn_features.py
        "df24": {"name": "EQMT", "pattern": "benchmarks/eq_multiturn_*.csv", "model_col": "model_name"},
    }


    for df_key, config in bench_configs.items():
        try:
            print(f"Loading latest file for {config['name']} from pattern: {config['pattern']}")
            latest_file = get_latest_file(config["pattern"])
            print(f"  > Found: {latest_file}")
            dfs[df_key] = _read_csv_resilient(latest_file)
            # Filter to specified columns if configured
            if "columns" in config:
                available = [c for c in config["columns"] if c in dfs[df_key].columns]
                missing = [c for c in config["columns"] if c not in dfs[df_key].columns]
                if missing:
                    print(f"  Warning: {config['name']} missing expected columns: {missing}")
                if not available:
                    print(f"  ERROR: {config['name']} has NONE of the expected columns. Wrong file selected?")
                    dfs[df_key] = pd.DataFrame()
                    continue
                dfs[df_key] = dfs[df_key][available]
            if df_key == "df1" and 'Model' in dfs[df_key].columns: # Specific filter for df1
                dfs[df_key] = dfs[df_key][dfs[df_key]['Model'] != 'DeepSeek V3']

            if df_key == "df15": # Specific filter for df1
                dfs[df_key] = dfs[df_key].rename(columns={'AI System': "Model"})
            if df_key == "df16": # Specific filter for df1
                dfs[df_key] = dfs[df_key].rename(columns={'name': "Model"})
            if df_key == "df17": # Ensure WeirdML uses consistent Model column for mapping
                dfs[df_key] = dfs[df_key].rename(columns={'display_name': "Model"})
            if df_key == "df18": # Normalize Yupp columns
                dfs[df_key] = dfs[df_key].rename(columns={'model': "Model", 'text_score': "Text_Score", 'coding_score': "Coding_Score"})
            if df_key == "df19":
                dfs[df_key] = dfs[df_key].rename(columns={
                    'author/model_name': 'Model',
                    'Writing \u270d\ufe0f': 'Writing',
                    'Is Finetuned': 'Is_Finetuned'
                })
                if 'Is_Finetuned' in dfs[df_key].columns:
                    try:
                        dfs[df_key]['Is_Finetuned'] = dfs[df_key]['Is_Finetuned'].astype(bool)
                        dfs[df_key] = dfs[df_key][dfs[df_key]['Is_Finetuned'] == False]
                        dfs[df_key] = dfs[df_key].drop(columns=['Is_Finetuned'])
                    except Exception:
                        dfs[df_key] = dfs[df_key].drop(columns=['Is_Finetuned'], errors='ignore')

        except ValueError as e:
            print(f"Error loading {config['name']}: {e}. Skipping this benchmark.")
            dfs[df_key] = pd.DataFrame() # Create empty DataFrame to avoid errors later


    # Ensure all necessary dataframes are loaded for mapping update
    required_dfs_for_mapping = ["df1", "df2", "df3", "df5", "df6", "df7", "df11", "df13", 'df14', 'df15', 'df16', 'df17', 'df18', 'df21', 'df22', 'df23']
    if not all(key in dfs and not dfs[key].empty for key in required_dfs_for_mapping):
        missing_keys = [bench_configs[key]['name'] for key in required_dfs_for_mapping if key not in dfs or dfs[key].empty]
        print(f"One or more required dataframes for mapping are missing or empty: {missing_keys}. Cannot proceed with mapping updates or full merge.")
        return pd.DataFrame(), {}


    # Identify new models needing mapping updates
    new_models_to_map_found = False
    print("\nChecking for new models that need mapping to OpenBench:")
    for df_key, config in bench_configs.items():
        if dfs[df_key].empty or config.get("is_base") or "map_dict" not in config:
            continue
        
        current_df = dfs[df_key]
        map_dict = config["map_dict"]
        bench_name = config["name"]
        
        model_col = 'Model' if 'Model' in current_df.columns else 'model' if 'model' in current_df.columns else None
        
        if not model_col:

            print(f"Warning: Neither 'Model' nor 'model' column found in {bench_name} DataFrame. Skipping mapping check for it.")
            continue

        new_models = [model for model in current_df[model_col].unique()
                      if isinstance(model, str) and (model not in map_dict or not map_dict.get(model))]
        
        if new_models:
            print(f"{bench_name} (to OpenBench): {len(new_models)} new models found (e.g., {new_models[:3]}{'...' if len(new_models) > 3 else ''})")
            new_models_to_map_found = True



    if new_models_to_map_found:
        print("\nAttempting to update mapping dictionaries...")
        # UPDATED: Pass df13 (LMArena) to the mapping function
        update_mapping_dictionaries(dfs["df1"], dfs["df2"], dfs["df3"],
                                    dfs["df5"], dfs["df6"], dfs["df7"], dfs["df11"],
                                    dfs["df13"], dfs["df14"], dfs["df15"], dfs["df16"], dfs["df17"], dfs["df18"], dfs.get("df19", pd.DataFrame(columns=['Model'])),
                                    dfs["df21"], dfs["df22"], dfs["df23"])
    else:
        print("\nNo new models found requiring mapping updates to OpenBench, or source benchmarks are empty.")

    # Apply mappings and create Unified_Name
    print("\nApplying mappings and creating Unified_Name...")
    for df_key, config in bench_configs.items():
        print(df_key)
        if dfs[df_key].empty:
            print(f"{config['name']} DataFrame is empty. Skipping.")
            if 'Unified_Name' not in dfs[df_key].columns:
                 dfs[df_key]['Unified_Name'] = pd.Series(dtype='object')
            continue

        current_df = dfs[df_key]
        bench_name_lower = config["name"].lower().replace(" ", "").replace("confabulations","")

        if config.get("is_base"): # This is OpenBench (df3)
            current_df['Unified_Name'] = current_df['Model'].str.strip()
        elif "map_dict" in config: # Benchmarks mapped via LLM (NOW INCLUDES LMARENA)

            map_dict = config["map_dict"]
            model_col = 'Model' if 'Model' in current_df.columns else 'model'
            
            if model_col not in current_df.columns:
                print(f"'{model_col}' column missing in {config['name']}. Cannot create 'Unified_Name'.")
                current_df['Unified_Name'] = pd.Series(dtype='object')
            else:
                current_df['Model_Mapped_to_OpenBench'] = current_df[model_col].apply(lambda x: map_dict.get(str(x), '') or str(x))
                current_df['Unified_Name'] = current_df['Model_Mapped_to_OpenBench'].str.strip()

        elif "model_col" in config: # Benchmarks with direct model name column
            model_col_name = config["model_col"]
            if model_col_name not in current_df.columns:
                print(f"Model column '{model_col_name}' missing in {config['name']}. Cannot create 'Unified_Name'.")
                current_df['Unified_Name'] = pd.Series(dtype='object')
            else:
                current_df['Unified_Name'] = current_df[model_col_name].str.strip()
        else:
            print(f"Configuration error for {config['name']}: No mapping rule.")
            current_df['Unified_Name'] = pd.Series(dtype='object')


        # Add prefixes to all columns except Unified_Name
        columns_to_rename = {col: f'{bench_name_lower}_{col}' for col in current_df.columns if col != 'Unified_Name'}
        current_df.rename(columns=columns_to_rename, inplace=True)

        # Deduplicate: if multiple source rows map to the same Unified_Name,
        # keep only the first occurrence to prevent row multiplication in merge.
        # The mapping file is authoritative; duplicates mean the mapping is
        # under-specified and needs curation (e.g. distinct effort levels
        # collapsing to one canonical name). Report them loudly so they can
        # be fixed in the mapping JSON rather than hidden here.
        before = len(current_df)
        current_df = current_df[current_df['Unified_Name'].notna() & (current_df['Unified_Name'] != '')]
        dup_mask = current_df['Unified_Name'].duplicated(keep=False)
        if dup_mask.any():
            src_col = None
            for c in (f'{bench_name_lower}_Model', f'{bench_name_lower}_model',
                      f'{bench_name_lower}_model_name', f'{bench_name_lower}_name',
                      f'{bench_name_lower}_display_name', f'{bench_name_lower}_AI System',
                      f'{bench_name_lower}_judged_model', f'{bench_name_lower}_writer_model',
                      f'{bench_name_lower}_author/model_name'):
                if c in current_df.columns:
                    src_col = c; break
            print(f"  [WARN] {config['name']}: {dup_mask.sum()} source rows map to duplicate Unified_Names — "
                  f"keeping first, dropping rest. Fix the mapping file to resolve:")
            dups = current_df.loc[dup_mask].groupby('Unified_Name', sort=False)
            for tgt, grp in dups:
                srcs = grp[src_col].tolist() if src_col else ['<unknown>'] * len(grp)
                print(f"    {tgt}  <-  {srcs}  (keeping: {srcs[0]})")
        current_df = current_df.drop_duplicates(subset='Unified_Name', keep='first')
        after = len(current_df)
        if before != after:
            print(f"  {config['name']}: deduplicated {before} -> {after} rows by Unified_Name")

        dfs[df_key] = current_df # Update the DataFrame in the dictionary

    # Merge all datasets on Unified_Name
    df_merged = pd.DataFrame(columns=['Unified_Name'])
    
    # Start with OpenBench if available
    if "df3" in dfs and not dfs["df3"].empty:
        df_merged = dfs["df3"]
        processed_keys = {"df3"}
    else:
        print("Warning: OpenBench (df3) data is missing or empty. Merged data might not be OpenBench-centric.")
        processed_keys = set()
        for df_key in bench_configs.keys():
             if df_key in dfs and not dfs[df_key].empty:
                 df_merged = dfs[df_key]
                 processed_keys.add(df_key)
                 break
        if df_merged.empty:
            print("All DataFrames are empty. Returning an empty merged DataFrame.")
            return pd.DataFrame(), {}


    for df_key, config in bench_configs.items():
        if df_key in processed_keys or dfs[df_key].empty:
            continue
        if 'Unified_Name' not in dfs[df_key].columns:
            print(f"Warning: {config['name']} DataFrame is missing 'Unified_Name' column. Skipping merge for it.")
            continue
        df_merged = pd.merge(df_merged, dfs[df_key], on='Unified_Name', how='outer')
        processed_keys.add(df_key)

    print("\nMerged DataFrame columns:", df_merged.columns.tolist())

    # Track source benchmarks using prefixed column presence
    print("\nTracking source benchmarks in merged data...")
    for df_key, config in bench_configs.items():
        bench_name_lower = config["name"].lower().replace(" ", "").replace("confabulations","")
        
        # Determine the original model column name before it was prefixed
        if "model_col" in config:
            original_model_col = config['model_col']
        elif 'Model' in dfs[df_key].columns:
            original_model_col = 'Model'
        elif 'model' in dfs[df_key].columns:
            original_model_col = 'model'
        else: # Fallback for empty/problematic dataframes
            original_model_col = 'Model' 

        original_model_col_prefixed = f"{bench_name_lower}_{original_model_col}"
        
        if original_model_col_prefixed in df_merged.columns:
            df_merged[f'In_{config["name"].replace(" ", "")}'] = ~df_merged[original_model_col_prefixed].isna()
        else:
            df_merged[f'In_{config["name"].replace(" ", "")}'] = False
            print(f"Warning: Original model column '{original_model_col_prefixed}' for {config['name']} not found in merged_df. 'In_{config['name']}' will be all False.")

    # Generate summary of mappings used for models that were mapped
    final_mappings_used_list = []
    for idx, row in df_merged.iterrows():
        for df_key, config in bench_configs.items():
            if dfs[df_key].empty or config.get("is_base") or "map_dict" not in config:
                continue

            bench_name = config["name"]
            bench_name_lower = bench_name.lower().replace(" ", "").replace("confabulations","")
            
            in_bench_col = f'In_{bench_name.replace(" ", "")}'
            if in_bench_col in row and row[in_bench_col]:
                # Determine original model column name
                original_model_col_base = 'Model' if f"{bench_name_lower}_Model" in row else 'model'
                original_model_col = f"{bench_name_lower}_{original_model_col_base}"
                mapped_model_col = f"{bench_name_lower}_Model_Mapped_to_OpenBench"

                if original_model_col in row and mapped_model_col in row:
                    original_name = row[original_model_col]
                    mapped_name = row[mapped_model_col]
                    unified_name = row['Unified_Name']
                    
                    if pd.notna(original_name) and original_name != unified_name and pd.notna(mapped_name) and mapped_name == unified_name:
                        final_mappings_used_list.append(f"{bench_name}: {original_name} → {unified_name}")
                elif original_model_col in row and pd.notna(row[original_model_col]) and row[original_model_col] != row['Unified_Name']:
                     final_mappings_used_list.append(f"({bench_name} - direct to Unified_Name): {row[original_model_col]} → {row['Unified_Name']}")


    if final_mappings_used_list:
        with open('mappings/final_openbench_mappings_used.txt', 'w') as f:
            f.write("Final mappings to OpenBench used in this run (Original Source Name → Unified OpenBench Name):\n")
            f.write("\n".join(sorted(list(set(final_mappings_used_list)))))
    else:
        if os.path.exists('mappings/final_openbench_mappings_used.txt'):
            os.remove('mappings/final_openbench_mappings_used.txt')


    return df_merged, bench_configs


def find_mapping_issues(df, key="Unified_Name"):
    """Detect mapping conflicts where one model maps to multiple data sources.

    After merging benchmarks, ideally each Unified_Name should appear exactly
    once. When a model appears in multiple rows with different values, this
    indicates either:
    1. A mapping conflict (two source models incorrectly mapped to same unified name)
    2. Duplicate entries in source data
    3. Version/date differences not captured in the unified name

    Args:
        df: Combined benchmark DataFrame.
        key: Column name to check for duplicates. Defaults to "Unified_Name".

    Returns:
        DataFrame with columns [Unified_Name, rows, differing_cols, sample_values]
        for each conflicting model, sorted by row count descending.
        Returns None if no conflicts found.

    Example:
        >>> issues = find_mapping_issues(df_combined)
        >>> if issues is not None:
        ...     print(f"Found {len(issues)} models with conflicts")
        ...     issues.to_csv("mapping_issues.csv")
    """
    dupes = df[df.duplicated(key, keep=False)].copy()
    if dupes.empty:
        return None
        
    records = []
    for name, grp in dupes.groupby(key):
        differing_cols = [
            col for col in grp.columns if col != key
            and grp[col].nunique(dropna=True) > 1
        ]
        if differing_cols:
            samples = {
                col: grp[col].dropna().unique()[:5].tolist()
                for col in differing_cols
            }
            records.append(
                {
                    "Unified_Name": name,
                    "rows": len(grp),
                    "differing_cols": differing_cols,
                    "sample_values": samples,
                }
            )
    if records:
        return pd.DataFrame(records).sort_values("rows", ascending=False)
    return None

# ------------------------------------------------------------
# Main execution
if __name__ == "__main__":
    print("Starting benchmark combination process...")
    print("Step 1: Loading existing mappings (to OpenBench)...")
    load_existing_mappings()
    
    print("\nStep 2: Combining benchmarks with auto-mapping to OpenBench...")
    df_all, bench_configs_used = combine_benchmarks_with_auto_mapping()
    
    if not df_all.empty:
        issues = find_mapping_issues(df_all)
        if issues is not None:
            print(f"\nFound {len(issues)} models split across multiple rows due to differing data.")
            print("Saving details to mappings/mapping_issues.csv")
            issues.to_csv("mappings/mapping_issues.csv", index=False)
        else:
            print("\nNo mapping issues found (no models split across multiple rows).")

        print(f"\nLength before filtering 'o3 Pro': {len(df_all)}")
        df_all = df_all[df_all['Unified_Name'] != 'o3 Pro']
        print(f"Length after filtering 'o3 Pro': {len(df_all)}")

        # Manual fill for AA pricing columns when data is missing
        manual_aa_pricing_updates = {
            "o3 High": {"input": 2.0, "output": 8.0},
            "GPT-5.2 (high)": {"input": 1.75, "output": 14.00},
            "GPT-5.1 Codex Max": (1.25, 10.0),
            "DeepSeek R1": (0.55, 2.19),
            "Gemma 4 31B": (0.14, 0.40),

            # "Unified Model Name": {"input": 0.0, "output": 0.0},
            # "Another Model": (0.12, 0.34),
        }
        aa_input_col = 'aa_pricing_price_1m_input_tokens'
        aa_output_col = 'aa_pricing_price_1m_output_tokens'
        if manual_aa_pricing_updates:
            missing_pricing_cols = [col for col in (aa_input_col, aa_output_col) if col not in df_all.columns]
            if missing_pricing_cols:
                print(f"AA pricing columns missing in combined data: {missing_pricing_cols}. Skipping manual AA pricing updates.")
            else:
                for model_name, pricing in manual_aa_pricing_updates.items():
                    if isinstance(pricing, (list, tuple)) and len(pricing) == 2:
                        input_price, output_price = pricing
                    elif isinstance(pricing, dict):
                        input_price = pricing.get("input")
                        output_price = pricing.get("output")
                    else:
                        print(f"Unrecognized pricing format for {model_name}; expected dict or 2-tuple.")
                        continue

                    row_condition_unified_name = (df_all['Unified_Name'] == model_name)
                    if not row_condition_unified_name.any():
                        print(f"Model '{model_name}' not found for manual AA pricing update.")
                        continue

                    aa_input_missing_mask = row_condition_unified_name & (df_all[aa_input_col].isna() | (df_all[aa_input_col] == 0))
                    if aa_input_missing_mask.any() and input_price is not None:
                        df_all.loc[aa_input_missing_mask, aa_input_col] = input_price
                        print(f"Applied manual AA input pricing update for {model_name}.")
                    elif not aa_input_missing_mask.any():
                        print(f"Skipping AA input pricing update for {model_name}: value already present.")

                    aa_output_missing_mask = row_condition_unified_name & (df_all[aa_output_col].isna() | (df_all[aa_output_col] == 0))
                    if aa_output_missing_mask.any() and output_price is not None:
                        df_all.loc[aa_output_missing_mask, aa_output_col] = output_price
                        print(f"Applied manual AA output pricing update for {model_name}.")
                    elif not aa_output_missing_mask.any():
                        print(f"Skipping AA output pricing update for {model_name}: value already present.")

        # Manual fill for token counts on non-reasoning variants that were
        # never benchmarked separately.  We know reasoning tokens = 0 for
        # non-reasoning models; answer tokens are left for the imputer.
        manual_token_fills = {
            "DeepSeek V3.2 (Non-Reasoning)": {
                "logic_avg_reasoning_tokens": 0.0,
            },
        }
        for model_name, fills in manual_token_fills.items():
            mask = df_all["Unified_Name"] == model_name
            if not mask.any():
                print(f"Token-fill target '{model_name}' not found.")
                continue
            for col, val in fills.items():
                if col not in df_all.columns:
                    continue
                missing = mask & df_all[col].isna()
                if missing.any():
                    df_all.loc[missing, col] = val
                    print(f"Set {col} = {val} for '{model_name}'.")

        # Example data correction
        lmsys_arena_score_col = 'lmsys_arena_score'
        if 'Unified_Name' in df_all.columns and lmsys_arena_score_col in df_all.columns:
            manual_input_dict = {
                "Gemini 2.5 Pro Preview (2025-03-25)": 1439,
            }
            for manual_model, score in manual_input_dict.items():
                row_condition_unified_name = (df_all['Unified_Name'] == manual_model)
                if row_condition_unified_name.any():
                    df_all.loc[row_condition_unified_name, lmsys_arena_score_col] = score
                    print(f"Applied manual lmsys_arena_score update for {manual_model}.")
                else:
                    print(f"Model '{manual_model}' not found for manual score update.")

        output_filename = 'benchmarks/combined_all_benches.csv'
        df_all.to_csv(output_filename, index=False)
        print(f"\nCombined benchmark data saved to {output_filename}")

        total_models = len(df_all)
        print(f"Total unique model entries (Unified_Name) in combined data: {total_models}")

        print("\nSummary of model presence from different benchmarks:")
        if bench_configs_used:
            for df_key, config in bench_configs_used.items():
                in_col_name = f'In_{config["name"].replace(" ", "")}'
                if in_col_name in df_all.columns:
                    count = df_all[in_col_name].sum()
                    print(f" - Models from {config['name']}: {int(count)}")

        print(f"\nSee 'mappings/unmapped_to_openbench_models.txt' for models from source benchmarks without a good OpenBench equivalent.")
        print(f"See 'mappings/new_openbench_mappings_for_review.txt' for new OpenBench mappings generated in this run.")
        print(f"See 'mappings/final_openbench_mappings_used.txt' for a list of mappings applied in this run.")
    else:
        print("Process finished, but the combined DataFrame is empty. Please check input files and logs.")

    if not df_all.empty:
        numeric_cols = df_all.select_dtypes(include='number').columns
        pct_filled = df_all[numeric_cols].notna().mean().mean() * 100
        print(f"\nBenchmark combination finished: {len(df_all)} models, {len(numeric_cols)} numeric columns, {pct_filled:.0f}% filled.")
    else:
        print("\nBenchmark combination finished (empty).")
