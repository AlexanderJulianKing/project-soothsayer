import pandas as pd
import os

BATTLE_HISTORY_CSV = "results/battle_history.csv"

def analyze_history():
    if not os.path.exists(BATTLE_HISTORY_CSV):
        print("No history file found.")
        return

    df = pd.read_csv(BATTLE_HISTORY_CSV)
    print(f"Total rows: {len(df)}")
    
    # Check for duplicates based on (prompt_id, story_a_model, story_b_model, judge_model)
    df['prompt_id'] = df['prompt_id'].astype(str)
    
    # Create a key for each row
    df['key'] = df.apply(lambda row: (str(row['prompt_id']), row['story_a_model'], row['story_b_model'], row['judge_model']), axis=1)
    
    duplicates = df[df.duplicated('key', keep=False)]
    if not duplicates.empty:
        print(f"Found {len(duplicates)} duplicate battles (same prompt, models in same order, judge).")
        print(duplicates[['prompt_id', 'story_a_model', 'story_b_model', 'judge_model']].head())
    else:
        print("No exact duplicates found.")

    # Check for pairs
    # A pair is (A, B) and (B, A) for same prompt and judge
    
    pairs = set()
    singles = set()
    
    for _, row in df.iterrows():
        p = str(row['prompt_id'])
        a = row['story_a_model']
        b = row['story_b_model']
        j = row['judge_model']
        
        if pd.isna(a) or pd.isna(b) or pd.isna(j):
            continue
            
        key = (p, tuple(sorted((a, b))), j)
        
        # We need to track orientation to know if it's a pair
        # Actually, let's just store the set of orientations seen for each (prompt, pair, judge)
        
        if key not in pairs:
            pairs.add(key) # This is just tracking unique pairs of models
            
    # Re-scan to count orientations
    orientation_counts = {}
    for _, row in df.iterrows():
        p = str(row['prompt_id'])
        a = row['story_a_model']
        b = row['story_b_model']
        j = row['judge_model']
        
        if pd.isna(a) or pd.isna(b) or pd.isna(j):
            continue

        pair_key = (p, tuple(sorted((a, b))), j)
        orientation = (a, b)
        
        if pair_key not in orientation_counts:
            orientation_counts[pair_key] = set()
        orientation_counts[pair_key].add(orientation)
        
    paired_count = 0
    unpaired_count = 0
    
    for key, orientations in orientation_counts.items():
        if len(orientations) >= 2:
            paired_count += 1
        else:
            unpaired_count += 1
            
    print(f"Paired pairs: {paired_count}")
    print(f"Unpaired pairs: {unpaired_count}")
    
    # Check for potential matches that are missed
    # e.g. same prompt, same models, but maybe different judge name?
    # or maybe (A, B) exists but (B, A) is missing? (That's just unpaired)
    
    # User suspects "battles in the history that could be pair matches for each other but aren't logged as such"
    # Maybe (A, B) is logged as (A, B) but (B, A) is logged as (B, A) but with a slightly different model name?
    
    all_models = set(df['story_a_model'].dropna().unique()) | set(df['story_b_model'].dropna().unique())
    print(f"Unique models: {len(all_models)}")
    
    sorted_models = sorted(list(all_models))
    print("First 20 models:", sorted_models[:20])
    print("Last 20 models:", sorted_models[-20:])
    
    # Check for similar names
    import difflib
    for i in range(len(sorted_models)):
        for j in range(i + 1, len(sorted_models)):
            m1 = sorted_models[i]
            m2 = sorted_models[j]
            if m1.lower() == m2.lower() and m1 != m2:
                print(f"Potential duplicate model name: '{m1}' vs '{m2}'")
            elif m1 in m2 or m2 in m1:
                 # simple substring check can be noisy but helpful
                 pass

    print("\nUnique Judges:")
    print(df['judge_model'].unique())

    # Debug scheduling logic
    print("\n--- Debugging Scheduling Logic ---")
    try:
        import super_bench
        
        # Mock story index based on history to avoid reading all files if not needed, 
        # but better to read real files if possible. 
        # Let's try to load real story index if directory exists.
        if os.path.isdir(super_bench.STORIES_DIR):
            print("Loading real story index...")
            story_index = super_bench.load_story_index()
        else:
            print("Stories dir not found, mocking story index from history...")
            # Mocking is risky if history doesn't contain all models/prompts
            story_index = {}
            for _, row in df.iterrows():
                p = str(row['prompt_id'])
                ma = row['story_a_model']
                mb = row['story_b_model']
                if pd.isna(ma) or pd.isna(mb): continue
                
                if ma not in story_index: story_index[ma] = {}
                if mb not in story_index: story_index[mb] = {}
                # We don't have the story text, but we need the key to exist
                story_index[ma][p] = "mock story"
                story_index[mb][p] = "mock story"
        
        existing_orientations = super_bench.extract_existing_orientations(df)
        judge_name = "Grok 4 Fast" # Hardcoded based on previous output
        
        print(f"Running list_pending_matches for judge '{judge_name}'...")
        matches, priority = super_bench.list_pending_matches(
            story_index=story_index,
            existing_orientations=existing_orientations,
            judge_name=judge_name,
            paired_mode=True
        )
        
        print(f"Pending matches: {len(matches)}")
        print(f"Priority orientations: {len(priority)}")
        
        if len(priority) > 0:
            print("Sample priority items:")
            print(list(priority)[:5])
            
            # Check if these priority items are in matches
            in_matches = 0
            for p in priority:
                # priority item is (prompt_id, model_a, model_b)
                # matches item is (prompt_id, model_a, model_b)
                if p in matches:
                    in_matches += 1
            print(f"Priority items found in matches list: {in_matches}")
            
    except ImportError as e:
        print(f"Could not import super_bench: {e}")
    except Exception as e:
        print(f"Error running super_bench logic: {e}")

if __name__ == "__main__":
    analyze_history()
