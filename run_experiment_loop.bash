#!/usr/bin/env bash
# Autonomous experiment loop: 12 cycles of brainstorm → implement → run → analyze
# Runs on the tower via Claude Code in -p mode
set -euo pipefail

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

CLAUDE="$HOME/.local/bin/claude"
PROJECT="$HOME/project_soothsayer"
ARENA="$PROJECT/arena_predictor"
LOG_DIR="/tmp/experiment_loop_logs"
mkdir -p "$LOG_DIR"

# Determine starting cycle by scanning existing bash scripts
START_CYCLE=10
for f in "$ARENA"/run_cycle*_experiments.bash; do
    if [[ -f "$f" ]]; then
        num=$(echo "$f" | grep -oP 'cycle\K[0-9]+')
        if (( num >= START_CYCLE )); then
            START_CYCLE=$((num + 1))
        fi
    fi
done
END_CYCLE=$((START_CYCLE + 11))

echo "================================================================"
echo "AUTONOMOUS EXPERIMENT LOOP"
echo "Cycles: $START_CYCLE through $END_CYCLE"
echo "Started: $(date)"
echo "================================================================"

for cycle in $(seq "$START_CYCLE" "$END_CYCLE"); do
    echo ""
    echo "================================================================"
    echo "=== CYCLE $cycle ==="
    echo "Started: $(date)"
    echo "================================================================"

    CYCLE_LOG="$LOG_DIR/cycle${cycle}.log"

    # ──────────────────────────────────────────────────────────────
    # PHASE 1: Brainstorm and implement (Claude Code + Codex)
    # ──────────────────────────────────────────────────────────────
    echo "[Phase 1] Brainstorming and implementing experiments..."

    cd "$ARENA"
    "$CLAUDE" -p --dangerously-skip-permissions --model opus --max-budget-usd 20 \
        "$(cat <<PROMPT
You are running autonomous Experiment Cycle $cycle for Project Soothsayer's arena predictor.
Working directory: $(pwd)

GOAL: Design and implement 6 new experiments to try to beat RMSE 21.48.

STEPS — execute ALL of these:

1. READ ../docs/FINDINGS.md to understand ALL prior experiments. Count them. Do NOT repeat any.

2. BRAINSTORM 3 novel experiment ideas yourself. Think creatively about what might help at n=90.
   Consider: regularization tricks, ensemble methods, feature interactions, loss functions,
   data augmentation, Bayesian approaches, information-theoretic methods, robustness tricks,
   imputation improvements, transfer learning concepts, etc.

3. USE CODEX to get 3 more ideas. Call the codex_start MCP tool with model="gpt-5.4".
   Share a summary of what's been tried (from FINDINGS.md) and ask for 3 novel ideas.
   IMPORTANT: Share enough context that Codex doesn't repeat prior experiments.

4. IMPLEMENT all 6 experiments in predict.py. For each experiment:
   a) Add a global flag near other experiment flags (search for "ENABLED = False" to find them)
   b) Add argparse argument in the argparse section (search for "add_argument" to find the section)
   c) Wire the flag in main() (search for "ENABLED = args" to find where other flags are wired)
   d) Implement the logic — INSIDE _precompute_single_fold() if it uses target/ALT values
   e) Verify syntax: run "python3 -c \"import py_compile; py_compile.compile('predict.py')\""

5. CREATE run_cycle${cycle}_experiments.bash following the template in CLAUDE.md.
   - Use output dirs: analysis_output/_c${cycle}_exp{1..6}
   - Use log files: /tmp/cycle${cycle}_exp{1..6}.log
   - Run 3 at a time (batches of 3 with & and wait)

6. VERIFY the bash script is executable and syntactically correct.

When done, output a brief summary of what the 6 experiments are, then "PHASE 1 COMPLETE".
PROMPT
)" > "$CYCLE_LOG.phase1" 2>&1

    echo "[Phase 1] Done. Output in $CYCLE_LOG.phase1"

    # Verify the experiment script was created
    EXP_SCRIPT="$ARENA/run_cycle${cycle}_experiments.bash"
    if [[ ! -f "$EXP_SCRIPT" ]]; then
        echo "ERROR: $EXP_SCRIPT was not created! Skipping cycle $cycle."
        continue
    fi
    chmod +x "$EXP_SCRIPT"

    # Verify predict.py syntax
    if ! python3 -c "import py_compile; py_compile.compile('$ARENA/predict.py')" 2>/dev/null; then
        echo "ERROR: predict.py has syntax errors after Phase 1! Attempting fix..."
        cd "$ARENA"
        "$CLAUDE" -p --dangerously-skip-permissions --model opus --max-budget-usd 5 \
            "predict.py has a syntax error. Run 'python3 -c \"import py_compile; py_compile.compile(\\\"predict.py\\\")\"' to see the error, then fix it. Only fix the syntax error, don't change any logic." \
            > "$CYCLE_LOG.fix" 2>&1
        if ! python3 -c "import py_compile; py_compile.compile('$ARENA/predict.py')" 2>/dev/null; then
            echo "ERROR: Could not fix predict.py syntax. Skipping cycle $cycle."
            continue
        fi
    fi

    # ──────────────────────────────────────────────────────────────
    # PHASE 2: Run experiments (direct bash, no Claude needed)
    # ──────────────────────────────────────────────────────────────
    echo "[Phase 2] Running experiments (this takes ~60 minutes)..."
    echo "[Phase 2] Started: $(date)"

    cd "$ARENA"
    bash "$EXP_SCRIPT" > "$CYCLE_LOG.phase2" 2>&1 || true

    echo "[Phase 2] Done: $(date)"
    echo "[Phase 2] Output in $CYCLE_LOG.phase2"
    cat "$CYCLE_LOG.phase2" | tail -20

    # ──────────────────────────────────────────────────────────────
    # PHASE 3: Analyze results and log to FINDINGS.md
    # ──────────────────────────────────────────────────────────────
    echo "[Phase 3] Analyzing results and updating FINDINGS.md..."

    cd "$ARENA"
    "$CLAUDE" -p --dangerously-skip-permissions --model opus --max-budget-usd 10 \
        "$(cat <<PROMPT
You just ran Experiment Cycle $cycle for Project Soothsayer's arena predictor.
Working directory: $(pwd)

TASK: Collect results and update ../docs/FINDINGS.md.

STEPS:

1. For each experiment 1-6, find the output directory analysis_output/_c${cycle}_exp{i}/output_*/
   and read metadata.json to extract oof_rmse. If no output exists, check /tmp/cycle${cycle}_exp{i}.log for errors.

2. Also check the experiment bash script run_cycle${cycle}_experiments.bash to know what each experiment name/flag was.

3. Read ../docs/FINDINGS.md to see the current format and content.

4. ADD a new "Cycle $cycle" section to FINDINGS.md with:
   - A results table showing all 6 experiments, their RMSE, and delta vs baseline (21.48)
   - Brief analysis of what worked/didn't and why
   - Update the cumulative experiment count in the summary section
   - Mark any winners (RMSE < 21.48) or interesting findings

5. Output the results table when done.
PROMPT
)" > "$CYCLE_LOG.phase3" 2>&1

    echo "[Phase 3] Done. Output in $CYCLE_LOG.phase3"
    echo ""
    echo "=== CYCLE $cycle COMPLETE ==="
    echo "Finished: $(date)"
    echo ""

    # Brief pause between cycles
    sleep 5
done

echo ""
echo "================================================================"
echo "ALL 12 CYCLES COMPLETE"
echo "Finished: $(date)"
echo "================================================================"
echo ""
echo "Results are in ../docs/FINDINGS.md"
echo "Logs are in $LOG_DIR/"
