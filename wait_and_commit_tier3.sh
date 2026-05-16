#!/bin/bash
# Waits for training_ladder PID to exit, then commits results and pushes.
set -e
cd /home/user/spo-reasoning-training-regimen

TRAINING_PID=2356671
LOG=output/ladder_run_latest.log

echo "[wait_and_commit] Waiting for PID $TRAINING_PID to finish..."
while kill -0 $TRAINING_PID 2>/dev/null; do
    sleep 30
done
echo "[wait_and_commit] Training complete. Checking results..."

tail -40 "$LOG"

# Check if ladder passed
if grep -q "LADDER PASSED" "$LOG"; then
    echo "[wait_and_commit] Ladder PASSED. Committing results..."
    
    ADAPTER_DIR=$(grep "adapter=" "$LOG" | tail -1 | sed 's/.*adapter=//')
    SUMMARY_FILE="output/ladder_run_latest/ladder_summary.json"
    
    git add "$LOG" "$SUMMARY_FILE" output/ladder_run_latest/tier3_convergence/ src/run_spo_training.py 2>/dev/null || true
    git add -A output/ladder_run_latest/tier3_convergence/adapter/ 2>/dev/null || true
    
    git commit -m "feat: Tier 3 training ladder passed — v9 canonical adapter

Full 900-record × 5-epoch convergence run passed all 10 checks.
Adapter saved to output/ladder_run_latest/tier3_convergence/adapter/

Also fix: add flush=True to run_spo_training.py step logging so
progress is visible in redirected log files.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
    
    git push origin verbatim-extraction-gate
    echo "[wait_and_commit] ✓ Committed and pushed."
    
elif grep -q "LADDER STOPPED" "$LOG"; then
    STOPPED_AT=$(grep "LADDER STOPPED" "$LOG" | tail -1)
    echo "[wait_and_commit] Ladder STOPPED: $STOPPED_AT"
    echo "[wait_and_commit] Check the Tier 3 failure lines above to calibrate thresholds."
    
    git add src/run_spo_training.py
    git commit -m "fix: flush=True on run_spo_training step logging

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
    git push origin verbatim-extraction-gate
    
    # Print the failures so calibration can be done
    echo "=== TIER 3 CHECK RESULTS ==="
    grep -A 20 "Tier 3:" "$LOG" | tail -20
else
    echo "[wait_and_commit] Training ended but no LADDER result found in log."
    tail -20 "$LOG"
fi
