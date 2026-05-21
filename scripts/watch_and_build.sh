#!/bin/bash
set -e
DB="data/gen_structured_v20_checkpoint.db"
LOG="logs/watch_and_build.log"
cd /home/user/spo-reasoning-training-regimen

echo "[$(date)] Watcher started. Waiting for 200 records..." | tee -a $LOG

while true; do
    cnt=$(sqlite3 $DB "SELECT COUNT(*) FROM done" 2>/dev/null || echo 0)
    echo "[$(date +%H:%M:%S)] records: $cnt/200" | tee -a $LOG
    if [[ $cnt -ge 200 ]]; then
        echo "[$(date)] Reached 200 records. Writing JSONL..." | tee -a $LOG
        /home/user/mamba-venv/bin/python3 -c "
import sqlite3, json
conn = sqlite3.connect('$DB')
rows = conn.execute('SELECT result FROM done').fetchall()
with open('data/train_structured_v20.jsonl', 'w') as f:
    for (r,) in rows:
        rec = json.loads(r)
        # write full structured record — build_training_regimens expects
        # quote / entailed_premises / non_entailed_premises / syllogism
        f.write(json.dumps(rec) + '\n')
print(f'Wrote {len(rows)} structured records')
" | tee -a $LOG
        wc -l data/train_structured_v20.jsonl | tee -a $LOG
        break
    fi
    sleep 60
done
