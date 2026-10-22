#!/bin/bash
# Full real-data pipeline: download -> organize -> landmarks -> splits -> train -> evaluate.
# Idempotent-ish: curl resumes (-C -), landmark extraction skips existing .npy.
set -u
cd ~/workspace/sign-language-translator
mkdir -p logs data/raw_dl
LOG="logs/pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
PY=.venv/bin/python
echo "=== PIPELINE START $(date) ==="

echo "--- step 1: download zips (resume enabled) ---"
for z in Greetings_1of2 Greetings_2of2 Pronouns_1of2 Pronouns_2of2; do
  echo "downloading ${z}.zip ..."
  curl -L --retry 5 --retry-delay 10 -C - -o "data/raw_dl/${z}.zip" \
    "https://zenodo.org/api/records/4010759/files/${z}.zip/content"
  sz=$(stat -c%s "data/raw_dl/${z}.zip")
  echo "  ${z}.zip size=${sz} bytes"
  if [ "$sz" -lt 100000000 ]; then
    echo "ERROR: ${z}.zip suspiciously small (${sz} bytes)"; exit 1
  fi
done

echo "--- step 2: organize into data/raw/<label>/ ---"
$PY scripts/organize_include.py || { echo "ERROR: organize failed"; exit 1; }

echo "--- step 3: extract landmarks (slow, CPU MediaPipe) ---"
$PY src/data/extract_landmarks.py || { echo "ERROR: landmark extraction failed"; exit 1; }

echo "--- step 4: build official-split dataset ---"
$PY src/data/build_official_splits.py || { echo "ERROR: split build failed"; exit 1; }

echo "--- step 5: train ---"
$PY src/train.py || { echo "ERROR: training failed"; exit 1; }

echo "--- step 6: evaluate on official test split ---"
$PY src/evaluate.py || { echo "ERROR: evaluation failed"; exit 1; }

echo "=== PIPELINE COMPLETE $(date) ==="
