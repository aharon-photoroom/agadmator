#!/usr/bin/env bash
# Download data from Google Drive and run the full GPU pipeline.
# Usage: bash scripts/run_gpu_pipeline.sh
set -euo pipefail

GDRIVE_FILE_ID="1CYkYkbNEMQ0fHIaM4xMpuEDL1GZVOs4T"
WORKSPACE="/workspace/agadmator"

cd "$WORKSPACE"

# 1. Download from Google Drive
echo "=== [1/7] Downloading data from Google Drive ==="
pip install gdown --quiet
gdown "$GDRIVE_FILE_ID" -O /tmp/raw.tar.gz

# 2. Extract to data/raw/
echo "=== [2/7] Extracting data ==="
mkdir -p data
# Detect archive type and extract
FILE_TYPE=$(file /tmp/raw.tar.gz)
if echo "$FILE_TYPE" | grep -q "Zip archive"; then
    unzip -o /tmp/raw.tar.gz -d data/
elif echo "$FILE_TYPE" | grep -q "gzip"; then
    tar xzf /tmp/raw.tar.gz -C data/
elif echo "$FILE_TYPE" | grep -q "tar"; then
    tar xf /tmp/raw.tar.gz -C data/
else
    echo "Unknown archive format: $FILE_TYPE"
    echo "Trying tar xzf..."
    tar xzf /tmp/raw.tar.gz -C data/
fi
rm /tmp/raw.tar.gz

echo "Audio files: $(ls data/raw/audio/*.wav 2>/dev/null | wc -l)"
echo "PGN files:   $(ls data/raw/pgn/*.pgn 2>/dev/null | wc -l)"

# 3. Isolate vocals
echo "=== [3/7] Isolating vocals (Demucs) ==="
agadmator isolate-vocals

# 4. Transcribe
echo "=== [4/7] Transcribing (faster-whisper) ==="
agadmator transcribe

# 5. Align PGN with transcripts
echo "=== [5/7] Aligning PGN ↔ transcripts ==="
agadmator align

# 6. Prepare LLM training data
echo "=== [6/7] Preparing LLM training data ==="
agadmator prepare-llm-data

# 7. Train LLM
echo "=== [7/7] Training LLM ==="
agadmator train-llm --phase pretrain --config configs/default.yaml
agadmator train-llm --phase style   --config configs/default.yaml

echo ""
echo "=== Pipeline complete ==="
agadmator status
