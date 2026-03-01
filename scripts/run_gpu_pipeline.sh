#!/usr/bin/env bash
# Download data from Google Drive and run the full GPU pipeline.
# Usage: bash scripts/run_gpu_pipeline.sh
set -euo pipefail

GDRIVE_FILE_ID="1CYkYkbNEMQ0fHIaM4xMpuEDL1GZVOs4T"
WORKSPACE="/workspace/agadmator"

cd "$WORKSPACE"

# 1. Download from Google Drive
echo "=== [1/8] Downloading data from Google Drive ==="
pip install gdown --quiet
gdown "$GDRIVE_FILE_ID" -O /tmp/raw_data.archive

# 2. Extract to data/raw/
echo "=== [2/8] Extracting data ==="
mkdir -p data
# Detect archive type and extract
FILE_TYPE=$(file /tmp/raw_data.archive)
if echo "$FILE_TYPE" | grep -q "Zip archive"; then
    unzip -o /tmp/raw_data.archive -d data/
elif echo "$FILE_TYPE" | grep -q "gzip"; then
    tar xzf /tmp/raw_data.archive -C data/
elif echo "$FILE_TYPE" | grep -q "tar"; then
    tar xf /tmp/raw_data.archive -C data/
else
    echo "Unknown archive format: $FILE_TYPE"
    echo "Trying tar xzf..."
    tar xzf /tmp/raw_data.archive -C data/
fi
rm /tmp/raw_data.archive

echo "Audio files: $(ls data/raw/audio/*.wav 2>/dev/null | wc -l)"
echo "PGN files:   $(ls data/raw/pgn/*.pgn 2>/dev/null | wc -l)"

# 3. Isolate vocals
echo "=== [3/8] Isolating vocals (Demucs) ==="
agadmator isolate-vocals

# 4. Transcribe
echo "=== [4/8] Transcribing (faster-whisper) ==="
agadmator transcribe

# 5. Align PGN with transcripts
echo "=== [5/8] Aligning PGN ↔ transcripts ==="
agadmator align

# 6. Prepare LLM training data
echo "=== [6/8] Preparing LLM training data ==="
agadmator prepare-llm-data

# 7. Train LLM (generates validation samples at each epoch + final)
echo "=== [7/8] Training LLM ==="
agadmator train-llm --phase pretrain --config configs/default.yaml
agadmator train-llm --phase style   --config configs/default.yaml

# 8. Package all outputs
echo "=== [8/8] Packaging checkpoints & validation samples ==="
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_ZIP="agadmator_results_${TIMESTAMP}.tar.gz"

tar czf "$OUTPUT_ZIP" \
    data/models/llm/chessgpt_lora/ \
    data/models/llm/agadmator_lora/ \
    data/models/llm/validation_samples/ \
    data/processed/llm_training/

echo ""
echo "=== Pipeline complete ==="
echo ""
agadmator status
echo ""
echo "Checkpoints + validation samples packaged: $OUTPUT_ZIP"
echo "  - data/models/llm/chessgpt_lora/          (pretrain LoRA)"
echo "  - data/models/llm/agadmator_lora/         (style LoRA)"
echo "  - data/models/llm/validation_samples/      (generated commentary vs ground truth)"
echo "  - data/processed/llm_training/             (training data)"
echo ""
echo "Download with: scp root@\$(hostname):${WORKSPACE}/${OUTPUT_ZIP} ."
