#!/usr/bin/env bash
# Setup script for cloud GPU environments (Lambda Labs, RunPod, Vast.ai, etc.)
# Assumes: Ubuntu 22.04+, NVIDIA GPU, CUDA drivers installed
set -euo pipefail

echo "=== Agadmator Pipeline: Cloud GPU Setup ==="

# 1. System dependencies
echo "[1/6] Installing system packages..."
apt-get update -qq
apt-get install -y -qq ffmpeg git fonts-dejavu-core curl unzip >/dev/null

# Install deno (required by yt-dlp for YouTube JS challenges)
if ! command -v deno &>/dev/null; then
    curl -fsSL https://deno.land/install.sh | sh -s -- --yes
    export DENO_INSTALL="$HOME/.deno"
    export PATH="$DENO_INSTALL/bin:$PATH"
    # Make persistent
    echo 'export DENO_INSTALL="$HOME/.deno"' >> ~/.bashrc
    echo 'export PATH="$DENO_INSTALL/bin:$PATH"' >> ~/.bashrc
fi

# 2. Verify GPU
echo "[2/6] Checking GPU..."
if ! nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. CUDA drivers required."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# 3. Install Python package (editable)
echo "[3/6] Installing agadmator package..."
pip install -e "." --quiet

# 4. Data collection dependencies
echo "[4/6] Installing data collection tools..."
pip install yt-dlp demucs faster-whisper torchcodec --quiet

# 5. LLM training dependencies
echo "[5/6] Installing LLM training tools..."
pip install "unsloth[colab-new]" datasets transformers trl peft bitsandbytes pyyaml --quiet

# 6. TTS dependencies (install fish-speech when ready)
echo "[6/6] Installing TTS tools..."
pip install soundfile librosa pyloudnorm --quiet
echo "NOTE: Install Fish Speech manually when ready: pip install fish-speech"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Verify with: agadmator status"
echo ""
echo "Quick start:"
echo "  agadmator fetch-metadata        # Get video catalog (4855 videos)"
echo "  agadmator collect-pgn            # Extract PGN files"
echo "  agadmator download-audio --limit 100"
echo "  agadmator isolate-vocals"
echo "  agadmator transcribe"
echo "  agadmator align"
echo "  agadmator prepare-llm-data"
echo "  agadmator train-llm --phase pretrain --config configs/default.yaml"
echo "  agadmator train-llm --phase style   --config configs/default.yaml"
