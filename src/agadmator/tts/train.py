"""Step 2: Fine-tune TTS model (Fish Speech / OpenAudio S1)."""

import json
import logging
import subprocess
import sys
from pathlib import Path

from agadmator.config import TTS_SEGMENTS_DIR, TTS_OUTPUT_DIR

log = logging.getLogger(__name__)


def _load_config(config_path: str) -> dict:
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def train_tts(config_path: str):
    """Fine-tune TTS model using Fish Speech's training pipeline.

    Fish Speech expects a directory of .wav + .lab pairs.
    We invoke their training CLI with LoRA configuration.
    """
    cfg = _load_config(config_path)["tts"]
    TTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data_dir = TTS_SEGMENTS_DIR
    manifest = data_dir / "manifest.json"
    if not manifest.exists():
        log.error("TTS manifest not found. Run 'prepare-tts-data' first.")
        return

    with open(manifest) as f:
        clips = json.load(f)

    total_hours = sum(c["duration"] for c in clips) / 3600
    log.info("Training data: %d clips, %.1f hours", len(clips), total_hours)

    # Generate Fish Speech training config
    fish_config = {
        "model": cfg["model"],
        "data_dir": str(data_dir),
        "output_dir": str(TTS_OUTPUT_DIR / "fish_lora"),
        "lora": {
            "enabled": True,
            "rank": 64,
            "alpha": 128,
        },
        "training": {
            "batch_size": 4,
            "learning_rate": 1e-4,
            "epochs": 10,
            "save_every_n_epochs": 2,
            "gradient_accumulation_steps": 4,
        },
        "audio": {
            "sample_rate": cfg["sample_rate"],
            "target_lufs": cfg["target_lufs"],
        },
    }

    config_out = TTS_OUTPUT_DIR / "fish_train_config.json"
    with open(config_out, "w") as f:
        json.dump(fish_config, f, indent=2)

    # Fish Speech training is invoked via their CLI
    # The exact command depends on the installed version
    log.info("Starting Fish Speech LoRA training...")
    log.info("Config saved to %s", config_out)

    try:
        # Try the Fish Speech CLI
        cmd = [
            sys.executable, "-m", "fish_speech.train",
            "--config", str(config_out),
        ]
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        log.info(
            "Fish Speech CLI not found. To train manually:\n"
            "  1. Install fish-speech: pip install fish-speech\n"
            "  2. Run: python -m fish_speech.train --config %s\n"
            "  Or use the Fish Speech Web UI for training.",
            config_out,
        )
    except subprocess.CalledProcessError as e:
        log.error("Training failed: %s", e)

    log.info("TTS training config saved to %s", config_out)
