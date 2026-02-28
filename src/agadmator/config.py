"""Central configuration for the pipeline."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# agadmator-library clone (db/ directory with per-video JSON files)
LIBRARY_DB_DIR = RAW_DIR / "library_db"

# Data subdirectories
AUDIO_DIR = RAW_DIR / "audio"
VOCALS_DIR = PROCESSED_DIR / "vocals"
TRANSCRIPTS_DIR = PROCESSED_DIR / "transcripts"
PGN_DIR = RAW_DIR / "pgn"
ALIGNED_DIR = PROCESSED_DIR / "aligned"
TTS_SEGMENTS_DIR = PROCESSED_DIR / "tts_segments"

# Model output directories
LLM_OUTPUT_DIR = DATA_DIR / "models" / "llm"
TTS_OUTPUT_DIR = DATA_DIR / "models" / "tts"

# Video output
VIDEO_OUTPUT_DIR = DATA_DIR / "output"

AGADMATOR_LIBRARY_REPO = "https://github.com/agadmator-library/agadmator-library.github.io"
LIBRARY_RAW_BASE = (
    "https://raw.githubusercontent.com/"
    "agadmator-library/agadmator-library.github.io/master"
)
CHESSGPT_DATASET = "Waterhorse/chess_data"


def ensure_dirs():
    """Create all data directories if they don't exist."""
    for d in [
        LIBRARY_DB_DIR, AUDIO_DIR, VOCALS_DIR, TRANSCRIPTS_DIR, PGN_DIR,
        ALIGNED_DIR, TTS_SEGMENTS_DIR, LLM_OUTPUT_DIR,
        TTS_OUTPUT_DIR, VIDEO_OUTPUT_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)
