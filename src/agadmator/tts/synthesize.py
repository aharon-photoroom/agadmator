"""Generate speech from transcript using fine-tuned TTS model."""

import json
import logging
import subprocess
import tempfile
from pathlib import Path

from agadmator.config import TTS_OUTPUT_DIR

log = logging.getLogger(__name__)

# Maximum duration per segment (seconds) for reliable generation
MAX_SEGMENT_SECONDS = 45


def _split_into_paragraphs(text: str) -> list[str]:
    """Split transcript into natural paragraphs for TTS generation."""
    # Split on double newlines or sentence-ending periods followed by newlines
    paragraphs = []
    current = []

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            if current:
                paragraphs.append(" ".join(current))
                current = []
        else:
            current.append(line)

    if current:
        paragraphs.append(" ".join(current))

    # Further split paragraphs that are too long
    result = []
    for para in paragraphs:
        if len(para) > 500:  # Roughly 30-45 seconds of speech
            sentences = para.replace(". ", ".\n").split("\n")
            chunk = []
            chunk_len = 0
            for sent in sentences:
                if chunk_len + len(sent) > 500 and chunk:
                    result.append(" ".join(chunk))
                    chunk = []
                    chunk_len = 0
                chunk.append(sent)
                chunk_len += len(sent)
            if chunk:
                result.append(" ".join(chunk))
        else:
            result.append(para)

    return [p for p in result if p.strip()]


def _synthesize_segment(
    text: str,
    reference_audio: str,
    output_path: Path,
    model_path: str | None = None,
):
    """Synthesize a single text segment."""
    import sys
    cmd = [
        sys.executable, "-m", "fish_speech.inference",
        "--text", text,
        "--reference-audio", reference_audio,
        "--output", str(output_path),
    ]
    if model_path:
        cmd.extend(["--checkpoint", model_path])

    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except FileNotFoundError:
        raise RuntimeError(
            "Fish Speech not found. Install with: pip install fish-speech"
        )


def _concatenate_segments(segment_paths: list[Path], output_path: str):
    """Concatenate audio segments with natural pauses."""
    if not segment_paths:
        return

    # Create ffmpeg concat file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for path in segment_paths:
            f.write(f"file '{path}'\n")
        concat_file = f.name

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_file,
        "-c:a", "pcm_s16le",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    Path(concat_file).unlink()


def synthesize(transcript_file: str, reference_audio: str, output: str):
    """Generate full-length speech from a transcript file.

    Uses segment-based generation with audio-prefix conditioning
    for coherent long-form output.
    """
    with open(transcript_file) as f:
        text = f.read()

    paragraphs = _split_into_paragraphs(text)
    log.info("Split transcript into %d segments", len(paragraphs))

    # Check for fine-tuned model
    lora_path = TTS_OUTPUT_DIR / "fish_lora"
    model_path = str(lora_path) if lora_path.exists() else None

    segment_paths = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, para in enumerate(paragraphs):
            seg_path = Path(tmp_dir) / f"seg_{i:04d}.wav"
            log.info("Generating segment %d/%d (%d chars)...", i + 1, len(paragraphs), len(para))

            try:
                # Use end of previous segment as reference for continuity
                ref = (
                    str(segment_paths[-1]) if segment_paths
                    else reference_audio
                )
                _synthesize_segment(para, ref, seg_path, model_path)
                segment_paths.append(seg_path)
            except subprocess.CalledProcessError as e:
                log.error("Failed to generate segment %d: %s", i, e)
                continue

        # Concatenate all segments
        _concatenate_segments(segment_paths, output)

    log.info("Saved synthesized audio to %s", output)

    # Also save timestamps for board sync
    timestamps = []
    offset = 0.0
    for i, para in enumerate(paragraphs):
        # Estimate duration from text length (~150 words per minute)
        word_count = len(para.split())
        est_duration = word_count / 2.5  # seconds
        timestamps.append({
            "segment": i,
            "start": offset,
            "end": offset + est_duration,
            "text": para,
        })
        offset += est_duration

    ts_path = Path(output).with_suffix(".timestamps.json")
    with open(ts_path, "w") as f:
        json.dump(timestamps, f, indent=2)
    log.info("Saved timestamps to %s", ts_path)
