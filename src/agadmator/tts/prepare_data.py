"""Prepare TTS training data: segment vocals into clips with transcripts."""

import json
import logging
import subprocess
from pathlib import Path

from tqdm import tqdm

from agadmator.config import VOCALS_DIR, TRANSCRIPTS_DIR, TTS_SEGMENTS_DIR

log = logging.getLogger(__name__)

MIN_DURATION = 3.0   # seconds
MAX_DURATION = 15.0  # seconds
TARGET_LUFS = -23
SAMPLE_RATE = 44100


def _get_audio_duration(path: Path) -> float:
    """Get audio duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


def _extract_segment(
    input_path: Path, output_path: Path, start: float, end: float
):
    """Extract an audio segment and normalize loudness."""
    duration = end - start
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-t", str(duration),
        "-i", str(input_path),
        "-ar", str(SAMPLE_RATE),
        "-ac", "1",
        "-af", f"loudnorm=I={TARGET_LUFS}:TP=-1.5:LRA=11",
        str(output_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def segment_audio_with_transcript(
    vocal_path: Path,
    transcript_path: Path,
    output_dir: Path,
) -> list[dict]:
    """Segment a vocal file into TTS training clips."""
    with open(transcript_path) as f:
        transcript = json.load(f)

    segments = transcript.get("segments", [])
    if not segments:
        return []

    stem = vocal_path.stem
    clips = []

    # Group consecutive segments into clips of 3-15 seconds
    current_start = None
    current_end = None
    current_text = []

    for seg in segments:
        seg_start = seg["start"]
        seg_end = seg["end"]
        seg_text = seg["text"].strip()
        if not seg_text:
            continue

        if current_start is None:
            current_start = seg_start
            current_end = seg_end
            current_text = [seg_text]
            continue

        proposed_duration = seg_end - current_start

        if proposed_duration <= MAX_DURATION:
            current_end = seg_end
            current_text.append(seg_text)
        else:
            # Flush current clip if it meets minimum duration
            duration = current_end - current_start
            if duration >= MIN_DURATION:
                clips.append({
                    "start": current_start,
                    "end": current_end,
                    "text": " ".join(current_text),
                })
            # Start new clip
            current_start = seg_start
            current_end = seg_end
            current_text = [seg_text]

    # Flush last clip
    if current_start is not None:
        duration = current_end - current_start
        if duration >= MIN_DURATION:
            clips.append({
                "start": current_start,
                "end": current_end,
                "text": " ".join(current_text),
            })

    # Extract audio segments
    extracted = []
    for i, clip in enumerate(clips):
        clip_name = f"{stem}_{i:04d}"
        wav_path = output_dir / f"{clip_name}.wav"
        lab_path = output_dir / f"{clip_name}.lab"

        try:
            _extract_segment(vocal_path, wav_path, clip["start"], clip["end"])
        except subprocess.CalledProcessError:
            log.warning("Failed to extract %s", clip_name)
            continue

        # Write text label file (Fish Speech format)
        with open(lab_path, "w") as f:
            f.write(clip["text"])

        extracted.append({
            "name": clip_name,
            "wav": str(wav_path),
            "lab": str(lab_path),
            "text": clip["text"],
            "duration": clip["end"] - clip["start"],
        })

    return extracted


def prepare_tts_data():
    """Prepare TTS training data from all vocal files."""
    TTS_SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)

    vocal_files = {f.stem: f for f in VOCALS_DIR.glob("*.wav")}
    transcript_files = {f.stem: f for f in TRANSCRIPTS_DIR.glob("*.json")}
    common = set(vocal_files.keys()) & set(transcript_files.keys())

    log.info("Preparing TTS data for %d files...", len(common))

    all_clips = []
    total_duration = 0.0

    for stem in tqdm(sorted(common), desc="Segmenting"):
        clips = segment_audio_with_transcript(
            vocal_files[stem], transcript_files[stem], TTS_SEGMENTS_DIR
        )
        all_clips.extend(clips)
        total_duration += sum(c["duration"] for c in clips)

    # Save manifest
    manifest_path = TTS_SEGMENTS_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(all_clips, f, indent=2)

    log.info(
        "Prepared %d clips, %.1f hours total",
        len(all_clips), total_duration / 3600,
    )
