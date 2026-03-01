"""Step 0.4: Transcribe audio with faster-whisper."""

import json
import logging
from pathlib import Path

from tqdm import tqdm

from agadmator.config import VOCALS_DIR, TRANSCRIPTS_DIR

log = logging.getLogger(__name__)


def _detect_whisper_settings() -> tuple[str, str, int]:
    """Detect optimal device, compute type, and batch size.

    Returns (device, compute_type, batch_size).
    """
    try:
        import torch
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            gpu_name = torch.cuda.get_device_name(0)
            log.info("GPU: %s (%.0f GB)", gpu_name, vram_gb)

            # Whisper large-v3 in float16 uses ~3GB VRAM
            # Each batch slot uses ~0.5GB additional
            batch_size = min(32, max(1, int((vram_gb - 4) / 0.5)))
            return "cuda", "float16", batch_size
    except ImportError:
        pass
    return "auto", "auto", 1


def transcribe_single(
    audio_path: Path, output_dir: Path, model, batch_size: int
) -> Path | None:
    """Transcribe a single audio file."""
    stem = audio_path.stem
    out_path = output_dir / f"{stem}.json"
    if out_path.exists():
        return out_path

    if batch_size > 1:
        # Use batched pipeline for GPU acceleration
        from faster_whisper import BatchedInferencePipeline
        pipeline = BatchedInferencePipeline(model=model)
        segments_raw, info = pipeline.transcribe(
            str(audio_path),
            batch_size=batch_size,
            word_timestamps=True,
            language="en",
        )
    else:
        segments_raw, info = model.transcribe(
            str(audio_path),
            beam_size=5,
            word_timestamps=True,
            language="en",
        )

    segments = []
    for seg in segments_raw:
        seg_data = {
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
            "words": [],
        }
        if seg.words:
            seg_data["words"] = [
                {
                    "word": w.word.strip(),
                    "start": w.start,
                    "end": w.end,
                    "probability": round(w.probability, 3),
                }
                for w in seg.words
            ]
        segments.append(seg_data)

    result = {
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "duration": info.duration,
        "segments": segments,
        "full_text": " ".join(s["text"] for s in segments),
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info("Transcribed: %s (%.1f min)", stem, info.duration / 60)
    return out_path


def transcribe_all(input_dir: str | None = None, model_size: str = "large-v3"):
    """Transcribe all vocal files."""
    from faster_whisper import WhisperModel

    in_path = Path(input_dir) if input_dir else VOCALS_DIR
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    device, compute_type, batch_size = _detect_whisper_settings()
    log.info("Whisper: device=%s, compute=%s, batch_size=%d", device, compute_type, batch_size)

    audio_files = sorted(in_path.glob("*.wav"))
    existing = {f.stem for f in TRANSCRIPTS_DIR.glob("*.json")}
    todo = [f for f in audio_files if f.stem not in existing]
    log.info("Transcribe: %d total, %d done, %d to process", len(audio_files), len(existing), len(todo))

    if not todo:
        return

    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    done = 0
    for f in tqdm(todo, desc="Transcribing"):
        if transcribe_single(f, TRANSCRIPTS_DIR, model, batch_size):
            done += 1

    log.info("Transcribed %d/%d files", done + len(existing), len(audio_files))
