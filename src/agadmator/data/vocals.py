"""Step 0.3: Voice isolation using Demucs."""

import logging
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from agadmator.config import AUDIO_DIR, VOCALS_DIR

log = logging.getLogger(__name__)


def _detect_device() -> str:
    """Detect best available device for Demucs."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _gpu_parallel_slots() -> int:
    """How many Demucs processes can run in parallel on the GPU.

    Demucs htdemucs uses ~3-4 GB VRAM per process.
    """
    try:
        import torch
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
            return max(1, int(vram_gb // 5))  # ~5GB headroom per slot
    except ImportError:
        pass
    return 1


def _isolate_one(args: tuple) -> tuple[str, bool]:
    """Worker function for parallel vocal isolation."""
    audio_path, output_dir, device = args
    stem = Path(audio_path).stem
    vocal_path = Path(output_dir) / f"{stem}.wav"
    if vocal_path.exists():
        return stem, True

    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems=vocals",
        "--device", device,
        "-o", str(Path(output_dir) / "_demucs_tmp"),
        str(audio_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=1800)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return stem, False

    for model_name in ["htdemucs", "htdemucs_ft", "mdx_extra"]:
        candidate = Path(output_dir) / "_demucs_tmp" / model_name / stem / "vocals.wav"
        if candidate.exists():
            candidate.rename(vocal_path)
            return stem, True

    return stem, False


def isolate_vocals(input_dir: str | None = None, output_dir: str | None = None):
    """Isolate vocals for all audio files (parallel on GPU)."""
    in_path = Path(input_dir) if input_dir else AUDIO_DIR
    out_path = Path(output_dir) if output_dir else VOCALS_DIR
    out_path.mkdir(parents=True, exist_ok=True)

    device = _detect_device()
    workers = _gpu_parallel_slots() if device == "cuda" else 1
    log.info("Device: %s, parallel workers: %d", device, workers)

    audio_files = sorted(in_path.glob("*.wav"))
    existing = {f.stem for f in out_path.glob("*.wav")}
    todo = [f for f in audio_files if f.stem not in existing]
    log.info("Demucs: %d total, %d done, %d to process", len(audio_files), len(existing), len(todo))

    if not todo:
        return

    args_list = [(str(f), str(out_path), device) for f in todo]
    done = 0

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_isolate_one, a): a[0] for a in args_list}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Isolating vocals"):
                stem, ok = future.result()
                if ok:
                    done += 1
                else:
                    log.error("Failed: %s", stem)
    else:
        for a in tqdm(args_list, desc="Isolating vocals"):
            stem, ok = _isolate_one(a)
            if ok:
                done += 1
            else:
                log.error("Failed: %s", stem)

    log.info("Isolated vocals for %d/%d files", done + len(existing), len(audio_files))
