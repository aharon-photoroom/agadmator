"""Step 0.2: Download audio from YouTube using yt-dlp."""

import json
import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from agadmator.config import RAW_DIR, AUDIO_DIR

log = logging.getLogger(__name__)

# Network-bound — safe to run many in parallel
DEFAULT_WORKERS = 8


def _env_with_deno() -> dict[str, str]:
    """Return env dict with deno on PATH if installed."""
    env = os.environ.copy()
    deno_bin = Path.home() / ".deno" / "bin"
    if deno_bin.is_dir() and str(deno_bin) not in env.get("PATH", ""):
        env["PATH"] = f"{deno_bin}:{env.get('PATH', '')}"
    return env


def download_single(video_id: str, output_dir: Path) -> Path | None:
    """Download audio for a single YouTube video."""
    output_path = output_dir / f"{video_id}.wav"
    if output_path.exists():
        return output_path

    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "-o", str(output_dir / f"{video_id}.%(ext)s"),
        "--no-playlist",
        "--remote-components", "ejs:github",
        "--remote-components", "ejs:npm",
        url,
    ]
    try:
        subprocess.run(
            cmd, check=True, capture_output=True, text=True, timeout=300,
            env=_env_with_deno(),
        )
        return output_path
    except FileNotFoundError:
        log.error("yt-dlp not found. Install with: pip install yt-dlp")
        return None
    except subprocess.CalledProcessError as e:
        log.error("Failed to download %s: %s", video_id, e.stderr[:200])
        return None
    except subprocess.TimeoutExpired:
        log.error("Timeout downloading %s", video_id)
        return None


def download_audio(
    metadata_path: str, limit: int | None = None, workers: int = DEFAULT_WORKERS
):
    """Download audio for all videos in metadata (concurrent)."""
    metadata_file = Path(metadata_path)
    if not metadata_file.exists():
        log.error("Metadata file not found: %s", metadata_file)
        return

    with open(metadata_file) as f:
        games = json.load(f)

    video_ids = [g["video_id"] for g in games if g.get("video_id")]
    if limit:
        video_ids = video_ids[:limit]

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    # Skip already downloaded
    existing = {f.stem for f in AUDIO_DIR.glob("*.wav")}
    todo = [v for v in video_ids if v not in existing]
    log.info(
        "Downloading audio: %d total, %d already done, %d to download (%d workers)",
        len(video_ids), len(existing), len(todo), workers,
    )

    if not todo:
        log.info("All files already downloaded.")
        return

    downloaded = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(download_single, vid, AUDIO_DIR): vid for vid in todo
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Downloading audio"
        ):
            if future.result():
                downloaded += 1
            else:
                failed += 1

    log.info("Downloaded: %d, Failed: %d, Already had: %d", downloaded, failed, len(existing))
