"""Full end-to-end pipeline: PGN → agadmator-style video."""

import json
import logging
import tempfile
from pathlib import Path

from agadmator.config import VIDEO_OUTPUT_DIR

log = logging.getLogger(__name__)


def full_pipeline(
    pgn_file: str,
    llm_model: str,
    reference_audio: str,
    output: str,
):
    """Generate a complete agadmator-style video from a PGN file.

    Steps:
        1. Generate commentary transcript (LLM)
        2. Synthesize speech (TTS)
        3. Render board video
        4. Compose final video
    """
    VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pgn_path = Path(pgn_file)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        # Step 1: Generate commentary
        log.info("Step 1/4: Generating commentary...")
        from agadmator.llm.generate import generate_commentary
        transcript_path = tmp / "commentary.txt"
        generate_commentary(pgn_file, llm_model, str(transcript_path))

        # Step 2: Synthesize speech
        log.info("Step 2/4: Synthesizing speech...")
        from agadmator.tts.synthesize import synthesize
        audio_path = tmp / "speech.wav"
        synthesize(str(transcript_path), reference_audio, str(audio_path))

        # Load timestamps for board sync
        ts_path = tmp / "speech.timestamps.json"
        if ts_path.exists():
            with open(ts_path) as f:
                timestamps = json.load(f)
        else:
            timestamps = [{"segment": 0, "start": 0, "end": 600}]

        # Step 3: Render board
        log.info("Step 3/4: Rendering board...")
        from agadmator.render.board import render_board_video
        board_video_path = tmp / "board.mp4"
        # Write timestamps to a temp file for the renderer
        ts_file = tmp / "timestamps.json"
        with open(ts_file, "w") as f:
            json.dump(timestamps, f)
        render_board_video(pgn_file, str(ts_file))
        # render_board_video saves next to the PGN; move it
        rendered = pgn_path.with_suffix(".board.mp4")
        if rendered.exists():
            rendered.rename(board_video_path)

        # Step 4: Compose
        log.info("Step 4/4: Composing final video...")
        from agadmator.compose.video import compose_video
        compose_video(str(board_video_path), str(audio_path), output)

    log.info("Pipeline complete! Video saved to %s", output)
