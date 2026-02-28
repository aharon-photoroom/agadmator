"""Step 4: Compose final video from board rendering + TTS audio."""

import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

# Final video layout: 1920x1080
# Board (720x720) centered left, info panel right
RESOLUTION = (1920, 1080)
BOARD_X = 120
BOARD_Y = (1080 - 720) // 2  # vertically centered


def compose_video(board_video: str, audio: str, output: str):
    """Compose final video: board video + audio track.

    Overlays the board video on a dark background and adds the audio track.
    Layout matches agadmator's style: large board on the left.
    """
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use ffmpeg to:
    # 1. Create a dark background at target resolution
    # 2. Overlay the board video
    # 3. Add audio track
    filter_complex = (
        f"color=c=0x1a1a2e:s={RESOLUTION[0]}x{RESOLUTION[1]}:r=30[bg];"
        f"[bg][0:v]overlay={BOARD_X}:{BOARD_Y}[v]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", board_video,
        "-i", audio,
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-map", "1:a",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        output,
    ]

    log.info("Composing final video...")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        log.info("Final video saved to %s", output)
    except subprocess.CalledProcessError as e:
        log.error("FFmpeg failed: %s", e.stderr[:500])
        raise


def compose_with_overlay(
    board_video: str,
    audio: str,
    overlay_image: str | None,
    output: str,
):
    """Compose video with optional overlay image (e.g., talking head placeholder).

    Places a small overlay in the bottom-right corner of the board area.
    """
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Overlay position: bottom-right of the board area
    overlay_size = 200
    overlay_x = BOARD_X + 720 - overlay_size - 10
    overlay_y = BOARD_Y + 720 - overlay_size - 10

    if overlay_image:
        filter_complex = (
            f"color=c=0x1a1a2e:s={RESOLUTION[0]}x{RESOLUTION[1]}:r=30[bg];"
            f"[2:v]scale={overlay_size}:{overlay_size}[ov];"
            f"[bg][0:v]overlay={BOARD_X}:{BOARD_Y}[tmp];"
            f"[tmp][ov]overlay={overlay_x}:{overlay_y}[v]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-i", board_video,
            "-i", audio,
            "-i", overlay_image,
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-map", "1:a",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            output,
        ]
    else:
        # No overlay, just board + audio
        compose_video(board_video, audio, output)
        return

    log.info("Composing video with overlay...")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        log.info("Final video saved to %s", output)
    except subprocess.CalledProcessError as e:
        log.error("FFmpeg failed: %s", e.stderr[:500])
        raise
