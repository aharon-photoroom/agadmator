"""Step 3: Programmatic chess board rendering."""

import io
import json
import logging
import math
import subprocess
import tempfile
from pathlib import Path

import chess
import chess.pgn
from PIL import Image, ImageDraw, ImageFont

from agadmator.render.pieces import get_piece_image

log = logging.getLogger(__name__)

_LABEL_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
    "C:/Windows/Fonts/arial.ttf",
]


def _find_label_font(size: int) -> ImageFont.FreeTypeFont:
    for path in _LABEL_FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


# Default board appearance
BOARD_SIZE = 720
SQUARE_SIZE = BOARD_SIZE // 8
LIGHT_COLOR = (240, 217, 181)   # #F0D9B5
DARK_COLOR = (181, 136, 99)     # #B58863
HIGHLIGHT_COLOR = (255, 255, 0, 100)  # Semi-transparent yellow
MOVE_FROM_COLOR = (100, 180, 100, 120)
MOVE_TO_COLOR = (100, 200, 100, 150)
FPS = 30
ANIMATION_FRAMES = 10


def _square_to_pixel(square: int, flipped: bool = False) -> tuple[int, int]:
    """Convert chess square index to pixel coordinates (top-left of square)."""
    col = chess.square_file(square)
    row = chess.square_rank(square)
    if flipped:
        col = 7 - col
        row = 7 - row
    else:
        row = 7 - row
    return col * SQUARE_SIZE, row * SQUARE_SIZE


def draw_empty_board() -> Image.Image:
    """Draw an empty chess board."""
    img = Image.new("RGB", (BOARD_SIZE, BOARD_SIZE))
    draw = ImageDraw.Draw(img)

    for rank in range(8):
        for file in range(8):
            x = file * SQUARE_SIZE
            y = rank * SQUARE_SIZE
            color = LIGHT_COLOR if (rank + file) % 2 == 0 else DARK_COLOR
            draw.rectangle([x, y, x + SQUARE_SIZE, y + SQUARE_SIZE], fill=color)

    # Draw file/rank labels
    font = _find_label_font(14)


    for i in range(8):
        # File labels (a-h) at bottom
        label = chr(ord("a") + i)
        lx = i * SQUARE_SIZE + SQUARE_SIZE - 14
        ly = BOARD_SIZE - 16
        color = DARK_COLOR if i % 2 == 0 else LIGHT_COLOR
        draw.text((lx, ly), label, fill=color, font=font)

        # Rank labels (1-8) at left
        label = str(8 - i)
        lx = 2
        ly = i * SQUARE_SIZE + 2
        color = DARK_COLOR if i % 2 == 0 else LIGHT_COLOR
        draw.text((lx, ly), label, fill=color, font=font)

    return img


def draw_board_position(
    board: chess.Board,
    last_move: chess.Move | None = None,
    flipped: bool = False,
) -> Image.Image:
    """Draw a board with pieces at the current position."""
    img = draw_empty_board()

    # Highlight last move squares
    if last_move:
        overlay = Image.new("RGBA", (BOARD_SIZE, BOARD_SIZE), (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)

        fx, fy = _square_to_pixel(last_move.from_square, flipped)
        overlay_draw.rectangle(
            [fx, fy, fx + SQUARE_SIZE, fy + SQUARE_SIZE],
            fill=MOVE_FROM_COLOR,
        )
        tx, ty = _square_to_pixel(last_move.to_square, flipped)
        overlay_draw.rectangle(
            [tx, ty, tx + SQUARE_SIZE, ty + SQUARE_SIZE],
            fill=MOVE_TO_COLOR,
        )

        img = img.convert("RGBA")
        img = Image.alpha_composite(img, overlay)
        img = img.convert("RGB")

    # Draw pieces
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue

        px, py = _square_to_pixel(square, flipped)
        piece_img = get_piece_image(piece, SQUARE_SIZE)
        if piece_img:
            img.paste(piece_img, (px, py), piece_img)

    return img


def draw_animated_move(
    board_before: chess.Board,
    move: chess.Move,
    n_frames: int = ANIMATION_FRAMES,
    flipped: bool = False,
) -> list[Image.Image]:
    """Generate frames for a move animation."""
    frames = []

    from_sq = move.from_square
    to_sq = move.to_square
    piece = board_before.piece_at(from_sq)

    # Board without the moving piece
    temp_board = board_before.copy()
    temp_board.remove_piece_at(from_sq)

    from_x, from_y = _square_to_pixel(from_sq, flipped)
    to_x, to_y = _square_to_pixel(to_sq, flipped)

    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)
        # Ease-in-out interpolation
        t = t * t * (3 - 2 * t)

        # Draw board without moving piece
        img = draw_board_position(temp_board, flipped=flipped)

        # Draw piece at interpolated position
        if piece:
            cx = from_x + (to_x - from_x) * t
            cy = from_y + (to_y - from_y) * t
            piece_img = get_piece_image(piece, SQUARE_SIZE)
            if piece_img:
                img.paste(piece_img, (int(cx), int(cy)), piece_img)

        frames.append(img)

    return frames


def render_game_frames(
    pgn_path: str,
    move_timestamps: list[dict],
) -> list[tuple[Image.Image, float]]:
    """Render all frames for a game synchronized with audio timestamps.

    Args:
        pgn_path: Path to PGN file.
        move_timestamps: List of {move_index, timestamp} dicts indicating
            when each move should appear in the video.

    Returns:
        List of (frame_image, timestamp) tuples.
    """
    with open(pgn_path) as f:
        game = chess.pgn.read_game(io.StringIO(f.read()))

    if not game:
        raise ValueError(f"Could not parse PGN: {pgn_path}")

    board = game.board()
    moves = list(game.mainline_moves())

    # Build timestamp lookup: move_index → timestamp
    ts_map = {}
    for entry in move_timestamps:
        ts_map[entry["move_index"]] = entry["timestamp"]

    frames = []

    # Initial position
    start_time = 0.0
    first_move_time = ts_map.get(0, 5.0)
    initial_frame = draw_board_position(board)

    # Hold initial position until first move
    n_hold = int(first_move_time * FPS)
    for i in range(n_hold):
        frames.append((initial_frame, start_time + i / FPS))

    # Render each move
    for i, move in enumerate(moves):
        move_time = ts_map.get(i, first_move_time + i * 3.0)
        next_time = ts_map.get(i + 1, move_time + 3.0)

        # Animation frames
        anim_frames = draw_animated_move(board, move)
        anim_duration = 0.4  # seconds for animation
        for j, anim_frame in enumerate(anim_frames):
            t = move_time + j * (anim_duration / len(anim_frames))
            frames.append((anim_frame, t))

        # Apply the move
        board.push(move)

        # Static frame after move (hold until next move)
        static = draw_board_position(board, last_move=move)
        hold_start = move_time + anim_duration
        hold_end = next_time
        n_hold = max(1, int((hold_end - hold_start) * FPS))
        for j in range(n_hold):
            frames.append((static, hold_start + j / FPS))

    return frames


def render_board_video(pgn_file: str, timestamps_file: str):
    """Render a full board video from PGN + timestamps."""
    with open(timestamps_file) as f:
        timestamps_data = json.load(f)

    # Parse timestamps: map move mentions in transcript to move indices
    move_timestamps = _build_move_timestamps(pgn_file, timestamps_data)

    log.info("Rendering board video...")
    frames = render_game_frames(pgn_file, move_timestamps)
    log.info("Generated %d frames", len(frames))

    # Encode frames to video using ffmpeg
    output_path = Path(pgn_file).with_suffix(".board.mp4")
    _encode_frames_to_video(frames, str(output_path))
    log.info("Board video saved to %s", output_path)


def _build_move_timestamps(
    pgn_path: str, timestamps_data: list[dict]
) -> list[dict]:
    """Map transcript timestamps to move indices."""
    # Simple approach: evenly distribute moves across the audio duration
    with open(pgn_path) as f:
        game = chess.pgn.read_game(io.StringIO(f.read()))

    n_moves = sum(1 for _ in game.mainline_moves())
    total_duration = max(
        (ts.get("end", 0) for ts in timestamps_data), default=600
    )

    # Reserve first 10% for intro, distribute rest evenly
    intro_time = total_duration * 0.1
    move_interval = (total_duration - intro_time) / max(n_moves, 1)

    return [
        {"move_index": i, "timestamp": intro_time + i * move_interval}
        for i in range(n_moves)
    ]


def _encode_frames_to_video(
    frames: list[tuple[Image.Image, float]], output_path: str
):
    """Encode PIL frames to an MP4 video using ffmpeg."""
    if not frames:
        return

    width, height = frames[0][0].size

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-pix_fmt", "rgb24",
        "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        output_path,
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for img, _ in frames:
        proc.stdin.write(img.convert("RGB").tobytes())
    proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        stderr = proc.stderr.read().decode(errors="replace")
        raise RuntimeError(f"ffmpeg exited with code {rc}: {stderr[-500:]}")
