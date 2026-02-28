"""Step 0.5: Extract and validate PGN from metadata.

Most games already have PGN embedded in the agadmator-library metadata
(videoGame[].pgn). This module extracts them, validates with python-chess,
and falls back to Lichess for any missing ones.
"""

import io
import json
import logging
from pathlib import Path

import chess
import chess.pgn
import requests

from agadmator.config import RAW_DIR, PGN_DIR

log = logging.getLogger(__name__)


def validate_pgn(pgn_text: str) -> chess.pgn.Game | None:
    """Parse and validate a PGN string by replaying all moves."""
    try:
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        if game is None:
            return None
        board = game.board()
        for move in game.mainline_moves():
            if move not in board.legal_moves:
                return None
            board.push(move)
        return game
    except Exception:
        return None


def _pgn_text_to_full(pgn_moves: str, meta: dict) -> str:
    """Wrap bare move text with PGN headers from metadata."""
    games = meta.get("games", [])
    game_info = games[0] if games else {}

    headers = []
    if meta.get("event"):
        headers.append(f'[Event "{meta["event"]}"]')
    if game_info.get("white"):
        headers.append(f'[White "{game_info["white"]}"]')
    if game_info.get("black"):
        headers.append(f'[Black "{game_info["black"]}"]')
    if game_info.get("date"):
        headers.append(f'[Date "{game_info["date"]}"]')
    if meta.get("result"):
        headers.append(f'[Result "{meta["result"]}"]')
    if meta.get("eco"):
        headers.append(f'[ECO "{meta["eco"]}"]')

    if headers:
        return "\n".join(headers) + "\n\n" + pgn_moves
    return pgn_moves


def _fetch_from_lichess(lichess_id: str) -> str | None:
    """Fetch PGN from Lichess by game ID."""
    url = f"https://lichess.org/game/export/{lichess_id}"
    try:
        resp = requests.get(
            url,
            headers={"Accept": "application/x-chess-pgn"},
            timeout=15,
        )
        if resp.status_code == 200:
            return resp.text
    except Exception:
        pass
    return None


def extract_single(video_meta: dict, output_dir: Path) -> Path | None:
    """Extract/validate PGN for a single video."""
    video_id = video_meta.get("video_id", "")
    if not video_id:
        return None

    out_path = output_dir / f"{video_id}.pgn"
    if out_path.exists():
        return out_path

    games = video_meta.get("games", [])
    if not games:
        return None

    # Try embedded PGN first (most common case)
    for game_info in games:
        pgn_moves = game_info.get("pgn", "").strip()
        if not pgn_moves:
            continue

        full_pgn = _pgn_text_to_full(pgn_moves, video_meta)
        game = validate_pgn(full_pgn)
        if game:
            with open(out_path, "w") as f:
                print(game, file=f, end="\n\n")
            return out_path

    # Fallback: fetch from Lichess if we have an ID
    for lic_id in video_meta.get("lichess_ids", []):
        pgn_text = _fetch_from_lichess(lic_id)
        if pgn_text:
            game = validate_pgn(pgn_text)
            if game:
                with open(out_path, "w") as f:
                    f.write(pgn_text)
                log.info("Fetched PGN from Lichess for %s", video_id)
                return out_path

    return None


def collect_pgn(metadata_path: str):
    """Extract/collect PGN files for all games in metadata."""
    meta_file = Path(metadata_path)
    if not meta_file.exists():
        log.error("Metadata file not found: %s", meta_file)
        return

    with open(meta_file) as f:
        videos = json.load(f)

    PGN_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Extracting PGN for %d videos...", len(videos))

    from_embedded = 0
    from_lichess = 0
    failed = 0

    for v in videos:
        result = extract_single(v, PGN_DIR)
        if result:
            # Check if it came from embedded or Lichess
            has_embedded = any(g.get("pgn") for g in v.get("games", []))
            if has_embedded:
                from_embedded += 1
            else:
                from_lichess += 1
        else:
            failed += 1

    log.info(
        "PGN results: %d from metadata, %d from Lichess, %d failed",
        from_embedded, from_lichess, failed,
    )
