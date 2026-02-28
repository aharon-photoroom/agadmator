"""Step 0.6: Align PGN moves with transcripts.

Uses the agadmator-library's embedded Lichess evaluations when available,
falling back to local Stockfish analysis. Detects move mentions in the
transcript text and segments commentary around them.
"""

import io
import json
import logging
import re
from pathlib import Path

import chess
import chess.pgn

from agadmator.config import TRANSCRIPTS_DIR, PGN_DIR, ALIGNED_DIR, RAW_DIR

log = logging.getLogger(__name__)

# Patterns that match chess move mentions in natural language
MOVE_PATTERNS = [
    # Standard algebraic: e4, Nf3, Bxd5, O-O, O-O-O
    r"\b([KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)\b",
    # Castles
    r"\b(O-O(?:-O)?|castles?\s+(?:king|queen)\s*side)\b",
    # Descriptive: "knight to f3", "bishop takes d5", "pawn to e4"
    r"\b((?:knight|bishop|rook|queen|king|pawn)\s+(?:to|takes?|captures?)\s+[a-h][1-8])\b",
    # Spoken: "plays e4", "goes to d5"
    r"\b(?:plays?|moves?\s+to|goes?\s+to)\s+([a-h][1-8])\b",
]

COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in MOVE_PATTERNS]


def find_move_mentions(text: str) -> list[dict]:
    """Find all chess move mentions in text with their character positions."""
    mentions = []
    for pattern in COMPILED_PATTERNS:
        for match in pattern.finditer(text):
            mentions.append({
                "text": match.group(0),
                "move_text": match.group(1),
                "start_char": match.start(),
                "end_char": match.end(),
            })
    # Sort by position and deduplicate overlaps
    mentions.sort(key=lambda m: m["start_char"])
    deduped = []
    last_end = -1
    for m in mentions:
        if m["start_char"] >= last_end:
            deduped.append(m)
            last_end = m["end_char"]
    return deduped


def enrich_pgn_with_lichess_eval(
    pgn_text: str, lichess_eval: dict | None = None
) -> list[dict]:
    """Parse PGN and enrich each move with FEN + evaluation.

    Uses pre-computed Lichess analysis when available (from the
    agadmator-library metadata), avoiding expensive local Stockfish runs.
    """
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if not game:
        return []

    # Build eval lookup from Lichess analysis
    eval_by_ply = {}
    if lichess_eval and lichess_eval.get("analysis"):
        for i, entry in enumerate(lichess_eval["analysis"]):
            if entry.get("eval") is not None:
                eval_by_ply[i] = entry["eval"] / 100.0  # centipawns → pawns
            elif entry.get("mate") is not None:
                eval_by_ply[i] = f"M{entry['mate']}"

    enriched_moves = []
    board = game.board()
    ply = 0

    for node in game.mainline():
        move = node.move
        san = board.san(move)
        board.push(move)

        entry = {
            "ply": ply,
            "move_number": (ply // 2) + 1,
            "color": "white" if ply % 2 == 0 else "black",
            "san": san,
            "uci": move.uci(),
            "fen": board.fen(),
        }

        if ply in eval_by_ply:
            entry["eval"] = eval_by_ply[ply]

        enriched_moves.append(entry)
        ply += 1

    return enriched_moves


def enrich_pgn_with_stockfish(
    pgn_text: str, stockfish_path: str, depth: int = 20
) -> list[dict]:
    """Fallback: enrich PGN using local Stockfish engine."""
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if not game:
        return []

    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    except Exception as e:
        log.warning("Could not start Stockfish: %s", e)
        return enrich_pgn_with_lichess_eval(pgn_text)

    enriched_moves = []
    board = game.board()
    ply = 0

    for node in game.mainline():
        move = node.move
        san = board.san(move)
        board.push(move)

        entry = {
            "ply": ply,
            "move_number": (ply // 2) + 1,
            "color": "white" if ply % 2 == 0 else "black",
            "san": san,
            "uci": move.uci(),
            "fen": board.fen(),
        }

        try:
            info = engine.analyse(board, chess.engine.Limit(depth=depth))
            score = info["score"].white()
            if score.is_mate():
                entry["eval"] = f"M{score.mate()}"
            else:
                entry["eval"] = score.score() / 100.0
        except Exception:
            pass

        enriched_moves.append(entry)
        ply += 1

    engine.quit()
    return enriched_moves


def segment_commentary(
    full_text: str, move_mentions: list[dict]
) -> list[dict]:
    """Segment transcript text between consecutive move mentions."""
    segments = []
    prev_end = 0

    for mention in move_mentions:
        commentary = full_text[prev_end:mention["start_char"]].strip()
        if commentary:
            segments.append({
                "type": "commentary",
                "text": commentary,
                "before_move": mention["move_text"],
            })
        segments.append({
            "type": "move",
            "text": mention["text"],
            "move_text": mention["move_text"],
        })
        prev_end = mention["end_char"]

    trailing = full_text[prev_end:].strip()
    if trailing:
        segments.append({"type": "commentary", "text": trailing})

    return segments


def align_single(
    video_id: str,
    transcript_path: Path,
    pgn_path: Path,
    output_dir: Path,
    lichess_eval: dict | None = None,
) -> Path | None:
    """Align a single transcript with its PGN."""
    out_path = output_dir / f"{video_id}.json"
    if out_path.exists():
        return out_path

    with open(transcript_path) as f:
        transcript = json.load(f)
    with open(pgn_path) as f:
        pgn_text = f.read()

    full_text = transcript.get("full_text", "")
    if not full_text:
        return None

    mentions = find_move_mentions(full_text)
    if not mentions:
        log.debug("No move mentions found in %s", video_id)
        return None

    # Enrich PGN — prefer Lichess evals over local Stockfish
    enriched_moves = enrich_pgn_with_lichess_eval(pgn_text, lichess_eval)
    segments = segment_commentary(full_text, mentions)

    result = {
        "video_id": video_id,
        "enriched_moves": enriched_moves,
        "transcript_segments": segments,
        "move_mentions_count": len(mentions),
        "total_moves": len(enriched_moves),
        "full_text": full_text,
        "has_evals": any("eval" in m for m in enriched_moves),
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info(
        "Aligned %s: %d mentions / %d moves (evals: %s)",
        video_id, len(mentions), len(enriched_moves), result["has_evals"],
    )
    return out_path


def align_all(transcripts_dir: str | None = None, pgn_dir: str | None = None):
    """Align all transcript-PGN pairs."""
    t_dir = Path(transcripts_dir) if transcripts_dir else TRANSCRIPTS_DIR
    p_dir = Path(pgn_dir) if pgn_dir else PGN_DIR
    ALIGNED_DIR.mkdir(parents=True, exist_ok=True)

    # Load Lichess evals if available
    evals_path = RAW_DIR / "lichess_evals.json"
    lichess_evals = {}
    if evals_path.exists():
        with open(evals_path) as f:
            raw_evals = json.load(f)
        # Each video can have multiple evals; use the first
        for vid, evals_list in raw_evals.items():
            if evals_list:
                lichess_evals[vid] = evals_list[0]
        log.info("Loaded Lichess evals for %d videos", len(lichess_evals))

    transcript_files = {f.stem: f for f in t_dir.glob("*.json")}
    pgn_files = {f.stem: f for f in p_dir.glob("*.pgn")}
    common = set(transcript_files.keys()) & set(pgn_files.keys())
    log.info("Found %d matching transcript-PGN pairs", len(common))

    aligned = 0
    with_evals = 0
    for stem in sorted(common):
        result = align_single(
            stem,
            transcript_files[stem],
            pgn_files[stem],
            ALIGNED_DIR,
            lichess_eval=lichess_evals.get(stem),
        )
        if result:
            aligned += 1
            if lichess_evals.get(stem):
                with_evals += 1

    log.info(
        "Aligned %d/%d pairs (%d with Lichess evals)",
        aligned, len(common), with_evals,
    )
