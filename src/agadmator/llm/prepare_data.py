"""Prepare training data for LLM fine-tuning."""

import io
import json
import logging
from pathlib import Path

import chess
import chess.pgn

from agadmator.config import ALIGNED_DIR, PROCESSED_DIR, CHESSGPT_DATASET

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are agadmator, a chess commentator who recaps games in an engaging, "
    "conversational style. You explain moves clearly, tell the story of the game, "
    "and use your signature phrases naturally."
)


def format_enriched_pgn(aligned: dict) -> str:
    """Format enriched PGN as model input."""
    moves = aligned.get("enriched_moves", [])
    if not moves:
        return ""

    # Build header from first move's context
    lines = []
    # Try to reconstruct PGN headers from the data
    lines.append(f"[White \"{aligned.get('white', 'White')}\"]")
    lines.append(f"[Black \"{aligned.get('black', 'Black')}\"]")
    if aligned.get("event"):
        lines.append(f"[Event \"{aligned['event']}\"]")
    lines.append("")

    # Format moves with annotations
    move_strs = []
    for m in moves:
        prefix = f"{m['move_number']}. " if m["color"] == "white" else ""
        eval_str = ""
        if "eval" in m:
            eval_str = f" {{eval: {m['eval']}}}"
        fen_short = m["fen"].split(" ")[0]  # Just piece placement
        move_strs.append(f"{prefix}{m['san']}{eval_str} {{FEN: {fen_short}}}")

    lines.append(" ".join(move_strs))
    return "\n".join(lines)


def make_conversation(pgn_input: str, transcript: str) -> dict:
    """Create a conversation in the training format."""
    return {
        "conversations": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": pgn_input},
            {"role": "assistant", "content": transcript},
        ]
    }


def prepare_agadmator_pairs() -> list[dict]:
    """Prepare training pairs from aligned agadmator data."""
    pairs = []
    for f in sorted(ALIGNED_DIR.glob("*.json")):
        with open(f) as fh:
            aligned = json.load(fh)

        pgn_input = format_enriched_pgn(aligned)
        transcript = aligned.get("full_text", "")
        if not pgn_input or not transcript:
            continue

        pairs.append(make_conversation(pgn_input, transcript))

    log.info("Prepared %d agadmator training pairs", len(pairs))
    return pairs


def prepare_chessgpt_data() -> list[dict]:
    """Prepare domain pre-training data from ChessGPT dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        log.error("Install 'datasets' package: pip install datasets")
        return []

    log.info("Loading ChessGPT dataset from HuggingFace...")
    ds = load_dataset(CHESSGPT_DATASET, split="train")

    pairs = []
    for item in ds:
        # ChessGPT has various formats; extract move-commentary pairs
        pgn = item.get("input", "") or item.get("pgn", "")
        commentary = item.get("output", "") or item.get("commentary", "")
        if not pgn or not commentary:
            continue

        pairs.append(make_conversation(pgn, commentary))

    log.info("Prepared %d ChessGPT training pairs", len(pairs))
    return pairs


def prepare_llm_data():
    """Prepare all LLM training data and save as JSONL files."""
    output_dir = PROCESSED_DIR / "llm_training"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: ChessGPT domain pre-training data
    chessgpt_pairs = prepare_chessgpt_data()
    chessgpt_path = output_dir / "chessgpt_train.jsonl"
    with open(chessgpt_path, "w") as f:
        for pair in chessgpt_pairs:
            f.write(json.dumps(pair) + "\n")
    log.info("Saved %d ChessGPT pairs to %s", len(chessgpt_pairs), chessgpt_path)

    # Phase 2: Agadmator style transfer data
    agad_pairs = prepare_agadmator_pairs()

    # Split 90/10 for train/eval
    split_idx = max(1, int(len(agad_pairs) * 0.9))
    train_pairs = agad_pairs[:split_idx]
    eval_pairs = agad_pairs[split_idx:]

    train_path = output_dir / "agadmator_train.jsonl"
    eval_path = output_dir / "agadmator_eval.jsonl"

    with open(train_path, "w") as f:
        for pair in train_pairs:
            f.write(json.dumps(pair) + "\n")
    with open(eval_path, "w") as f:
        for pair in eval_pairs:
            f.write(json.dumps(pair) + "\n")

    log.info(
        "Saved %d train / %d eval agadmator pairs",
        len(train_pairs), len(eval_pairs),
    )
