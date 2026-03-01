"""Generate commentary from PGN using the fine-tuned LLM."""

import io
import logging
from pathlib import Path

import chess
import chess.pgn
import torch

from agadmator.llm.prepare_data import SYSTEM_PROMPT

log = logging.getLogger(__name__)


def enrich_pgn_for_inference(pgn_path: str) -> str:
    """Read a PGN file and produce enriched input for the model."""
    with open(pgn_path) as f:
        pgn_text = f.read()

    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if not game:
        raise ValueError(f"Could not parse PGN: {pgn_path}")

    headers = game.headers
    lines = []
    for key in ["Event", "Site", "Date", "White", "Black", "Result"]:
        if key in headers:
            lines.append(f'[{key} "{headers[key]}"]')
    lines.append("")

    board = game.board()
    move_strs = []
    move_num = 0
    for node in game.mainline():
        move = node.move
        san = board.san(move)
        board.push(move)
        move_num += 1
        prefix = f"{(move_num + 1) // 2}. " if move_num % 2 == 1 else ""
        fen_short = board.fen().split(" ")[0]
        move_strs.append(f"{prefix}{san} {{FEN: {fen_short}}}")

    lines.append(" ".join(move_strs))
    return "\n".join(lines)


def generate_commentary(
    pgn_file: str, model_path: str, output_path: str | None = None
):
    """Generate commentary transcript from a PGN file."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU required for LLM inference. "
            "Run this on a machine with an NVIDIA GPU."
        )

    from unsloth import FastLanguageModel

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    # Skip quantization on large-VRAM GPUs (7B bf16 ≈ 14GB)
    use_4bit = vram_gb < 30

    log.info("Loading model from %s (4bit=%s, %.0fGB VRAM)", model_path, use_4bit, vram_gb)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=8192 if vram_gb >= 70 else 4096,
        dtype=None,
        load_in_4bit=use_4bit,
    )
    FastLanguageModel.for_inference(model)

    pgn_input = enrich_pgn_for_inference(pgn_file)

    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{pgn_input}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    log.info("Generating commentary...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=4096,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True,
    )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=False)
    # Extract assistant response
    assistant_start = generated.rfind("<|im_start|>assistant\n")
    if assistant_start != -1:
        commentary = generated[assistant_start + len("<|im_start|>assistant\n"):]
        commentary = commentary.split("<|im_end|>")[0].strip()
    else:
        commentary = generated[len(prompt):].strip()

    if output_path:
        out = Path(output_path)
    else:
        out = Path(pgn_file).with_suffix(".txt")

    with open(out, "w") as f:
        f.write(commentary)
    log.info("Saved commentary to %s (%d chars)", out, len(commentary))

    return commentary
