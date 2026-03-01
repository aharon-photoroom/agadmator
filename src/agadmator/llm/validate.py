"""Generate validation samples from eval PGNs at various training stages."""

import json
import logging
from datetime import datetime
from pathlib import Path

import torch

from agadmator.config import PROCESSED_DIR, LLM_OUTPUT_DIR
from agadmator.llm.prepare_data import SYSTEM_PROMPT

log = logging.getLogger(__name__)

SAMPLES_DIR = LLM_OUTPUT_DIR / "validation_samples"


def _generate_from_eval_pair(model, tokenizer, pgn_input: str) -> str:
    """Generate commentary from a single eval PGN input."""
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{pgn_input}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=False)
    assistant_start = generated.rfind("<|im_start|>assistant\n")
    if assistant_start != -1:
        commentary = generated[assistant_start + len("<|im_start|>assistant\n"):]
        commentary = commentary.split("<|im_end|>")[0].strip()
    else:
        commentary = generated[len(prompt):].strip()

    return commentary


def generate_validation_samples(
    model, tokenizer, phase: str, step: str, num_samples: int = 5
):
    """Generate commentary for eval PGNs and save alongside ground truth.

    Args:
        model: The loaded model (already in inference mode).
        tokenizer: The tokenizer.
        phase: "pretrain" or "style".
        step: Label for this checkpoint (e.g. "epoch_1", "final").
        num_samples: How many eval examples to generate.
    """
    eval_path = PROCESSED_DIR / "llm_training" / "agadmator_eval.jsonl"
    if not eval_path.exists():
        log.warning("No eval data found at %s, skipping validation", eval_path)
        return

    # Load eval pairs
    eval_pairs = []
    with open(eval_path) as f:
        for line in f:
            eval_pairs.append(json.loads(line))

    if not eval_pairs:
        log.warning("Eval file is empty, skipping validation")
        return

    num_samples = min(num_samples, len(eval_pairs))
    samples_dir = SAMPLES_DIR / phase / step
    samples_dir.mkdir(parents=True, exist_ok=True)

    log.info("Generating %d validation samples for %s/%s...", num_samples, phase, step)

    for i, pair in enumerate(eval_pairs[:num_samples]):
        convos = pair["conversations"]
        pgn_input = convos[1]["content"]  # user message = PGN
        ground_truth = convos[2]["content"]  # assistant message = real transcript

        generated = _generate_from_eval_pair(model, tokenizer, pgn_input)

        sample = {
            "pgn_input": pgn_input,
            "ground_truth": ground_truth,
            "generated": generated,
            "phase": phase,
            "step": step,
            "timestamp": datetime.now().isoformat(),
        }

        out_path = samples_dir / f"sample_{i:02d}.json"
        with open(out_path, "w") as f:
            json.dump(sample, f, indent=2)

        # Also write a human-readable comparison
        txt_path = samples_dir / f"sample_{i:02d}.txt"
        with open(txt_path, "w") as f:
            f.write(f"=== VALIDATION SAMPLE {i} — {phase}/{step} ===\n\n")
            f.write(f"--- PGN INPUT ---\n{pgn_input}\n\n")
            f.write(f"--- GROUND TRUTH (agadmator) ---\n{ground_truth}\n\n")
            f.write(f"--- GENERATED ---\n{generated}\n")

        log.info(
            "Sample %d: generated %d chars (ground truth: %d chars)",
            i, len(generated), len(ground_truth),
        )

    log.info("Saved validation samples to %s", samples_dir)
