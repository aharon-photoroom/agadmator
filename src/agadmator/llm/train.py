"""Step 1: Fine-tune LLM with QLoRA via Unsloth."""

import json
import logging
from pathlib import Path

import torch

from agadmator.config import PROCESSED_DIR, LLM_OUTPUT_DIR

log = logging.getLogger(__name__)


def _load_config(config_path: str) -> dict:
    """Load YAML config."""
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def _check_gpu() -> tuple[float, str]:
    """Verify CUDA is available. Returns (vram_gb, gpu_name)."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU required for QLoRA training. "
            "Run this on a machine with an NVIDIA GPU."
        )
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    log.info("GPU: %s (%.1f GB VRAM)", gpu, vram)
    return vram, gpu


def _plan_training(vram_gb: float, cfg: dict) -> dict:
    """Adapt training params to available VRAM.

    H200 (80-141 GB): full bf16, no quantization, large batches, no grad checkpointing
    A100 (40-80 GB):  full bf16, no quantization, moderate batches
    RTX 4090 (24 GB): 4-bit quantization, small batches, grad checkpointing
    RTX 3090 (24 GB): 4-bit quantization, small batches, grad checkpointing
    <16 GB:           4-bit quantization, minimal batches, grad checkpointing
    """
    max_seq = cfg["max_seq_length"]

    if vram_gb >= 70:
        # H200 / A100-80GB: bf16, large batches with grad checkpointing
        return {
            "load_in_4bit": False,
            "batch_size": 8,
            "grad_accum": 4,
            "max_seq_length": 4096,
            "grad_checkpointing": True,
            "note": "H200/A100-80GB: bf16, no quant, batch 8×4, seq 4096",
        }
    elif vram_gb >= 35:
        # A100-40GB
        return {
            "load_in_4bit": False,
            "batch_size": 16,
            "grad_accum": 2,
            "max_seq_length": max_seq,
            "grad_checkpointing": False,
            "note": "A100-40GB: bf16, no quant, batch 16",
        }
    elif vram_gb >= 20:
        # RTX 4090 / 3090
        return {
            "load_in_4bit": True,
            "batch_size": cfg["batch_size"],
            "grad_accum": cfg["gradient_accumulation_steps"],
            "max_seq_length": max_seq,
            "grad_checkpointing": True,
            "note": "24GB GPU: 4-bit, batch %d" % cfg["batch_size"],
        }
    else:
        return {
            "load_in_4bit": True,
            "batch_size": 2,
            "grad_accum": 8,
            "max_seq_length": min(max_seq, 2048),
            "grad_checkpointing": True,
            "note": "<20GB: 4-bit, batch 2, seq capped",
        }


def train_llm(config_path: str, phase: str):
    """Fine-tune the LLM.

    Args:
        config_path: Path to YAML config file.
        phase: "pretrain" for ChessGPT domain training,
               "style" for agadmator style transfer.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("Install pyyaml: pip install pyyaml")

    vram_gb, gpu_name = _check_gpu()
    cfg = _load_config(config_path)["llm"]
    plan = _plan_training(vram_gb, cfg)
    log.info("Training plan: %s", plan["note"])

    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import load_dataset

    # Select data file based on phase
    data_dir = PROCESSED_DIR / "llm_training"
    if phase == "pretrain":
        data_file = str(data_dir / "chessgpt_train.jsonl")
        output_name = "chessgpt_lora"
        epochs = cfg.get("epochs", 3)
    else:
        data_file = str(data_dir / "agadmator_train.jsonl")
        output_name = "agadmator_lora"
        epochs = cfg.get("epochs", 5)

    output_dir = LLM_OUTPUT_DIR / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    max_seq = plan["max_seq_length"]
    log.info("Loading base model: %s (4bit=%s, seq=%d)", cfg["base_model"], plan["load_in_4bit"], max_seq)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["base_model"],
        max_seq_length=max_seq,
        dtype=None,  # auto-detect (bf16 on Hopper/Ampere)
        load_in_4bit=plan["load_in_4bit"],
    )

    # For style phase, load pretrained LoRA if it exists
    pretrained_lora = LLM_OUTPUT_DIR / "chessgpt_lora"
    if phase == "style" and pretrained_lora.exists():
        log.info("Loading pre-trained LoRA from %s", pretrained_lora)
        model = FastLanguageModel.from_pretrained(
            model_name=str(pretrained_lora),
            max_seq_length=max_seq,
            dtype=None,
            load_in_4bit=plan["load_in_4bit"],
        )[0]

    checkpointing = "unsloth" if plan["grad_checkpointing"] else False
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_rank"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=checkpointing,
    )

    # Load training data
    dataset = load_dataset("json", data_files=data_file, split="train")

    def format_conversation(example):
        """Format conversation for training."""
        convos = example["conversations"]
        text = ""
        for msg in convos:
            if msg["role"] == "system":
                text += f"<|im_start|>system\n{msg['content']}<|im_end|>\n"
            elif msg["role"] == "user":
                text += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
            elif msg["role"] == "assistant":
                text += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
        return {"text": text}

    dataset = dataset.map(format_conversation)

    use_bf16 = torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=plan["batch_size"],
        gradient_accumulation_steps=plan["grad_accum"],
        num_train_epochs=epochs,
        learning_rate=cfg["learning_rate"],
        fp16=not use_bf16,
        bf16=use_bf16,
        logging_steps=10,
        save_strategy="epoch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        seed=42,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq,
        args=training_args,
    )

    log.info(
        "Starting %s training: %d epochs, batch %d×%d, seq %d, bf16=%s",
        phase, epochs, plan["batch_size"], plan["grad_accum"], max_seq, use_bf16,
    )

    # Add validation callback to generate samples each epoch
    from transformers import TrainerCallback
    from agadmator.llm.validate import generate_validation_samples

    class ValidationSampleCallback(TrainerCallback):
        def on_epoch_end(self, args, state, control, model=None, **kwargs):
            epoch = int(state.epoch)
            log.info("Generating validation samples at epoch %d...", epoch)
            FastLanguageModel.for_inference(model)
            generate_validation_samples(
                model, tokenizer, phase, f"epoch_{epoch}", num_samples=3,
            )
            # Switch back to training mode
            model.train()

    trainer.add_callback(ValidationSampleCallback())
    trainer.train()

    # Save LoRA adapter
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    log.info("Saved model to %s", output_dir)

    # Final validation with more samples
    log.info("Generating final validation samples...")
    FastLanguageModel.for_inference(model)
    generate_validation_samples(
        model, tokenizer, phase, "final", num_samples=5,
    )
