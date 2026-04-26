"""
fine_tune.py
PolicyPal AI — QLoRA Fine-tuning on LLaMA-3-8B using PEFT + TRL.

Why QLoRA:
  - Full fine-tuning of LLaMA-3-8B needs ~80 GB VRAM. QLoRA brings it to ~12 GB.
  - 4-bit NF4 quantization (bitsandbytes) + LoRA adapters on all attention + MLP layers.
  - Trained on (policy_text → simplified_summary) pairs from India Code PDFs.
  - Compared alternatives:
      * LoRA (16-bit) : ~24 GB VRAM — too large for a single A100-40GB
      * Prompt Tuning : insufficient domain adaptation for dense legal text
      * QLoRA         : best quality-to-cost ratio for this domain corpus size

Run (single GPU):
  python src/fine_tune.py \
      --base_model meta-llama/Meta-Llama-3-8B-Instruct \
      --train_file data/splits/train.json \
      --val_file   data/splits/val.json \
      --output_dir models/policypal-qlora \
      --epochs 3 --batch_size 4

Run (multi-GPU with accelerate):
  accelerate launch src/fine_tune.py --base_model ... --epochs 3

Dependencies:
  pip install transformers peft bitsandbytes accelerate trl datasets torch
"""

import argparse, json, os
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer


# ─────────────────────────────────────────────
#  PROMPT TEMPLATE  (LLaMA-3 Instruct format)
# ─────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are PolicyPal, an expert at explaining Indian government policies in clear, "
    "simple English. Given an official policy text from India Code, explain it concisely "
    "so that any citizen can understand it. Include: what the rule/scheme is, "
    "who it applies to, key amounts or dates, and how to benefit from it. "
    "Keep your answer to 4–6 sentences."
)


def format_record(rec: dict) -> str:
    """
    Wrap a PolicyRecord into the LLaMA-3 Instruct chat template.
    Input  : rec['category'] + rec['raw_text']
    Output : rec['simple_summary']
    """
    # Truncate very long raw_text to ~700 words to stay within 1024-token budget
    words    = rec["raw_text"].split()
    excerpt  = " ".join(words[:700]) + ("…" if len(words) > 700 else "")

    return (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"Policy category: {rec['category']}\n"
        f"Source: {rec.get('source', 'IndiaCode')}\n"
        f"Title: {rec.get('title', '')}\n\n"
        f"Official text:\n{excerpt}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
        f"{rec['simple_summary']}<|eot_id|>"
    )


# ─────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────

def load_split(path: str) -> Dataset:
    """Load a JSON split file and format each record for SFT."""
    with open(path) as f:
        records = json.load(f)

    # Filter out records without a summary (safety guard)
    records = [r for r in records if r.get("simple_summary", "").strip()]
    print(f"  Loaded {len(records)} records from {path}")

    formatted = [{"text": format_record(r)} for r in records]
    return Dataset.from_list(formatted)


# ─────────────────────────────────────────────
#  QLoRA CONFIGURATION
# ─────────────────────────────────────────────

def get_bnb_config() -> BitsAndBytesConfig:
    """
    4-bit NF4 quantization for QLoRA.
    NF4 (Normal Float 4) outperforms FP4 for LLM weight distributions.
    double_quant saves an additional ~0.4 GB by quantizing the quantization constants.
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def get_lora_config() -> LoraConfig:
    """
    LoRA adapter configuration.
    r=16, alpha=32  →  effective rank ratio = 2.0  (standard for instruction-tuning).
    target_modules covers all linear projections in LLaMA-3 (attention + MLP gates).
    dropout=0.05 adds light regularization suited to our moderate corpus size.
    Trainable params ≈ 20 M out of 8 B total (~0.25%).
    """
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",   # attention
            "gate_proj", "up_proj", "down_proj",        # MLP
        ],
    )


# ─────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────

def train(args):
    print("\n" + "="*60)
    print("  PolicyPal — QLoRA Fine-tuning")
    print("="*60)
    print(f"  Base model : {args.base_model}")
    print(f"  Train file : {args.train_file}")
    print(f"  Val file   : {args.val_file}")
    print(f"  Output dir : {args.output_dir}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {args.batch_size}  (eff. batch = {args.batch_size * 4})\n")

    # ── Tokenizer ─────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"   # required for causal LM with packing

    # ── Model (4-bit quantized) ────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=get_bnb_config(),
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if args.flash_attn else "eager",
    )
    model.config.use_cache = False          # disable KV-cache for training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, get_lora_config())
    model.print_trainable_parameters()

    # ── Datasets ──────────────────────────────────────────────
    train_ds = load_split(args.train_file)
    val_ds   = load_split(args.val_file)

    # ── Training Arguments ────────────────────────────────────
    training_args = TrainingArguments(
        output_dir                  = args.output_dir,
        num_train_epochs            = args.epochs,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size  = args.batch_size,
        gradient_accumulation_steps = 4,        # effective batch = batch_size × 4
        learning_rate               = 2e-4,
        lr_scheduler_type           = "cosine",
        warmup_ratio                = 0.05,
        weight_decay                = 0.01,
        fp16                        = False,
        bf16                        = True,     # bfloat16 for Ampere+ GPUs
        logging_steps               = 10,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        save_total_limit            = 2,
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
        report_to                   = "none",   # change to "wandb" for tracking
        dataloader_num_workers      = 2,
        gradient_checkpointing      = True,     # saves ~30% VRAM
        optim                       = "paged_adamw_32bit",  # QLoRA-optimised
        seed                        = 42,
        group_by_length             = True,     # reduces padding waste
    )

    # ── Trainer ───────────────────────────────────────────────
    trainer = SFTTrainer(
        model               = model,
        tokenizer           = tokenizer,
        train_dataset       = train_ds,
        eval_dataset        = val_ds,
        dataset_text_field  = "text",
        max_seq_length      = 1024,
        args                = training_args,
        callbacks           = [EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # ── Train ─────────────────────────────────────────────────
    print("[QLoRA] Starting training …")
    trainer.train()

    # ── Save adapter ──────────────────────────────────────────
    out = Path(args.output_dir)
    trainer.save_model(str(out / "final"))
    tokenizer.save_pretrained(str(out / "final"))
    print(f"\n[QLoRA] Adapter saved to: {out / 'final'}")

    # ── Training loss summary ─────────────────────────────────
    log_history = trainer.state.log_history
    train_losses = [e["loss"]      for e in log_history if "loss"      in e]
    eval_losses  = [e["eval_loss"] for e in log_history if "eval_loss" in e]
    if train_losses:
        print(f"  Final train loss : {train_losses[-1]:.4f}")
    if eval_losses:
        print(f"  Final eval loss  : {eval_losses[-1]:.4f}")

    # Save loss log
    loss_path = out / "loss_log.json"
    with open(loss_path, "w") as f:
        json.dump(log_history, f, indent=2)
    print(f"  Loss log saved to: {loss_path}")


# ─────────────────────────────────────────────
#  MERGE ADAPTERS (for deployment without PEFT)
# ─────────────────────────────────────────────

def merge_and_save(base_model: str, adapter_path: str, output_path: str):
    """
    Merge LoRA weights back into the base model for faster inference
    (removes the PEFT dependency at serving time).
    Requires enough CPU RAM to load the full 16-bit model.
    """
    from peft import PeftModel

    print(f"[Merge] Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    print(f"[Merge] Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base, adapter_path)

    print("[Merge] Merging and unloading …")
    merged = model.merge_and_unload()
    merged.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"[Merge] Saved merged model to: {output_path}")


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PolicyPal QLoRA Fine-tuning")
    parser.add_argument("--base_model",  default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--train_file",  default="data/splits/train.json")
    parser.add_argument("--val_file",    default="data/splits/val.json")
    parser.add_argument("--output_dir",  default="models/policypal-qlora")
    parser.add_argument("--epochs",      type=int,   default=3)
    parser.add_argument("--batch_size",  type=int,   default=4)
    parser.add_argument("--flash_attn",  action="store_true",
                        help="Enable Flash Attention 2 (requires flash-attn package)")
    parser.add_argument("--merge",       action="store_true",
                        help="After training, merge adapter into base model")
    parser.add_argument("--merge_output", default="models/policypal-merged")
    args = parser.parse_args()

    train(args)

    if args.merge:
        merge_and_save(
            base_model   = args.base_model,
            adapter_path = str(Path(args.output_dir) / "final"),
            output_path  = args.merge_output,
        )
