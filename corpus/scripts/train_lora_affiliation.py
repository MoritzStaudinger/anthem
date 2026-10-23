"""LoRA fine-tune Qwen2.5-7B-Instruct for affiliation extraction.

Reads (input, output) pairs produced by scripts/fetch_oa_training.py and
trains a LoRA adapter on top of a 4-bit-quantised Qwen2.5-7B-Instruct.

Prompt format (Qwen chat template):
    system  : "You are an expert at extracting author affiliations
               from scientific paper headers..."
    user    : "Extract author affiliations from this paper header:\\n\\n<input>"
    assistant: "<json output>"

Train target: only the assistant turn's tokens contribute to the loss
(handled by TRL's SFTTrainer with `completion_only_loss=True` when we
use the chat template).

Hyperparams (sensible defaults — A100 80GB, 7B+LoRA at 4-bit fits with
headroom):
    LoRA  : r=16, alpha=32, dropout=0.05, on q/k/v/o + gate/up/down
    seq   : 2048 tokens
    batch : per-device 1, grad_accum 16 → effective 16
    lr    : 2e-4, cosine, 3% warmup
    epochs: 3
    eval  : every 200 steps on val.jsonl

Outputs:
    <out-dir>/adapter/   PEFT LoRA adapter (load with PeftModel)
    <out-dir>/tokenizer/ saved tokenizer (handy if base ever changes)
    <out-dir>/training_log.jsonl  loss curve

Inference: see scripts/infer_lora_affiliation.py (load adapter, generate).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger("lora_train")


def _format_for_chat(row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert {input, output, source} into a chat-template messages list."""
    system_msg = (
        "You are an expert at extracting author affiliations from "
        "scientific paper headers. Given the front-matter text of a "
        "paper (the title block, authors, and affiliation lines as they "
        "appear on the PDF before the abstract), produce a JSON array "
        "of authors. Each author has a `name` and a list of "
        "`affiliations`. Each affiliation has `raw` (the substring "
        "where the affiliation appears), `institution` (the normalised "
        "institution name), `ror_id` (the full ROR URL), and "
        "`country_code` (ISO 3166-1 alpha-2). Output ONLY the JSON; no "
        "preamble or explanation."
    )
    user_msg = (
        "Extract author affiliations from this paper header:\n\n"
        f"{row['input']}"
    )
    # Compact JSON (no indent) — saves tokens.
    assistant_msg = json.dumps(row["output"], ensure_ascii=False, separators=(",", ":"))
    return {
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
    }


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path,
                    default=Path("data/lora_train/train.jsonl"))
    ap.add_argument("--val", type=Path,
                    default=Path("data/lora_train/val.jsonl"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("models/qwen25_7b_aff_lora"))
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=1,
                    help="Per-device train batch size.")
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--no-4bit", action="store_true",
                    help="Disable 4-bit quantisation (needs ≥40 GB GPU).")
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--eval-steps", type=int, default=200)
    ap.add_argument("--logging-steps", type=int, default=20)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    # Heavy imports — only here so `--help` is fast and the data-fetch
    # script doesn't need torch installed.
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    )
    from trl import SFTConfig, SFTTrainer

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --------------------------- data ---------------------------
    if not args.train.is_file():
        raise SystemExit(f"missing {args.train} — run fetch_oa_training.py first")
    train_rows = _load_jsonl(args.train)
    val_rows = _load_jsonl(args.val) if args.val.is_file() else []
    logger.info("loaded %d train + %d val rows", len(train_rows), len(val_rows))

    train_ds = Dataset.from_list([_format_for_chat(r) for r in train_rows])
    val_ds = Dataset.from_list([_format_for_chat(r) for r in val_rows]) if val_rows else None

    # --------------------------- model ---------------------------
    logger.info("loading base model: %s", args.base_model)
    bnb = None
    dtype = torch.bfloat16
    if not args.no_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb,
        torch_dtype=dtype,
        attn_implementation="sdpa",
        device_map="auto",
    )
    model.config.use_cache = False

    # --------------------------- LoRA config ---------------------------
    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    # --------------------------- training config ---------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sft_cfg = SFTConfig(
        output_dir=str(args.out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=True,
        max_length=args.max_seq_len,
        packing=False,  # one example per sequence — clearer loss accounting
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps" if val_ds is not None else "no",
        eval_steps=args.eval_steps,
        report_to=[],  # no wandb by default
        seed=args.seed,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit" if not args.no_4bit else "adamw_torch",
        # SFTTrainer + chat-template messages automatically masks the
        # user/system tokens from the loss; only the assistant tokens
        # contribute. This is the right default for instruction-tuning.
        assistant_only_loss=True,
        dataset_text_field=None,  # forces messages-style processing
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_cfg,
        processing_class=tok,
    )

    logger.info("starting training: %d steps total",
                len(train_ds) * args.epochs //
                (args.batch_size * args.grad_accum * max(torch.cuda.device_count(), 1)))
    train_out = trainer.train()
    logger.info("training done. metrics: %s", train_out.metrics)

    # Save adapter only (small, ~40 MB) + tokenizer for completeness.
    trainer.model.save_pretrained(str(args.out_dir / "adapter"))
    tok.save_pretrained(str(args.out_dir / "tokenizer"))
    # Persist training log (loss curve) so we don't lose it.
    with (args.out_dir / "training_log.jsonl").open("w") as fh:
        for entry in trainer.state.log_history:
            fh.write(json.dumps(entry) + "\n")

    if val_ds is not None:
        logger.info("running final eval...")
        eval_out = trainer.evaluate()
        (args.out_dir / "final_eval.json").write_text(json.dumps(eval_out, indent=2))
        logger.info("final eval: %s", eval_out)

    logger.info("done. adapter in %s/adapter", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
