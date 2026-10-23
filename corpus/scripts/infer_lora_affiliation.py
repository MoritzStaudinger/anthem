"""Run the trained LoRA adapter on paper-header inputs.

Loads Qwen2.5-7B-Instruct (or whatever base you trained on) + the LoRA
adapter, generates JSON output for each row in the input JSONL, and
writes predictions alongside the input for downstream comparison.

Usage:
    python scripts/infer_lora_affiliation.py \\
        --inputs data/lora_train/val.jsonl \\
        --adapter models/qwen25_7b_aff_lora/adapter \\
        --out predictions/val_lora.jsonl

Each output row mirrors the input plus a `prediction` field:
    {"input": "...", "output": [...gold...], "prediction": [...],
     "raw_completion": "...", "parse_error": null|"...", "source": {...}}
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger("lora_infer")


def _build_prompt(tok, input_text: str) -> str:
    messages = [
        {"role": "system",
         "content": (
             "You are an expert at extracting author affiliations from "
             "scientific paper headers. Given the front-matter text of a "
             "paper, produce a JSON array of authors. Each author has a "
             "`name` and a list of `affiliations`. Each affiliation has "
             "`raw`, `institution`, `ror_id`, and `country_code`. "
             "Output ONLY the JSON; no preamble."
         )},
        {"role": "user",
         "content": f"Extract author affiliations from this paper header:\n\n{input_text}"},
    ]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.+?)```", re.DOTALL)


def _parse_json(completion: str):
    """Extract a JSON array from a model completion. Handles raw arrays,
    fenced code blocks, and trailing/leading whitespace."""
    s = completion.strip()
    m = _JSON_FENCE.search(s)
    if m:
        s = m.group(1).strip()
    # Find the outermost [ ... ] structure
    start = s.find("[")
    if start == -1:
        return None, "no '[' in completion"
    depth = 0
    for i, ch in enumerate(s[start:], start=start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(s[start:i + 1]), None
                except json.JSONDecodeError as e:
                    return None, f"json decode: {e}"
    return None, "unterminated '['"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", type=Path, required=True,
                    help="JSONL with at least an `input` field per row.")
    ap.add_argument("--adapter", type=Path, required=True,
                    help="Path to the saved LoRA adapter dir.")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=1,
                    help="Inference batch size. Generation is sequential; "
                         "set 1 unless you've benchmarked.")
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    rows: List[Dict[str, Any]] = []
    with args.inputs.open() as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    logger.info("loaded %d rows", len(rows))

    bnb = None
    dtype = torch.bfloat16
    if not args.no_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True,
        )
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb,
        torch_dtype=dtype, device_map="auto", attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base, str(args.adapter))
    model.eval()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fout:
        for i, row in enumerate(rows):
            prompt = _build_prompt(tok, row["input"])
            inputs = tok(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
            # Strip the prompt off the front of the generation
            gen = out[0, inputs.input_ids.shape[1]:]
            completion = tok.decode(gen, skip_special_tokens=True)
            pred, err = _parse_json(completion)
            fout.write(json.dumps({
                **row,
                "prediction": pred,
                "raw_completion": completion,
                "parse_error": err,
            }, ensure_ascii=False) + "\n")
            if (i + 1) % 50 == 0:
                logger.info("  %d / %d", i + 1, len(rows))
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
