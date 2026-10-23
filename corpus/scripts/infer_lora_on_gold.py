"""Run the trained LoRA adapter on the 781-paper gold set and emit
predictions in the aff-comparison leaderboard schema, so the result
joins the table next to `ours`, `openalex`, `s2_graph`, etc.

End-to-end: gold CSV  →  pymu_llm headers  →  LoRA inference  →
predictions_lora_qwen25_7b_aff.jsonl (comparison schema).

Output rows look like the other passthrough_file pipelines:
    {
      "paper_id": "<anthology_id>",
      "authors": [{
          "name": "...",
          "affiliations": [{
              "raw_affiliation": "...",
              "clean_institution_name": "...",
              "ror_id": "...",
              "country_code": "..."
          }]
      }],
      "_pipeline": {"id": "lora_qwen25_7b_aff",
                    "extraction_seconds": float,
                    "tokens_in": int|None,
                    "tokens_out": int|None,
                    "error": str|None}
    }

The LoRA's own training schema uses (`raw`, `institution`); this script
renames those to (`raw_affiliation`, `clean_institution_name`) so the
scorer's existing track logic (raw / ror / hier / name) just works.

Usage (after training finishes and `models/qwen25_7b_aff_lora/adapter`
exists):
    python scripts/infer_lora_on_gold.py \\
        --gold-csv reports/ground_truth_merged.csv \\
        --pymu-dir /share/moritz.staudinger/acl_data/acl_pdfs_pymu_llm \\
        --adapter models/qwen25_7b_aff_lora/adapter \\
        --out reports/pipeline_comparison/predictions_lora_qwen25_7b_aff.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("lora_infer_gold")

PIPELINE_ID = "lora_qwen25_7b_aff"  # overridable via --pipeline-id


def _read_pymu_header(md_path: Path, max_chars: int) -> Optional[str]:
    """First N characters of the pymu_llm markdown for this paper.
    Same shape the LoRA was trained on (see scripts/fetch_oa_training.py)."""
    if not md_path.is_file():
        return None
    try:
        txt = md_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return txt[:max_chars].strip() or None


def _build_prompt(tok, input_text: str) -> str:
    """Must match the prompt template used during training
    (scripts/train_lora_affiliation.py)."""
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
    s = completion.strip()
    m = _JSON_FENCE.search(s)
    if m:
        s = m.group(1).strip()
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


def _to_comparison_schema(paper_id: str, pred: Any, wall: float,
                          tokens_in: Optional[int], tokens_out: Optional[int],
                          err: Optional[str],
                          pipeline_id: str = PIPELINE_ID) -> Dict[str, Any]:
    """Rename the LoRA's (`raw`, `institution`) keys to the scorer's
    (`raw_affiliation`, `clean_institution_name`)."""
    authors_out: List[Dict[str, Any]] = []
    if isinstance(pred, list):
        for a in pred:
            if not isinstance(a, dict):
                continue
            affs_in = a.get("affiliations") or []
            affs_out: List[Dict[str, Any]] = []
            for aff in affs_in:
                if not isinstance(aff, dict):
                    continue
                affs_out.append({
                    "raw_affiliation": aff.get("raw") or aff.get("raw_affiliation") or "",
                    "clean_institution_name": (aff.get("institution")
                                               or aff.get("clean_institution_name") or ""),
                    "ror_id": aff.get("ror_id") or "",
                    "country_code": aff.get("country_code") or "",
                })
            authors_out.append({"name": a.get("name") or "", "affiliations": affs_out})
    return {
        "paper_id": paper_id,
        "authors": authors_out,
        "_pipeline": {
            "id": pipeline_id,
            "extraction_seconds": wall,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "error": err,
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold-csv", type=Path,
                    default=Path("reports/ground_truth_merged.csv"))
    ap.add_argument("--pymu-dir", type=Path,
                    default=Path("/share/moritz.staudinger/acl_data/acl_pdfs_pymu_llm"))
    ap.add_argument("--adapter", type=Path,
                    default=Path("models/qwen25_7b_aff_lora/adapter"))
    ap.add_argument("--no-adapter", action="store_true",
                    help="Run the base model WITHOUT the LoRA adapter — "
                         "used as the zero-shot control row for the LoRA "
                         "ablation. Pair with --pipeline-id "
                         "qwen25_7b_instruct and --out predictions_"
                         "qwen25_7b_instruct.jsonl so it shows up as its "
                         "own leaderboard row.")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--pipeline-id", default=None,
                    help="Pipeline id written into each row's _pipeline.id. "
                         "Defaults to lora_qwen25_7b_aff with --adapter, "
                         "qwen25_7b_instruct with --no-adapter.")
    ap.add_argument("--out", type=Path,
                    default=Path("reports/pipeline_comparison/predictions_lora_qwen25_7b_aff.jsonl"))
    ap.add_argument("--max-header-chars", type=int, default=5000,
                    help="Same window as the LoRA was trained on (default 5k).")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--no-4bit", action="store_true",
                    help="Disable bitsandbytes 4-bit quantization. Slower but "
                         "higher fidelity. On a 24G A40 with Qwen2.5-7B base, "
                         "fp16 fits; default 4-bit is just to leave headroom.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    if args.pipeline_id is None:
        args.pipeline_id = ("qwen25_7b_instruct" if args.no_adapter
                            else "lora_qwen25_7b_aff")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    if not args.no_adapter and not args.adapter.is_dir():
        raise SystemExit(f"adapter dir not found: {args.adapter} "
                         "(is training finished? or pass --no-adapter for the "
                         "zero-shot base-model control)")
    if not args.gold_csv.is_file():
        raise SystemExit(f"gold csv not found: {args.gold_csv}")
    if not args.pymu_dir.is_dir():
        raise SystemExit(f"pymu dir not found: {args.pymu_dir}")

    paper_ids: List[str] = []
    with args.gold_csv.open() as fh:
        for row in csv.DictReader(fh):
            pid = (row.get("anthology_id") or "").strip()
            if pid:
                paper_ids.append(pid)
    logger.info("loaded %d gold paper_ids", len(paper_ids))

    # Heavy imports last so --help / missing-file errors are fast.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb = None
    dtype = torch.bfloat16
    if not args.no_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True,
        )
    if args.no_adapter:
        logger.info("loading tokenizer + base %s (NO adapter — zero-shot control)",
                    args.base_model)
    else:
        logger.info("loading tokenizer + base %s + adapter %s",
                    args.base_model, args.adapter)
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb,
        torch_dtype=dtype, device_map="auto", attn_implementation="sdpa",
    )
    if args.no_adapter:
        model = base
    else:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, str(args.adapter))
    model.eval()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0; n_no_header = 0; n_parse_err = 0
    t_start = time.time()
    with args.out.open("w", encoding="utf-8") as fout:
        for i, pid in enumerate(paper_ids):
            t0 = time.perf_counter()
            err: Optional[str] = None
            pred = None
            tokens_in = tokens_out = None

            header = _read_pymu_header(args.pymu_dir / f"{pid}.md",
                                       args.max_header_chars)
            if not header:
                n_no_header += 1
                err = "pymu_llm header not found"
            else:
                try:
                    prompt = _build_prompt(tok, header)
                    inputs = tok(prompt, return_tensors="pt").to(model.device)
                    tokens_in = int(inputs.input_ids.shape[1])
                    with torch.no_grad():
                        out = model.generate(
                            **inputs,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=False,
                            pad_token_id=tok.pad_token_id,
                        )
                    gen = out[0, inputs.input_ids.shape[1]:]
                    tokens_out = int(gen.shape[0])
                    completion = tok.decode(gen, skip_special_tokens=True)
                    pred, parse_err = _parse_json(completion)
                    if parse_err:
                        n_parse_err += 1
                        err = f"parse: {parse_err}"
                    else:
                        n_ok += 1
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"

            wall = time.perf_counter() - t0
            fout.write(json.dumps(
                _to_comparison_schema(pid, pred, wall, tokens_in, tokens_out, err,
                                      pipeline_id=args.pipeline_id),
                ensure_ascii=False,
            ) + "\n")
            fout.flush()
            if (i + 1) % 25 == 0:
                rate = (i + 1) / (time.time() - t_start)
                logger.info("  %d/%d  ok=%d  no_header=%d  parse_err=%d  "
                            "rate=%.2f papers/s",
                            i + 1, len(paper_ids), n_ok, n_no_header,
                            n_parse_err, rate)

    elapsed = time.time() - t_start
    logger.info("done in %.1fs (%.2f papers/s): ok=%d  no_header=%d  parse_err=%d",
                elapsed, len(paper_ids) / max(elapsed, 1e-9),
                n_ok, n_no_header, n_parse_err)
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
