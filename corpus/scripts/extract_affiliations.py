#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extraction-only: authors + affiliations (+ country string if explicitly present)
from three sources (TEI, PyMu4LLM layout, raw), using a local Qwen model.

This version supports PyMu4LLM outputs in either:
- Markdown (.md), or
- Page-chunk JSONL (.jsonl): one JSON object per page with keys {metadata, text}

For JSONL, it concatenates per-page text in page order (metadata.page_index),
optionally inserting page markers.

No country normalization/alignment in this version (country is a free string).
"""

import os
import re
import json
import argparse
from dataclasses import dataclass
from typing import Optional, Dict, List, Any, Tuple, Union

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# -----------------------------
# IO / indexing
# -----------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()

def write_json(path: str, obj: Any):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def stem_no_ext(filename: str) -> str:
    base = os.path.basename(filename)
    parts = base.split(".")
    return parts[0] if len(parts) > 1 else base

def build_index_by_stem(dir_path: str, exts: Tuple[str, ...]) -> Dict[str, str]:
    idx = {}
    if not dir_path:
        return idx
    for root, _, files in os.walk(dir_path):
        for fn in files:
            lfn = fn.lower()
            if any(lfn.endswith(e) for e in exts):
                idx[stem_no_ext(fn)] = os.path.join(root, fn)
    return idx

def discover_paper_ids(tei_idx, pymu_idx, raw_idx) -> List[str]:
    ids = set()
    ids.update(tei_idx.keys())
    ids.update(pymu_idx.keys())
    ids.update(raw_idx.keys())
    return sorted(ids)


# -----------------------------
# PyMu JSONL (page chunks) -> text
# -----------------------------

def pymu_jsonl_to_text(jsonl_path: str, add_page_markers: bool = True) -> str:
    """
    Reads PyMu4LLM page-chunk JSONL and concatenates per-page text in order.
    Expected lines: {"metadata": {"page_index": int, ...}, "text": "..."}
    """
    rows: List[Tuple[Optional[int], str]] = []
    with open(jsonl_path, "r", encoding="utf-8", errors="replace") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            obj = json.loads(ln)
            meta = obj.get("metadata", {}) or {}
            page = meta.get("page_index", None)
            text = obj.get("text", "") or ""
            rows.append((page, text))

    # Sort: pages with known indices first in numeric order, then unknown.
    rows.sort(key=lambda x: (x[0] is None, x[0] if x[0] is not None else 10**9))

    parts: List[str] = []
    for page, text in rows:
        if add_page_markers:
            if page is None:
                parts.append("\n\n=== PAGE ===\n\n")
            else:
                parts.append(f"\n\n=== PAGE {page} ===\n\n")
        parts.append(text)

    return "".join(parts).strip()


def read_pymu_any(path: str, add_page_markers: bool = True) -> str:
    """
    Reads PyMu output in either .md or .jsonl format, returning a single text blob.
    """
    lp = path.lower()
    if lp.endswith(".jsonl"):
        return pymu_jsonl_to_text(path, add_page_markers=add_page_markers)
    return read_text(path)


# -----------------------------
# Text preparation
# -----------------------------

_ABSTRACT_RE = re.compile(r"\babstract\b", re.IGNORECASE)

def truncate_before_abstract(text: str) -> str:
    m = _ABSTRACT_RE.search(text)
    if not m:
        return text
    return text[: m.start()]

def compact_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def cap_chars(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n[TRUNCATED]\n"


@dataclass
class EvidenceBundle:
    paper_id: str
    tei: str
    layout: str
    raw: str


def make_bundle(
    paper_id: str,
    tei_text: str,
    layout_text: str,
    raw_text: str,
    stop_at_abstract: bool,
    max_chars: int,
) -> EvidenceBundle:
    tei = compact_whitespace(tei_text or "")
    layout = compact_whitespace(layout_text or "")
    raw = compact_whitespace(raw_text or "")

    if stop_at_abstract:
        if layout:
            layout = truncate_before_abstract(layout)
        if raw:
            raw = truncate_before_abstract(raw)

    tei = cap_chars(tei, max_chars)
    layout = cap_chars(layout, max_chars)
    raw = cap_chars(raw, max_chars)

    return EvidenceBundle(paper_id=paper_id, tei=tei, layout=layout, raw=raw)


# -----------------------------
# Prompting
# -----------------------------

SYSTEM_PROMPT = (
    "You extract author and affiliation metadata from scientific papers.\n"
    "Output ONLY valid JSON. Do not include any commentary.\n"
    "You must cross-check the three provided sources (TEI, LAYOUT, RAW).\n"
    "If sources disagree, choose the best-supported value and record the disagreement.\n"
    "Do NOT invent information. Every affiliation and country must have verbatim evidence spans.\n"
    "If a field is not supported by evidence, set it to null.\n"
)

def build_user_prompt(bundle: EvidenceBundle) -> str:
    schema = {
        "paper_id": bundle.paper_id,
        "authors": [
            {
                "name": "string",
                "markers": ["string"],
                "affiliations": [
                    {
                        "marker": "string or null",
                        "raw_affiliation": "string",
                        "country": "string or null",
                        "evidence": [
                            {"source": "tei|layout|raw", "span": "verbatim span"}
                        ]
                    }
                ]
            }
        ],
        "affiliation_index": {
            "marker_string_or_generated_key": {
                "raw_affiliation": "string",
                "country": "string or null",
                "evidence": [
                    {"source": "tei|layout|raw", "span": "verbatim span"}
                ]
            }
        },
        "conflicts": [
            {
                "field": "author_name|affiliation_text|country|marker_mapping|other",
                "description": "string",
                "tei_value": "string or null",
                "layout_value": "string or null",
                "raw_value": "string or null",
                "resolution": "string (what you chose and why, based on evidence)",
                "evidence": [
                    {"source": "tei|layout|raw", "span": "verbatim span"}
                ]
            }
        ]
    }

    rules = [
        "Use all three sources. Treat TEI as useful but not authoritative; verify against LAYOUT and RAW.",
        "Prefer the version that has the clearest, most complete affiliation lines and marker mapping.",
        "If authors use markers (1,2,* or †), build an affiliation_index and map authors to it. If no markers exist, still output affiliations per author.",
        "Country: only if explicitly stated in an evidence span (e.g., 'Austria', 'USA'). Otherwise null.",
        "Every extracted raw_affiliation must be supported by evidence spans copied verbatim.",
        "Record disagreements between TEI / LAYOUT / RAW in conflicts, even if you can resolve them."
    ]

    return (
        "Task: Extract authors and affiliations, cross-checking TEI vs LAYOUT vs RAW.\n"
        "Rules:\n- " + "\n- ".join(rules) + "\n\n"
        "Return JSON matching this shape (keys must exist; use null/[] when missing):\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
        "INPUTS:\n"
        f"=== TEI ===\n{bundle.tei}\n\n"
        f"=== LAYOUT ===\n{bundle.layout}\n\n"
        f"=== RAW ===\n{bundle.raw}\n"
    )


# -----------------------------
# JSON extraction and validation
# -----------------------------

_JSON_RE = re.compile(r"\{.*\}\s*$", re.DOTALL)

def extract_json_from_text(text: str) -> Optional[dict]:
    text = text.strip()
    m = _JSON_RE.search(text)
    if not m:
        return None
    blob = m.group(0)
    try:
        return json.loads(blob)
    except Exception:
        return None

def evidence_span_exists(bundle: EvidenceBundle, source: str, span: str) -> bool:
    if not span:
        return False
    span = span.strip()
    if not span:
        return False
    hay = bundle.tei if source == "tei" else bundle.layout if source == "layout" else bundle.raw
    return span in (hay or "")

def harden_output(obj: dict, bundle: EvidenceBundle) -> dict:
    out = obj if isinstance(obj, dict) else {}
    out.setdefault("paper_id", bundle.paper_id)
    out.setdefault("authors", [])
    out.setdefault("affiliation_index", {})
    out.setdefault("conflicts", [])

    def clean_evidence(ev_list):
        cleaned = []
        if not isinstance(ev_list, list):
            return cleaned
        for ev in ev_list:
            if not isinstance(ev, dict):
                continue
            src = ev.get("source")
            span = ev.get("span")
            if src in ("tei", "layout", "raw") and isinstance(span, str) and evidence_span_exists(bundle, src, span):
                cleaned.append({"source": src, "span": span})
        return cleaned

    authors = []
    for a in out["authors"] if isinstance(out["authors"], list) else []:
        if not isinstance(a, dict):
            continue
        a.setdefault("name", None)
        a.setdefault("markers", [])
        a.setdefault("affiliations", [])
        affs = []
        for aff in a["affiliations"] if isinstance(a["affiliations"], list) else []:
            if not isinstance(aff, dict):
                continue
            aff.setdefault("marker", None)
            aff.setdefault("raw_affiliation", "")
            aff.setdefault("country", None)
            aff["evidence"] = clean_evidence(aff.get("evidence", []))
            affs.append(aff)
        a["affiliations"] = affs
        authors.append(a)
    out["authors"] = authors

    idx = out["affiliation_index"] if isinstance(out["affiliation_index"], dict) else {}
    new_idx = {}
    for k, v in idx.items():
        if not isinstance(k, str) or not isinstance(v, dict):
            continue
        v.setdefault("raw_affiliation", "")
        v.setdefault("country", None)
        v["evidence"] = clean_evidence(v.get("evidence", []))
        new_idx[k] = v
    out["affiliation_index"] = new_idx

    conflicts = []
    for c in out["conflicts"] if isinstance(out["conflicts"], list) else []:
        if not isinstance(c, dict):
            continue
        c.setdefault("field", "other")
        c.setdefault("description", "")
        c.setdefault("tei_value", None)
        c.setdefault("layout_value", None)
        c.setdefault("raw_value", None)
        c.setdefault("resolution", "")
        c["evidence"] = clean_evidence(c.get("evidence", []))
        conflicts.append(c)
    out["conflicts"] = conflicts

    return out


# -----------------------------
# Local LLM wrapper
# -----------------------------

class LocalLLM:
    def __init__(self, model_id: str, device: str = "cuda", dtype: str = "auto", max_new_tokens: int = 900):
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

        if dtype == "auto":
            torch_dtype = torch.float16 if device == "cuda" else torch.float32
        elif dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="auto" if device == "cuda" else None,
        )
        if device != "cuda":
            self.model.to(device)

    @torch.inference_mode()
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        out_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        gen = self.tokenizer.decode(out_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        return gen.strip()


# -----------------------------
# Main
# -----------------------------

def load_inputs_for_id(
    paper_id: str,
    tei_idx: Dict[str, str],
    pymu_idx: Dict[str, str],
    raw_idx: Dict[str, str],
    pymu_add_page_markers: bool,
) -> Tuple[str, str, str]:
    tei = read_text(tei_idx[paper_id]) if paper_id in tei_idx else ""
    pymu = read_pymu_any(pymu_idx[paper_id], add_page_markers=pymu_add_page_markers) if paper_id in pymu_idx else ""
    raw = read_text(raw_idx[paper_id]) if paper_id in raw_idx else ""
    return tei, pymu, raw

def main():
    ap = argparse.ArgumentParser("Extraction-only affiliation metadata via local Qwen, cross-checking 3 sources.")
    ap.add_argument("--tei_dir", required=True)
    ap.add_argument("--pymu_dir", required=True)
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--list_file", default=None, help="Optional file with paper IDs (stems) to process.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max_chars", type=int, default=20000)
    ap.add_argument("--stop_at_abstract", action="store_true")

    # NEW: JSONL concat behavior
    ap.add_argument("--pymu_page_markers", action="store_true",
                    help="If PyMu input is .jsonl, insert page markers between pages.")

    ap.add_argument("--model_id", required=True)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    ap.add_argument("--max_new_tokens", type=int, default=900)

    ap.add_argument("--log_file", default="logs/affiliation_extract.log")

    args = ap.parse_args()

    ensure_dir(args.out_dir)
    ensure_dir(os.path.dirname(args.log_file) or ".")

    tei_idx = build_index_by_stem(args.tei_dir, (".tei.xml", ".xml"))
    pymu_idx = build_index_by_stem(args.pymu_dir, (".md", ".jsonl"))
    raw_idx = build_index_by_stem(args.raw_dir, (".txt", ".text"))

    if args.list_file:
        with open(args.list_file, "r", encoding="utf-8") as f:
            paper_ids = [ln.strip() for ln in f if ln.strip()]
    else:
        paper_ids = discover_paper_ids(tei_idx, pymu_idx, raw_idx)

    llm = LocalLLM(
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
    )

    n_ok, n_skip, n_fail = 0, 0, 0
    with open(args.log_file, "a", encoding="utf-8") as flog:
        for pid in paper_ids:
            out_path = os.path.join(args.out_dir, f"{pid}.json")
            if (not args.overwrite) and os.path.exists(out_path):
                n_skip += 1
                continue

            try:
                tei, pymu, raw = load_inputs_for_id(
                    pid, tei_idx, pymu_idx, raw_idx,
                    pymu_add_page_markers=args.pymu_page_markers
                )

                bundle = make_bundle(
                    paper_id=pid,
                    tei_text=tei,
                    layout_text=pymu,
                    raw_text=raw,
                    stop_at_abstract=args.stop_at_abstract,
                    max_chars=args.max_chars,
                )

                user_prompt = build_user_prompt(bundle)
                gen = llm.generate(SYSTEM_PROMPT, user_prompt)

                obj = extract_json_from_text(gen)
                if obj is None:
                    repair_user = (
                        "Your previous output was not valid JSON.\n"
                        "Return ONLY valid JSON matching the schema. Do not add any facts.\n\n"
                        f"Previous output:\n{gen}\n"
                    )
                    gen2 = llm.generate(SYSTEM_PROMPT, repair_user)
                    obj = extract_json_from_text(gen2)

                if obj is None:
                    raise ValueError("LLM did not return valid JSON after repair attempt.")

                hardened = harden_output(obj, bundle=bundle)
                hardened["_provenance"] = {
                    "tei_present": bool(bundle.tei),
                    "layout_present": bool(bundle.layout),
                    "raw_present": bool(bundle.raw),
                    "stop_at_abstract": args.stop_at_abstract,
                    "max_chars": args.max_chars,
                    "pymu_page_markers": args.pymu_page_markers,
                    "model_id": args.model_id,
                    "pymu_input_kind": "jsonl_or_md",
                }

                # If any affiliation has zero evidence, add a conflict entry so you see it.
                for a in hardened.get("authors", []):
                    for aff in a.get("affiliations", []):
                        if not aff.get("evidence"):
                            hardened.setdefault("conflicts", []).append({
                                "field": "affiliation_text",
                                "description": "Affiliation extracted but no evidence span validated as substring; likely parsing/model issue.",
                                "tei_value": None, "layout_value": None, "raw_value": None,
                                "resolution": "Kept affiliation, but flagged for review.",
                                "evidence": []
                            })

                write_json(out_path, hardened)
                n_ok += 1

            except Exception as e:
                n_fail += 1
                flog.write(f"[FAIL] {pid}: {repr(e)}\n")

    print(f"Done. ok={n_ok} skipped={n_skip} failed={n_fail}")
    print(f"Outputs: {os.path.abspath(args.out_dir)}")
    if n_fail:
        print(f"See log: {os.path.abspath(args.log_file)}")


if __name__ == "__main__":
    main()
