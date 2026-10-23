#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import random
import argparse
import asyncio
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Any

import aiofiles

# -----------------------------
# TARGET IDS (default)
# -----------------------------
TARGET_IDS = [
    "2021.acl-long.236", "2021.acl-long.66", "2021.adaptnlp-1.23", "2021.ccl-1.5",
    "2021.emnlp-main.174", "2021.emnlp-main.515", "2021.emnlp-main.555", "2021.emnlp-main.744",
    "2021.emnlp-main.97", "2021.findings-acl.113", "2021.findings-acl.182", "2021.findings-acl.330",
    "2021.findings-emnlp.236", "2021.findings-emnlp.63", "2021.findings-emnlp.98", "2021.inlg-1.34",
    "2021.mmsr-1.5", "2021.mrl-1.1", "2021.naacl-main.174", "2021.naacl-main.176", "2022.acl-long.317",
    "2022.acl-long.47", "2022.acl-long.98", "2022.acl-short.65", "2022.bionlp-1.17", "2022.coling-1.240",
    "2022.coling-1.278", "2022.coling-1.612", "2022.ecnlp-1.1", "2022.emnlp-main.114", "2022.emnlp-main.255",
    "2022.emnlp-main.527", "2022.findings-acl.28", "2022.findings-emnlp.204", "2022.fnp-1.20",
    "2022.jeptalnrecital-taln.2", "2022.law-1.5", "2022.lrec-1.78", "2022.rocling-1.19", "2022.sumeval-1.2",
    "2023.acl-long.299", "2023.acl-long.508", "2023.acl-long.568", "2023.americasnlp-1.2", "2023.codi-1.20",
    "2023.emnlp-main.683", "2023.emnlp-main.782", "2023.findings-acl.490", "2023.findings-acl.578",
    "2023.findings-acl.79", "2023.findings-emnlp.320", "2023.icnlsp-1.16", "2023.icon-1.1",
    "2023.ijcnlp-main.36", "2023.jeptalnrecital-short.5", "2023.paclic-1.15", "2023.pandl-1.9",
    "2023.semeval-1.279", "2023.yrrsds-1.23", "2024.acl-long.31", "2024.acl-long.603", "2024.ccl-3.5",
    "2024.eacl-short.29", "2024.emnlp-main.1196", "2024.emnlp-main.586", "2024.emnlp-main.598",
    "2024.findings-acl.30", "2024.findings-acl.585", "2024.findings-acl.63", "2024.findings-acl.641",
    "2024.findings-acl.734", "2024.icnlsp-1.26", "2024.jeptalnrecital-jep.19", "2024.lrec-main.1287",
    "2024.lrec-main.909", "2024.ltedi-1.20", "2024.mrl-1.3", "2024.naacl-industry.24", "2024.sigdial-1.65",
    "2025.acl-demo.8", "2025.acl-industry.109", "2025.acl-long.1359", "2025.aisd-main.5", "2025.argmining-1.20",
    "2025.bea-1.69", "2025.coling-main.110", "2025.coling-main.480", "2025.coling-main.488",
    "2025.emnlp-demos.56", "2025.emnlp-industry.158", "2025.emnlp-main.297", "2025.findings-acl.167",
    "2025.findings-naacl.100", "2025.genaidetect-1.9", "2025.iwpt-1.1", "2025.jeptalnrecital-evalllm.20",
    "2025.mtsummit-1.16", "2025.mwe-1.1", "2025.sdp-1.12"
]

# -----------------------------
# Prompting
# -----------------------------
SYSTEM_PROMPT = (
    "You extract author and affiliation metadata from scientific papers.\n"
    "Output ONLY valid JSON. Do not include any commentary.\n"
    "You must cross-check the provided sources.\n"
    "If sources disagree, choose the best-supported value and record the disagreement.\n"
    "Do NOT invent information. Every affiliation and country must have verbatim evidence spans.\n"
    "If a field is not supported by evidence, set it to null.\n"
)

# -----------------------------
# Helpers: robust JSON parse
# -----------------------------
def clean_and_parse_json(raw_text: str) -> Any:
    text = (raw_text or "").strip()

    # strip markdown fences
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) > 2:
            text = "\n".join(lines[1:-1]).strip()

    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end <= 0:
        raise ValueError("No JSON object found in response")

    json_str = text[start:end]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # try json_repair if available
        try:
            import json_repair  # type: ignore
            return json_repair.loads(json_str)
        except Exception:
            # patch invalid escapes
            fixed = re.sub(r'\\(?![/u"\\bfnrt])', r"\\\\", json_str)
            return json.loads(fixed)

# -----------------------------
# Data structures
# -----------------------------
@dataclass
class EvidenceBundle:
    paper_id: str
    tei: str
    layout: str
    raw: str

# -----------------------------
# Indexing
# -----------------------------
TEI_SUFFIXES  = (".tei.xml", ".xml")
PYMU_SUFFIXES = (".md", ".jsonl")

def get_clean_key(filename: str, suffixes: Tuple[str, ...]) -> str:
    name = filename
    lower = name.lower()
    changed = True
    while changed:
        changed = False
        for suf in suffixes:
            if lower.endswith(suf):
                name = name[:-len(suf)]
                lower = lower[:-len(suf)]
                changed = True
                break
    return name

def index_directory(path: Path, extensions: Tuple[str, ...], suffixes: Tuple[str, ...]) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    if not path.exists():
        print(f"[WARN] Directory not found: {path}")
        return mapping
    for p in path.rglob("*"):
        if p.is_file() and p.name.lower().endswith(extensions):
            key = get_clean_key(p.name, suffixes)
            mapping[key] = p
    return mapping

# -----------------------------
# Reading TEI / MD / JSONL (layout)
# -----------------------------
ABSTRACT_PATTERNS = (
    "\nabstract\n",
    "\nabstract:",
    "\nabstract—",
    "\nabstract –",
)

def cut_at_abstract_md(text: str) -> str:
    if not text:
        return ""
    lower = text.lower()
    best = None
    for pat in ABSTRACT_PATTERNS:
        idx = lower.find(pat)
        if idx != -1 and (best is None or idx < best):
            best = idx
    if best is None:
        return text
    cut = text.rfind("\n", 0, best)
    return text[:cut if cut != -1 else best].rstrip()

def md_blocks(text: str) -> List[str]:
    lines = text.splitlines(True)
    blocks: List[str] = []
    buf: List[str] = []
    in_fence = False
    fence_marker = "```"

    def flush():
        nonlocal buf
        if buf:
            blocks.append("".join(buf))
            buf = []

    i = 0
    while i < len(lines):
        ln = lines[i]
        stripped = ln.strip()

        if stripped.startswith(fence_marker):
            if not in_fence:
                flush()
                in_fence = True
                buf.append(ln)
            else:
                buf.append(ln)
                flush()
                in_fence = False
            i += 1
            continue

        if in_fence:
            buf.append(ln)
            i += 1
            continue

        if stripped == "":
            buf.append(ln)
            flush()
            i += 1
            continue

        if "|" in ln:
            if buf:
                buf.append(ln)
            else:
                buf.append(ln)
            i += 1
            while i < len(lines) and lines[i].strip() != "" and "|" in lines[i]:
                buf.append(lines[i])
                i += 1
            flush()
            continue

        if stripped.startswith("#"):
            flush()
            buf.append(ln)
            flush()
            i += 1
            continue

        buf.append(ln)
        i += 1

    flush()
    return blocks

def truncate_md_preserving_blocks(text: str, max_chars: int) -> str:
    if not text:
        return ""
    blocks = md_blocks(text)
    out: List[str] = []
    n = 0
    open_fences = 0

    for blk in blocks:
        # track fences (best effort)
        fence_lines = [ln for ln in blk.splitlines() if ln.strip().startswith("```")]
        if fence_lines:
            open_fences ^= (len(fence_lines) % 2)

        if n + len(blk) > max_chars:
            remaining = max_chars - n
            if remaining > 0:
                out.append(blk[:remaining])
            break
        out.append(blk)
        n += len(blk)
        if n >= max_chars:
            break

    s = "".join(out).rstrip()
    if open_fences:
        s += "\n```\n"
    return s

def extract_tei_authorblock_xml(tei_path: str, max_bytes: int = 1_000_000, prefer_analytic: bool = True) -> str:
    if not tei_path:
        return ""
    try:
        raw = Path(tei_path).read_bytes()[:max_bytes]
        xml = raw.decode("utf-8", errors="ignore")
    except Exception:
        return ""

    mh = re.search(r"(?is)<teiHeader\b.*?</teiHeader>", xml)
    scope = mh.group(0) if mh else xml

    if prefer_analytic:
        m = re.search(r"(?is)<analytic\b[^>]*>.*?</analytic>", scope)
        if m:
            return m.group(0).strip()

    m = re.search(r"(?is)<fileDesc\b[^>]*>.*?</fileDesc>", scope)
    if m:
        return m.group(0).strip()

    if mh:
        return scope.strip()

    return scope.strip()

def read_text_file(path: str, max_chars: int, stop_at_abstract: bool = True) -> str:
    if not path:
        return ""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    if stop_at_abstract:
        text = cut_at_abstract_md(text)
    return text[:max_chars]

def read_pymu_jsonl(path: str, max_chars: int, add_page_markers: bool = False, stop_at_abstract: bool = True) -> str:
    parts: List[str] = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    obj = json.loads(ln)
                except Exception:
                    continue
                txt = obj.get("text", "") or ""
                if not txt:
                    continue
                parts.append(txt)
    except Exception:
        return ""
    text = "\n\n".join(parts)
    if stop_at_abstract:
        text = cut_at_abstract_md(text)
    # keep markdown-ish structure
    text = truncate_md_preserving_blocks(text, max_chars=max_chars)
    return text

# -----------------------------
# Prompt builder
# -----------------------------
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
                        "evidence": [{"source": "tei|layout|raw", "span": "verbatim span"}]
                    }
                ]
            }
        ],
        "affiliation_index": {},
        "conflicts": []
    }

    return (
        "Task: Extract authors and affiliations.\n"
        "Return JSON matching this shape:\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        "INPUTS:\n"
        f"=== TEI ===\n{bundle.tei}\n\n"
        f"=== LAYOUT ===\n{bundle.layout}\n\n"
        f"=== RAW ===\n{bundle.raw}\n"
    )

# -----------------------------
# Backends
# -----------------------------
class LLMBackend:
    async def generate(self, system_prompt: str, user_prompt: str, max_output_tokens: int) -> str:
        raise NotImplementedError

class GeminiBackend(LLMBackend):
    def __init__(self, model_name: str, api_key: str):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(
            model_name,
            system_instruction=system_prompt,
            generation_config={"response_mime_type": "application/json"},
        )

    async def generate(self, system_prompt: str, user_prompt: str, max_output_tokens: int) -> str:
        # max_output_tokens is not always honored across Gemini variants; keep for interface parity.
        resp = await self._model.generate_content_async(user_prompt)
        if not getattr(resp, "parts", None):
            return ""
        return resp.text or ""

class OpenAIBackend(LLMBackend):
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        # Uses the official OpenAI Python SDK (Responses API).
        # API key is typically read from OPENAI_API_KEY env var.
        from openai import AsyncOpenAI
        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        self._client = AsyncOpenAI(**kwargs)
        self._model = model_name

    async def generate(self, system_prompt: str, user_prompt: str, max_output_tokens: int) -> str:
        # Structured outputs in Responses API: use `text.format`.
        # Here we request JSON object mode; still keep robust parsing.
        resp = await self._client.responses.create(
            model=self._model,
            instructions=system_prompt,
            input=user_prompt,
            max_output_tokens=max_output_tokens,
            text={"format": {"type": "json_object"}},
            store=False,
        )
        return getattr(resp, "output_text", "") or ""

class LocalQwenBackend(LLMBackend):
    def __init__(self, model_id: str, device: str = "cuda", dtype: str = "float16"):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.torch = torch
        self.device = device

        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "auto": "auto",
        }.get(dtype, torch.float16)

        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        if device != "cuda":
            self.model.to(device)

    async def generate(self, system_prompt: str, user_prompt: str, max_output_tokens: int) -> str:
        # Run generation in a thread to avoid blocking the event loop.
        return await asyncio.to_thread(self._generate_sync, system_prompt, user_prompt, max_output_tokens)

    def _generate_sync(self, system_prompt: str, user_prompt: str, max_output_tokens: int) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Qwen instruct models usually ship a chat template.
        input_ids = self.tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        with self.torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_output_tokens,
                do_sample=False,
                temperature=0.0,
            )

        # Strip the prompt tokens
        gen_ids = out[0][input_ids.shape[-1]:]
        return self.tok.decode(gen_ids, skip_special_tokens=True).strip()

# -----------------------------
# Runner
# -----------------------------
async def append_failure(failure_log: Path, key: str, reason: str):
    async with aiofiles.open(failure_log, "a", encoding="utf-8") as f:
        await f.write(json.dumps({"key": key, "reason": reason}, ensure_ascii=False) + "\n")

async def process_one(
    sem: asyncio.Semaphore,
    backend: LLMBackend,
    out_dir: Path,
    failure_log: Path,
    key: str,
    tei_path: Optional[Path],
    layout_path: Optional[Path],
    layout_kind: str,   # "md" or "jsonl" (if layout_path provided)
    mode: str,          # "tei" | "md" | "jsonl" | "all"
    stop_at_abstract: bool,
    max_chars_tei: int,
    max_chars_layout: int,
    max_output_tokens: int,
    max_retries: int,
):
    out_path = out_dir / f"{key}.json"
    if out_path.exists():
        return

    async with sem:
        try:
            # Build bundle based on mode
            tei_text = ""
            layout_text = ""
            raw_text = ""  # kept for schema parity; left empty in this test script

            if mode in ("tei", "all") and tei_path and tei_path.exists():
                tei_text = await asyncio.to_thread(extract_tei_authorblock_xml, str(tei_path))

            if mode in ("md", "all") and layout_path and layout_path.exists() and layout_kind == "md":
                layout_text = await asyncio.to_thread(read_text_file, str(layout_path), max_chars_layout, stop_at_abstract)

            if mode in ("jsonl", "all") and layout_path and layout_path.exists() and layout_kind == "jsonl":
                layout_text = await asyncio.to_thread(read_pymu_jsonl, str(layout_path), max_chars_layout, False, stop_at_abstract)

            # cap
            tei_text = (tei_text or "")[:max_chars_tei]
            layout_text = (layout_text or "")[:max_chars_layout]

            if not tei_text and not layout_text:
                await append_failure(failure_log, key, "No input text found for requested mode (missing files?)")
                return

            bundle = EvidenceBundle(paper_id=key, tei=tei_text, layout=layout_text, raw=raw_text)
            user_prompt = build_user_prompt(bundle)

            attempt = 0
            while attempt < max_retries:
                try:
                    resp_text = await backend.generate(SYSTEM_PROMPT, user_prompt, max_output_tokens)
                    if not resp_text:
                        raise ValueError("Empty model response (blocked or failed)")

                    data = clean_and_parse_json(resp_text)
                    data["_provenance"] = {
                        "paper_id": key,
                        "mode": mode,
                        "tei_path": str(tei_path) if tei_path else None,
                        "layout_path": str(layout_path) if layout_path else None,
                    }

                    async with aiofiles.open(out_path, "w", encoding="utf-8") as f:
                        await f.write(json.dumps(data, ensure_ascii=False, indent=2))
                    return

                except Exception as e:
                    attempt += 1
                    if attempt >= max_retries:
                        # save raw dump if we have something that looks like a response
                        dump_path = out_dir / f"{key}_ERROR_RAW.txt"
                        try:
                            async with aiofiles.open(dump_path, "w", encoding="utf-8") as f:
                                await f.write(str(e))
                        except Exception:
                            pass
                        await append_failure(failure_log, key, f"Max retries hit: {e}")
                        return
                    await asyncio.sleep((2 ** attempt) + random.uniform(0, 0.5))

        except Exception as e:
            await append_failure(failure_log, key, f"Pre-processing error: {e}")

async def run(args):
    tei_dir = Path(args.tei_dir)
    pymu_dir = Path(args.pymu_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    failure_log = out_dir / "_failures.jsonl"

    # Choose IDs
    ids = args.ids if args.ids else TARGET_IDS
    ids = list(dict.fromkeys(ids))  # dedupe preserve order

    # Index
    tei_map = index_directory(tei_dir, (".xml",), TEI_SUFFIXES)

    if args.mode == "md":
        pymu_map = index_directory(pymu_dir, (".md",), (".md",))
        layout_kind = "md"
    elif args.mode == "jsonl":
        pymu_map = index_directory(pymu_dir, (".jsonl",), (".jsonl",))
        layout_kind = "jsonl"
    else:
        # for "all" we accept either; prefer md if both exist, else jsonl
        md_map = index_directory(pymu_dir, (".md",), (".md",))
        jl_map = index_directory(pymu_dir, (".jsonl",), (".jsonl",))
        pymu_map = {**jl_map, **md_map}  # md overwrites jsonl on collisions
        layout_kind = "mixed"

    # Filter to target ids only
    keys = [k for k in ids if (k in tei_map or k in pymu_map)]
    missing = [k for k in ids if (k not in tei_map and k not in pymu_map)]
    print(f"[INFO] Requested IDs: {len(ids)} | Found at least one file for: {len(keys)} | Missing: {len(missing)}")

    if missing:
        print("[INFO] Missing examples (first 20):")
        for m in missing[:20]:
            print("  -", m)

    # Backend
    if args.backend == "gemini":
        api_key = args.gemini_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Gemini backend selected but no API key provided (use --gemini-key or GEMINI_API_KEY).")
        backend: LLMBackend = GeminiBackend(args.model, api_key)

    elif args.backend == "openai":
        # OPENAI_API_KEY is standard; --openai-key optional
        backend = OpenAIBackend(args.model, api_key=args.openai_key or None)

    elif args.backend == "local":
        backend = LocalQwenBackend(args.model, device=args.device, dtype=args.dtype)

    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    # Concurrency
    sem = asyncio.Semaphore(args.concurrency)
    tasks = []

    for key in keys:
        tei_path = tei_map.get(key)
        layout_path = pymu_map.get(key)

        # if mode == all and mixed layout, detect actual kind
        lk = layout_kind
        if args.mode == "all" and layout_path:
            lk = "jsonl" if layout_path.name.lower().endswith(".jsonl") else "md"

        tasks.append(
            process_one(
                sem=sem,
                backend=backend,
                out_dir=out_dir,
                failure_log=failure_log,
                key=key,
                tei_path=tei_path,
                layout_path=layout_path,
                layout_kind=lk if lk != "mixed" else ("jsonl" if (layout_path and layout_path.name.lower().endswith(".jsonl")) else "md"),
                mode=args.mode,
                stop_at_abstract=args.stop_at_abstract,
                max_chars_tei=args.max_chars_tei,
                max_chars_layout=args.max_chars_layout,
                max_output_tokens=args.max_output_tokens,
                max_retries=args.max_retries,
            )
        )

    print(f"[INFO] Running {len(tasks)} tasks with concurrency={args.concurrency} ...")
    await asyncio.gather(*tasks)
    print(f"[DONE] Outputs in: {out_dir}")
    print(f"[DONE] Failures log: {failure_log}")

def parse_args():
    p = argparse.ArgumentParser(description="Affiliation extraction test runner (TARGET_IDS only).")

    p.add_argument("--tei-dir", required=True, help="Directory containing TEI XML files.")
    p.add_argument("--pymu-dir", required=True, help="Directory containing PyMu outputs (.md or .jsonl).")
    p.add_argument("--out-dir", required=True, help="Output directory.")

    p.add_argument("--mode", choices=["tei", "md", "jsonl", "all"], default="tei",
                   help="Which input(s) to use. 'tei'/'md'/'jsonl' run individually; 'all' uses TEI + whichever layout is found.")

    p.add_argument("--backend", choices=["gemini", "openai", "local"], default="gemini",
                   help="LLM backend to use.")

    p.add_argument("--model", default="gemini-2.5-flash-lite",
                   help="Model name/id. Examples: gemini-2.5-flash-lite | gpt-5-nano | Qwen/Qwen2.5-7B-Instruct")

    p.add_argument("--gemini-key", default="", help="Gemini API key (or set GEMINI_API_KEY).")
    p.add_argument("--openai-key", default="", help="OpenAI API key (or set OPENAI_API_KEY).")

    p.add_argument("--device", default="cuda", help="Local model device hint: cuda|cpu (local backend only).")
    p.add_argument("--dtype", default="float16", choices=["auto", "float16", "bfloat16", "float32"],
                   help="Local model dtype (local backend only).")

    p.add_argument("--concurrency", type=int, default=16, help="Max concurrent requests.")
    p.add_argument("--max-retries", type=int, default=5, help="Retries per paper.")
    p.add_argument("--max-output-tokens", type=int, default=2000, help="Generation cap.")
    p.add_argument("--max-chars-tei", type=int, default=8000, help="Max chars fed from TEI.")
    p.add_argument("--max-chars-layout", type=int, default=8000, help="Max chars fed from MD/JSONL.")
    p.add_argument("--stop-at-abstract", action="store_true", help="Cut inputs at Abstract marker (MD/JSONL).")

    p.add_argument("--ids", nargs="*", default=None,
                   help="Optional override list of paper IDs. If omitted, uses embedded TARGET_IDS.")

    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run(args))
