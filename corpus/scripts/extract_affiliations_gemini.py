"""Async Gemini affiliation extractor — lifted from extractor.ipynb
cells 4, 5, 12. Same logic that built the existing
ror_parsed_affiliations_90_with_topics corpus, packaged as a CLI for
cluster use.

Cross-checks three sources per paper (TEI XML / PyMu_LLM markdown /
raw text) in ONE Gemini call. The model picks the best-supported
value per field and records disagreements in a `conflicts` block.

Output: one <paper_id>.json per paper in --out-dir, matching the
schema the existing corpus uses (authors[].affiliations[] with
raw_affiliation + country + evidence spans; affiliation_index;
conflicts; _provenance.model).

Idempotent: skips papers whose output JSON already exists in
--out-dir. Failures are persisted to <out-dir>/_failures.jsonl so
re-running picks up only what's missing.

Reads GEMINI_API_KEY from the environment (never hardcode keys in
this script — the original notebook leaked a key).

Usage (cluster — see slurm/extract_affiliations_gemini.sbatch):
    python scripts/extract_affiliations_gemini.py \\
        --tei-dir  /share/.../acl_pdfs_parsed \\
        --pymu-dir /share/.../acl_pdfs_pymu_llm \\
        --raw-dir  /share/.../acl_pdfs_hybrid \\
        --out-dir  /share/.../gemini_affiliations_NEW \\
        --list-file reports/new_papers.txt \\
        --model gemini-1.5-flash
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import google.generativeai as genai  # type: ignore
from google.api_core.exceptions import (  # type: ignore
    InternalServerError, ResourceExhausted, ServiceUnavailable,
)

logger = logging.getLogger("extract_aff_gemini")

# ---------------------------------------------------------------------------
# File suffix chains (notebook cell 12)
# ---------------------------------------------------------------------------
TEI_SUFFIXES = (".tei.xml", ".xml")
PYMU_SUFFIXES = (".md", ".jsonl")
RAW_SUFFIXES = (".txt", ".text", ".jsonl")

ABSTRACT_PATTERNS = (
    "\nabstract\n",
    "\nabstract:",
    "\nabstract—",
    "\nabstract –",
)


@dataclass
class EvidenceBundle:
    paper_id: str
    tei: str
    layout: str
    raw: str


# ---------------------------------------------------------------------------
# Source readers (lifted from notebook cell 4)
# ---------------------------------------------------------------------------

def extract_tei_authorblock_xml(tei_path: str, max_bytes: int = 1_000_000,
                                prefer_analytic: bool = True) -> str:
    """Extract analytic / fileDesc / teiHeader XML snippet from a GROBID
    TEI file. Returns the XML string (un-flattened — keeps author↔aff
    mapping). Regex-based, no XML parser dependency."""
    if not tei_path:
        return ""
    try:
        with open(tei_path, "rb") as f:
            raw = f.read(max_bytes)
        xml = raw.decode("utf-8", errors="ignore")
    except OSError:
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
    return (mh.group(0) if mh else scope).strip()


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


def read_pymu_any(path: str) -> str:
    """Read pymu_llm output — supports both .md and .jsonl (page chunks).
    For .jsonl, concatenates per-page text in order. Always trims at the
    first 'Abstract' marker so we don't burn tokens on body text."""
    if not path:
        return ""
    if path.lower().endswith(".jsonl"):
        parts: List[str] = []
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                rows = []
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        obj = json.loads(ln)
                    except json.JSONDecodeError:
                        continue
                    page = (obj.get("metadata") or {}).get("page_index")
                    text = obj.get("text") or ""
                    rows.append((page, text))
            rows.sort(key=lambda x: (x[0] is None, x[0] if x[0] is not None else 10**9))
            for _, t in rows:
                parts.append(t)
        except OSError:
            return ""
        text = "\n\n".join(parts)
        return cut_at_abstract_md(text)
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return ""
    return cut_at_abstract_md(text)


def read_raw_any(path: str) -> str:
    if not path:
        return ""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return ""
    return cut_at_abstract_md(text)


# ---------------------------------------------------------------------------
# Prompt (lifted from notebook cell 5 + cell 12)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You extract author and affiliation metadata from scientific papers.\n"
    "Output ONLY valid JSON. Do not include any commentary.\n"
    "You must cross-check the three provided sources (TEI, LAYOUT, RAW).\n"
    "If sources disagree, choose the best-supported value and record the "
    "disagreement.\n"
    "Do NOT invent information. Every affiliation and country must have "
    "verbatim evidence spans.\n"
    "If a field is not supported by evidence, set it to null.\n"
)


def build_user_prompt(bundle: EvidenceBundle) -> str:
    schema = {
        "paper_id": bundle.paper_id,
        "authors": [{
            "name": "string",
            "markers": ["string"],
            "affiliations": [{
                "marker": "string or null",
                "raw_affiliation": "string",
                "country": "string or null",
                "evidence": [{"source": "tei|layout|raw", "span": "verbatim span"}],
            }],
        }],
        "affiliation_index": {
            "marker_string_or_generated_key": {
                "raw_affiliation": "string",
                "country": "string or null",
                "evidence": [{"source": "tei|layout|raw", "span": "verbatim span"}],
            },
        },
        "conflicts": [{
            "field": "author_name|affiliation_text|country|marker_mapping|other",
            "description": "string",
            "tei_value": "string or null",
            "layout_value": "string or null",
            "raw_value": "string or null",
            "resolution": "string",
            "evidence": [{"source": "tei|layout|raw", "span": "verbatim span"}],
        }],
    }
    rules = [
        "Use all three sources. Treat TEI as useful but not authoritative; "
        "verify against LAYOUT and RAW.",
        "Prefer the version that has the clearest, most complete affiliation "
        "lines and marker mapping.",
        "If authors use markers (1,2,* or †), build an affiliation_index and "
        "map authors to it. If no markers exist, still output affiliations "
        "per author.",
        "Country: only if explicitly stated in an evidence span (e.g., "
        "'Austria', 'USA'). Otherwise null.",
        "Every extracted raw_affiliation must be supported by evidence spans "
        "copied verbatim.",
        "Record disagreements between TEI / LAYOUT / RAW in conflicts, even "
        "if you can resolve them.",
        "Before adding affiliations as: 'Not explicitly stated in the "
        "sources.' you have to set them to null instead.",
    ]
    return (
        "Task: Extract authors and affiliations, cross-checking TEI vs "
        "LAYOUT vs RAW.\n"
        "Rules:\n- " + "\n- ".join(rules) + "\n\n"
        "Return JSON matching this shape (keys must exist; use null/[] when "
        "missing):\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
        "INPUTS:\n"
        f"=== TEI ===\n{bundle.tei}\n\n"
        f"=== LAYOUT ===\n{bundle.layout}\n\n"
        f"=== RAW ===\n{bundle.raw}\n"
    )


# ---------------------------------------------------------------------------
# JSON robustness (notebook cell 12 + cell 13)
# ---------------------------------------------------------------------------

def clean_and_parse_json(raw_text: str) -> dict:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) > 2:
            text = "\n".join(lines[1:-1])
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON object found in response")
    json_str = text[start:end]
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            import json_repair  # type: ignore
            return json_repair.loads(json_str)
        except ImportError:
            fixed = re.sub(r'\\(?![/u"\\bfnrt])', r"\\\\", json_str)
            return json.loads(fixed)


# ---------------------------------------------------------------------------
# Directory indexing (notebook cell 12)
# ---------------------------------------------------------------------------

def _get_clean_key(filename: str, suffixes: Tuple[str, ...]) -> str:
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


def index_directory(path: Path, extensions: Tuple[str, ...],
                    suffixes: Tuple[str, ...]) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    if not path.exists():
        logger.warning("source dir not found: %s", path)
        return mapping
    for p in path.rglob("*"):
        if p.is_file() and p.name.lower().endswith(extensions):
            mapping[_get_clean_key(p.name, suffixes)] = p
    return mapping


# ---------------------------------------------------------------------------
# Async runner (notebook cell 12, adapted)
# ---------------------------------------------------------------------------

def _append_failure(failure_log: Path, key: str, reason: str) -> None:
    """Sync append — sub-ms per call. Async file I/O isn't worth the
    aiofiles dependency when the API call dominates wall time."""
    with failure_log.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "reason": reason}) + "\n")


async def _process_paper(
    sem: asyncio.Semaphore, model, key: str, paths: Dict[str, Optional[Path]],
    out_dir: Path, failure_log: Path, model_name: str,
    max_retries: int, tei_chars: int, layout_chars: int, raw_chars: int,
) -> None:
    out_path = out_dir / f"{key}.json"
    if out_path.exists():
        return
    async with sem:
        try:
            task_tei = (asyncio.to_thread(extract_tei_authorblock_xml, str(paths["tei"]))
                        if paths.get("tei") else asyncio.sleep(0, result=""))
            task_pymu = (asyncio.to_thread(read_pymu_any, str(paths["pymu"]))
                         if paths.get("pymu") else asyncio.sleep(0, result=""))
            task_raw = (asyncio.to_thread(read_raw_any, str(paths["raw"]))
                        if paths.get("raw") else asyncio.sleep(0, result=""))
            tei_t, pymu_t, raw_t = await asyncio.gather(task_tei, task_pymu, task_raw)
            bundle = EvidenceBundle(
                paper_id=key,
                tei=str(tei_t)[:tei_chars],
                layout=str(pymu_t)[:layout_chars],
                raw=str(raw_t)[:raw_chars],
            )
            user_prompt = build_user_prompt(bundle)

            attempt = 0
            while attempt < max_retries:
                try:
                    response = await model.generate_content_async(user_prompt)
                    if not response.parts:
                        _append_failure(failure_log, key,
                                        "blocked by safety filters")
                        return
                    try:
                        data = clean_and_parse_json(response.text)
                    except Exception as parse_error:
                        # Save raw text for manual recovery — same pattern as
                        # the notebook (don't waste API spend on parse fails).
                        raw_dump = out_dir / f"{key}_ERROR_RAW.txt"
                        raw_dump.write_text(response.text, encoding="utf-8")
                        raise ValueError(
                            f"JSON parse: {parse_error} "
                            f"(raw saved {raw_dump.name})")
                    data["_provenance"] = {"key": key, "model": model_name}
                    out_path.write_text(json.dumps(data, ensure_ascii=False),
                                        encoding="utf-8")
                    return
                except (ResourceExhausted, InternalServerError,
                        ServiceUnavailable) as e:
                    attempt += 1
                    if attempt >= max_retries:
                        _append_failure(failure_log, key,
                                        f"max retries: {e}")
                        return
                    await asyncio.sleep((2 ** attempt) + random.uniform(0, 1))
                except Exception as e:
                    _append_failure(failure_log, key, f"fatal: {e}")
                    return
        except Exception as e:
            _append_failure(failure_log, key, f"preprocess: {e}")


def _load_env(cwd: Path) -> None:
    env = cwd / ".env"
    if not env.is_file():
        return
    for raw in env.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


async def _main_async(args: argparse.Namespace) -> int:
    _load_env(Path.cwd())
    api_key = (os.environ.get("GEMINI_API_KEY")
               or os.environ.get("GOOGLE_API_KEY"))
    if not api_key:
        raise SystemExit("GEMINI_API_KEY (or GOOGLE_API_KEY) not in env / .env")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        args.model,
        system_instruction=SYSTEM_PROMPT,
        generation_config={"response_mime_type": "application/json"},
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    failure_log = args.out_dir / "_failures.jsonl"

    logger.info("indexing source dirs ...")
    tei_map = index_directory(args.tei_dir, (".xml",), TEI_SUFFIXES)
    pymu_map = index_directory(args.pymu_dir, (".md", ".jsonl"), PYMU_SUFFIXES)
    raw_map = index_directory(args.raw_dir, (".txt",), RAW_SUFFIXES)
    logger.info("indexed: tei=%d pymu=%d raw=%d",
                len(tei_map), len(pymu_map), len(raw_map))

    if args.list_file and args.list_file.is_file():
        keys = [ln.strip() for ln in args.list_file.read_text().splitlines()
                if ln.strip()]
        logger.info("restricting to %d paper_ids from %s",
                    len(keys), args.list_file)
    else:
        keys = sorted(set(tei_map) | set(pymu_map) | set(raw_map))
        logger.info("processing all %d paper_ids found across source dirs",
                    len(keys))

    sem = asyncio.Semaphore(args.max_concurrent)
    tasks = []
    for key in keys:
        paths = {"tei": tei_map.get(key),
                 "pymu": pymu_map.get(key),
                 "raw": raw_map.get(key)}
        if not any(paths.values()):
            continue
        tasks.append(_process_paper(
            sem, model, key, paths, args.out_dir, failure_log,
            args.model, args.max_retries,
            args.tei_chars, args.layout_chars, args.raw_chars,
        ))

    logger.info("dispatching %d tasks (concurrency=%d, model=%s)",
                len(tasks), args.max_concurrent, args.model)
    done = 0
    for fut in asyncio.as_completed(tasks):
        await fut
        done += 1
        if done % 200 == 0:
            logger.info("  %d / %d done", done, len(tasks))
    logger.info("run complete; failures in %s", failure_log)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tei-dir", type=Path, required=True)
    ap.add_argument("--pymu-dir", type=Path, required=True)
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--list-file", type=Path, default=None,
                    help="Optional newline-separated paper_ids to restrict "
                         "to. Omit to process every paper found across the "
                         "three source dirs.")
    ap.add_argument("--model", default="gemini-1.5-flash",
                    help="Gemini model id. The existing corpus was built with "
                         "gemini-2.5-flash-lite; gemini-1.5-flash is the "
                         "stable, more conservative choice.")
    ap.add_argument("--max-concurrent", type=int, default=50,
                    help="Concurrent API requests. Notebook used 200; default "
                         "lowered to 50 to stay well under per-minute quotas "
                         "on shared keys.")
    ap.add_argument("--max-retries", type=int, default=10)
    ap.add_argument("--tei-chars", type=int, default=8000)
    ap.add_argument("--layout-chars", type=int, default=8000)
    ap.add_argument("--raw-chars", type=int, default=4000)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
