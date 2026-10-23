"""Configurable LLM-based affiliation extractor.

One class, many pipelines: each (model, backend, input_template) combo
becomes its own pipeline_id. Lets you run a single matrix-style experiment
across {gpt-4o, claude-3-5-sonnet, gemini-1.5-pro, qwen-3.5-397b, ...} ×
{abstract-only, first-page, GROBID-header, PyMu-first-N, hybrid-OCR-first-N}.

Backends supported:
    openai      — official OpenAI API
    anthropic   — Anthropic API
    google      — Google Generative AI
    aqueduct    — TU Wien Aqueduct gateway (Qwen + others, free for TU staff)
    openrouter  — OpenRouter (any model on their catalogue)

Input templates:
    title_abstract       — title + abstract only
    pymu_first_n         — first N characters of PyMu_LLM markdown
    hybrid_first_n       — first N characters of hybrid OCR text
    grobid_header        — only the TEI header section text
    pdf_first_page_text  — first page of PDF as extracted by PyMu (no LLM)
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import xml.etree.ElementTree as ET

import requests

from . import (
    PaperPrediction, PaperSources, PredictedAffiliation, PredictedAuthor,
)

logger = logging.getLogger("aff_llm")


# -------------------------- input template construction --------------------------

def _title_abstract(sources: PaperSources, **_) -> str:
    parts = []
    if sources.title:
        parts.append(f"Title: {sources.title.strip()}")
    if sources.abstract:
        parts.append(f"Abstract: {sources.abstract.strip()}")
    return "\n\n".join(parts) or "(no title/abstract available)"


def _pymu_first_n(sources: PaperSources, n: int = 5000, **_) -> str:
    if not sources.pymu_md:
        return "(no PyMu markdown available)"
    return sources.pymu_md[:n]


def _hybrid_first_n(sources: PaperSources, n: int = 5000, **_) -> str:
    if not sources.hybrid_txt:
        return "(no hybrid OCR text available)"
    return sources.hybrid_txt[:n]


_TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}

def _grobid_header_text(sources: PaperSources, **_) -> str:
    if not sources.tei_xml:
        return "(no TEI XML available)"
    try:
        root = ET.fromstring(sources.tei_xml)
    except ET.ParseError:
        return "(TEI parse failed)"
    header = root.find(".//tei:teiHeader", _TEI_NS)
    if header is None:
        return "(no teiHeader)"
    return " ".join(t.strip() for t in header.itertext() if t.strip())


def _combined(sources: PaperSources, pymu_n: int = 3000,
              hybrid_n: int = 2000, **_) -> str:
    """All available views stacked into one prompt — title + abstract +
    GROBID header + first N chars of PyMu markdown (+ hybrid OCR if it's
    the only body we have). Tests whether richer context buys accuracy
    beyond a single source view. Sections clearly delimited so the LLM
    can pick the cleanest one."""
    parts = []
    if sources.title:
        parts.append(f"=== TITLE ===\n{sources.title.strip()}")
    if sources.abstract:
        parts.append(f"=== ABSTRACT ===\n{sources.abstract.strip()}")
    header = _grobid_header_text(sources) if sources.tei_xml else None
    if header and not header.startswith("("):
        parts.append(f"=== GROBID HEADER ===\n{header[:5000]}")
    if sources.pymu_md:
        parts.append(f"=== PYMU BODY (first {pymu_n} chars) ===\n{sources.pymu_md[:pymu_n]}")
    elif sources.hybrid_txt:
        # Hybrid only when PyMu is missing — they're alternative renderings
        # of the same content, so concatenating both would just duplicate text.
        parts.append(f"=== HYBRID OCR (first {hybrid_n} chars) ===\n{sources.hybrid_txt[:hybrid_n]}")
    return "\n\n".join(parts) or "(no source views available)"


# Registry of available input renderers
INPUT_TEMPLATES: Dict[str, Callable[..., str]] = {
    "title_abstract":  _title_abstract,
    "pymu_first_n":    _pymu_first_n,
    "hybrid_first_n":  _hybrid_first_n,
    "grobid_header":   _grobid_header_text,
    "combined":        _combined,
}


# -------------------------- backend abstraction --------------------------

@dataclass
class LLMCallResult:
    text: str
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None


class _Backend:
    """Minimal backend interface — call(prompt) -> text."""
    name: str = "?"
    def call(self, system_prompt: str, user_prompt: str, model: str,
             temperature: float = 0.0, max_tokens: int = 1500) -> LLMCallResult:
        raise NotImplementedError


class AqueductBackend(_Backend):
    """TU Wien Aqueduct gateway. OpenAI-compatible API. Free for TU staff.

    Note on response shapes: Qwen reasoning models (qwen-3.5-397b,
    qwen-3.6-35b) sometimes return `reasoning_content` alongside or
    instead of `content`. Gemma + classic chat models return plain
    `content`. We fall back through both to handle either layout.
    """
    name = "aqueduct"
    URL = "https://aqueduct.ai.datalab.tuwien.ac.at/v1/chat/completions"

    def call(self, system_prompt, user_prompt, model, temperature=0.0, max_tokens=1500):
        api_key = os.environ.get("AQUEDUCT_API_KEY")
        if not api_key:
            raise RuntimeError("AQUEDUCT_API_KEY not set")
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature, "max_tokens": max_tokens,
        }
        r = requests.post(self.URL, json=body, timeout=180,
                          headers={"Authorization": f"Bearer {api_key}",
                                   "Content-Type": "application/json"})
        r.raise_for_status()
        obj = r.json()
        usage = obj.get("usage") or {}
        choice = (obj.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        # Qwen reasoning models can return reasoning_content with content=null;
        # we just want whichever non-empty text the model emitted.
        text = msg.get("content") or msg.get("reasoning_content") or ""
        return LLMCallResult(
            text=text,
            tokens_in=usage.get("prompt_tokens"),
            tokens_out=usage.get("completion_tokens"),
        )


class OpenAIBackend(_Backend):
    """OpenAI Chat Completions. Handles the param-name + temperature
    constraints introduced by reasoning-class models (o1/o3/o4/gpt-5/*-nano):
      - use `max_completion_tokens` instead of `max_tokens`
      - omit `temperature` (only the default 1 is accepted; sending 0 errors)
    """
    name = "openai"

    @staticmethod
    def _is_reasoning_model(model: str) -> bool:
        m = model.lower()
        return (m.startswith("o1") or m.startswith("o3") or m.startswith("o4")
                or m.startswith("gpt-5") or m.startswith("gpt-6"))

    def call(self, system_prompt, user_prompt, model, temperature=0.0, max_tokens=1500):
        from openai import OpenAI
        client = OpenAI()  # picks up OPENAI_API_KEY
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if self._is_reasoning_model(model):
            # New API surface: max_completion_tokens, default-only temperature.
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = temperature
        resp = client.chat.completions.create(**kwargs)
        usage = resp.usage
        return LLMCallResult(
            text=resp.choices[0].message.content or "",
            tokens_in=getattr(usage, "prompt_tokens", None),
            tokens_out=getattr(usage, "completion_tokens", None),
        )


class AnthropicBackend(_Backend):
    name = "anthropic"
    def call(self, system_prompt, user_prompt, model, temperature=0.0, max_tokens=1500):
        import anthropic
        client = anthropic.Anthropic()  # picks up ANTHROPIC_API_KEY
        resp = client.messages.create(
            model=model, max_tokens=max_tokens, temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = "".join(b.text for b in resp.content if hasattr(b, "text"))
        usage = resp.usage
        return LLMCallResult(
            text=text,
            tokens_in=getattr(usage, "input_tokens", None),
            tokens_out=getattr(usage, "output_tokens", None),
        )


class GoogleBackend(_Backend):
    """Google Generative AI (Gemini) via the native google-genai SDK."""
    name = "google"

    def call(self, system_prompt, user_prompt, model, temperature=0.0, max_tokens=1500):
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY (or GEMINI_API_KEY) not set")
        try:
            from google import genai
            from google.genai import types as gtypes
        except ImportError as e:
            raise RuntimeError(
                "google-genai package not installed. Run: pip install google-genai"
            ) from e
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=gtypes.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        # token counts live under usage_metadata; the field names changed
        # across SDK versions so we defensively pull whichever exists.
        usage = getattr(resp, "usage_metadata", None)
        tin = getattr(usage, "prompt_token_count", None) if usage else None
        tout = getattr(usage, "candidates_token_count", None) if usage else None
        return LLMCallResult(text=resp.text or "", tokens_in=tin, tokens_out=tout)


class OpenRouterBackend(_Backend):
    """OpenRouter — single key, any model. Great for trying open-weights
    models (Llama, Mistral, Qwen) without managing per-provider keys."""
    name = "openrouter"
    URL = "https://openrouter.ai/api/v1/chat/completions"

    def call(self, system_prompt, user_prompt, model, temperature=0.0, max_tokens=1500):
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")
        body = {
            "model": model, "temperature": temperature, "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        r = requests.post(self.URL, json=body, timeout=180,
                          headers={"Authorization": f"Bearer {api_key}",
                                   "Content-Type": "application/json"})
        r.raise_for_status()
        obj = r.json()
        usage = obj.get("usage") or {}
        return LLMCallResult(
            text=obj["choices"][0]["message"]["content"],
            tokens_in=usage.get("prompt_tokens"),
            tokens_out=usage.get("completion_tokens"),
        )


BACKENDS = {
    "aqueduct":   AqueductBackend(),
    "openai":     OpenAIBackend(),
    "anthropic":  AnthropicBackend(),
    "google":     GoogleBackend(),
    "openrouter": OpenRouterBackend(),
}


# -------------------------- prompt --------------------------

_SYSTEM_PROMPT = (
    "You are extracting author affiliations from the front matter of a "
    "scientific paper. Output strictly valid JSON. Do not invent "
    "affiliations not present in the input. Do not include emails or "
    "postal addresses — only institution names."
)


def _build_user_prompt(input_text: str, no_think: bool = False) -> str:
    """Builds the extraction prompt.

    `no_think`: prepends `/no_think` — a Qwen3-family directive that
    bypasses the model's reasoning loop. Without it, Qwen3 models can
    spend the entire token budget in <think> blocks and emit empty
    content. Has no effect on non-Qwen models (they ignore unknown
    directives in the user message).
    """
    prefix = "/no_think\n\n" if no_think else ""
    return prefix + (
        f"Below is the available text from the paper's front matter:\n"
        f"---\n{input_text}\n---\n\n"
        f"Extract every author and their affiliated institution(s). "
        f"Return a JSON object exactly matching this schema, and NOTHING else:\n"
        f'{{\n  "authors": [\n    {{"name": "<full name>", '
        f'"affiliations": ["<institution 1>", "<institution 2>"]}},\n    ...\n  ]\n}}\n\n'
        f"If an author has no affiliation visible in the text, set their "
        f"affiliations to []. If no authors at all are extractable, return "
        f'{{"authors": []}}. Output ONLY the JSON — no prose, no markdown '
        f"code fences, no commentary."
    )


_JSON_RE = re.compile(r"\{[\s\S]*?\}", re.DOTALL)
_THINK_TAG_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)


def _parse_llm_json(text: str) -> dict:
    """Tolerant JSON extraction from LLM output.

    Reasoning models (Qwen-3.x, R1-style) emit `<think>...</think>` blocks
    before the final JSON, sometimes containing partial JSON-looking text
    that would mislead a first-match regex. We:
      1. Strip <think> blocks if present.
      2. Strip code fences.
      3. Try whole-text parse.
      4. Fallback: find every {...} candidate, return the LAST one that
         parses AND contains an 'authors' key (the actual answer; earlier
         candidates may be intermediate reasoning artifacts).
    """
    text = text.strip()
    # Remove reasoning blocks first — they often contain partial JSON
    text = _THINK_TAG_RE.sub("", text).strip()
    # Strip code fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # Whole-text parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Find every candidate JSON block; prefer the LAST one with 'authors'
    candidates = []
    # Use a balanced-brace scan since regex can't handle nested braces.
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start:i + 1])
                start = None
    # Walk from the END — final answer is most likely at the bottom.
    for cand in reversed(candidates):
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        # Prefer candidates that look like the schema we asked for
        if isinstance(obj, dict) and "authors" in obj:
            return obj
        # Fallback: any parseable dict
        if not candidates[0] == cand:  # last-resort
            continue
        return obj
    return {}


# -------------------------- pipeline --------------------------

@dataclass
class LLMPipeline:
    """One LLM × one input-template combo. Optionally ROR-link the raw
    extracted strings via a post-processor.

    Construct with whatever pipeline_id you want; e.g. 'gpt4_titleabstract'.
    """
    pipeline_id: str
    backend: str        # key into BACKENDS
    model: str          # model id at the chosen backend
    input_template: str # key into INPUT_TEMPLATES
    input_kwargs: dict = field(default_factory=dict)  # e.g. {"n": 5000} for *_first_n
    temperature: float = 0.0
    # 4000 instead of 1500: reasoning models (Qwen-3.x) consume token
    # budget for <think> blocks before emitting the actual JSON answer.
    # Non-reasoning models won't notice the bigger ceiling.
    max_tokens: int = 4000
    ror_resolver: Optional[Callable[[str], dict]] = None

    def __post_init__(self):
        if self.backend not in BACKENDS:
            raise ValueError(f"unknown backend {self.backend!r}, want one of {list(BACKENDS)}")
        if self.input_template not in INPUT_TEMPLATES:
            raise ValueError(f"unknown input_template {self.input_template!r}, "
                             f"want one of {list(INPUT_TEMPLATES)}")
        self._backend_impl = BACKENDS[self.backend]
        self._input_fn = INPUT_TEMPLATES[self.input_template]

    def predict(self, sources: PaperSources) -> PaperPrediction:
        t0 = time.time()
        input_text = self._input_fn(sources, **self.input_kwargs)
        # Qwen3-family models default to a reasoning loop that consumes the
        # entire token budget before emitting JSON. Bypass with /no_think.
        # Easy heuristic: any model id containing "qwen" + "3" qualifies.
        m_lower = self.model.lower()
        no_think = ("qwen" in m_lower and "3" in m_lower
                    and "embedding" not in m_lower)
        user_prompt = _build_user_prompt(input_text, no_think=no_think)
        try:
            result = self._backend_impl.call(
                _SYSTEM_PROMPT, user_prompt, self.model,
                temperature=self.temperature, max_tokens=self.max_tokens,
            )
        except Exception as e:
            return PaperPrediction(
                paper_id=sources.paper_id, pipeline_id=self.pipeline_id,
                extraction_seconds=time.time() - t0,
                error=f"{type(e).__name__}: {e}",
            )
        obj = _parse_llm_json(result.text)
        authors_struct = obj.get("authors") or []
        authors = []
        for a in authors_struct:
            if not isinstance(a, dict):
                continue
            name = (a.get("name") or "").strip()
            if not name:
                continue
            affs = []
            for af_str in (a.get("affiliations") or []):
                if not isinstance(af_str, str):
                    continue
                af_str = af_str.strip()
                if not af_str:
                    continue
                pa = PredictedAffiliation(raw=af_str)
                if self.ror_resolver:
                    resolved = self.ror_resolver(af_str) or {}
                    pa.clean_institution_name = resolved.get("clean_institution_name")
                    pa.ror_id = resolved.get("ror_id")
                    pa.country_code = resolved.get("country_code")
                affs.append(pa)
            authors.append(PredictedAuthor(name=name, affiliations=affs))
        return PaperPrediction(
            paper_id=sources.paper_id, pipeline_id=self.pipeline_id,
            authors=authors,
            extraction_seconds=time.time() - t0,
            tokens_in=result.tokens_in, tokens_out=result.tokens_out,
        )
