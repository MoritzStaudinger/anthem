"""Pass-through pipelines — read pre-extracted affiliations from disk.

These don't run any extraction at eval time; they just adapt the
existing per-paper JSON formats (ours, S2ORC normalized) into the
common PaperPrediction schema so the eval harness can score them
alongside live LLM pipelines.

Pipelines:
  ours                — current SciCo pipeline output (ROR-linked)
  s2orc               — S2ORC `authoraffiliation` annotations (post-normalize)
  s2_papers_only      — S2 papers dataset, no S2ORC structured affs
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from . import (
    PaperPrediction, PaperSources, PredictedAffiliation, PredictedAuthor,
)


class OursPassthrough:
    """Reads our pipeline's per-paper JSON (Qwen-extracted + ROR-linked)."""
    pipeline_id = "ours"

    def __init__(self, scico_root: Path):
        self.aff_dir = Path(scico_root) / "affiliations"
        if not self.aff_dir.is_dir():
            # Allow direct pointer to the dir itself
            self.aff_dir = Path(scico_root)

    def predict(self, sources: PaperSources) -> PaperPrediction:
        t0 = time.time()
        fp = self.aff_dir / f"{sources.paper_id}.json"
        if not fp.is_file():
            return PaperPrediction(
                paper_id=sources.paper_id, pipeline_id=self.pipeline_id,
                extraction_seconds=time.time() - t0,
                error=f"missing JSON: {fp}",
            )
        try:
            d = json.loads(fp.read_text(encoding="utf-8", errors="replace"))
        except json.JSONDecodeError as e:
            return PaperPrediction(
                paper_id=sources.paper_id, pipeline_id=self.pipeline_id,
                extraction_seconds=time.time() - t0, error=f"bad JSON: {e}",
            )
        authors = []
        for a in (d.get("authors") or []):
            affs = []
            for af in (a.get("affiliations") or []):
                affs.append(PredictedAffiliation(
                    raw=af.get("raw_affiliation") or af.get("clean_institution_name") or "",
                    clean_institution_name=af.get("clean_institution_name"),
                    ror_id=af.get("ror_id"),
                    country_code=af.get("country_code"),
                ))
            authors.append(PredictedAuthor(name=a.get("name") or "", affiliations=affs))
        return PaperPrediction(
            paper_id=sources.paper_id, pipeline_id=self.pipeline_id,
            authors=authors, extraction_seconds=time.time() - t0,
        )


class S2ORCPassthrough:
    """Reads S2ORC-normalized records (output of scripts/s2orc_normalize.py).

    Our normalizer extracts a flat list of affiliation strings per paper
    (S2ORC's `authoraffiliation` spans aren't per-author-aligned in the
    raw release). We attribute every extracted affiliation to a single
    synthetic 'paper-level' author — match scoring is on the union of
    affiliation strings, so per-author attribution doesn't affect F1.

    When a `ror_resolver` is supplied, each extracted affiliation string
    is ROR-linked at read time so the ROR-based scoring track has a
    populated `ror_id` field per affiliation. Without a resolver, only
    the raw strings are emitted and ROR scoring will see this pipeline
    as "no ROR coverage" (zero precision/recall on ROR axis).
    """
    pipeline_id = "s2orc"

    def __init__(self, s2orc_norm_dir: Path, ror_resolver=None):
        self.aff_dir = Path(s2orc_norm_dir) / "affiliations"
        if not self.aff_dir.is_dir():
            self.aff_dir = Path(s2orc_norm_dir)
        self.ror_resolver = ror_resolver

    def predict(self, sources: PaperSources) -> PaperPrediction:
        t0 = time.time()
        fp = self.aff_dir / f"{sources.paper_id}.json"
        if not fp.is_file():
            return PaperPrediction(
                paper_id=sources.paper_id, pipeline_id=self.pipeline_id,
                extraction_seconds=time.time() - t0,
                error=f"S2ORC has no record for {sources.paper_id}",
            )
        try:
            d = json.loads(fp.read_text(encoding="utf-8", errors="replace"))
        except json.JSONDecodeError as e:
            return PaperPrediction(
                paper_id=sources.paper_id, pipeline_id=self.pipeline_id,
                extraction_seconds=time.time() - t0, error=f"bad JSON: {e}",
            )
        # S2ORC normalize.py output: affiliations as a flat list of strings.
        flat_affs = d.get("affiliations") or []
        authors_raw = d.get("authors") or []
        if not flat_affs and not authors_raw:
            return PaperPrediction(
                paper_id=sources.paper_id, pipeline_id=self.pipeline_id,
                extraction_seconds=time.time() - t0,
            )
        # Pseudo-author bucket — every affiliation under one umbrella name.
        # The eval scorer compares the union of affiliation strings.
        affs = []
        for s in flat_affs:
            pa = PredictedAffiliation(raw=s)
            if self.ror_resolver:
                resolved = self.ror_resolver(s) or {}
                pa.clean_institution_name = resolved.get("clean_institution_name")
                pa.ror_id = resolved.get("ror_id")
                pa.country_code = resolved.get("country_code")
            affs.append(pa)
        synth_author = PredictedAuthor(
            name="(s2orc-aggregate)", affiliations=affs,
        )
        return PaperPrediction(
            paper_id=sources.paper_id, pipeline_id=self.pipeline_id,
            authors=[synth_author], extraction_seconds=time.time() - t0,
        )
