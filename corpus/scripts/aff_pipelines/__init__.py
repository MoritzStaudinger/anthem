"""Pipeline registry + common types for affiliation-extraction comparison.

Every pipeline is a class with:
    pipeline_id (str)
    predict(paper_id: str, sources: PaperSources) -> PaperPrediction

`PaperSources` is a bundle of every available view of a paper (TEI XML,
PyMu Markdown, Hybrid OCR text, S2ORC record, abstract, etc.) so that
each pipeline picks the inputs it needs without re-fetching from disk.

`PaperPrediction` is the canonical output schema — same shape regardless
of pipeline, so the eval harness can score any pipeline uniformly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol


@dataclass
class PaperSources:
    """All available representations of a paper. Pipelines pick what they need."""
    paper_id: str
    pdf_path: Optional[str] = None
    tei_xml: Optional[str] = None        # GROBID TEI XML body
    pymu_md: Optional[str] = None        # PyMu_LLM markdown text
    hybrid_txt: Optional[str] = None     # Hybrid OCR text
    s2orc_record: Optional[dict] = None  # S2ORC raw record (if available)
    title: Optional[str] = None
    abstract: Optional[str] = None
    year: Optional[int] = None


@dataclass
class PredictedAffiliation:
    raw: str                              # Verbatim extracted string
    clean_institution_name: Optional[str] = None  # ROR-canonical name when linked
    ror_id: Optional[str] = None
    country_code: Optional[str] = None


@dataclass
class PredictedAuthor:
    name: str
    affiliations: List[PredictedAffiliation] = field(default_factory=list)


@dataclass
class PaperPrediction:
    paper_id: str
    authors: List[PredictedAuthor] = field(default_factory=list)
    pipeline_id: str = ""
    # Lightweight telemetry — for cost/throughput comparison across pipelines
    extraction_seconds: Optional[float] = None
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    error: Optional[str] = None           # When extraction fails (we still emit a row)

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "authors": [
                {
                    "name": a.name,
                    "affiliations": [
                        {
                            "raw_affiliation": af.raw,
                            "clean_institution_name": af.clean_institution_name,
                            "ror_id": af.ror_id,
                            "country_code": af.country_code,
                        } for af in a.affiliations
                    ],
                } for a in self.authors
            ],
            "_pipeline": {
                "id": self.pipeline_id,
                "extraction_seconds": self.extraction_seconds,
                "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out,
                "error": self.error,
            },
        }


class Pipeline(Protocol):
    """Pipeline interface — implement predict()."""
    pipeline_id: str
    def predict(self, sources: PaperSources) -> PaperPrediction: ...


# Module-level registry. Subpackages register on import.
PIPELINES: Dict[str, "Pipeline"] = {}


def register(pipeline: "Pipeline") -> "Pipeline":
    """Decorator-friendly registration: `@register` on a class with a
    `pipeline_id` attribute adds it to the global registry."""
    pid = getattr(pipeline, "pipeline_id", None)
    if not pid:
        raise ValueError(f"Pipeline {pipeline!r} missing pipeline_id")
    if pid in PIPELINES:
        raise ValueError(f"Pipeline id {pid!r} already registered")
    PIPELINES[pid] = pipeline
    return pipeline
