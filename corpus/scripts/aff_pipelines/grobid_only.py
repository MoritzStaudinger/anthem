"""GROBID-only pipeline — parse TEI XML headers, extract author + affiliation
strings, optionally ROR-link the raw strings (offline lookup).

This is the 'naive' baseline: just GROBID without the Qwen LLM cleanup
step our main pipeline uses. Gives us a clean "what does GROBID alone
buy you?" data point in the comparison table.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET

from . import (
    PaperPrediction, PaperSources, PredictedAffiliation, PredictedAuthor,
)

# TEI namespace handling
_TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag


def _all_text(el) -> str:
    return " ".join(t.strip() for t in el.itertext() if t.strip())


class GrobidOnly:
    """No LLM, no normalisation — just GROBID's parsed TEI headers."""
    pipeline_id = "grobid_only"

    def __init__(self, tei_dir: Optional[Path] = None,
                 ror_resolver=None):
        """
        Args:
            tei_dir: fallback dir to read <paper_id>.tei.xml when
                     sources.tei_xml is None.
            ror_resolver: optional callable (str) -> dict that returns
                     {ror_id, clean_institution_name, country_code} for
                     a raw affiliation string. None disables ROR linking
                     (raw_only_baseline).
        """
        self.tei_dir = Path(tei_dir) if tei_dir else None
        self.ror_resolver = ror_resolver

    def _load_tei(self, sources: PaperSources) -> Optional[str]:
        if sources.tei_xml:
            return sources.tei_xml
        if self.tei_dir:
            for fname in (f"{sources.paper_id}.tei.xml", f"{sources.paper_id}.xml"):
                fp = self.tei_dir / fname
                if fp.is_file():
                    return fp.read_text(encoding="utf-8", errors="replace")
        return None

    def predict(self, sources: PaperSources) -> PaperPrediction:
        t0 = time.time()
        xml = self._load_tei(sources)
        if not xml:
            return PaperPrediction(
                paper_id=sources.paper_id, pipeline_id=self.pipeline_id,
                extraction_seconds=time.time() - t0,
                error="no TEI XML available",
            )
        try:
            root = ET.fromstring(xml)
        except ET.ParseError as e:
            return PaperPrediction(
                paper_id=sources.paper_id, pipeline_id=self.pipeline_id,
                extraction_seconds=time.time() - t0,
                error=f"TEI parse error: {e}",
            )

        # Walk the GROBID author/affiliation structure under teiHeader.
        # GROBID emits one <author> per person, with one or more nested
        # <affiliation> elements. <orgName> children carry the institution
        # text (with type='institution','department','laboratory').
        authors = []
        for author_el in root.iter("{http://www.tei-c.org/ns/1.0}author"):
            persname = author_el.find("tei:persName", _TEI_NS)
            if persname is None:
                continue
            name = " ".join(_all_text(persname).split())
            if not name:
                continue
            affs = []
            for aff_el in author_el.findall("tei:affiliation", _TEI_NS):
                # Prefer institution-typed orgName; fall back to all text
                inst_names = []
                for org in aff_el.findall("tei:orgName", _TEI_NS):
                    if (org.attrib.get("type") or "") in ("institution", "laboratory", ""):
                        txt = _all_text(org)
                        if txt:
                            inst_names.append(txt)
                if not inst_names:
                    raw = _all_text(aff_el)
                else:
                    raw = ", ".join(inst_names)
                if not raw:
                    continue
                pa = PredictedAffiliation(raw=raw)
                if self.ror_resolver:
                    resolved = self.ror_resolver(raw) or {}
                    pa.clean_institution_name = resolved.get("clean_institution_name")
                    pa.ror_id = resolved.get("ror_id")
                    pa.country_code = resolved.get("country_code")
                affs.append(pa)
            authors.append(PredictedAuthor(name=name, affiliations=affs))

        return PaperPrediction(
            paper_id=sources.paper_id, pipeline_id=self.pipeline_id,
            authors=authors, extraction_seconds=time.time() - t0,
        )
