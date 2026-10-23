"""Unit tests for the citation-format validator (no LLM calls)."""

from backend.rag.rag_utils import (
    _AUTHOR_YEAR_RE,
    _BARE_DOI_RE,
    _DOI_URL_RE,
    _citation_violations,
    _split_body_refs,
    _strip_dois_in_body,
)


def test_doi_url_regex_matches_common_forms():
    assert _DOI_URL_RE.search("see https://doi.org/10.1234/abc.")
    assert _DOI_URL_RE.search("(http://dx.doi.org/10.1/x)")
    assert not _DOI_URL_RE.search("the dois are listed below")


def test_bare_doi_regex():
    assert _BARE_DOI_RE.search("paper 10.18653/v1/N19-1423 is great")
    assert not _BARE_DOI_RE.search("year 2024 mattered")


def test_author_year_regex():
    assert _AUTHOR_YEAR_RE.search("BERT (Devlin et al., 2019) reshaped NLP")
    assert _AUTHOR_YEAR_RE.search("(Smith and Jones, 2020) showed")
    assert _AUTHOR_YEAR_RE.search("(Müller, 2021)")
    # parens around plain year alone — not a citation, must not match
    assert not _AUTHOR_YEAR_RE.search("(2023) saw new methods")


def test_split_body_refs_no_header():
    text = "Just a body, no references."
    body, refs = _split_body_refs(text)
    assert body == text
    assert refs == ""


def test_split_body_refs_with_header():
    text = "Body para [1].\n\n## References\n[1] Smith. (2020). https://doi.org/10.1/x"
    body, refs = _split_body_refs(text)
    assert body.strip().endswith("[1].")
    assert "References" in refs
    assert "[1] Smith" in refs


def test_violations_caught_in_body_only():
    text = (
        "Foo (Smith et al., 2023) and https://doi.org/10.1/x.\n\n"
        "## References\n"
        "[1] Smith. (2023). https://doi.org/10.1/x\n"
    )
    body, _ = _split_body_refs(text)
    issues = _citation_violations(body)
    kinds = {kind for kind, _ in issues}
    assert "doi_url_in_body" in kinds
    assert "author_year_in_body" in kinds


def test_strip_dois_in_body_keeps_refs_intact():
    text = (
        "Foo https://doi.org/10.1/x bar.\n\n"
        "## References\n"
        "[1] Smith. (2023). https://doi.org/10.1/x\n"
    )
    cleaned = _strip_dois_in_body(text)
    body, refs = _split_body_refs(cleaned)
    assert "doi.org" not in body
    assert "doi.org/10.1/x" in refs


def test_strip_dois_no_op_without_refs_section():
    text = "Body with https://doi.org/10.1/x and no references list."
    assert _strip_dois_in_body(text) == text
