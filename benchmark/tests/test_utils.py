import io
from types import SimpleNamespace

from backend.utils.utils import (
    compute_upload_hash,
    extract_year,
    remove_duplicates_by_doi,
    remove_duplicates_by_doi_dict,
)


class _FakeFileStorage:
    def __init__(self, content: bytes, filename: str = "x.pdf"):
        self.filename = filename
        self.stream = io.BytesIO(content)

    def read(self, *a, **kw):
        return self.stream.read(*a, **kw)


def test_compute_upload_hash_is_deterministic():
    a = _FakeFileStorage(b"hello world")
    b = _FakeFileStorage(b"hello world")
    assert compute_upload_hash(a) == compute_upload_hash(b)


def test_compute_upload_hash_differs_for_different_content():
    a = _FakeFileStorage(b"hello world")
    b = _FakeFileStorage(b"hello world!")
    assert compute_upload_hash(a) != compute_upload_hash(b)


def test_compute_upload_hash_rewinds_stream():
    fs = _FakeFileStorage(b"hello world")
    compute_upload_hash(fs)
    assert fs.stream.read() == b"hello world"


def test_extract_year_iso():
    assert extract_year("2024-05-12") == 2024


def test_extract_year_invalid_returns_none():
    assert extract_year("not a date") is None
    assert extract_year(None) is None


def test_remove_duplicates_by_doi_dict_keeps_first():
    papers = [
        {"DOI": "10.1/a", "title": "first"},
        {"DOI": "10.1/a", "title": "dup"},
        {"DOI": "10.2/b", "title": "second"},
    ]
    out = remove_duplicates_by_doi_dict(papers)
    assert [p["title"] for p in out] == ["first", "second"]


def test_remove_duplicates_by_doi_dict_drops_missing_doi():
    papers = [{"DOI": None, "title": "no-doi"}, {"DOI": "", "title": "empty"}]
    assert remove_duplicates_by_doi_dict(papers) == []


def test_remove_duplicates_by_doi_for_node_objects():
    def make(doi):
        return SimpleNamespace(node=SimpleNamespace(metadata={"doi": doi}))

    nodes = [make("10.1/a"), make("10.1/a"), make("10.2/b")]
    out = remove_duplicates_by_doi(nodes)
    assert [n.node.metadata["doi"] for n in out] == ["10.1/a", "10.2/b"]
