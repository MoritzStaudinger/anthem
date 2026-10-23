"""Unit tests for the SciCo eval metrics. Deterministic, no LLM, no index."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.eval.metrics import (
    affiliation_correctness,
    estimate_cost_usd,
    institution_list_metrics,
    parse_references,
    phantom_citation_rate,
    precision_recall_mrr,
)


# --- precision / recall / mrr -----------------------------------------------

def test_prm_full_recall_and_precision():
    res = precision_recall_mrr(["a", "b"], ["a", "b"])
    assert res["recall"] == 1.0
    assert res["precision"] == 1.0
    assert res["mrr"] == 1.0
    assert res["f1"] == 1.0


def test_prm_partial_with_noise():
    res = precision_recall_mrr(["x", "a", "y", "b"], ["a", "b", "c"], k=4)
    # 2 of 3 expected found, 2 of 4 retrieved are relevant
    assert math.isclose(res["recall"], 2 / 3)
    assert math.isclose(res["precision"], 0.5)
    # First hit at rank 2
    assert math.isclose(res["mrr"], 0.5)


def test_prm_no_gold_returns_nan():
    res = precision_recall_mrr(["a"], [])
    assert math.isnan(res["recall"])
    assert math.isnan(res["precision"])
    assert math.isnan(res["mrr"])


def test_prm_mrr_only_first_hit_counts():
    res = precision_recall_mrr(["x", "y", "a", "a"], ["a"])
    assert math.isclose(res["mrr"], 1 / 3)


# --- institution_list -------------------------------------------------------

def test_institution_list_substring_match_p_r_f1():
    sys_affs = ["Alibaba Group (China)", "Stanford University", "Zhejiang Lab"]
    expected = ["Alibaba Group", "Zhejiang Lab", "Microsoft Research"]
    res = institution_list_metrics(sys_affs, expected)
    assert math.isclose(res["recall"], 2 / 3)
    # 2 of the 3 sys affs match an expected (Stanford doesn't)
    assert math.isclose(res["precision"], 2 / 3)
    assert res["system_size"] == 3


def test_institution_list_no_gold():
    res = institution_list_metrics(["X"], [])
    assert math.isnan(res["recall"])
    assert math.isnan(res["f1"])


# --- phantom citations ------------------------------------------------------

def test_parse_references_extracts_paper_id_and_url():
    text = (
        "Body [1] [2].\n\n"
        "### References\n\n"
        "[1] Smith, J. (2021). Thing. https://doi.org/10.18653/v1/2021.acl-long.236\n"
        "\n"
        "[2] Jones, K. et al. (2021). Other. https://aclanthology.org/2021.acl-long.541/\n"
    )
    refs = parse_references(text)
    assert len(refs) == 2
    assert refs[0]["url"].startswith("https://doi.org/")
    assert refs[1]["paper_id"] == "2021.acl-long.541"


def test_phantom_citation_rate_zero_when_all_in_evidence():
    answer = (
        "Body [1].\n\n"
        "### References\n\n"
        "[1] Smith. (2021). https://aclanthology.org/2021.acl-long.236/\n"
    )
    evidence = [{"url": "https://aclanthology.org/2021.acl-long.236/", "paper_id": "2021.acl-long.236"}]
    res = phantom_citation_rate(answer, evidence)
    assert res["phantom_rate"] == 0.0
    assert res["total_refs"] == 1


def test_phantom_citation_rate_detects_missing():
    answer = (
        "Body [1] [2].\n\n"
        "### References\n\n"
        "[1] In evidence. https://aclanthology.org/2021.acl-long.236/\n"
        "\n"
        "[2] Phantom. https://aclanthology.org/9999.acl-long.000/\n"
    )
    evidence = [{"url": "https://aclanthology.org/2021.acl-long.236/", "paper_id": "2021.acl-long.236"}]
    res = phantom_citation_rate(answer, evidence)
    assert res["phantom_rate"] == 0.5
    assert res["phantom_count"] == 1
    assert res["total_refs"] == 2


def test_phantom_citation_rate_nan_when_no_refs_section():
    res = phantom_citation_rate("Just prose.", [{"url": "x"}])
    assert math.isnan(res["phantom_rate"])


# --- affiliation correctness -----------------------------------------------

def test_affiliation_correctness_perfect_match():
    gold = {"2021.acl-long.236": {"alibaba group", "zhejiang lab"}}
    retrieved = [
        {"paper_id": "2021.acl-long.236", "affiliations": ["Alibaba Group", "Zhejiang Lab"]},
    ]
    res = affiliation_correctness(retrieved, gold)
    assert math.isclose(res["mean_f1"], 1.0)
    assert res["graded"] == 1
    assert res["skipped"] == 0


def test_affiliation_correctness_partial():
    # System has Alibaba but missed Zhejiang Lab; gold has both
    gold = {"x": {"alibaba", "zhejiang"}}
    retrieved = [{"paper_id": "x", "affiliations": ["Alibaba"]}]
    res = affiliation_correctness(retrieved, gold)
    # tp=1, sys=1, gold=2 → P=1, R=0.5, F1=0.667
    assert math.isclose(res["mean_f1"], 2 / 3, rel_tol=1e-6)


def test_affiliation_correctness_skips_when_paper_not_in_gold():
    gold = {"x": {"alibaba"}}
    retrieved = [{"paper_id": "y", "affiliations": ["Anything"]}]
    res = affiliation_correctness(retrieved, gold)
    assert math.isnan(res["mean_f1"])
    assert res["skipped"] == 1


# --- pricing ----------------------------------------------------------------

def test_estimate_cost_usd_known_model():
    # gpt-4o-mini at 1k input + 1k output = $0.00075 input + $0.0006 output? wait
    # rate is per-1K — so 1000 in tokens = 0.00015 * 1 = 0.00015
    # 1000 out tokens = 0.00060 * 1 = 0.00060 => total 0.00075
    cost = estimate_cost_usd("gpt-4o-mini", 1000, 1000)
    assert math.isclose(cost, 0.00015 + 0.00060, rel_tol=1e-6)


def test_estimate_cost_usd_unknown_model():
    assert estimate_cost_usd("unknown-model-xyz", 100, 100) is None


def test_estimate_cost_usd_aqueduct_unpriced():
    assert estimate_cost_usd("qwen-3.6-35b", 100, 100) is None
