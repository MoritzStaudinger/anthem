import json

import pytest

from backend.rag.rag_utils import extract_first_json_object


def test_extracts_plain_json():
    assert extract_first_json_object('{"a": 1}') == {"a": 1}


def test_extracts_json_with_surrounding_text():
    raw = 'Here is the answer:\n{"type": "paper", "year_start": 2020}\nThanks!'
    assert extract_first_json_object(raw) == {"type": "paper", "year_start": 2020}


def test_handles_nested_objects():
    raw = '{"a": {"b": {"c": 1}}, "d": 2}'
    assert extract_first_json_object(raw) == {"a": {"b": {"c": 1}}, "d": 2}


def test_raises_when_no_json():
    with pytest.raises(ValueError):
        extract_first_json_object("no braces here")


def test_raises_on_unbalanced_braces():
    with pytest.raises((ValueError, json.JSONDecodeError)):
        extract_first_json_object("{ unbalanced")
