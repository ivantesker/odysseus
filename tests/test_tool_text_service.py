"""Tool text service — pure parsing/sniffing helpers."""
import pytest

from src.services import tool_text_service as t


def test_truncate():
    assert t.truncate("abc", 10) == "abc"
    out = t.truncate("x" * 20, 5)
    assert out.startswith("xxxxx") and "truncated, 20 chars" in out


def test_parse_tool_args_json_and_dict():
    assert t.parse_tool_args('{"action": "go"}') == {"action": "go"}
    assert t.parse_tool_args({"a": 1}) == {"a": 1}
    assert t.parse_tool_args("") == {}
    assert t.parse_tool_args(None) == {}


def test_parse_tool_args_unwraps_body_envelope():
    assert t.parse_tool_args({"body": {"action": "x", "v": 1}}) == {"action": "x", "v": 1}
    # Not unwrapped: body without 'action', or alongside other keys.
    assert t.parse_tool_args({"body": {"v": 1}}) == {"body": {"v": 1}}
    assert t.parse_tool_args({"body": {"action": "x"}, "to": "a"}) == {"body": {"action": "x"}, "to": "a"}


def test_parse_tool_args_bad_json_raises():
    with pytest.raises(ValueError):
        t.parse_tool_args("{not json}")


def test_looks_like_email_document():
    assert t.looks_like_email_document(title="New Email")
    assert t.looks_like_email_document("To: a@b.com\nSubject: hi\n---\nbody")
    assert not t.looks_like_email_document("just prose")


@pytest.mark.parametrize("text,lang", [
    ("def foo():\n    pass", "python"),
    ("function f(){}", "javascript"),
    ('{"a": 1}', "json"),
    ("<svg viewBox='0'></svg>", "svg"),
    ("<!doctype html><html></html>", "html"),
    ("SELECT * FROM t", "sql"),
    ("just some prose here", "markdown"),
    ("To: a@b.com\nSubject: hi", "email"),
    ("", "markdown"),
])
def test_sniff_doc_language(text, lang):
    assert t.sniff_doc_language(text) == lang


def test_coerce_email_document_keeps_separator_shape():
    # Already shaped: returned as-is.
    shaped = "To: a@b.com\nSubject: hi\n---\nbody"
    assert t.coerce_email_document_content("", shaped) == shaped
    # Body-only incoming gets the existing header + separator.
    out = t.coerce_email_document_content("To: a@b.com\nSubject: hi\n---\nold", "new body")
    assert out.endswith("\n---\nnew body")
    assert out.startswith("To: a@b.com")
