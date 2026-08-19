#!/usr/bin/env python3
"""Tests for the multi-agent washer (scripts/multi_agent_washer.py).

Covers the pure selection/formatting helpers and the parallel orchestration
path with ``call_ollama`` / ``analyze_text`` mocked out (no Ollama needed).
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.multi_agent_washer import (  # noqa: E402
    AGENT_MODELS,
    WashCandidate,
    _wash_with_model,
    build_parser,
    format_candidate_summary,
    format_results_table,
    multi_agent_wash,
    rank_candidates,
)


# --------------------------------------------------------------------------- #
# Model pool
# --------------------------------------------------------------------------- #
def test_agent_models_has_three_entries():
    assert len(AGENT_MODELS) == 3


def test_agent_models_have_distinct_labels():
    labels = [m["label"] for m in AGENT_MODELS]
    assert sorted(labels) == ["fast", "premium", "standard"]


def test_agent_models_have_required_keys():
    for m in AGENT_MODELS:
        for key in ("name", "temperature", "max_tokens", "label"):
            assert key in m


# --------------------------------------------------------------------------- #
# WashCandidate
# --------------------------------------------------------------------------- #
def test_wash_candidate_is_error_false_when_no_error():
    c = WashCandidate("text", "llama3.2", "standard", 1.0, 42.0)
    assert not c.is_error
    assert c.error is None


def test_wash_candidate_is_error_true_when_error_set():
    c = WashCandidate("", "llama3.2", "standard", 1.0, 999.0, error="boom")
    assert c.is_error


# --------------------------------------------------------------------------- #
# rank_candidates
# --------------------------------------------------------------------------- #
def _mk(model: str, score: float, label: str = "x", err: str | None = None) -> WashCandidate:
    return WashCandidate("t", model, label, 1.0, score, error=err)


def test_rank_candidates_picks_lowest_score():
    cands = [_mk("a", 60.0), _mk("b", 20.0), _mk("c", 40.0)]
    best = rank_candidates(cands)
    assert best.model == "b"
    assert best.ai_score == 20.0


def test_rank_candidates_excludes_errors_when_success_exists():
    cands = [_mk("a", 999.0, err="fail"), _mk("b", 20.0)]
    best = rank_candidates(cands)
    assert best.model == "b"
    assert not best.is_error


def test_rank_candidates_all_failed_returns_first():
    cands = [_mk("a", 999.0, err="fail1"), _mk("b", 999.0, err="fail2")]
    best = rank_candidates(cands)
    assert best.is_error
    assert best.error == "fail1"


def test_rank_candidates_empty_list_returns_empty():
    best = rank_candidates([])
    assert best.is_error
    assert best.ai_score == 999.0


# --------------------------------------------------------------------------- #
# format_candidate_summary
# --------------------------------------------------------------------------- #
def test_format_summary_success():
    c = WashCandidate("hello world", "eurollm-9b", "premium", 2.5, 12.0)
    s = format_candidate_summary(c)
    assert "eurollm-9b" in s
    assert "premium" in s
    assert "AI=12.0" in s
    assert "chars=11" in s
    assert "FAILED" not in s


def test_format_summary_error():
    c = WashCandidate("", "lfm25-tool", "fast", 0.3, 999.0, error="timeout")
    s = format_candidate_summary(c)
    assert "FAILED" in s
    assert "lfm25-tool" in s
    assert "timeout" in s


# --------------------------------------------------------------------------- #
# format_results_table
# --------------------------------------------------------------------------- #
def test_format_results_table_lists_all_and_winner():
    c1 = WashCandidate("a", "llama3.2", "standard", 1.0, 40.0)
    c2 = WashCandidate("b", "eurollm-9b", "premium", 2.0, 20.0)
    table = format_results_table([c1, c2], c2)
    assert "Multi-Agent Wash Results" in table
    assert "llama3.2" in table
    assert "eurollm-9b" in table
    assert "2.0s" in table
    assert "WINNER" in table
    # winner summary line shows model + timing
    assert "eurollm-9b" in table


def test_format_results_table_error_winner():
    c = WashCandidate("", "x", "standard", 0.1, 999.0, error="down")
    table = format_results_table([c], c)
    assert "Winner:" in table
    assert "down" in table


# --------------------------------------------------------------------------- #
# _wash_with_model (mocked Ollama + stat engine)
# --------------------------------------------------------------------------- #
def _fake_report(score):
    return SimpleNamespace(ai_score=score)


def test_wash_with_model_success():
    with patch(
        "scripts.multi_agent_washer.call_ollama", return_value="clean text"
    ) as mock_call, patch(
        "scripts.multi_agent_washer.analyze_text", return_value=_fake_report(33.0)
    ) as mock_analyze:
        c = _wash_with_model("dirty text", AGENT_MODELS[1])
    assert c.text == "clean text"
    assert c.model == AGENT_MODELS[1]["name"]
    assert c.ai_score == 33.0
    assert not c.is_error
    mock_call.assert_called_once()
    mock_analyze.assert_called_once_with("clean text")


def test_wash_with_model_failure_returns_error_candidate():
    with patch(
        "scripts.multi_agent_washer.call_ollama", side_effect=RuntimeError("ollama down")
    ):
        c = _wash_with_model("text", AGENT_MODELS[0])
    assert c.is_error
    assert c.error == "ollama down"
    assert c.text == ""
    assert c.ai_score == 999.0


# --------------------------------------------------------------------------- #
# multi_agent_wash (end-to-end, mocked)
# --------------------------------------------------------------------------- #
def test_multi_agent_wash_picks_best_and_verbose_runs():
    results = {"lfm25-tool": "A", "llama3.2": "B", "eurollm-9b": "C"}
    scores = {"A": 60.0, "B": 40.0, "C": 20.0}

    def fake_call(prompt, model, **kwargs):
        return results[model]

    def fake_analyze(text):
        return _fake_report(scores[text])

    with patch("scripts.multi_agent_washer.call_ollama", side_effect=fake_call), patch(
        "scripts.multi_agent_washer.analyze_text", side_effect=fake_analyze
    ):
        best = multi_agent_wash("input text", verbose=True)

    assert best.model == "eurollm-9b"
    assert best.text == "C"
    assert best.ai_score == 20.0
    assert not best.is_error


def test_multi_agent_wash_handles_one_failure():
    def fake_call(prompt, model, **kwargs):
        if model == "lfm25-tool":
            raise RuntimeError("transient")
        return model

    with patch("scripts.multi_agent_washer.call_ollama", side_effect=fake_call), patch(
        "scripts.multi_agent_washer.analyze_text", side_effect=lambda t: _fake_report(10.0)
    ):
        best = multi_agent_wash("input text", verbose=False)

    # The failed agent is excluded; the surviving best wins.
    assert best.model in ("llama3.2", "eurollm-9b")
    assert not best.is_error


def test_multi_agent_wash_all_fail():
    with patch(
        "scripts.multi_agent_washer.call_ollama",
        side_effect=RuntimeError("service down"),
    ):
        best = multi_agent_wash("input text", verbose=False)
    assert best.is_error


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_build_parser_defaults():
    p = build_parser()
    args = p.parse_args(["input.txt"])
    assert args.input == "input.txt"
    assert args.verbose is False
    assert args.dry_run is False
    assert args.output is None


def test_build_parser_dry_run():
    p = build_parser()
    args = p.parse_args(["input.txt", "--dry-run", "-v"])
    assert args.dry_run is True
    assert args.verbose is True


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
