#!/usr/bin/env python3
"""Tests for the ``--benchmark`` feature of the multi-agent washer."""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.multi_agent_washer import (  # noqa: E402
    BenchmarkResult,
    _percentile,
    build_parser,
    run_benchmark,
)
from scripts.ollama_utils import reset_circuit_breakers  # noqa: E402


# --------------------------------------------------------------------------- #
# _percentile
# --------------------------------------------------------------------------- #
def test_percentile_single_element():
    assert _percentile([42.0], 95) == 42.0


def test_percentile_empty_is_zero():
    assert _percentile([], 50) == 0.0


def test_percentile_known_values():
    data = list(range(1, 11))  # [1..10]
    assert _percentile(data, 50) == 5.5
    # p95 of 1..10 → k=8.55 → interpolate between 9 and 10.
    assert _percentile(data, 95) == pytest.approx(9.55)
    assert _percentile(data, 0) == 1.0
    assert _percentile(data, 100) == 10.0


@pytest.mark.parametrize("pct", [50, 95, 99])
def test_percentile_monotonic_in_pct(pct):
    data = [3.0, 1.0, 4.0, 1.6, 5.0, 9.0]
    assert _percentile(data, 0) <= _percentile(data, pct) <= _percentile(data, 100)


# --------------------------------------------------------------------------- #
# BenchmarkResult
# --------------------------------------------------------------------------- #
def test_benchmark_result_computes_stats():
    res = BenchmarkResult(
        iterations=3,
        durations=[1.0, 2.0, 3.0],
        ai_scores=[40.0, 50.0, 60.0],
        winners=["eurollm-9b", "llama3.2", "lfm25-tool"],
    )
    assert res.mean_latency == 2.0
    assert res.p50_latency == 2.0
    assert res.min_latency == 1.0
    assert res.max_latency == 3.0
    assert res.mean_ai_score == pytest.approx(50.0)
    assert res.min_ai_score == 40.0
    assert res.max_ai_score == 60.0
    assert res.errors == 0


def test_benchmark_result_with_errors():
    res = BenchmarkResult(
        iterations=2,
        durations=[0.5, 0.5],
        ai_scores=[999.0, 999.0],
        winners=["ERROR", "ERROR"],
        errors=2,
    )
    assert res.errors == 2
    assert res.mean_ai_score == 999.0
    report = res.format_report()
    assert "Failed runs:  2" in report


# --------------------------------------------------------------------------- #
# run_benchmark (inject fake wash_fn — no Ollama needed)
# --------------------------------------------------------------------------- #
def _fake_candidate(model="eurollm-9b", score=40.0, dur=0.25):
    from scripts.multi_agent_washer import WashCandidate

    return WashCandidate("clean text", model, "premium", dur, score)


def test_run_benchmark_aggregates_iterations():
    calls = {"n": 0}

    def fake_wash(text, verbose=False):
        calls["n"] += 1
        return _fake_candidate(score=40.0 + calls["n"], dur=0.1 * calls["n"])

    res = run_benchmark("input", iterations=5, wash_fn=fake_wash)
    assert calls["n"] == 5
    assert res.iterations == 5
    assert len(res.durations) == 5
    assert res.errors == 0
    # p50 of [0.1,0.2,0.3,0.4,0.5] == 0.3
    assert res.p50_latency == pytest.approx(0.3)


def test_run_benchmark_counts_error_candidates():
    from scripts.multi_agent_washer import WashCandidate

    def fake_wash(text, verbose=False):
        # Every other run fails.
        return WashCandidate(
            "", "eurollm-9b", "premium", 0.1, 999.0, error="down"
        )

    res = run_benchmark("input", iterations=3, wash_fn=fake_wash)
    assert res.errors == 3
    assert "ERROR" in res.winners
    assert res.mean_ai_score == 999.0


# --------------------------------------------------------------------------- #
# CLI wiring (--benchmark via main(argv))
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _reset_cb():
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()


def test_main_benchmark_flag_parses():
    parser = build_parser()
    args = parser.parse_args(["in.txt", "--benchmark", "10"])
    assert args.benchmark == 10
    assert args.verbose is False


def _ok_urlopen(req, timeout=None):
    return io.BytesIO(json.dumps({"response": "benchmark wash"}).encode("utf-8"))


def test_main_benchmark_runs_and_reports(tmp_path, capsys):
    """claude-washer multi-agent --benchmark N prints latency/p50/p95/ai stats."""
    inp = tmp_path / "in.txt"
    inp.write_text("dirty input text")

    # urlopen returns a 200-style stream; use a real http response stand-in.
    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def getcode(self):
            return 200

        def read(self):
            return json.dumps({"response": "benchmark wash"}).encode("utf-8")

    # urlopen returns a 200-style response; all models succeed, so no backoff
    # sleeps are needed.  (urllib.request is a shared global module, so patching
    # it here covers whichever ollama_utils instance the washer imported.)
    with patch(
        "scripts.ollama_utils.urllib.request.urlopen",
        return_value=_Resp(),
    ):
        from scripts.multi_agent_washer import main as ma_main

        rc = ma_main([str(inp), "--benchmark", "3", "-v"])

    out = capsys.readouterr().out
    assert rc == 0
    assert "Benchmark Results (3 iterations)" in out
    assert "p50" in out
    assert "p95" in out
    assert "AI score" in out


def test_main_benchmark_rejects_zero(tmp_path):
    inp = tmp_path / "in.txt"
    inp.write_text("dirty input text")
    from scripts.multi_agent_washer import main as ma_main

    with pytest.raises(SystemExit) as exc:
        ma_main([str(inp), "--benchmark", "0"])
    assert exc.value.code == 2
