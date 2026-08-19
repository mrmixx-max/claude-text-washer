#!/usr/bin/env python3
"""Tests for model pool management (scripts/ollama_utils.py)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ollama_utils import (
    format_model_list,
    get_default_model,
    get_model_names,
    load_models,
    resolve_model,
    validate_model,
)

# Expected models from the task specification
EXPECTED_MODELS = [
    "llama3.2",
    "qwen-coder-7b",
    "qwen-coder",
    "eurollm-9b",
    "nemo-heretic",
    "darkest",
    "gutenberg-26b",
    "lfm2-24b-a2b",
    "qwen3-30b-a3b",
    "lfm25-tool",
    "gemma-4-e4b",
]


def test_load_models_returns_dict():
    data = load_models()
    assert isinstance(data, dict)
    assert "models" in data
    assert "default" in data


def test_get_model_names():
    names = get_model_names()
    assert isinstance(names, list)
    assert len(names) == 11


def test_all_expected_models_present():
    names = set(get_model_names())
    for expected in EXPECTED_MODELS:
        assert expected in names, f"Missing model: {expected}"


def test_default_model():
    default = get_default_model()
    assert default == "llama3.2"


def test_default_model_matches_top_level():
    data = load_models()
    assert data["default"] == "llama3.2"


def test_validate_existing_model():
    assert validate_model("llama3.2") is True
    assert validate_model("qwen-coder-7b") is True
    assert validate_model("gemma-4-e4b") is True


def test_validate_nonexistent_model():
    assert validate_model("fake-model") is False
    assert validate_model("gpt-4") is False
    assert validate_model("") is False


def test_resolve_with_explicit_valid_model():
    assert resolve_model("qwen-coder", script_default="llama3.2") == "qwen-coder"
    assert resolve_model("darkest", script_default="llama3.2") == "darkest"


def test_resolve_without_request_returns_default():
    result = resolve_model(None, script_default="llama3.2")
    assert result == "llama3.2"


def test_resolve_with_invalid_model_raises():
    try:
        resolve_model("nonexistent")
        assert False, "Should have raised ValueError"
    except ValueError as exc:
        assert "not in the configured pool" in str(exc)


def test_resolve_fallback_when_yaml_missing():
    """When models.yaml is unavailable, resolve_model falls back gracefully."""
    from unittest.mock import patch

    with patch("scripts.ollama_utils.load_models", side_effect=FileNotFoundError("missing")):
        result = resolve_model(None, script_default="fallback-model")
    assert result == "fallback-model"


def test_format_model_list_contains_all_models():
    listing = format_model_list()
    for name in EXPECTED_MODELS:
        assert name in listing, f"Model '{name}' missing from listing"


def test_format_model_list_marks_default():
    listing = format_model_list()
    # The default model should be marked with *
    assert "*" in listing


def test_model_configs_have_size_and_description():
    data = load_models()
    for name, cfg in data["models"].items():
        assert "size" in cfg, f"Model '{name}' missing 'size'"
        assert "description" in cfg, f"Model '{name}' missing 'description'"


def test_exactly_one_default_model():
    data = load_models()
    defaults = [n for n, c in data["models"].items() if c.get("default")]
    assert len(defaults) == 1, f"Expected 1 default model, found {len(defaults)}: {defaults}"
    assert defaults[0] == "llama3.2"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
