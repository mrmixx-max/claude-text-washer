"""Shared pytest setup for the claude-text-washer test-suite.

Ensures the repository root is importable so ``scripts`` and its submodules
import whether pytest is invoked from the repo root or elsewhere.
"""
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
