"""Shared pytest setup for the claude-text-washer test-suite.

Ensures the repository root and scripts directory are importable so ``scripts``
and its submodules import whether pytest is invoked from the repo root or elsewhere.
"""
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[1])
_SCRIPTS = str(Path(_ROOT) / "scripts")

if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
