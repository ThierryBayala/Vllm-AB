"""Pytest configuration and shared fixtures."""

import sys
from pathlib import Path

# Ensure src is on path when running tests from project root
root = Path(__file__).resolve().parent.parent
if str(root / "src") not in sys.path:
    sys.path.insert(0, str(root / "src"))