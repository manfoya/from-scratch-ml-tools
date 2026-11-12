"""Configuration commune des tests."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

RACINE = Path(__file__).resolve().parents[1]
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))
