"""Compatibility entry point for leakage-resistant model evaluation.

The earlier script mixed row-wise random splits with a manual locus split that
was never passed to the trainer. It could therefore report leaked metrics.
Use grouped outer cross-validation for all current model evaluation.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.phase3.evaluate_grouped_cross_validation import main


if __name__ == "__main__":
    main()
