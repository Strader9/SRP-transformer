"""A1: Compute gc_content and avg_conservation from 8-genotype targets,
fill the 7-dim hbv_features vector.

hbv_features fixed order (per README / annotation.py):
    [S, P, C, X, is_overlap, gc_content, avg_conservation]

S/P/C/X/is_overlap are set to 0.0 here; they are filled in A2 via BED
annotation.  gc_content is computed from the reference genotype target
(first in target_sequences).  avg_conservation is the mean of the 19-dim
per-position majority-base frequency across all 8 genotypes.

Output: data/processed/candidates_with_features.csv
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from hbv_sirna.sequence import conservation_profile

SRC = REPO / "data" / "processed" / "candidates_multi_genotype.csv"
DST = REPO / "data" / "processed" / "candidates_with_features.csv"

FEATURE_DIM = 7  # [S, P, C, X, is_overlap, gc, avg_cons]


def gc_content(seq: str) -> float:
    seq = seq.upper()
    gc = sum(1 for b in seq if b in "GC")
    total = sum(1 for b in seq if b in "ACGT")
    return 0.0 if total == 0 else gc / total


def main() -> None:
    df = pd.read_csv(SRC)
    print(f"Read {len(df)} multi-genotype candidates")

    rows = []
    gc_vals = []
    cons_vals = []

    for _, rec in df.iterrows():
        targets = json.loads(rec["target_sequences"])
        ref_target = targets[0]

        # 19-dim conservation across all genotypes
        aligned = [seq[:19].ljust(19, "-") for seq in targets]
        cons_profile = conservation_profile(aligned)
        avg_cons = float(np.mean(cons_profile))

        gc = gc_content(ref_target)

        features = [0.0, 0.0, 0.0, 0.0, 0.0, gc, avg_cons]
        assert len(features) == FEATURE_DIM

        gc_vals.append(gc)
        cons_vals.append(avg_cons)

        row = rec.to_dict()
        row["hbv_features"] = json.dumps(features)
        rows.append(row)

    out = pd.DataFrame(rows)
    DST.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(DST, index=False)

    print(f"Wrote {len(out)} candidates with 7-dim hbv_features to {DST}")
    print(f"  gc_content:     min={min(gc_vals):.3f}, max={max(gc_vals):.3f}, "
          f"mean={np.mean(gc_vals):.3f}")
    print(f"  avg_conservation: min={min(cons_vals):.3f}, max={max(cons_vals):.3f}, "
          f"mean={np.mean(cons_vals):.3f}")
    print(f"  (S/P/C/X/is_overlap = 0.0, to be filled in A2 via BED annotation)")


if __name__ == "__main__":
    main()
