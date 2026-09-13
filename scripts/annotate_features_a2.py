"""A2b: Annotate candidates with S/P/C/X functional-region features using
the genotype-A BED file, merge with A1 gc/conservation to produce complete
7-dim hbv_features.

For each candidate, the reference target is mapped onto genotype A (AM282986)
via best-match window (circular genome aware).  The resulting interval is
queried against the BED to determine overlapping S/P/C/X classes.

hbv_features fixed order:
    [S, P, C, X, is_overlap, gc_content, avg_conservation]

Input:  data/processed/candidates_with_features.csv (A1 output)
Output: data/processed/candidates_full_features.csv
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(r"C:\Users\34359\Documents\SRP\SRP-transformer-HBV-v0.1")
sys.path.insert(0, str(REPO))

from hbv_sirna.annotation import parse_bed, overlapping_classes
from hbv_sirna.sequence import conservation_profile

SRC = REPO / "data" / "processed" / "candidates_with_features.csv"
BED_PATH = REPO / "data" / "raw" / "hbv_regions_genotypeA.bed"
REF_GENOME = REPO / "data" / "raw" / "genomes" / "hbv_genotype_A.fasta"
DST = REPO / "data" / "processed" / "candidates_full_features.csv"

WINDOW = 19


def load_ref_genome() -> str:
    lines = REF_GENOME.read_text(encoding="utf-8").strip().splitlines()
    return "".join(lines[1:]).upper().replace("U", "T")


def best_match_position(target: str, genome: str) -> int:
    """Return 0-based start of best-match 19-nt window in circular genome."""
    circular = genome + genome[: WINDOW - 1]
    best_dist = WINDOW + 1
    best_pos = 0
    for i in range(len(genome)):
        window = circular[i : i + WINDOW]
        d = sum(1 for a, b in zip(target, window) if a != b)
        if d < best_dist:
            best_dist = d
            best_pos = i
            if d == 0:
                break
    return best_pos


def gc_content(seq: str) -> float:
    seq = seq.upper()
    gc = sum(1 for b in seq if b in "GC")
    total = sum(1 for b in seq if b in "ACGT")
    return 0.0 if total == 0 else gc / total


def main() -> None:
    df = pd.read_csv(SRC)
    print(f"Read {len(df)} candidates (with gc+conservation from A1)")

    regions = parse_bed(BED_PATH)
    print(f"Loaded {len(regions)} BED intervals: "
          f"{sorted(set(r.name for r in regions))}")

    ref_genome = load_ref_genome()
    print(f"Reference genome (A): {len(ref_genome)} bp")

    rows = []
    class_counts = {"S": 0, "P": 0, "C": 0, "X": 0}
    overlap_count = 0

    for _, rec in df.iterrows():
        targets = json.loads(rec["target_sequences"])
        ref_target = targets[0]

        # Map target onto genotype A to get genomic interval
        start = best_match_position(ref_target, ref_genome)
        end = start + WINDOW
        # Handle wrap-around: if end exceeds genome length, split is not needed
        # for BED query because overlapping_classes handles simple intervals;
        # we query [start, min(end, genome_len)) and if wrapped also [0, end-genome_len)
        classes = set()
        if end <= len(ref_genome):
            classes |= overlapping_classes(regions, start, end)
        else:
            classes |= overlapping_classes(regions, start, len(ref_genome))
            classes |= overlapping_classes(regions, 0, end - len(ref_genome))

        # Recompute gc and conservation (consistent with A1, but we can also
        # just carry over from rec["hbv_features"] — recompute for clarity)
        gc = gc_content(ref_target)
        aligned = [seq[:WINDOW].ljust(WINDOW, "-") for seq in targets]
        avg_cons = float(np.mean(conservation_profile(aligned)))

        is_overlap = 1.0 if len(classes) > 1 else 0.0
        features = [
            1.0 if "S" in classes else 0.0,
            1.0 if "P" in classes else 0.0,
            1.0 if "C" in classes else 0.0,
            1.0 if "X" in classes else 0.0,
            is_overlap,
            gc,
            avg_cons,
        ]

        for cls in classes:
            class_counts[cls] = class_counts.get(cls, 0) + 1
        if is_overlap:
            overlap_count += 1

        row = rec.to_dict()
        row["hbv_features"] = json.dumps(features)
        row["_ref_position"] = f"{start}-{end % len(ref_genome) if end > len(ref_genome) else end}"
        row["_classes"] = ",".join(sorted(classes)) if classes else "none"
        rows.append(row)

    out = pd.DataFrame(rows)
    DST.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(DST, index=False)

    print(f"\nWrote {len(out)} candidates with full 7-dim hbv_features to {DST}")
    print(f"  Functional class distribution: {class_counts}")
    print(f"  Overlapping-region candidates: {overlap_count}/{len(out)}")
    print(f"  Sample features (first 3):")
    for i in range(min(3, len(out))):
        print(f"    {out.iloc[i]['sample_id']}: {out.iloc[i]['hbv_features']} "
              f"classes={out.iloc[i]['_classes']} pos={out.iloc[i]['_ref_position']}")


if __name__ == "__main__":
    main()
