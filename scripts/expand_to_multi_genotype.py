"""Expand single-genotype candidates to multi-genotype using sliding-window
best-match against 8 HBV reference genomes (A-H).

For each candidate, the 19-nt reference target is matched against every
genome by sliding a 19-nt window (circular genome handled by concat).
The best-match window (minimum Hamming distance) becomes that genotype's
target sequence.  HBV genomes are >90% identical, so the correct homolog
is almost always the unique best match.

Output: data/processed/candidates_multi_genotype.csv
  - target_sequences: list of 8 strings (A,B,C,D,E,F,G,H order)
  - genotype_ids: ["A","B","C","D","E","F","G","H"]
  - hbv_features: [] (filled later by prepare --annotation)
  - A new column _match_distances records per-genotype Hamming distances
    for QC (not consumed by the model).
"""

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(r"C:\Users\34359\Documents\SRP\SRP-transformer-HBV-v0.1")
sys.path.insert(0, str(REPO))

GENOMES_DIR = REPO / "data" / "raw" / "genomes"
SRC = REPO / "data" / "processed" / "candidates_labelled.csv"
DST = REPO / "data" / "processed" / "candidates_multi_genotype.csv"

GENOTYPE_ORDER = ["A", "B", "C", "D", "E", "F", "G", "H"]
WINDOW = 19


def load_genomes() -> dict[str, str]:
    """Load 8 genotype genomes from individual FASTA files."""
    genomes = {}
    for gt in GENOTYPE_ORDER:
        path = GENOMES_DIR / f"hbv_genotype_{gt}.fasta"
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        seq = "".join(lines[1:]).upper().replace("U", "T")
        genomes[gt] = seq
    return genomes


def revcomp(s: str) -> str:
    return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def normalise_orientation(genomes: dict[str, str], ref_key: str = "A") -> dict[str, str]:
    """Ensure all genomes are in the same (positive-sense) orientation as ref.

    GenBank HBV accessions are inconsistent: some submit the complete long (-)
    strand, others the (+) strand.  We detect orientation by sliding 3 conserved
    100-nt anchors from the reference against each genome (and its revcomp),
    then majority-vote.
    """
    ref = genomes[ref_key]
    anchors = [ref[500:600], ref[1500:1600], ref[2500:2600]]
    result = {}
    for gt, seq in genomes.items():
        if gt == ref_key:
            result[gt] = seq
            continue
        votes_forward = 0
        votes_reverse = 0
        for anchor in anchors:
            _, d_fwd, _ = best_match_window(anchor, seq)
            _, d_rev, _ = best_match_window(anchor, revcomp(seq))
            if d_fwd <= d_rev:
                votes_forward += 1
            else:
                votes_reverse += 1
        if votes_reverse > votes_forward:
            result[gt] = revcomp(seq)
            print(f"  genotype {gt}: reverse-complemented ({votes_reverse}/{len(anchors)} anchors)")
        else:
            result[gt] = seq
    return result


def hamming(a: str, b: str) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)


def best_match_window(target: str, genome: str) -> tuple[str, int, int]:
    """Find the 19-nt window in circular genome with minimum Hamming distance.

    Returns (window_sequence, distance, start_position_0based).
    """
    # Handle circular genome by appending first WINDOW-1 bases
    circular = genome + genome[: WINDOW - 1]
    best_dist = WINDOW + 1
    best_seq = ""
    best_pos = -1
    for i in range(len(genome)):
        window = circular[i : i + WINDOW]
        d = hamming(target, window)
        if d < best_dist:
            best_dist = d
            best_seq = window
            best_pos = i
            if d == 0:
                break  # exact match, can't do better
    return best_seq, best_dist, best_pos


def main() -> None:
    genomes = load_genomes()
    print(f"Loaded {len(genomes)} genomes: "
          + ", ".join(f"{g}({len(s)}bp)" for g, s in genomes.items()))
    print("Normalising orientation to positive-sense (reference = genotype A)...")
    genomes = normalise_orientation(genomes, ref_key="A")

    df = pd.read_csv(SRC)
    print(f"Read {len(df)} single-genotype candidates")

    rows = []
    total_match_dist = []
    ref_exact = 0

    for _, rec in df.iterrows():
        ref_target = json.loads(rec["target_sequences"])[0]
        ref_genotype = json.loads(rec["genotype_ids"])[0]

        targets = []
        distances = []
        positions = []
        for gt in GENOTYPE_ORDER:
            seq, dist, pos = best_match_window(ref_target, genomes[gt])
            targets.append(seq)
            distances.append(dist)
            positions.append(pos)

        # QC: reference genotype should have exact match (dist=0)
        ref_idx = GENOTYPE_ORDER.index(ref_genotype)
        if distances[ref_idx] == 0:
            ref_exact += 1
        else:
            print(f"  WARNING: {rec['sample_id']} ref genotype {ref_genotype} "
                  f"match dist={distances[ref_idx]} (not exact), pos={positions[ref_idx]}")

        total_match_dist.extend(distances)
        rows.append({
            "sample_id": rec["sample_id"],
            "sirna": rec["sirna"],
            "target_sequences": json.dumps(targets),
            "genotype_ids": json.dumps(GENOTYPE_ORDER),
            "hbv_features": json.dumps([]),
            "label": rec["label"],
            "_match_distances": json.dumps(distances),
            "_match_positions": json.dumps(positions),
        })

    out = pd.DataFrame(rows)
    DST.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(DST, index=False)

    print(f"\nWrote {len(out)} multi-genotype candidates to {DST}")
    print(f"  Reference genotype exact matches: {ref_exact}/{len(out)}")
    print(f"  Match distance stats: min={min(total_match_dist)}, "
          f"max={max(total_match_dist)}, mean={sum(total_match_dist)/len(total_match_dist):.2f}")
    # Per-genotype mean distance
    for i, gt in enumerate(GENOTYPE_ORDER):
        dists = [json.loads(r["_match_distances"])[i] for r in rows]
        print(f"  genotype {gt}: mean dist={sum(dists)/len(dists):.2f}, "
              f"exact={sum(1 for d in dists if d==0)}/{len(dists)}")


if __name__ == "__main__":
    main()
