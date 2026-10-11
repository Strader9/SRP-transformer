"""Phase 3 Direction 4: Build HUVK pretraining dataset.

For each (siRNA, mRNA, efficacy) pair:
1. Take the central 19-mer of the siRNA (RNA -> DNA by U->T).
2. Find the siRNA's target site on the mRNA by sliding-window best match of
   the reverse-complemented 19-mer (circular mRNA not assumed).
3. Emit one row per pair with sirna / target_sequences / efficacy.

Output: data/external/huvk/huvk_pretrain.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(r"C:\Users\34359\Documents\SRP\SRP-transformer-HBV-v0.1")
sys.path.insert(0, str(REPO))

HUVK_DIR = REPO / "data" / "external" / "huvk"
OUT_CSV = HUVK_DIR / "huvk_pretrain.csv"
WINDOW = 19


def read_fasta(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    name, seq = None, []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                records[name] = "".join(seq)
            name = line[1:].split()[0]
            seq = []
        else:
            seq.append(line)
    if name is not None:
        records[name] = "".join(seq)
    return records


def revcomp(s: str) -> str:
    return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def best_match_window(target: str, mrna: str) -> tuple[str, int]:
    """Best 19-nt window of mRNA for target; returns (window, hamming_dist)."""
    best_dist = WINDOW + 1
    best_seq = ""
    for i in range(len(mrna) - WINDOW + 1):
        window = mrna[i : i + WINDOW]
        d = sum(1 for a, b in zip(target, window) if a != b)
        if d < best_dist:
            best_dist, best_seq = d, window
            if d == 0:
                break
    return best_seq, best_dist


def main() -> None:
    sirnas = read_fasta(HUVK_DIR / "siRNA_HUVK.fas")
    mrnas = read_fasta(HUVK_DIR / "mRNA_HUVK.fas")
    print(f"siRNAs: {len(sirnas)}, mRNAs: {len(mrnas)}")

    eff = pd.read_csv(HUVK_DIR / "siRNA_mRNA_Efficacy.csv")
    print(f"Pairs: {len(eff)}")

    rows = []
    skipped = 0
    for _, rec in eff.iterrows():
        sirna_id, mrna_id, efficacy = rec["siRNA"], rec["mRNA"], rec["efficacy"]
        if sirna_id not in sirnas or mrna_id not in mrnas:
            skipped += 1
            continue
        sirna = sirnas[sirna_id].upper().replace("U", "T")
        mrna = mrnas[mrna_id].upper().replace("U", "T")
        if len(sirna) < WINDOW:
            skipped += 1
            continue
        # central 19-mer
        start = (len(sirna) - WINDOW) // 2
        sirna19 = sirna[start : start + WINDOW]
        # target = revcomp of sirna19, matched on mRNA
        target_rc = revcomp(sirna19)
        target_window, dist = best_match_window(target_rc, mrna)
        if target_window == "":
            skipped += 1
            continue
        rows.append({
            "sample_id": f"{sirna_id}__{mrna_id}",
            "sirna": sirna19,
            "target_sequences": f'["{target_window}"]',
            "genotype_ids": '["D"]',
            "hbv_features": "[]",
            "label": float(efficacy),
            "_match_dist": int(dist),
        })

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(out)} pretrain pairs to {OUT_CSV} (skipped {skipped})")
    print(f"Match distance distribution: {out['_match_dist'].value_counts().sort_index().to_dict()}")
    print(f"Label stats: mean={out['label'].mean():.3f}, min={out['label'].min():.3f}, max={out['label'].max():.3f}")


if __name__ == "__main__":
    main()
