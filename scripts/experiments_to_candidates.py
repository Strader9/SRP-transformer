"""Convert hbv_experiments.csv (raw literature records) into model-ready
candidates_labelled.csv (single-genotype wrapper).

Each experiment record becomes one candidate sample:
  - target_sequences = [target_sequence]  (single reference genotype)
  - genotype_ids = [genotype]
  - hbv_features = []  (to be filled later by prepare --annotation)
  - sirna = given antisense, or reverse_complement(target) if not given
  - label = fractional inhibition (empty rows are dropped for training)

Usage:
  python scripts/experiments_to_candidates.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(r"C:\Users\34359\Documents\SRP\SRP-transformer-HBV-v0.1")
sys.path.insert(0, str(REPO))

from hbv_sirna.sequence import reverse_complement

SRC = REPO / "data" / "raw" / "hbv_experiments.csv"
DST = REPO / "data" / "processed" / "candidates_labelled.csv"

SEQ_LEN = 19


def trim_or_pad(seq: str, length: int = SEQ_LEN) -> str:
    seq = seq.upper().replace("U", "T")
    if len(seq) >= length:
        return seq[:length]
    return seq + "-" * (length - seq)


def main() -> None:
    df = pd.read_csv(SRC)
    print(f"Read {len(df)} experiment records")

    # Drop rows without target sequence or without label (training needs both)
    before = len(df)
    df = df.dropna(subset=["target_sequence", "label"])
    df = df[df["target_sequence"].str.strip() != ""]
    print(f"Dropped {before - len(df)} rows missing target_sequence or label")

    rows = []
    for _, rec in df.iterrows():
        target = trim_or_pad(str(rec["target_sequence"]))
        sirna = str(rec["sirna"]).strip() if pd.notna(rec["sirna"]) else ""
        if not sirna or sirna == "" or sirna.lower() == "nan":
            sirna = reverse_complement(target)
        else:
            sirna = trim_or_pad(sirna)

        genotype = str(rec["genotype"]).strip() if pd.notna(rec["genotype"]) else "unknown"

        rows.append({
            "sample_id": str(rec["experiment_id"]),
            "sirna": sirna,
            "target_sequences": json.dumps([target]),
            "genotype_ids": json.dumps([genotype]),
            "hbv_features": json.dumps([]),
            "label": float(rec["label"]),
        })

    out = pd.DataFrame(rows)
    DST.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(DST, index=False)

    print(f"Wrote {len(out)} labelled candidates to {DST}")
    print(f"  label range: [{out['label'].min():.3f}, {out['label'].max():.3f}]")
    print(f"  label mean: {out['label'].mean():.3f}")
    print(f"  genotypes: {out['genotype_ids'].apply(lambda x: json.loads(x)[0]).value_counts().to_dict()}")


if __name__ == "__main__":
    main()
