"""Convert hbv_experiments.csv (raw literature records) into model-ready
candidates_labelled.csv (single-genotype wrapper).

Each experiment record becomes one candidate sample:
  - guide and target are trimmed together to the most complementary 19-mer pair
  - target_sequences = [target_sequence]  (single reference genotype)
  - genotype_ids = [genotype]
  - hbv_features = []  (to be filled later by prepare --annotation)
  - missing sirna values are marked as inferred from reverse_complement(target)
  - label = fractional inhibition (empty rows are dropped for training)

Usage:
  python scripts/experiments_to_candidates.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from hbv_sirna.sequence import align_sirna_target, normalize_sequence, reverse_complement

SRC = REPO / "data" / "raw" / "hbv_experiments.csv"
DST = REPO / "data" / "processed" / "candidates_labelled.csv"

SEQ_LEN = 19


def center_window(seq: str, length: int = SEQ_LEN) -> str:
    seq = normalize_sequence(seq)
    if len(seq) < length:
        raise ValueError(f"Sequence must contain at least {length} bases")
    start = (len(seq) - length) // 2
    return seq[start : start + length]


def main() -> None:
    df = pd.read_csv(SRC)
    print(f"Read {len(df)} experiment records")

    # Drop rows without target sequence or without label (training needs both)
    before = len(df)
    df = df.dropna(subset=["target_sequence", "label"])
    df = df[df["target_sequence"].str.strip() != ""]
    print(f"Dropped {before - len(df)} rows missing target_sequence or label")

    rows = []
    skipped = []
    for _, rec in df.iterrows():
        try:
            raw_target = normalize_sequence(str(rec["target_sequence"]))
            raw_sirna = str(rec["sirna"]).strip() if pd.notna(rec["sirna"]) else ""
            if not raw_sirna or raw_sirna.lower() == "nan":
                target = center_window(raw_target)
                sirna = reverse_complement(target)
                pair_identity = 1.0
                guide_inferred = True
            else:
                sirna, target, pair_identity = align_sirna_target(raw_sirna, raw_target, SEQ_LEN)
                guide_inferred = False
        except ValueError as exc:
            skipped.append((str(rec["experiment_id"]), str(exc)))
            continue

        genotype = str(rec["genotype"]).strip() if pd.notna(rec["genotype"]) else "unknown"

        row = {
            "sample_id": str(rec["experiment_id"]),
            "sirna": sirna,
            "target_sequences": json.dumps([target]),
            "genotype_ids": json.dumps([genotype]),
            "hbv_features": json.dumps([]),
            "label": float(rec["label"]),
            "_pair_identity": pair_identity,
            "_guide_inferred": guide_inferred,
        }
        for column in ("reference_accession", "assay_type", "dose_nm", "time_h", "cell_line", "source"):
            if column in rec.index:
                row[column] = rec[column]
        rows.append(row)

    out = pd.DataFrame(rows)
    DST.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(DST, index=False)

    print(f"Wrote {len(out)} labelled candidates to {DST}")
    print(f"  skipped invalid sequence rows: {len(skipped)}")
    for sample_id, reason in skipped[:10]:
        print(f"    {sample_id}: {reason}")
    print(f"  guide-target complementarity below 80%: {int((out['_pair_identity'] < 0.8).sum())}")
    print(f"  label range: [{out['label'].min():.3f}, {out['label'].max():.3f}]")
    print(f"  label mean: {out['label'].mean():.3f}")
    print(f"  genotypes: {out['genotype_ids'].apply(lambda x: json.loads(x)[0]).value_counts().to_dict()}")


if __name__ == "__main__":
    main()
