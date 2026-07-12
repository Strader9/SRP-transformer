from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .sequence import read_fasta, reverse_complement


def build_candidate_table(
    aligned_fasta: str | Path,
    output_csv: str | Path,
    reference_id: str,
    window: int = 19,
    stride: int = 1,
) -> pd.DataFrame:
    """Create candidate siRNAs from a whole-genome MSA (gaps retained)."""
    records = read_fasta(aligned_fasta)
    if reference_id not in records:
        raise ValueError(f"Reference id {reference_id!r} is absent from FASTA")
    lengths = {len(seq) for seq in records.values()}
    if len(lengths) != 1:
        raise ValueError("Input must be a precomputed multiple-sequence alignment")
    genotype_ids = list(records)
    rows = []
    reference = records[reference_id]
    for start in range(0, len(reference) - window + 1, stride):
        targets = [records[name][start : start + window] for name in genotype_ids]
        ref_target = targets[genotype_ids.index(reference_id)]
        if "-" in ref_target or "N" in ref_target:
            continue
        rows.append(
            {
                "sample_id": f"{reference_id}:{start + 1}-{start + window}",
                "start_1based": start + 1,
                "sirna": reverse_complement(ref_target),
                "target_sequences": json.dumps(targets),
                "genotype_ids": json.dumps(genotype_ids),
                "hbv_features": json.dumps([]),
            }
        )
    frame = pd.DataFrame(rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    return frame

