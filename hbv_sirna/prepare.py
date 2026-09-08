from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .annotation import (
    build_msa_to_ref_map,
    compute_hbv_features,
    msa_window_to_ref_interval,
    parse_bed,
)
from .sequence import conservation_profile, read_fasta, reverse_complement


def build_candidate_table(
    aligned_fasta: str | Path,
    output_csv: str | Path,
    reference_id: str,
    window: int = 19,
    stride: int = 1,
    annotation_bed: str | Path | None = None,
) -> pd.DataFrame:
    """Create candidate siRNAs from a whole-genome MSA (gaps retained).

    If ``annotation_bed`` is given, each candidate gets the 7-dim
    ``hbv_features`` vector (S/P/C/X, overlap, GC, mean conservation).
    Otherwise ``hbv_features`` is an empty list (backward compatible).
    """
    records = read_fasta(aligned_fasta)
    if reference_id not in records:
        raise ValueError(f"Reference id {reference_id!r} is absent from FASTA")
    lengths = {len(seq) for seq in records.values()}
    if len(lengths) != 1:
        raise ValueError("Input must be a precomputed multiple-sequence alignment")
    genotype_ids = list(records)
    reference = records[reference_id]

    regions = None
    msa_to_ref = None
    if annotation_bed is not None:
        regions = parse_bed(annotation_bed)
        msa_to_ref = build_msa_to_ref_map(reference)

    rows = []
    for start in range(0, len(reference) - window + 1, stride):
        targets = [records[name][start : start + window] for name in genotype_ids]
        ref_target = targets[genotype_ids.index(reference_id)]
        if "-" in ref_target or "N" in ref_target:
            continue

        hbv_features: list[float] = []
        if regions is not None and msa_to_ref is not None:
            ref_interval = msa_window_to_ref_interval(start, start + window, msa_to_ref)
            if ref_interval is not None:
                conservation = conservation_profile(targets)
                hbv_features = compute_hbv_features(
                    reference_target=ref_target,
                    regions=regions,
                    ref_start=ref_interval[0],
                    ref_end=ref_interval[1],
                    conservation=conservation,
                )

        rows.append(
            {
                "sample_id": f"{reference_id}:{start + 1}-{start + window}",
                "start_1based": start + 1,
                "sirna": reverse_complement(ref_target),
                "target_sequences": json.dumps(targets),
                "genotype_ids": json.dumps(genotype_ids),
                "hbv_features": json.dumps(hbv_features),
            }
        )
    frame = pd.DataFrame(rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    return frame
