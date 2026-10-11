"""Attach assay task labels to aligned candidate records for multitask training.

The source assay table is authoritative for endpoint names. This script only
joins by experiment ID; it never infers an endpoint from the label value.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build(candidates: Path, experiments: Path, output: Path, assay_filter: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(candidates)
    raw = pd.read_csv(experiments)
    if "sample_id" not in frame or "experiment_id" not in raw or "assay_type" not in raw:
        raise ValueError("Expected sample_id in candidates and experiment_id/assay_type in raw data")
    if raw["experiment_id"].duplicated().any():
        raise ValueError("Raw experiment_id values must be unique for an unambiguous join")
    metadata_columns = [column for column in (
        "assay_type", "genotype", "reference_accession", "dose_nm", "time_h", "cell_line", "source"
    ) if column in raw]
    raw_index = raw.set_index("experiment_id")
    merged = frame.copy()
    for column in metadata_columns:
        values = merged["sample_id"].map(raw_index[column])
        if column in merged:
            both = merged[column].notna() & values.notna()
            conflicts = both & (merged[column].astype(str) != values.astype(str))
            if conflicts.any():
                ids = merged.loc[conflicts, "sample_id"].tolist()
                raise ValueError(f"Conflicting {column} metadata for samples: {ids[:10]}")
            merged[column] = merged[column].fillna(values)
        else:
            merged[column] = values
    if merged["assay_type"].isna().any():
        missing = merged.loc[merged["assay_type"].isna(), "sample_id"].tolist()
        raise ValueError(f"Candidates missing assay_type metadata: {missing[:10]}")
    if assay_filter:
        merged = merged[merged["assay_type"] == assay_filter].copy()
        if merged.empty:
            raise ValueError(f"No candidate records found for assay_type={assay_filter!r}")
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="data/processed/candidates_full_features.csv")
    parser.add_argument("--experiments", default="data/raw/hbv_experiments.csv")
    parser.add_argument("--output", default="data/processed/candidates_multitask.csv")
    parser.add_argument("--assay", help="Keep one primary endpoint, e.g. HBsAg; omit to retain all task labels")
    args = parser.parse_args()
    frame = build(Path(args.candidates), Path(args.experiments), Path(args.output), args.assay)
    print(f"wrote {len(frame)} records; labels by assay:\n{frame.groupby('assay_type').size().to_string()}")


if __name__ == "__main__":
    main()
