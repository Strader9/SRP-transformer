from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .sequence import conservation_profile, normalize_sequence, tokenize


REQUIRED_COLUMNS = {"sample_id", "sirna", "target_sequences"}


@dataclass(frozen=True)
class Sample:
    sample_id: str
    sirna: str
    target_sequences: list[str]
    genotype_ids: list[str]
    hbv_features: list[float]
    label: float | None
    assay_type: str | None = None
    genotype: str | None = None
    ref_interval: tuple[int, int] | None = None
    sirna_inferred: bool = False


def parse_sample(row: pd.Series) -> Sample:
    def decode_list(value, field: str, default: str = "[]") -> list:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            value = default
        try:
            result = json.loads(value) if isinstance(value, str) else value
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"{row.get('sample_id', '<unknown>')}: {field} must be a JSON array") from exc
        if not isinstance(result, list):
            raise ValueError(f"{row.get('sample_id', '<unknown>')}: {field} must be a JSON array")
        return result

    targets = decode_list(row["target_sequences"], "target_sequences")
    genotype_ids = decode_list(row.get("genotype_ids", "[]"), "genotype_ids")
    features = decode_list(row.get("hbv_features", "[]"), "hbv_features")
    if not targets or any(not isinstance(seq, str) or not seq for seq in targets):
        raise ValueError(f"{row['sample_id']}: target_sequences must be a non-empty JSON list")
    try:
        targets = [normalize_sequence(seq) for seq in targets]
    except ValueError as exc:
        raise ValueError(f"{row['sample_id']}: invalid target sequence: {exc}") from exc
    if genotype_ids and (len(genotype_ids) != len(targets) or
                         any(not isinstance(name, str) or not name for name in genotype_ids) or
                         len(set(genotype_ids)) != len(genotype_ids)):
        raise ValueError(f"{row['sample_id']}: genotype_ids and target_sequences differ in length")
    try:
        features = [float(value) for value in features]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{row['sample_id']}: hbv_features must contain numeric values") from exc
    if not all(math.isfinite(value) for value in features):
        raise ValueError(f"{row['sample_id']}: hbv_features must be finite")
    raw_sirna = row.get("sirna")
    if raw_sirna is None or pd.isna(raw_sirna) or not str(raw_sirna).strip():
        raise ValueError(f"{row['sample_id']}: sirna must be present")
    try:
        sirna = normalize_sequence(str(raw_sirna))
    except ValueError as exc:
        raise ValueError(f"{row['sample_id']}: invalid sirna sequence: {exc}") from exc
    try:
        label = None if "label" not in row or pd.isna(row["label"]) else float(row["label"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{row['sample_id']}: label must be numeric") from exc
    if label is not None and not math.isfinite(label):
        raise ValueError(f"{row['sample_id']}: label must be finite")
    assay = row.get("assay_type")
    assay_type = None if assay is None or pd.isna(assay) or not str(assay).strip() else str(assay).strip()
    genotype_value = row.get("genotype")
    genotype = None if genotype_value is None or pd.isna(genotype_value) else str(genotype_value).strip()
    position_value = row.get("_ref_position")
    position = None if position_value is None or pd.isna(position_value) else str(position_value)
    match = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", position or "")
    if not match:
        match = re.search(r":(\d+)-(\d+)$", str(row["sample_id"]))
        interval = None if not match else (int(match.group(1)) - 1, int(match.group(2)))
    else:
        interval = (int(match.group(1)), int(match.group(2)))
    inferred_value = row.get("_guide_inferred", False)
    sirna_inferred = (str(inferred_value).strip().lower() in {"true", "1", "yes"}
                      if not isinstance(inferred_value, bool) else inferred_value)
    return Sample(str(row["sample_id"]), sirna, targets, genotype_ids, features,
                  label, assay_type, genotype, interval, sirna_inferred)


class HBVMultiGenotypeDataset(Dataset):
    def __init__(self, csv_path: str | Path, sequence_length: int = 19, max_genotypes: int = 10,
                 visible_genotypes: set[str] | None = None):
        if sequence_length < 1 or max_genotypes < 1:
            raise ValueError("sequence_length and max_genotypes must be positive")
        frame = pd.read_csv(csv_path)
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"Missing dataset columns: {sorted(missing)}")
        self.samples = [parse_sample(row) for _, row in frame.iterrows()]
        if "assay_type" in frame.columns and any(sample.assay_type is None for sample in self.samples):
            raise ValueError("assay_type is present but missing for one or more rows; task assignment must be explicit")
        self.task_names = sorted({sample.assay_type for sample in self.samples if sample.assay_type}) or ["efficacy"]
        self.task_ids = {name: i for i, name in enumerate(self.task_names)}
        self.sequence_length = sequence_length
        self.max_genotypes = max_genotypes
        self.visible_genotypes = None if visible_genotypes is None else set(visible_genotypes)
        feature_dims = {len(sample.hbv_features) for sample in self.samples}
        if len(feature_dims) > 1:
            raise ValueError("All samples must have the same hbv_features length")
        self.feature_dim = next(iter(feature_dims), 0)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        if self.visible_genotypes is not None:
            if not sample.genotype_ids:
                raise ValueError(f"{sample.sample_id}: genotype filtering requires genotype_ids")
            pairs = [(genotype, target) for genotype, target in zip(sample.genotype_ids, sample.target_sequences)
                     if genotype in self.visible_genotypes]
            if not pairs:
                raise ValueError(f"{sample.sample_id}: no target sequences remain after genotype filtering")
            targets = [target for _, target in pairs[: self.max_genotypes]]
        else:
            targets = sample.target_sequences[: self.max_genotypes]
        mask = np.zeros(self.max_genotypes, dtype=np.bool_)
        mask[: len(targets)] = True
        padded_targets = targets + ["-"] * (self.max_genotypes - len(targets))
        target_tokens = np.stack([tokenize(seq, self.sequence_length) for seq in padded_targets])
        aligned = []
        for seq in targets:
            if len(seq) > self.sequence_length:
                left = (len(seq) - self.sequence_length) // 2
                seq = seq[left : left + self.sequence_length]
            aligned.append(seq.ljust(self.sequence_length, "-"))
        conservation = conservation_profile(aligned)
        item: dict[str, torch.Tensor | str] = {
            "sample_id": sample.sample_id,
            "sirna_tokens": torch.from_numpy(tokenize(sample.sirna, self.sequence_length)),
            "target_tokens": torch.from_numpy(target_tokens),
            "genotype_mask": torch.from_numpy(mask),
            "conservation": torch.from_numpy(conservation),
            "hbv_features": torch.tensor(sample.hbv_features, dtype=torch.float32),
            "task_id": torch.tensor(self.task_ids.get(sample.assay_type, 0), dtype=torch.long),
        }
        if sample.label is not None:
            item["label"] = torch.tensor(sample.label, dtype=torch.float32)
        return item

