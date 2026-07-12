from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .sequence import conservation_profile, tokenize


REQUIRED_COLUMNS = {"sample_id", "sirna", "target_sequences"}


@dataclass(frozen=True)
class Sample:
    sample_id: str
    sirna: str
    target_sequences: list[str]
    genotype_ids: list[str]
    hbv_features: list[float]
    label: float | None


def parse_sample(row: pd.Series) -> Sample:
    targets = json.loads(row["target_sequences"])
    genotype_ids = json.loads(row.get("genotype_ids", "[]"))
    features = json.loads(row.get("hbv_features", "[]"))
    if not isinstance(targets, list) or not targets:
        raise ValueError(f"{row['sample_id']}: target_sequences must be a non-empty JSON list")
    if genotype_ids and len(genotype_ids) != len(targets):
        raise ValueError(f"{row['sample_id']}: genotype_ids and target_sequences differ in length")
    label = None if "label" not in row or pd.isna(row["label"]) else float(row["label"])
    return Sample(str(row["sample_id"]), str(row["sirna"]), targets, genotype_ids, features, label)


class HBVMultiGenotypeDataset(Dataset):
    def __init__(self, csv_path: str | Path, sequence_length: int = 19, max_genotypes: int = 10):
        frame = pd.read_csv(csv_path)
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"Missing dataset columns: {sorted(missing)}")
        self.samples = [parse_sample(row) for _, row in frame.iterrows()]
        self.sequence_length = sequence_length
        self.max_genotypes = max_genotypes
        feature_dims = {len(sample.hbv_features) for sample in self.samples}
        if len(feature_dims) > 1:
            raise ValueError("All samples must have the same hbv_features length")
        self.feature_dim = next(iter(feature_dims), 0)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        targets = sample.target_sequences[: self.max_genotypes]
        mask = np.zeros(self.max_genotypes, dtype=np.bool_)
        mask[: len(targets)] = True
        padded_targets = targets + ["-"] * (self.max_genotypes - len(targets))
        target_tokens = np.stack([tokenize(seq, self.sequence_length) for seq in padded_targets])
        aligned = [seq[: self.sequence_length].ljust(self.sequence_length, "-") for seq in targets]
        conservation = conservation_profile(aligned)
        item: dict[str, torch.Tensor | str] = {
            "sample_id": sample.sample_id,
            "sirna_tokens": torch.from_numpy(tokenize(sample.sirna, self.sequence_length)),
            "target_tokens": torch.from_numpy(target_tokens),
            "genotype_mask": torch.from_numpy(mask),
            "conservation": torch.from_numpy(conservation),
            "hbv_features": torch.tensor(sample.hbv_features, dtype=torch.float32),
        }
        if sample.label is not None:
            item["label"] = torch.tensor(sample.label, dtype=torch.float32)
        return item

