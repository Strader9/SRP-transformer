from __future__ import annotations

import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from .data import HBVMultiGenotypeDataset
from .model import HBVMultiGenotypeModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _locus_group(sample_id: str) -> str:
    """Extract a target-locus key from ``sample_id`` like ``ref:100-118``.

    The start position defines the locus (two candidates with the same start
    share the target site and must not be split across train/validation).
    Falls back to the full sample_id when the pattern is not present.
    """
    m = re.search(r":(\d+)-(\d+)", sample_id)
    return m.group(1) if m else sample_id


def _split_indices(
    dataset: HBVMultiGenotypeDataset,
    seed: int,
    split_by: str = "locus",
    val_ratio: float = 0.2,
) -> tuple[list[int], list[int]]:
    """Return (train_indices, valid_indices).

    ``locus`` mode groups samples by target site so the same locus never appears
    in both splits.  ``random`` mode is a plain row-wise shuffle (not recommended
    for siRNA data because of target-site leakage).
    """
    rng = np.random.default_rng(seed)
    n = len(dataset)
    if split_by == "random":
        indices = rng.permutation(n)
        cut = max(1, int(n * (1 - val_ratio)))
        return indices[:cut].tolist(), indices[cut:].tolist()

    groups = [_locus_group(s.sample_id) for s in dataset.samples]
    unique_groups = list(dict.fromkeys(groups))
    rng.shuffle(unique_groups)
    n_train_groups = max(1, int(len(unique_groups) * (1 - val_ratio)))
    train_groups = set(unique_groups[:n_train_groups])
    train_idx = [i for i, g in enumerate(groups) if g in train_groups]
    valid_idx = [i for i, g in enumerate(groups) if g not in train_groups]
    if not valid_idx:
        valid_idx = [train_idx.pop()]
    return train_idx, valid_idx


def train_model(
    data_csv: str,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    seed: int = 42,
    split_by: str = "locus",
    device: str | None = None,
    genotype_dropout: float = 0.0,
    consistency_weight: float = 0.0,
) -> dict[str, float]:
    set_seed(seed)
    dataset = HBVMultiGenotypeDataset(data_csv)
    if len(dataset) < 5:
        raise ValueError("At least five labelled samples are required")
    if any(sample.label is None for sample in dataset.samples):
        raise ValueError("Training CSV contains samples without label")
    train_idx, valid_idx = _split_indices(dataset, seed, split_by)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(Subset(dataset, valid_idx), batch_size=batch_size)
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = HBVMultiGenotypeModel(hbv_feature_dim=dataset.feature_dim).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            inputs = {key: value.to(target_device) for key, value in batch.items() if key not in {"sample_id", "label"}}
            prediction = model(**inputs)
            loss = criterion(prediction, batch["label"].to(target_device))
            if consistency_weight > 0 and genotype_dropout > 0:
                drop_inputs = dict(inputs)
                mask = drop_inputs["genotype_mask"]
                keep = torch.rand(mask.shape, device=mask.device) > genotype_dropout
                drop_mask = mask & keep
                row_any = drop_mask.any(dim=-1, keepdim=True)
                drop_mask = drop_mask | ((~row_any) & mask)
                drop_inputs["genotype_mask"] = drop_mask
                with torch.no_grad():
                    target_repr = prediction
                drop_prediction = model(**drop_inputs)
                loss = loss + consistency_weight * criterion(drop_prediction, target_repr.detach())
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        model.eval()
        valid_losses = []
        with torch.no_grad():
            for batch in valid_loader:
                prediction = model(**{key: value.to(target_device) for key, value in batch.items() if key not in {"sample_id", "label"}})
                valid_losses.append(criterion(prediction, batch["label"].to(target_device)).item())
        record = {"epoch": epoch, "train_mse": float(np.mean(train_losses)), "valid_mse": float(np.mean(valid_losses))}
        history.append(record)
        if record["valid_mse"] < best:
            best = record["valid_mse"]
            torch.save({"state_dict": model.state_dict(), "feature_dim": dataset.feature_dim}, out / "best.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {"best_valid_mse": best, "n_train": len(train_idx), "n_valid": len(valid_idx), "split_by": split_by}
