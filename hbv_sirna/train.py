from __future__ import annotations

import json
import random
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


def train_model(
    data_csv: str,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    seed: int = 42,
    device: str | None = None,
) -> dict[str, float]:
    set_seed(seed)
    dataset = HBVMultiGenotypeDataset(data_csv)
    if len(dataset) < 5:
        raise ValueError("At least five labelled samples are required")
    if any(sample.label is None for sample in dataset.samples):
        raise ValueError("Training CSV contains samples without label")
    indices = np.random.default_rng(seed).permutation(len(dataset))
    split = max(1, int(len(indices) * 0.8))
    train_loader = DataLoader(Subset(dataset, indices[:split]), batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(Subset(dataset, indices[split:]), batch_size=batch_size)
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = HBVMultiGenotypeModel(hbv_feature_dim=dataset.feature_dim).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
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
            prediction = model(**{key: value.to(target_device) for key, value in batch.items() if key not in {"sample_id", "label"}})
            loss = criterion(prediction, batch["label"].to(target_device))
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
    return {"best_valid_mse": best}

