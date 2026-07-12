from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import HBVMultiGenotypeDataset
from .model import HBVMultiGenotypeModel


def predict(data_csv: str, checkpoint: str, output_csv: str, batch_size: int = 64, device: str | None = None) -> pd.DataFrame:
    dataset = HBVMultiGenotypeDataset(data_csv)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    saved = torch.load(checkpoint, map_location=target_device, weights_only=True)
    model = HBVMultiGenotypeModel(hbv_feature_dim=int(saved["feature_dim"])).to(target_device)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            values = model(**{key: value.to(target_device) for key, value in batch.items() if key not in {"sample_id", "label"}})
            rows.extend(zip(batch["sample_id"], values.cpu().tolist()))
    result = pd.DataFrame(rows, columns=["sample_id", "hbv_score"]).sort_values("hbv_score", ascending=False)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return result

