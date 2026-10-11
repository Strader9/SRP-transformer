"""Phase 3 combined: pretrained sequence_encoder init + genotype-dropout consistency.

Compares 4 strategies (3 seeds each, locus split, lr=5e-4, bs=8, 25 epochs):
  1. baseline (scratch)
  2. dir2 only: gd0.25 + cw0.1
  3. dir4 only: pretrain init + freeze encoder
  4. combined: pretrain init (full finetune) + gd0.25 + cw0.1
"""
import sys, json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

REPO = Path(r"C:\Users\34359\Documents\SRP\SRP-transformer-HBV-v0.1")
sys.path.insert(0, str(REPO))

from hbv_sirna.data import HBVMultiGenotypeDataset
from hbv_sirna.model import HBVMultiGenotypeModel
from hbv_sirna.train import set_seed, _split_indices

HUVK = REPO / "data" / "external" / "huvk" / "huvk_pretrain.csv"
HBV = REPO / "data" / "processed" / "candidates_full_features.csv"
OUT = REPO / "results"

EPOCHS = 25
LR = 5e-4
BS = 8
WD = 1e-4
GD = 0.25
CW = 0.1
N_SEEDS = 3


def train_loop(dataset, train_idx, valid_idx, model, device, tag,
               genotype_dropout=0.0, consistency_weight=0.0, freeze_encoder=False):
    model = model.to(device)
    for p in model.parameters():
        p.requires_grad = True
    if freeze_encoder:
        for p in model.sequence_encoder.parameters():
            p.requires_grad = False
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BS, shuffle=True)
    valid_loader = DataLoader(Subset(dataset, valid_idx), batch_size=BS)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WD)
    criterion = nn.MSELoss()
    best = float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            inputs = {k: v.to(device) for k, v in batch.items() if k not in {"sample_id", "label"}}
            pred = model(**inputs)
            loss = criterion(pred, batch["label"].to(device))
            if consistency_weight > 0 and genotype_dropout > 0:
                drop_inputs = dict(inputs)
                mask = drop_inputs["genotype_mask"]
                keep = torch.rand(mask.shape, device=mask.device) > genotype_dropout
                drop_mask = mask & keep
                row_any = drop_mask.any(dim=-1, keepdim=True)
                drop_mask = drop_mask | ((~row_any) & mask)
                drop_inputs["genotype_mask"] = drop_mask
                drop_pred = model(**drop_inputs)
                loss = loss + consistency_weight * criterion(drop_pred, pred.detach())
            loss.backward()
            optimizer.step()
        model.eval()
        losses = []
        with torch.no_grad():
            for batch in valid_loader:
                pred = model(**{k: v.to(device) for k, v in batch.items() if k not in {"sample_id", "label"}})
                losses.append(criterion(pred, batch["label"].to(device)).item())
        mse = float(np.mean(losses))
        if mse < best:
            best = mse
            ckpt_dir = OUT / f"ckpt_p3_{tag}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "feature_dim": dataset.feature_dim},
                       ckpt_dir / "best.pt")
    return best


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Pretrain on HUVK (1 seed, reuse existing ckpt_p3_pretrain_huvk_s0)
    pretrained_ckpt = OUT / "ckpt_p3_pretrain_huvk_s0" / "best.pt"
    if not pretrained_ckpt.exists():
        print("Pretraining on HUVK (seed=0)...")
        pt_dataset = HBVMultiGenotypeDataset(HUVK)
        set_seed(0)
        train_idx, valid_idx = _split_indices(pt_dataset, 0, "locus")
        model = HBVMultiGenotypeModel(hbv_feature_dim=pt_dataset.feature_dim)
        train_loop(pt_dataset, train_idx, valid_idx, model, device, "pretrain_huvk_s0")
    saved = torch.load(pretrained_ckpt, map_location="cpu", weights_only=True)
    enc_state = {k: v for k, v in saved["state_dict"].items() if k.startswith("sequence_encoder.")}
    print(f"Loaded pretrained sequence_encoder ({len(enc_state)} tensors)")

    hbv_dataset = HBVMultiGenotypeDataset(HBV)
    print(f"HBV samples: {len(hbv_dataset)}")

    strategies = [
        ("baseline",        {"freeze_encoder": False, "genotype_dropout": 0.0,  "consistency_weight": 0.0,  "pretrain": False}),
        ("dir2_only",       {"freeze_encoder": False, "genotype_dropout": GD,   "consistency_weight": CW,   "pretrain": False}),
        ("dir4_freeze",     {"freeze_encoder": True,  "genotype_dropout": 0.0,  "consistency_weight": 0.0,  "pretrain": True}),
        ("combined",        {"freeze_encoder": False, "genotype_dropout": GD,   "consistency_weight": CW,   "pretrain": True}),
    ]

    results = {}
    for name, cfg in strategies:
        seed_mses = []
        for seed in range(N_SEEDS):
            set_seed(seed)
            train_idx, valid_idx = _split_indices(hbv_dataset, seed, "locus")
            model = HBVMultiGenotypeModel(hbv_feature_dim=hbv_dataset.feature_dim)
            if cfg["pretrain"]:
                model.load_state_dict(enc_state, strict=False)
            mse = train_loop(hbv_dataset, train_idx, valid_idx, model, device,
                             f"{name}_s{seed}",
                             genotype_dropout=cfg["genotype_dropout"],
                             consistency_weight=cfg["consistency_weight"],
                             freeze_encoder=cfg["freeze_encoder"])
            seed_mses.append(mse)
            print(f"  {name} seed={seed}: MSE={mse:.4f}")
        results[name] = {"mean": float(np.mean(seed_mses)), "std": float(np.std(seed_mses)),
                         "all": [float(x) for x in seed_mses]}

    with open(OUT / "phase3_combined_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Phase 3 Combined Results (3 seeds, HBV valid MSE) ===")
    for k, v in results.items():
        delta = (v["mean"] - results["baseline"]["mean"]) / results["baseline"]["mean"] * 100
        print(f"{k:14s}: {v['mean']:.4f} +/- {v['std']:.4f}  ({delta:+.1f}% vs baseline)")
    best = min(results.items(), key=lambda x: x[1]["mean"])
    print(f"\nBest: {best[0]} = {best[1]['mean']:.4f}")


if __name__ == "__main__":
    main()
