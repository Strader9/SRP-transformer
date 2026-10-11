"""Phase 3 Direction 4: Pretrain sequence encoder on HUVK, then finetune on HBV.

Stage 1: pretrain HBVMultiGenotypeModel (hbv_feature_dim=0, single genotype)
         on HUVK 2816 pairs -> results/ckpt_pretrain_huvk
Stage 2: finetune on HBV with pretrained sequence_encoder init.
         Modes: 'freeze' (encoder frozen) vs 'full' (all finetuned)
         -> compare vs baseline gd025_cw01 (0.0879)
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
N_SEEDS = 3


def train_loop(dataset, train_idx, valid_idx, model, device, batch_size, lr, wd, epochs, tag, freeze=False):
    model = model.to(device)
    for p in model.parameters():
        p.requires_grad = True
    if freeze:
        for p in model.sequence_encoder.parameters():
            p.requires_grad = False
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(Subset(dataset, valid_idx), batch_size=batch_size)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
    criterion = nn.MSELoss()
    best = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            pred = model(**{k: v.to(device) for k, v in batch.items() if k not in {"sample_id", "label"}})
            loss = criterion(pred, batch["label"].to(device))
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
            torch.save({"state_dict": model.state_dict(), "feature_dim": dataset.feature_dim},
                       OUT / f"ckpt_p3_{tag}" / "best.pt")
    return best


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Stage 1: pretrain on HUVK ----
    pt_dataset = HBVMultiGenotypeDataset(HUVK)
    print(f"HUVK pretrain samples: {len(pt_dataset)}")
    results = {}
    seed_mses = []
    for seed in range(N_SEEDS):
        set_seed(seed)
        train_idx, valid_idx = _split_indices(pt_dataset, seed, "locus")
        model = HBVMultiGenotypeModel(hbv_feature_dim=pt_dataset.feature_dim)
        mse = train_loop(pt_dataset, train_idx, valid_idx, model, device,
                         batch_size=64, lr=1e-3, wd=1e-4, epochs=EPOCHS,
                         tag=f"pretrain_huvk_s{seed}")
        seed_mses.append(mse)
        print(f"  pretrain seed={seed}: valid MSE={mse:.4f}")
    results["pretrain"] = {"mean": float(np.mean(seed_mses)), "std": float(np.std(seed_mses)),
                           "all": [float(x) for x in seed_mses]}

    # ---- Stage 2: finetune on HBV with pretrained init ----
    hbv_dataset = HBVMultiGenotypeDataset(HBV)
    print(f"\nHBV finetune samples: {len(hbv_dataset)}")

    pretrained_ckpt = OUT / f"ckpt_p3_pretrain_huvk_s0" / "best.pt"
    saved = torch.load(pretrained_ckpt, map_location="cpu", weights_only=True)

    for mode in ["freeze", "full"]:
        seed_mses = []
        for seed in range(N_SEEDS):
            set_seed(seed)
            train_idx, valid_idx = _split_indices(hbv_dataset, seed, "locus")
            model = HBVMultiGenotypeModel(hbv_feature_dim=hbv_dataset.feature_dim)
            # copy pretrained sequence_encoder weights
            enc_state = {k: v for k, v in saved["state_dict"].items()
                         if k.startswith("sequence_encoder.")}
            model.load_state_dict(enc_state, strict=False)
            mse = train_loop(hbv_dataset, train_idx, valid_idx, model, device,
                             batch_size=BS, lr=LR, wd=WD, epochs=EPOCHS,
                             tag=f"finetune_{mode}_s{seed}", freeze=(mode == "freeze"))
            seed_mses.append(mse)
            print(f"  finetune[{mode}] seed={seed}: valid MSE={mse:.4f}")
        results[f"finetune_{mode}"] = {"mean": float(np.mean(seed_mses)),
                                       "std": float(np.std(seed_mses)),
                                       "all": [float(x) for x in seed_mses]}

    # ---- Baseline for comparison: scratch training ----
    seed_mses = []
    for seed in range(N_SEEDS):
        set_seed(seed)
        train_idx, valid_idx = _split_indices(hbv_dataset, seed, "locus")
        model = HBVMultiGenotypeModel(hbv_feature_dim=hbv_dataset.feature_dim)
        mse = train_loop(hbv_dataset, train_idx, valid_idx, model, device,
                         batch_size=BS, lr=LR, wd=WD, epochs=EPOCHS,
                         tag=f"scratch_s{seed}")
        seed_mses.append(mse)
        print(f"  scratch seed={seed}: valid MSE={mse:.4f}")
    results["scratch"] = {"mean": float(np.mean(seed_mses)), "std": float(np.std(seed_mses)),
                          "all": [float(x) for x in seed_mses]}

    with open(OUT / "phase3_direction4_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n=== Direction 4: Pretrain + Finetune (3 seeds) ===")
    for k, v in results.items():
        print(f"{k:16s}: MSE={v['mean']:.4f} +/- {v['std']:.4f}")
    best = min(results.items(), key=lambda x: x[1]["mean"])
    print(f"\nBest: {best[0]} = {best[1]['mean']:.4f}")


if __name__ == "__main__":
    main()
