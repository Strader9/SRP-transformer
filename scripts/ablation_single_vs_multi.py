"""A3: Ablation study — single-genotype vs 8-genotype target input.

Creates a single-genotype variant of candidates_full_features.csv (only the
reference genotype target), then trains both variants with N random seeds
and compares best_valid_mse.

Usage:
  python scripts/ablation_single_vs_multi.py
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(r"C:\Users\34359\Documents\SRP\SRP-transformer-HBV-v0.1")
sys.path.insert(0, str(REPO))

from hbv_sirna.train import train_model

SRC = REPO / "data" / "processed" / "candidates_full_features.csv"
N_SEEDS = 5
EPOCHS = 15
BATCH_SIZE = 8
LR = 1e-4


def make_single_genotype_csv(src: Path, dst: Path) -> None:
    """Keep only the first (reference) target sequence per candidate."""
    df = pd.read_csv(src)
    rows = []
    for _, rec in df.iterrows():
        targets = json.loads(rec["target_sequences"])
        genotypes = json.loads(rec.get("genotype_ids", "[]"))
        row = rec.to_dict()
        row["target_sequences"] = json.dumps([targets[0]])
        row["genotype_ids"] = json.dumps([genotypes[0]] if genotypes else ["ref"])
        rows.append(row)
    pd.DataFrame(rows).to_csv(dst, index=False)


def run_ablation(csv_path: Path, seeds: list[int], tag: str) -> list[float]:
    mses = []
    for seed in seeds:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / f"ckpt_{tag}_{seed}"
            metrics = train_model(
                str(csv_path), str(ckpt),
                epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LR,
                seed=seed, split_by="random",
            )
            mse = metrics["best_valid_mse"]
            mses.append(mse)
            print(f"  [{tag}] seed={seed}: best_valid_mse={mse:.6f} "
                  f"(n_train={metrics['n_train']}, n_valid={metrics['n_valid']})")
    return mses


def main() -> None:
    df = pd.read_csv(SRC)
    print(f"Dataset: {len(df)} candidates, "
          f"n_genotypes={len(json.loads(df.iloc[0]['target_sequences']))}")

    seeds = list(range(42, 42 + N_SEEDS))

    # Single-genotype variant
    with tempfile.TemporaryDirectory() as tmpdir:
        single_csv = Path(tmpdir) / "candidates_single.csv"
        make_single_genotype_csv(SRC, single_csv)

        print(f"\n=== Single-genotype (1 target per candidate) ===")
        single_mses = run_ablation(single_csv, seeds, "single")

        print(f"\n=== 8-genotype (8 targets per candidate) ===")
        multi_mses = run_ablation(SRC, seeds, "multi")

    # Summary
    print(f"\n{'='*60}")
    print(f"ABLATION SUMMARY ({N_SEEDS} seeds, {EPOCHS} epochs)")
    print(f"{'='*60}")
    print(f"  Single-genotype: mean={np.mean(single_mses):.6f}  "
          f"std={np.std(single_mses):.6f}  "
          f"range=[{np.min(single_mses):.6f}, {np.max(single_mses):.6f}]")
    print(f"  8-genotype:      mean={np.mean(multi_mses):.6f}  "
          f"std={np.std(multi_mses):.6f}  "
          f"range=[{np.min(multi_mses):.6f}, {np.max(multi_mses):.6f}]")
    delta = np.mean(single_mses) - np.mean(multi_mses)
    print(f"  Delta (single - multi): {delta:+.6f} "
          f"({'multi better' if delta > 0 else 'single better/equal'})")

    # Paired comparison per seed
    print(f"\n  Per-seed comparison:")
    for i, seed in enumerate(seeds):
        diff = single_mses[i] - multi_mses[i]
        print(f"    seed={seed}: single={single_mses[i]:.6f}  "
              f"multi={multi_mses[i]:.6f}  diff={diff:+.6f}")

    wins = sum(1 for s, m in zip(single_mses, multi_mses) if m < s)
    print(f"\n  8-genotype wins (lower MSE): {wins}/{N_SEEDS} seeds")


if __name__ == "__main__":
    main()
