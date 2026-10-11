"""Evaluate assay-genotype transfer with grouped train/test folds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from hbv_sirna.data import HBVMultiGenotypeDataset
from hbv_sirna.train import train_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/candidates_hbsag.csv")
    parser.add_argument("--output-dir", default="results/phase3_leave_one_genotype_out")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--fusion", choices=["concat", "gated"], default="gated")
    parser.add_argument("--min-fold-samples", type=int, default=5,
                        help="Skip held-out genotypes with fewer than this many training or test records")
    parser.add_argument("--allow-inferred-guides", action="store_true",
                        help="Allow guide sequences inferred from target regions; exploratory only")
    args = parser.parse_args()

    dataset = HBVMultiGenotypeDataset(args.data)
    genotypes = sorted({sample.genotype for sample in dataset.samples if sample.genotype})
    if len(genotypes) < 2:
        raise SystemExit("Leave-one-genotype-out requires at least two assay genotypes in the data")
    results = {}
    for genotype in genotypes:
        n_test = sum(sample.genotype == genotype for sample in dataset.samples)
        n_train = len(dataset) - n_test
        if min(n_test, n_train) < args.min_fold_samples:
            results[genotype] = {
                "status": "skipped_insufficient_genotype_data",
                "n_train": n_train,
                "n_test": n_test,
                "minimum_per_fold": args.min_fold_samples,
            }
            print(f"held={genotype}: skipped (train={n_train}, test={n_test})", flush=True)
            continue
        fold = []
        for seed in range(args.seeds):
            result = train_model(
                data_csv=args.data,
                output_dir=str(Path(args.output_dir) / f"held_{genotype}_seed_{seed}"),
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=1e-4,
                seed=seed,
                split_by="genotype",
                held_out_genotype=genotype,
                genotype_dropout=0.25,
                consistency_weight=0.1,
                fusion=args.fusion,
                positional_encoding=True,
                allow_inferred_guides=args.allow_inferred_guides,
            )
            fold.append(result)
            print(f"held={genotype} seed={seed} metrics={result}", flush=True)
        results[genotype] = fold
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(results, indent=2, allow_nan=True), encoding="utf-8")
    printable = []
    for genotype, fold in results.items():
        if isinstance(fold, dict):
            continue
        printable.append(
        {"held_out_genotype": genotype,
         "n_test": int(pd.Series(fold).apply(lambda item: item["n_valid"]).iloc[0]),
         "mean_mse": float(pd.Series(fold).apply(lambda item: item["mse"]).mean()),
         "mean_spearman": float(pd.Series(fold).apply(lambda item: item["spearman"]).mean())}
        )
    print(pd.DataFrame(printable).to_string(index=False) if printable else "No genotype fold has enough labeled data.")


if __name__ == "__main__":
    main()
