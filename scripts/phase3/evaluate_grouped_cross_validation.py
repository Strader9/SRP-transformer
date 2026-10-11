"""Leakage-resistant grouped cross-validation with an outer test fold.

Each outer test fold is held out by target/guide similarity group. Training
uses a second grouped split only for epoch selection; the outer test labels are
not used until prediction is complete.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from hbv_sirna.data import HBVMultiGenotypeDataset
from hbv_sirna.infer import predict
from hbv_sirna.train import _group_kfold_indices, _locus_groups, train_model


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int | None]:
    if len(actual) == 0:
        return {"n": 0, "mse": None, "rmse": None, "mae": None, "pearson": None, "spearman": None}
    mse = float(np.mean((predicted - actual) ** 2))

    def correlation(a: np.ndarray, b: np.ndarray) -> float | None:
        if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
            return None
        return float(np.corrcoef(a, b)[0, 1])

    return {
        "n": int(len(actual)), "mse": mse, "rmse": math.sqrt(mse),
        "mae": float(np.mean(np.abs(predicted - actual))),
        "pearson": correlation(actual, predicted),
        "spearman": correlation(pd.Series(actual).rank().to_numpy(),
                                 pd.Series(predicted).rank().to_numpy()),
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/candidates_hbsag.csv")
    parser.add_argument("--output-dir", default="results/phase3_grouped_cv")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=1,
                        help="Repeated outer CV runs; increase to estimate split variability")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--fusion", choices=["concat", "gated"], default="gated")
    parser.add_argument("--model-size", choices=["compact", "base"], default="compact")
    parser.add_argument("--allow-inferred-guides", action="store_true",
                        help="Required for current literature records whose guide was inferred from target")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    data_path = Path(args.data)
    frame = pd.read_csv(data_path)
    dataset = HBVMultiGenotypeDataset(data_path)
    if len(dataset.task_names) != 1:
        raise SystemExit("This evaluator currently expects one assay endpoint per CSV; filter to one assay first")
    inferred_count = sum(sample.sirna_inferred for sample in dataset.samples)
    if inferred_count and not args.allow_inferred_guides:
        raise SystemExit(
            f"Found {inferred_count} inferred guide sequences. Pass --allow-inferred-guides only for exploratory analysis."
        )
    if any(sample.label is None for sample in dataset.samples):
        raise SystemExit("Cross-validation data must contain labels for every row")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    all_predictions: list[pd.DataFrame] = []
    all_baselines: list[pd.DataFrame] = []
    run_results = []

    for repeat in range(args.seeds):
        outer_folds = _group_kfold_indices(dataset, args.folds, seed=repeat)
        for fold_index, (train_indices, test_indices) in enumerate(outer_folds):
            fold_dir = output / f"repeat_{repeat}" / f"fold_{fold_index}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            train_csv = fold_dir / "outer_train.csv"
            test_csv = fold_dir / "outer_test.csv"
            frame.iloc[train_indices].to_csv(train_csv, index=False)
            frame.iloc[test_indices].to_csv(test_csv, index=False)
            checkpoint_dir = fold_dir / "checkpoint"
            validation = train_model(
                data_csv=str(train_csv), output_dir=str(checkpoint_dir),
                epochs=args.epochs, batch_size=args.batch_size,
                learning_rate=args.learning_rate, weight_decay=1e-4,
                seed=repeat * args.folds + fold_index, split_by="locus",
                fusion=args.fusion, positional_encoding=True, device=args.device,
                genotype_dropout=0.25, consistency_weight=0.1,
                model_config=({"d_model": 32, "nhead": 4, "sequence_layers": 1,
                               "genotype_layers": 1, "dropout": 0.2}
                              if args.model_size == "compact" else None),
                allow_inferred_guides=args.allow_inferred_guides,
            )
            predictions = predict(str(test_csv), str(checkpoint_dir / "best.pt"),
                                  str(fold_dir / "predictions.csv"), device=args.device)
            score_column = dataset.task_names[0]
            if score_column in {"efficacy", "hbv_score"}:
                score_column = "hbv_score"
            joined = frame.iloc[test_indices][["sample_id", "label"]].merge(
                predictions[["sample_id", score_column]], on="sample_id", how="left", validate="one_to_one"
            )
            if joined[score_column].isna().any():
                raise RuntimeError("A test row did not receive a prediction")
            joined = joined.rename(columns={score_column: "prediction"})
            joined["repeat"] = repeat
            joined["fold"] = fold_index
            all_predictions.append(joined)
            baseline = joined[["sample_id", "label", "repeat", "fold"]].copy()
            baseline["prediction"] = float(frame.iloc[train_indices]["label"].mean())
            all_baselines.append(baseline)
            fold_metrics = _metrics(joined["label"].to_numpy(), joined["prediction"].to_numpy())
            baseline_metrics = _metrics(baseline["label"].to_numpy(), baseline["prediction"].to_numpy())
            run_results.append({
                "repeat": repeat, "fold": fold_index,
                "n_train": len(train_indices), "n_test": len(test_indices),
                "inner_validation": validation, "outer_metrics": fold_metrics,
                "mean_baseline_metrics": baseline_metrics,
            })
            print(f"repeat={repeat} fold={fold_index} n_test={len(test_indices)} "
                  f"outer_mse={fold_metrics['mse']:.4f}", flush=True)

    predictions = pd.concat(all_predictions, ignore_index=True)
    baselines = pd.concat(all_baselines, ignore_index=True)
    predictions = predictions.merge(
        baselines[["sample_id", "repeat", "fold", "prediction"]].rename(
            columns={"prediction": "mean_baseline_prediction"}
        ), on=["sample_id", "repeat", "fold"], how="left", validate="one_to_one"
    )
    predictions.to_csv(output / "out_of_fold_predictions.csv", index=False)
    fold_summary = {
        "data": str(data_path), "folds": args.folds, "repeats": args.seeds,
        "model_size": args.model_size,
        "allow_inferred_guides": args.allow_inferred_guides,
        "inferred_guide_count": inferred_count,
        "n_records": len(dataset), "n_independent_groups": len(set(_locus_groups(dataset))),
        "outer_metrics_oof": _metrics(predictions["label"].to_numpy(),
                                       predictions["prediction"].to_numpy()),
        "mean_baseline_metrics_oof": _metrics(predictions["label"].to_numpy(),
                                              predictions["mean_baseline_prediction"].to_numpy()),
        "folds_detail": run_results,
    }
    (output / "summary.json").write_text(
        json.dumps(_json_safe(fold_summary), indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(fold_summary["outer_metrics_oof"], indent=2), flush=True)


if __name__ == "__main__":
    main()
