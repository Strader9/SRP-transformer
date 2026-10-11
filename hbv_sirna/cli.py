from __future__ import annotations

import argparse
import json

from .infer import predict
from .prepare import build_candidate_table
from .train import train_model


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="hbv-sirna")
    commands = root.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Build candidates from an aligned HBV FASTA")
    prepare.add_argument("--alignment", required=True)
    prepare.add_argument("--reference-id", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--window", type=int, default=19)
    prepare.add_argument("--stride", type=int, default=1)
    prepare.add_argument("--annotation", default=None, help="BED file of HBV functional regions (enables 7-dim hbv_features)")
    train = commands.add_parser("train", help="Train the HBV multi-genotype model")
    train.add_argument("--data", required=True)
    train.add_argument("--output-dir", default="checkpoints")
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--split-by", choices=["locus", "random", "genotype"], default="locus", help="Split by overlapping/exact target loci, random rows, or held-out assay genotype")
    train.add_argument("--held-out-genotype", default=None, help="Required when --split-by genotype")
    train.add_argument("--genotype-dropout", type=float, default=0.25)
    train.add_argument("--consistency-weight", type=float, default=0.1)
    train.add_argument("--fusion", choices=["concat", "gated"], default="gated")
    train.add_argument("--model-size", choices=["compact", "base"], default="compact",
                       help="compact (32-dim) is the recommended starting point for small HBV datasets")
    train.add_argument("--allow-inferred-guides", action="store_true",
                       help="Allow proxy guide sequences inferred from target regions; exploratory only")
    train.add_argument("--no-positional-encoding", action="store_false", dest="positional_encoding", help="Disable sequence position encoding for legacy comparisons")
    train.set_defaults(positional_encoding=True)
    infer = commands.add_parser("predict", help="Rank candidate siRNAs")
    infer.add_argument("--data", required=True)
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--output", default="results/ranked.csv")
    infer.add_argument("--rank-task", default=None, help="Task column to sort by for a multi-task checkpoint")
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "prepare":
        frame = build_candidate_table(args.alignment, args.output, args.reference_id, args.window, args.stride, args.annotation)
        print(f"wrote {len(frame)} candidates to {args.output}")
    elif args.command == "train":
        metrics = train_model(
            data_csv=args.data,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
            split_by=args.split_by,
            genotype_dropout=args.genotype_dropout,
            consistency_weight=args.consistency_weight,
            fusion=args.fusion,
            positional_encoding=args.positional_encoding,
            held_out_genotype=args.held_out_genotype,
            allow_inferred_guides=args.allow_inferred_guides,
            model_config=({"d_model": 32, "nhead": 4, "sequence_layers": 1,
                           "genotype_layers": 1, "dropout": 0.2}
                          if args.model_size == "compact" else None),
        )
        print(json.dumps(metrics, indent=2))
    elif args.command == "predict":
        frame = predict(args.data, args.checkpoint, args.output, rank_task=args.rank_task)
        print(frame.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

