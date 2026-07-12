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
    train = commands.add_parser("train", help="Train the HBV multi-genotype model")
    train.add_argument("--data", required=True)
    train.add_argument("--output-dir", default="checkpoints")
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--seed", type=int, default=42)
    infer = commands.add_parser("predict", help="Rank candidate siRNAs")
    infer.add_argument("--data", required=True)
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--output", default="results/ranked.csv")
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "prepare":
        frame = build_candidate_table(args.alignment, args.output, args.reference_id, args.window, args.stride)
        print(f"wrote {len(frame)} candidates to {args.output}")
    elif args.command == "train":
        metrics = train_model(args.data, args.output_dir, args.epochs, args.batch_size, args.learning_rate, args.seed)
        print(json.dumps(metrics, indent=2))
    elif args.command == "predict":
        frame = predict(args.data, args.checkpoint, args.output)
        print(frame.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

