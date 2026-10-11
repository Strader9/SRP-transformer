from __future__ import annotations

import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
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


def _locus_group(sample) -> str:
    """Return a compatibility key for sample IDs such as ``ref:100-118``.

    The start position defines the locus (two candidates with the same start
    share the target site and must not be split across train/validation).
    Falls back to the full sample_id when the pattern is not present.
    """
    sample_id = sample if isinstance(sample, str) else sample.sample_id
    m = re.search(r":(\d+)-(\d+)", sample_id)
    if m:
        return str(m.group(1))
    if isinstance(sample, str):
        return sample
    # Literature records often use experiment IDs rather than coordinates.
    # Group on the aligned target bundle alone. This also prevents leakage when
    # a locus has multiple guides, orientations, or assay records.
    targets = ",".join(sorted(set(seq.upper().replace("U", "T") for seq in getattr(sample, "target_sequences", []))))
    return f"target:{targets}" if targets else sample_id


def _locus_groups(dataset: HBVMultiGenotypeDataset) -> list[str]:
    """Merge homologous target bundles and overlapping annotated intervals."""
    n = len(dataset.samples)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    by_targets: dict[str, int] = {}
    by_guide: dict[str, int] = {}
    for i, sample in enumerate(dataset.samples):
        target_key = ",".join(sorted(set(seq.upper().replace("U", "T") for seq in getattr(sample, "target_sequences", []))))
        if target_key:
            if target_key in by_targets:
                union(i, by_targets[target_key])
            else:
                by_targets[target_key] = i
        guide = getattr(sample, "sirna", "").upper().replace("U", "T")
        if guide:
            if guide in by_guide:
                union(i, by_guide[guide])
            else:
                by_guide[guide] = i

    def bundle_identity(first, second) -> float:
        first_targets = getattr(first, "target_sequences", [])
        second_targets = getattr(second, "target_sequences", [])
        first_genotypes = getattr(first, "genotype_ids", [])
        second_genotypes = getattr(second, "genotype_ids", [])
        first_by_gt = dict(zip(first_genotypes, first_targets)) if first_genotypes else {}
        second_by_gt = dict(zip(second_genotypes, second_targets)) if second_genotypes else {}
        if first_by_gt and second_by_gt:
            shared = sorted(set(first_by_gt) & set(second_by_gt))
            pairs = [(first_by_gt[key], second_by_gt[key]) for key in shared]
        else:
            pairs = list(zip(first_targets, second_targets))
        matches = observed = 0
        for seq_a, seq_b in pairs:
            if len(seq_a) != len(seq_b):
                continue
            for base_a, base_b in zip(seq_a.upper(), seq_b.upper()):
                if base_a in "ACGT" and base_b in "ACGT":
                    observed += 1
                    matches += base_a == base_b
        return 0.0 if observed == 0 else matches / observed

    # Keep near-identical homologous target bundles in one split. At 19 nt,
    # 95% corresponds to allowing only a small number of substitutions while
    # comparing all shared genotypes together.
    for i in range(n):
        for j in range(i + 1, n):
            guide_i = getattr(dataset.samples[i], "sirna", "").upper().replace("U", "T")
            guide_j = getattr(dataset.samples[j], "sirna", "").upper().replace("U", "T")
            same_near_guide = (
                len(guide_i) == len(guide_j) and len(guide_i) > 0
                and sum(a == b for a, b in zip(guide_i, guide_j)) / len(guide_i) >= 0.95
            )
            if same_near_guide or bundle_identity(dataset.samples[i], dataset.samples[j]) >= 0.95:
                union(i, j)

    positioned = []
    for i, sample in enumerate(dataset.samples):
        interval = getattr(sample, "ref_interval", None)
        if interval is None:
            match = re.search(r":(\d+)-(\d+)$", sample.sample_id)
            if match:
                interval = (int(match.group(1)) - 1, int(match.group(2)))
        if interval and interval[1] > interval[0]:
            positioned.append((i, interval))
    for a in range(len(positioned)):
        i, interval_i = positioned[a]
        assert interval_i is not None
        for b in range(a + 1, len(positioned)):
            j, interval_j = positioned[b]
            assert interval_j is not None
            if interval_i[0] < interval_j[1] and interval_j[0] < interval_i[1]:
                union(i, j)
    return [f"group:{find(i)}" for i in range(n)]


def _split_indices(
    dataset: HBVMultiGenotypeDataset,
    seed: int,
    split_by: str = "locus",
    val_ratio: float = 0.2,
    held_out_genotype: str | None = None,
) -> tuple[list[int], list[int]]:
    """Return (train_indices, valid_indices).

    ``locus`` mode groups samples by target site so the same locus never appears
    in both splits.  ``random`` mode is a plain row-wise shuffle (not recommended
    for siRNA data because of target-site leakage).
    """
    rng = np.random.default_rng(seed)
    n = len(dataset)
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    if split_by == "genotype":
        if not held_out_genotype:
            raise ValueError("held_out_genotype is required for genotype split")
        missing = [s.sample_id for s in dataset.samples if not s.genotype]
        if missing:
            raise ValueError("Genotype split requires a genotype column in the training CSV")
        train = [i for i, sample in enumerate(dataset.samples) if sample.genotype != held_out_genotype]
        valid = [i for i, sample in enumerate(dataset.samples) if sample.genotype == held_out_genotype]
        if not train or not valid:
            raise ValueError(f"Genotype {held_out_genotype!r} must have both held-out and training complement")
        return train, valid
    if split_by == "random":
        indices = rng.permutation(n)
        cut = max(1, int(n * (1 - val_ratio)))
        return indices[:cut].tolist(), indices[cut:].tolist()

    groups = _locus_groups(dataset)
    unique_groups = list(dict.fromkeys(groups))
    rng.shuffle(unique_groups)
    n_train_groups = max(1, int(len(unique_groups) * (1 - val_ratio)))
    train_groups = set(unique_groups[:n_train_groups])
    train_idx = [i for i, g in enumerate(groups) if g in train_groups]
    valid_idx = [i for i, g in enumerate(groups) if g not in train_groups]
    if not valid_idx:
        raise ValueError("Target groups overlap into one cluster; a leakage-safe validation split is not possible")
    return train_idx, valid_idx


def _group_kfold_indices(dataset, n_splits: int, seed: int = 42) -> list[tuple[list[int], list[int]]]:
    """Create disjoint, approximately size-balanced folds from leakage groups."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    groups = _locus_groups(dataset)
    members: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        members.setdefault(group, []).append(index)
    if len(members) < n_splits:
        raise ValueError(f"Need at least {n_splits} independent target groups; found {len(members)}")
    rng = np.random.default_rng(seed)
    group_names = list(members)
    rng.shuffle(group_names)
    group_names.sort(key=lambda group: len(members[group]), reverse=True)
    fold_groups: list[list[str]] = [[] for _ in range(n_splits)]
    fold_sizes = [0] * n_splits
    for group in group_names:
        fold = min(range(n_splits), key=lambda index: (fold_sizes[index], index))
        fold_groups[fold].append(group)
        fold_sizes[fold] += len(members[group])
    all_indices = set(range(len(dataset)))
    folds = []
    for held_groups in fold_groups:
        valid = sorted(index for group in held_groups for index in members[group])
        train = sorted(all_indices - set(valid))
        folds.append((train, valid))
    return folds


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
    fusion: str = "gated",
    positional_encoding: bool = True,
    held_out_genotype: str | None = None,
    model_config: dict[str, object] | None = None,
    allow_inferred_guides: bool = False,
) -> dict[str, object]:
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative")
    if not 0.0 <= genotype_dropout < 1.0:
        raise ValueError("genotype_dropout must be in [0, 1)")
    if consistency_weight < 0:
        raise ValueError("consistency_weight must be non-negative")
    if (genotype_dropout == 0.0) != (consistency_weight == 0.0):
        raise ValueError("genotype_dropout and consistency_weight must both be zero or both be positive")
    architecture = {"d_model": 128, "nhead": 4, "sequence_layers": 2,
                    "genotype_layers": 2, "dropout": 0.1}
    if model_config:
        unknown = set(model_config) - set(architecture)
        if unknown:
            raise ValueError(f"Unsupported model_config keys: {sorted(unknown)}")
        architecture.update(model_config)
    if int(architecture["d_model"]) <= 0 or int(architecture["nhead"]) <= 0:
        raise ValueError("d_model and nhead must be positive")
    if int(architecture["d_model"]) % int(architecture["nhead"]):
        raise ValueError("d_model must be divisible by nhead")
    if not 0.0 <= float(architecture["dropout"]) < 1.0:
        raise ValueError("model dropout must be in [0, 1)")
    set_seed(seed)
    dataset = HBVMultiGenotypeDataset(data_csv)
    inferred_guides = [sample.sample_id for sample in dataset.samples if sample.sirna_inferred]
    if inferred_guides and not allow_inferred_guides:
        raise ValueError(
            f"Found {len(inferred_guides)} records with inferred (not experimentally supplied) guide sequences. "
            "Exclude them or explicitly set allow_inferred_guides=True for exploratory analysis."
        )
    if len(dataset) < 5:
        raise ValueError("At least five labelled samples are required")
    if any(sample.label is None for sample in dataset.samples):
        raise ValueError("Training CSV contains samples without label")
    task_counts = {name: sum(sample.assay_type == name for sample in dataset.samples)
                   for name in dataset.task_names}
    if len(task_counts) > 1 and min(task_counts.values()) < 10:
        raise ValueError(
            f"Multi-task training requires at least 10 labeled records per assay; got {task_counts}"
        )
    train_idx, valid_idx = _split_indices(dataset, seed, split_by, held_out_genotype=held_out_genotype)
    if split_by == "genotype" and len(train_idx) < 5:
        raise ValueError("At least five training records are required in a genotype holdout fold")
    train_dataset = valid_dataset = dataset
    train_input_genotypes = valid_input_genotypes = None
    if split_by == "genotype":
        observed_genotypes = {g for sample in dataset.samples for g in sample.genotype_ids}
        if not observed_genotypes:
            raise ValueError("Genotype holdout requires genotype_ids in target_sequences")
        train_input_genotypes = observed_genotypes - {held_out_genotype}
        valid_input_genotypes = {held_out_genotype}
        if not train_input_genotypes:
            raise ValueError("No input genotypes remain after holding out the requested genotype")
        # Prevent sequence-level leakage: held-out genotype targets are hidden
        # from every training record and are the only target sequences shown at
        # evaluation time.
        train_dataset = HBVMultiGenotypeDataset(data_csv, visible_genotypes=train_input_genotypes)
        valid_dataset = HBVMultiGenotypeDataset(data_csv, visible_genotypes=valid_input_genotypes)
    train_loader = DataLoader(Subset(train_dataset, train_idx), batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(Subset(valid_dataset, valid_idx), batch_size=batch_size)
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = HBVMultiGenotypeModel(
        **architecture,
        hbv_feature_dim=dataset.feature_dim,
        num_tasks=len(dataset.task_names),
        fusion=fusion,
        positional_encoding=positional_encoding,
    ).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch in train_loader:
            optimizer.zero_grad()
            inputs = {key: value.to(target_device) for key, value in batch.items() if key not in {"sample_id", "label", "task_id"}}
            task_ids = batch["task_id"].to(target_device)
            prediction = model(**inputs)
            if prediction.ndim == 1:
                selected_prediction = prediction
            else:
                selected_prediction = prediction.gather(1, task_ids.unsqueeze(1)).squeeze(1)
            loss = criterion(selected_prediction, batch["label"].to(target_device))
            if consistency_weight > 0 and genotype_dropout > 0:
                drop_inputs = dict(inputs)
                mask = drop_inputs["genotype_mask"]
                keep = torch.rand(mask.shape, device=mask.device) > genotype_dropout
                drop_mask = mask & keep
                row_any = drop_mask.any(dim=-1, keepdim=True)
                drop_mask = drop_mask | ((~row_any) & mask)
                drop_inputs["genotype_mask"] = drop_mask
                target_repr = selected_prediction
                drop_prediction = model(**drop_inputs)
                if drop_prediction.ndim > 1:
                    drop_prediction = drop_prediction.gather(1, task_ids.unsqueeze(1)).squeeze(1)
                loss = loss + consistency_weight * criterion(drop_prediction, target_repr.detach())
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(batch["label"])
            train_count += len(batch["label"])
        model.eval()
        valid_loss_sum = 0.0
        valid_count = 0
        with torch.no_grad():
            for batch in valid_loader:
                inputs = {key: value.to(target_device) for key, value in batch.items() if key not in {"sample_id", "label", "task_id"}}
                prediction = model(**inputs)
                task_ids = batch["task_id"].to(target_device)
                if prediction.ndim > 1:
                    prediction = prediction.gather(1, task_ids.unsqueeze(1)).squeeze(1)
                loss = criterion(prediction, batch["label"].to(target_device))
                valid_loss_sum += loss.item() * len(batch["label"])
                valid_count += len(batch["label"])
        record = {"epoch": epoch, "train_mse": train_loss_sum / train_count,
                  "valid_mse": valid_loss_sum / valid_count}
        history.append(record)
        if record["valid_mse"] < best:
            best = record["valid_mse"]
            torch.save({
                "state_dict": model.state_dict(),
                "output_activation": "identity",
                "feature_dim": dataset.feature_dim,
                "feature_names": ["S", "P", "C", "X", "is_overlap", "gc_content", "avg_conservation"]
                    if dataset.feature_dim == 7 else None,
                "task_names": dataset.task_names,
                "fusion": fusion,
                "num_tasks": len(dataset.task_names),
                "positional_encoding": positional_encoding,
                "sequence_length": dataset.sequence_length,
                "max_genotypes": dataset.max_genotypes,
                "model_config": architecture,
                "input_genotypes": sorted(train_input_genotypes) if train_input_genotypes is not None else None,
                "inferred_guide_count": len(inferred_guides),
            }, out / "best.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    best_checkpoint = torch.load(out / "best.pt", map_location=target_device, weights_only=True)
    model.load_state_dict(best_checkpoint["state_dict"])
    model.eval()
    actual, predicted, valid_task_ids = [], [], []
    with torch.no_grad():
        for batch in valid_loader:
            inputs = {key: value.to(target_device) for key, value in batch.items()
                      if key not in {"sample_id", "label", "task_id"}}
            output = model(**inputs)
            task_ids = batch["task_id"].to(target_device)
            if output.ndim > 1:
                output = output.gather(1, task_ids.unsqueeze(1)).squeeze(1)
            actual.extend(batch["label"].cpu().numpy().tolist())
            predicted.extend(output.cpu().numpy().tolist())
            valid_task_ids.extend(task_ids.cpu().numpy().tolist())
    y = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    def corr(a: np.ndarray, b: np.ndarray) -> float | None:
        if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
            return None
        return float(np.corrcoef(a, b)[0, 1])
    rank_y = pd.Series(y).rank(method="average").to_numpy()
    rank_p = pd.Series(p).rank(method="average").to_numpy()
    metrics = {
        "mse": float(np.mean((p - y) ** 2)),
        "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
        "mae": float(np.mean(np.abs(p - y))),
        "pearson": corr(y, p),
        "spearman": corr(rank_y, rank_p),
    }
    by_task = {}
    task_ids_array = np.asarray(valid_task_ids, dtype=int)
    for task_id, task_name in enumerate(dataset.task_names):
        selected = task_ids_array == task_id
        task_y, task_p = y[selected], p[selected]
        task_rank_y = pd.Series(task_y).rank(method="average").to_numpy()
        task_rank_p = pd.Series(task_p).rank(method="average").to_numpy()
        by_task[task_name] = {
            "n": int(selected.sum()),
            "mse": float(np.mean((task_p - task_y) ** 2)) if selected.any() else None,
            "rmse": float(np.sqrt(np.mean((task_p - task_y) ** 2))) if selected.any() else None,
            "mae": float(np.mean(np.abs(task_p - task_y))) if selected.any() else None,
            "pearson": corr(task_y, task_p),
            "spearman": corr(task_rank_y, task_rank_p),
        }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    (out / "metrics_by_task.json").write_text(json.dumps(by_task, indent=2, allow_nan=False), encoding="utf-8")
    return {"best_valid_mse": best, "n_train": len(train_idx), "n_valid": len(valid_idx), "split_by": split_by,
            "task_names": dataset.task_names, "fusion": fusion,
            "held_out_genotype": held_out_genotype, "inferred_guide_count": len(inferred_guides),
            **metrics, "metrics_by_task": by_task}
