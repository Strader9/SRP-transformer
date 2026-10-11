from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import HBVMultiGenotypeDataset
from .model import HBVMultiGenotypeModel, apply_checkpoint_output, load_compatible_state_dict
from .sequence import genotype_coverage


def predict(data_csv: str, checkpoint: str, output_csv: str, batch_size: int = 64,
            device: str | None = None, rank_task: str | None = None,
            visible_genotypes: set[str] | None = None) -> pd.DataFrame:
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    saved = torch.load(checkpoint, map_location=target_device, weights_only=True)
    if visible_genotypes is None and saved.get("input_genotypes") is not None:
        visible_genotypes = set(saved["input_genotypes"])
    dataset = HBVMultiGenotypeDataset(
        data_csv,
        sequence_length=int(saved.get("sequence_length", 19)),
        max_genotypes=int(saved.get("max_genotypes", 10)),
        visible_genotypes=visible_genotypes,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    task_names = saved.get("task_names", ["hbv_score"])
    if len(task_names) != int(saved.get("num_tasks", 1)):
        raise ValueError("Checkpoint task_names do not match num_tasks")
    if rank_task is not None and rank_task not in task_names:
        raise ValueError(f"rank_task {rank_task!r} is not in checkpoint tasks {task_names}")
    model_config = saved.get("model_config", {})
    model = HBVMultiGenotypeModel(
        **model_config,
        hbv_feature_dim=int(saved["feature_dim"]),
        num_tasks=int(saved.get("num_tasks", 1)),
        fusion=saved.get("fusion", "concat"),
        positional_encoding=bool(saved.get("positional_encoding", False)),
        sequence_length=int(saved.get("sequence_length", 19)),
    ).to(target_device)
    if dataset.feature_dim != int(saved["feature_dim"]):
        raise ValueError(
            f"Input feature dimension {dataset.feature_dim} does not match checkpoint feature_dim {saved['feature_dim']}"
        )
    load_compatible_state_dict(model, saved["state_dict"])
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            values = model(**{key: value.to(target_device) for key, value in batch.items() if key not in {"sample_id", "label", "task_id"}})
            values = apply_checkpoint_output(values, saved)
            if values.ndim == 1:
                values = values.unsqueeze(1)
            for sample_id, scores in zip(batch["sample_id"], values.cpu().tolist()):
                sample = dataset.samples[len(rows)]
                if visible_genotypes is None or not sample.genotype_ids:
                    targets = sample.target_sequences[: dataset.max_genotypes]
                else:
                    targets = [target for genotype, target in zip(sample.genotype_ids, sample.target_sequences)
                               if genotype in visible_genotypes][:dataset.max_genotypes]
                try:
                    coverage = genotype_coverage(sample.sirna, targets)
                except ValueError:
                    # Keep scoring available for non-19mer input; leave coverage
                    # missing instead of reporting a misleading value.
                    coverage = float("nan")
                rows.append((sample_id, *scores, coverage, len(targets), sample.sirna_inferred))
    columns = ["sample_id", *task_names, "genotype_coverage", "genotypes_observed", "guide_inferred"]
    result = pd.DataFrame(rows, columns=columns)
    if len(task_names) == 1:
        output_name = "hbv_score" if task_names[0] in {"hbv_score", "efficacy"} else task_names[0]
        if output_name != task_names[0]:
            result = result.rename(columns={task_names[0]: output_name})
        result = result.sort_values(output_name, ascending=False)
    elif rank_task is not None:
        result = result.sort_values(rank_task, ascending=False)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return result

