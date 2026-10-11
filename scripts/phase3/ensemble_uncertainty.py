"""Phase 3 Direction 5: Deep ensemble uncertainty on unlabeled candidates.

Pipeline:
1. Expand 11 unlabeled Ely2009 candidates to 8-genotype targets (same as
   expand_to_multi_genotype.py) using the reference genomes.
2. Compute 7-dim hbv_features (same as annotate_features_a2.py).
3. Load 5 checkpoints (ckpt_seq_s0..s4) and predict each candidate.
4. Report mean +/- std (uncertainty) ranked by uncertainty.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO = Path(r"C:\Users\34359\Documents\SRP\SRP-transformer-HBV-v0.1")
sys.path.insert(0, str(REPO))

from hbv_sirna.annotation import parse_bed, overlapping_classes
from hbv_sirna.data import HBVMultiGenotypeDataset
from hbv_sirna.model import HBVMultiGenotypeModel
from hbv_sirna.sequence import conservation_profile

GENOMES_DIR = REPO / "data" / "raw" / "genomes"
BED_PATH = REPO / "data" / "raw" / "hbv_regions_genotypeA.bed"
REF_GENOME = REPO / "data" / "raw" / "genomes" / "hbv_genotype_A.fasta"
RAW_CSV = REPO / "data" / "raw" / "hbv_experiments.csv"
OUT_CSV = REPO / "data" / "processed" / "candidates_unlabeled_ensemble.csv"

GENOTYPE_ORDER = ["A", "B", "C", "D", "E", "F", "G", "H"]
WINDOW = 19
CKPTS = [REPO / "results" / f"ckpt_seq_s{s}" / "best.pt" for s in range(5)]


def load_genomes() -> dict[str, str]:
    genomes = {}
    for gt in GENOTYPE_ORDER:
        lines = (GENOMES_DIR / f"hbv_genotype_{gt}.fasta").read_text(encoding="utf-8").strip().splitlines()
        genomes[gt] = "".join(lines[1:]).upper().replace("U", "T")
    return genomes


def revcomp(s: str) -> str:
    return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def normalise_orientation(genomes: dict[str, str], ref_key: str = "A") -> dict[str, str]:
    def best_match_window(target, genome):
        circular = genome + genome[: WINDOW - 1]
        best_dist = WINDOW + 1
        best_seq, best_pos = "", -1
        for i in range(len(genome)):
            window = circular[i : i + WINDOW]
            d = sum(1 for a, b in zip(target, window) if a != b)
            if d < best_dist:
                best_dist, best_seq, best_pos = d, window, i
                if d == 0:
                    break
        return best_seq, best_dist, best_pos

    ref = genomes[ref_key]
    anchors = [ref[500:600], ref[1500:1600], ref[2500:2600]]
    result = {}
    for gt, seq in genomes.items():
        if gt == ref_key:
            result[gt] = seq
            continue
        vf = vr = 0
        for anchor in anchors:
            _, df_, _ = best_match_window(anchor, seq)
            _, dr_, _ = best_match_window(anchor, revcomp(seq))
            if df_ <= dr_:
                vf += 1
            else:
                vr += 1
        result[gt] = seq if vf >= vr else revcomp(seq)
    return result


def best_match_window(target: str, genome: str) -> tuple[str, int, int]:
    circular = genome + genome[: WINDOW - 1]
    best_dist = WINDOW + 1
    best_seq, best_pos = "", -1
    for i in range(len(genome)):
        window = circular[i : i + WINDOW]
        d = sum(1 for a, b in zip(target, window) if a != b)
        if d < best_dist:
            best_dist, best_seq, best_pos = d, window, i
            if d == 0:
                break
    return best_seq, best_dist, best_pos


def gc_content(seq: str) -> float:
    gc = sum(1 for b in seq if b in "GC")
    total = sum(1 for b in seq if b in "ACGT")
    return 0.0 if total == 0 else gc / total


def main() -> None:
    raw = pd.read_csv(RAW_CSV)
    cand = raw[(raw["label"].isna()) & (raw["sirna"].notna())].copy()
    print(f"Unlabeled candidates with siRNA: {len(cand)}")

    genomes = normalise_orientation(load_genomes())
    regions = parse_bed(BED_PATH)
    ref_genome = genomes["A"]

    rows = []
    for _, rec in cand.iterrows():
        sirna = str(rec["sirna"]).upper()
        # 19-mer at 5' end, target = revcomp (consistent with training pipeline)
        sirna19 = sirna[:WINDOW]
        ref_target = revcomp(sirna19)

        targets, distances, positions = [], [], []
        for gt in GENOTYPE_ORDER:
            seq, dist, pos = best_match_window(ref_target, genomes[gt])
            targets.append(seq)
            distances.append(dist)
            positions.append(pos)

        # hbv_features: map ref target onto genotype A for BED query
        ref_seq_a, dist_a, pos_a = best_match_window(ref_target, ref_genome)
        start, end = pos_a, pos_a + WINDOW
        classes = set()
        if end <= len(ref_genome):
            classes |= overlapping_classes(regions, start, end)
        else:
            classes |= overlapping_classes(regions, start, len(ref_genome))
            classes |= overlapping_classes(regions, 0, end - len(ref_genome))

        gc = gc_content(ref_target)
        aligned = [s[:WINDOW].ljust(WINDOW, "-") for s in targets]
        avg_cons = float(np.mean(conservation_profile(aligned)))
        is_overlap = 1.0 if len(classes) > 1 else 0.0
        features = [
            1.0 if "S" in classes else 0.0,
            1.0 if "P" in classes else 0.0,
            1.0 if "C" in classes else 0.0,
            1.0 if "X" in classes else 0.0,
            is_overlap, gc, avg_cons,
        ]

        rows.append({
            "sample_id": f"{rec['experiment_id']}",
            "sirna": sirna19,
            "sirna_full": sirna,
            "target_sequences": json.dumps(targets),
            "genotype_ids": json.dumps(GENOTYPE_ORDER),
            "hbv_features": json.dumps(features),
            "_match_distances": json.dumps(distances),
            "_classes": ",".join(sorted(classes)) if classes else "none",
        })

    cand_df = pd.DataFrame(rows)
    cand_csv = REPO / "data" / "processed" / "candidates_unlabeled.csv"
    cand_df.to_csv(cand_csv, index=False)
    print(f"Wrote {len(cand_df)} expanded candidates to {cand_csv}")
    print(f"  Match distances: {cand_df['_match_distances'].tolist()}")

    # ---- Ensemble prediction ----
    dataset = HBVMultiGenotypeDataset(cand_csv)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preds = []
    for ckpt in CKPTS:
        saved = torch.load(ckpt, map_location=device, weights_only=True)
        model = HBVMultiGenotypeModel(hbv_feature_dim=int(saved["feature_dim"])).to(device)
        model.load_state_dict(saved["state_dict"])
        model.eval()
        with torch.no_grad():
            for batch in loader:
                out = model(**{k: v.to(device) for k, v in batch.items() if k not in {"sample_id", "label"}})
                preds.append(out.cpu().numpy())

    stack = np.stack(preds)  # (5, n)
    mean = stack.mean(axis=0)
    std = stack.std(axis=0)

    result = pd.DataFrame({
        "sample_id": [s.sample_id for s in dataset.samples],
        "sirna": cand_df["sirna"].tolist(),
        "sirna_full": cand_df["sirna_full"].tolist(),
        "ensemble_mean": mean,
        "ensemble_std": std,
        "n_models": len(CKPTS),
    })
    result = result.sort_values("ensemble_std", ascending=False).reset_index(drop=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_CSV, index=False)

    print(f"\n=== Deep Ensemble Uncertainty (5 models, ckpt_seq_s0..s4) ===")
    print("Ranked by uncertainty (std) - highest uncertainty = most informative to test first")
    print(f"{'sample_id':<22}{'mean':>8}{'std':>8}   sirna19")
    print("-" * 60)
    for _, r in result.iterrows():
        print(f"{r['sample_id']:<22}{r['ensemble_mean']:>8.4f}{r['ensemble_std']:>8.4f}   {r['sirna']}")
    print(f"\nSaved to {OUT_CSV}")


if __name__ == "__main__":
    main()
