"""Phase 3 Direction 2: Genotype dropout + consistency regularization search"""
import sys, json, argparse, numpy as np
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from hbv_sirna.train import train_model

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--allow-inferred-guides", action="store_true",
                    help="Allow proxy guides inferred from target regions; exploratory only")
args = parser.parse_args()
if not args.allow_inferred_guides:
    parser.error("current HBV records include inferred guides; pass --allow-inferred-guides for exploratory analysis")

DATA = REPO / 'data' / 'processed' / 'candidates_full_features.csv'
OUT = REPO / 'results'

LR = 5e-4
BS = 8
EPOCHS = 25

# Configs: (genotype_dropout, consistency_weight, tag)
CONFIGS = [
    (0.0,   0.0, 'baseline'),        # baseline
    (0.25,  0.1, 'gd025_cw01'),
    (0.25,  0.5, 'gd025_cw05'),
    (0.5,   0.1, 'gd05_cw01'),
    (0.5,   0.5, 'gd05_cw05'),
]
N_SEEDS = 3

results = {}
for gd, cw, tag in CONFIGS:
    seed_mses = []
    for seed in range(N_SEEDS):
        ckpt_dir = OUT / f'ckpt_p3_{tag}_s{seed}'
        m = train_model(str(DATA), str(ckpt_dir),
                        epochs=EPOCHS, batch_size=BS,
                        learning_rate=LR, weight_decay=1e-4,
                        seed=seed, split_by='locus',
                        genotype_dropout=gd, consistency_weight=cw,
                        allow_inferred_guides=args.allow_inferred_guides)
        seed_mses.append(m['best_valid_mse'])
        print(f'  {tag} seed={seed}: MSE={m["best_valid_mse"]:.4f}')
    results[tag] = {
        'genotype_dropout': gd, 'consistency_weight': cw,
        'mean': float(np.mean(seed_mses)),
        'std': float(np.std(seed_mses)),
        'all': [float(x) for x in seed_mses]
    }

out_path = OUT / 'phase3_direction2_search.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)

print('\n=== Direction 2: Genotype Dropout + Consistency (locus split, lr=5e-4, bs=8) ===')
for k, v in results.items():
    print(f'{k:14s} (gd={v["genotype_dropout"]}, cw={v["consistency_weight"]}): MSE={v["mean"]:.4f} +/- {v["std"]:.4f}')
best = min(results.items(), key=lambda x: x[1]['mean'])
print(f'\nBest: {best[0]} = {best[1]["mean"]:.4f} +/- {best[1]["std"]:.4f}')
