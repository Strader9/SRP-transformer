"""Phase 3 Direction 2: Genotype dropout + consistency regularization search"""
import sys, json, numpy as np
sys.path.insert(0, r'C:\Users\34359\Documents\SRP\SRP-transformer-HBV-v0.1')
from hbv_sirna.train import train_model

DATA = r'C:\Users\34359\Documents\SRP\SRP-transformer-HBV-v0.1\data\processed\candidates_full_features.csv'
OUT = r'C:\Users\34359\Documents\SRP\SRP-transformer-HBV-v0.1\results'

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
        ckpt_dir = f'{OUT}/ckpt_p3_{tag}_s{seed}'
        m = train_model(DATA, ckpt_dir,
                        epochs=EPOCHS, batch_size=BS,
                        learning_rate=LR, weight_decay=1e-4,
                        seed=seed, split_by='locus',
                        genotype_dropout=gd, consistency_weight=cw)
        seed_mses.append(m['best_valid_mse'])
        print(f'  {tag} seed={seed}: MSE={m["best_valid_mse"]:.4f}')
    results[tag] = {
        'genotype_dropout': gd, 'consistency_weight': cw,
        'mean': float(np.mean(seed_mses)),
        'std': float(np.std(seed_mses)),
        'all': [float(x) for x in seed_mses]
    }

out_path = f'{OUT}/phase3_direction2_search.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)

print('\n=== Direction 2: Genotype Dropout + Consistency (locus split, lr=5e-4, bs=8) ===')
for k, v in results.items():
    print(f'{k:14s} (gd={v["genotype_dropout"]}, cw={v["consistency_weight"]}): MSE={v["mean"]:.4f} +/- {v["std"]:.4f}')
best = min(results.items(), key=lambda x: x[1]['mean'])
print(f'\nBest: {best[0]} = {best[1]["mean"]:.4f} +/- {best[1]["std"]:.4f}')
