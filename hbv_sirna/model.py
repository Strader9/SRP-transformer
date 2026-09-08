from __future__ import annotations

import torch
from torch import nn


class SequenceEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, nhead: int, layers: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=6)
        block = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(block, layers)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        padding = tokens.eq(6)
        all_padding = padding.all(dim=-1)
        # Fully-padded rows (e.g. unused genotype slots) would make self-attention
        # softmax over all -inf -> NaN.  Give them one valid token so the encoder
        # runs normally; their pooled output is zeroed by the original padding mask.
        safe_tokens = tokens.clone()
        safe_tokens[all_padding, 0] = 0  # 'A' token id
        safe_padding = safe_tokens.eq(6)
        encoded = self.encoder(self.embedding(safe_tokens), src_key_padding_mask=safe_padding)
        valid = (~padding).unsqueeze(-1)
        pooled = (encoded * valid).sum(dim=-2) / valid.sum(dim=-2).clamp_min(1)
        pooled[all_padding] = 0.0
        return pooled


class HBVMultiGenotypeModel(nn.Module):
    """Two-level Transformer: bases within a target, then targets across genotypes."""

    def __init__(
        self,
        hbv_feature_dim: int = 0,
        vocab_size: int = 7,
        d_model: int = 128,
        nhead: int = 4,
        sequence_layers: int = 2,
        genotype_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.sequence_encoder = SequenceEncoder(vocab_size, d_model, nhead, sequence_layers, dropout)
        genotype_block = nn.TransformerEncoderLayer(d_model, nhead, d_model * 2, dropout, batch_first=True)
        self.genotype_encoder = nn.TransformerEncoder(genotype_block, genotype_layers)
        self.conservation_projection = nn.Linear(1, d_model)
        fusion_dim = d_model * 3 + hbv_feature_dim
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 1), nn.Sigmoid()
        )

    def forward(
        self,
        sirna_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        genotype_mask: torch.Tensor,
        conservation: torch.Tensor,
        hbv_features: torch.Tensor,
    ) -> torch.Tensor:
        batch, genotypes, length = target_tokens.shape
        sirna_repr = self.sequence_encoder(sirna_tokens)
        target_repr = self.sequence_encoder(target_tokens.reshape(batch * genotypes, length)).reshape(batch, genotypes, -1)
        genotype_repr = self.genotype_encoder(target_repr, src_key_padding_mask=~genotype_mask)
        weights = genotype_mask.unsqueeze(-1)
        pooled_genotypes = (genotype_repr * weights).sum(1) / weights.sum(1).clamp_min(1)
        conservation_repr = self.conservation_projection(conservation.mean(1, keepdim=True))
        fused = torch.cat([sirna_repr, pooled_genotypes, conservation_repr, hbv_features], dim=-1)
        return self.head(fused).squeeze(-1)

