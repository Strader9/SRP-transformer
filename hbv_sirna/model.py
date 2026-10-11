from __future__ import annotations

import torch
from torch import nn


class SequenceEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, nhead: int, layers: int, dropout: float,
                 positional_encoding: bool = True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=6)
        self.positional_encoding = positional_encoding
        self.d_model = d_model
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
        embedded = self.embedding(safe_tokens)
        if self.positional_encoding:
            position = torch.arange(tokens.shape[1], device=tokens.device, dtype=embedded.dtype).unsqueeze(1)
            scale = torch.exp(
                torch.arange(0, self.d_model, 2, device=tokens.device, dtype=embedded.dtype)
                * (-torch.log(torch.tensor(10000.0, device=tokens.device, dtype=embedded.dtype)) / self.d_model)
            )
            positional = torch.zeros(tokens.shape[1], self.d_model, device=tokens.device, dtype=embedded.dtype)
            positional[:, 0::2] = torch.sin(position * scale)
            positional[:, 1::2] = torch.cos(position * scale[: positional[:, 1::2].shape[1]])
            embedded = embedded + positional.unsqueeze(0)
        encoded = self.encoder(embedded, src_key_padding_mask=safe_padding)
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
        num_tasks: int = 1,
        fusion: str = "concat",
        positional_encoding: bool = True,
        sequence_length: int = 19,
    ):
        super().__init__()
        if d_model <= 0 or nhead <= 0 or d_model % nhead:
            raise ValueError("d_model must be positive and divisible by nhead")
        if sequence_length <= 0 or num_tasks <= 0:
            raise ValueError("sequence_length and num_tasks must be positive")
        self.sequence_encoder = SequenceEncoder(
            vocab_size, d_model, nhead, sequence_layers, dropout, positional_encoding=positional_encoding
        )
        self.positional_encoding = positional_encoding
        self.sequence_length = sequence_length
        genotype_block = nn.TransformerEncoderLayer(d_model, nhead, d_model * 2, dropout, batch_first=True)
        self.genotype_encoder = nn.TransformerEncoder(genotype_block, genotype_layers)
        self.conservation_projection = nn.Linear(1, d_model)
        self.conservation_profile_projection = nn.Sequential(
            nn.Linear(sequence_length, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        # Keep old checkpoints' predictions unchanged until this new
        # position-aware conservation pathway is learned during fine-tuning.
        nn.init.zeros_(self.conservation_profile_projection[-1].weight)
        nn.init.zeros_(self.conservation_profile_projection[-1].bias)
        if fusion not in {"concat", "gated"}:
            raise ValueError("fusion must be 'concat' or 'gated'")
        self.fusion = fusion
        self.hbv_feature_dim = hbv_feature_dim
        self.num_tasks = num_tasks
        if fusion == "gated":
            self.hbv_projection = nn.Linear(max(hbv_feature_dim, 1), d_model)
            self.modality_projection = nn.ModuleList(nn.Linear(d_model, d_model) for _ in range(4))
            self.modality_gate = nn.Sequential(
                nn.Linear(d_model * 4, d_model), nn.GELU(), nn.Linear(d_model, 4)
            )
            fusion_dim = d_model
        else:
            fusion_dim = d_model * 3 + hbv_feature_dim
        self.head = nn.Sequential(
            # Inhibition labels can legitimately be below zero (increased
            # expression relative to control), so do not constrain regression
            # predictions to [0, 1].
            nn.Linear(fusion_dim, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, num_tasks)
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
        all_missing = ~genotype_mask.any(dim=1)
        safe_mask = genotype_mask.clone()
        safe_targets = target_repr.clone()
        if all_missing.any():
            safe_mask[all_missing, 0] = True
            safe_targets[all_missing, 0] = 0.0
        genotype_repr = self.genotype_encoder(safe_targets, src_key_padding_mask=~safe_mask)
        weights = genotype_mask.unsqueeze(-1)
        pooled_genotypes = (genotype_repr * weights).sum(1) / weights.sum(1).clamp_min(1)
        if conservation.shape[1] != self.sequence_length:
            raise ValueError(
                f"Expected conservation length {self.sequence_length}, got {conservation.shape[1]}"
            )
        conservation_repr = (
            self.conservation_projection(conservation.mean(1, keepdim=True))
            + self.conservation_profile_projection(conservation)
        )
        if self.fusion == "gated":
            if hbv_features.shape[-1] == 0:
                hbv_input = hbv_features.new_zeros((batch, 1))
            else:
                hbv_input = hbv_features
            values = [sirna_repr, pooled_genotypes, conservation_repr, self.hbv_projection(hbv_input)]
            modalities = torch.stack([
                projection(value) for projection, value in zip(
                    self.modality_projection,
                    values,
                )
            ], dim=1)
            gate = torch.softmax(self.modality_gate(modalities.flatten(1)), dim=-1)
            fused = (modalities * gate.unsqueeze(-1)).sum(dim=1)
        else:
            fused = torch.cat([sirna_repr, pooled_genotypes, conservation_repr, hbv_features], dim=-1)
        output = self.head(fused)
        return output.squeeze(-1) if self.num_tasks == 1 else output


def load_compatible_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    """Load current or pre-position-aware checkpoints with strict validation."""
    result = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {
        "conservation_profile_projection.0.weight",
        "conservation_profile_projection.0.bias",
        "conservation_profile_projection.2.weight",
        "conservation_profile_projection.2.bias",
    }
    missing = set(result.missing_keys)
    unexpected = set(result.unexpected_keys)
    profile_missing = missing & allowed_missing
    if unexpected or missing - allowed_missing or (profile_missing and profile_missing != allowed_missing):
        raise RuntimeError(
            f"Checkpoint/model mismatch; missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )


def apply_checkpoint_output(values: torch.Tensor, checkpoint: dict) -> torch.Tensor:
    """Restore the output transform used when a checkpoint was trained.

    Checkpoints predating raw-scale regression had a sigmoid layer at the end
    of the head. That layer has no state-dict entries, so metadata is needed to
    preserve their inference behavior.
    """
    activation = checkpoint.get("output_activation", "sigmoid")
    if activation == "sigmoid":
        return torch.sigmoid(values)
    if activation == "identity":
        return values
    raise ValueError(f"Unsupported checkpoint output_activation: {activation!r}")

