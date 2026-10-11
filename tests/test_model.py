import torch

import pytest

from hbv_sirna.model import (
    HBVMultiGenotypeModel,
    SequenceEncoder,
    apply_checkpoint_output,
    load_compatible_state_dict,
)


def test_model_output_shape_and_finite_values():
    model = HBVMultiGenotypeModel(hbv_feature_dim=3, d_model=32, nhead=4, sequence_layers=1, genotype_layers=1)
    output = model(
        sirna_tokens=torch.randint(0, 6, (2, 19)),
        target_tokens=torch.randint(0, 6, (2, 4, 19)),
        genotype_mask=torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool),
        conservation=torch.rand(2, 19),
        hbv_features=torch.rand(2, 3),
    )
    assert output.shape == (2,)
    assert torch.isfinite(output).all()


def test_regression_head_can_predict_negative_inhibition():
    model = HBVMultiGenotypeModel(hbv_feature_dim=0, d_model=32, nhead=4,
                                  sequence_layers=1, genotype_layers=1)
    model.eval()
    with torch.no_grad():
        model.head[-1].weight.zero_()
        model.head[-1].bias.fill_(-0.1)
        output = model(
            sirna_tokens=torch.randint(0, 5, (1, 19)),
            target_tokens=torch.randint(0, 5, (1, 1, 19)),
            genotype_mask=torch.ones((1, 1), dtype=torch.bool),
            conservation=torch.rand(1, 19), hbv_features=torch.zeros(1, 0),
        )
    assert output.item() < 0


def test_model_handles_padded_genotypes_without_nan():
    """Unused genotype slots are filled with token 6 (padding).  The model must
    not produce NaN when some target rows are fully padded."""
    model = HBVMultiGenotypeModel(hbv_feature_dim=0, d_model=32, nhead=4, sequence_layers=1, genotype_layers=1)
    model.eval()
    # 2 samples, max 4 genotypes, but only 1 and 2 are real; rest are all-padding (token 6).
    target_tokens = torch.full((2, 4, 19), 6, dtype=torch.long)
    target_tokens[0, 0] = torch.randint(0, 5, (19,))
    target_tokens[1, 0] = torch.randint(0, 5, (19,))
    target_tokens[1, 1] = torch.randint(0, 5, (19,))
    genotype_mask = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0]], dtype=torch.bool)
    with torch.no_grad():
        output = model(
            sirna_tokens=torch.randint(0, 5, (2, 19)),
            target_tokens=target_tokens,
            genotype_mask=genotype_mask,
            conservation=torch.rand(2, 19),
            hbv_features=torch.zeros(2, 0),
        )
    assert torch.isfinite(output).all()
    assert output.shape == (2,)


def test_model_handles_sample_with_no_observed_genotypes():
    model = HBVMultiGenotypeModel(hbv_feature_dim=0, d_model=32, nhead=4,
                                  sequence_layers=1, genotype_layers=1)
    output = model(
        sirna_tokens=torch.randint(0, 5, (1, 19)),
        target_tokens=torch.full((1, 2, 19), 6, dtype=torch.long),
        genotype_mask=torch.zeros((1, 2), dtype=torch.bool),
        conservation=torch.zeros((1, 19)), hbv_features=torch.zeros((1, 0)),
    )
    assert torch.isfinite(output).all()


def test_sequence_encoder_uses_nucleotide_positions():
    encoder = SequenceEncoder(vocab_size=7, d_model=32, nhead=4, layers=1, dropout=0.0,
                              positional_encoding=True)
    encoder.eval()
    sequence = torch.tensor([[0, 1, 2, 3, 4]])
    permuted = sequence[:, [4, 3, 2, 1, 0]]
    with torch.no_grad():
        first = encoder(sequence)
        second = encoder(permuted)
    assert not torch.allclose(first, second)


def test_model_returns_assay_specific_task_scores():
    model = HBVMultiGenotypeModel(
        hbv_feature_dim=1, num_tasks=3, fusion="gated",
        d_model=32, nhead=4, sequence_layers=1, genotype_layers=1,
    )
    output = model(
        sirna_tokens=torch.randint(0, 5, (2, 19)),
        target_tokens=torch.randint(0, 5, (2, 2, 19)),
        genotype_mask=torch.tensor([[1, 1], [1, 0]], dtype=torch.bool),
        conservation=torch.rand(2, 19), hbv_features=torch.rand(2, 1),
    )
    assert output.shape == (2, 3)


def test_model_uses_position_specific_conservation():
    torch.manual_seed(7)
    model = HBVMultiGenotypeModel(hbv_feature_dim=0, d_model=32, nhead=4,
                                  sequence_layers=1, genotype_layers=1, dropout=0.0)
    model.eval()
    with torch.no_grad():
        model.conservation_profile_projection[-1].weight.normal_()
        inputs = {
            "sirna_tokens": torch.randint(0, 5, (1, 19)),
            "target_tokens": torch.randint(0, 5, (1, 2, 19)),
            "genotype_mask": torch.ones((1, 2), dtype=torch.bool),
            "hbv_features": torch.zeros((1, 0)),
        }
        forward = model(**inputs, conservation=torch.linspace(0, 1, 19).unsqueeze(0))
        reverse = model(**inputs, conservation=torch.linspace(1, 0, 19).unsqueeze(0))
    assert not torch.allclose(forward, reverse)


def test_legacy_checkpoint_loads_with_zero_initialized_conservation_pathway():
    source = HBVMultiGenotypeModel(hbv_feature_dim=0, d_model=32, nhead=4,
                                   sequence_layers=1, genotype_layers=1)
    legacy = {key: value for key, value in source.state_dict().items()
              if not key.startswith("conservation_profile_projection.")}
    restored = HBVMultiGenotypeModel(hbv_feature_dim=0, d_model=32, nhead=4,
                                     sequence_layers=1, genotype_layers=1)
    load_compatible_state_dict(restored, legacy)
    assert torch.count_nonzero(restored.conservation_profile_projection[-1].weight) == 0


def test_partial_conservation_checkpoint_is_rejected():
    model = HBVMultiGenotypeModel(hbv_feature_dim=0, d_model=32, nhead=4,
                                  sequence_layers=1, genotype_layers=1)
    state = model.state_dict()
    del state["conservation_profile_projection.0.weight"]
    with pytest.raises(RuntimeError, match="Checkpoint/model mismatch"):
        load_compatible_state_dict(model, state)


def test_old_checkpoint_keeps_sigmoid_output_and_new_checkpoint_is_raw():
    values = torch.tensor([-1.0, 0.0, 1.0])
    legacy = apply_checkpoint_output(values, {})
    torch.testing.assert_close(legacy, torch.sigmoid(values))
    current = apply_checkpoint_output(values, {"output_activation": "identity"})
    torch.testing.assert_close(current, values)

