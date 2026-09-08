import torch

from hbv_sirna.model import HBVMultiGenotypeModel


def test_model_output_shape_and_range():
    model = HBVMultiGenotypeModel(hbv_feature_dim=3, d_model=32, nhead=4, sequence_layers=1, genotype_layers=1)
    output = model(
        sirna_tokens=torch.randint(0, 6, (2, 19)),
        target_tokens=torch.randint(0, 6, (2, 4, 19)),
        genotype_mask=torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool),
        conservation=torch.rand(2, 19),
        hbv_features=torch.rand(2, 3),
    )
    assert output.shape == (2,)
    assert torch.all((output >= 0) & (output <= 1))


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

