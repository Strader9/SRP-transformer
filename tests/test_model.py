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

