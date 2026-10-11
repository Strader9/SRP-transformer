import numpy as np

from hbv_sirna.sequence import (
    align_sirna_target,
    conservation_profile,
    genotype_coverage,
    reverse_complement,
    tokenize,
)


def test_reverse_complement():
    assert reverse_complement("ACGTT") == "AACGT"


def test_conservation_profile():
    values = conservation_profile(["ACGT", "ACGT", "ATGT"])
    np.testing.assert_allclose(values, [1.0, 2 / 3, 1.0, 1.0])


def test_tokenize_pads_to_length():
    assert tokenize("AC", 4).shape == (4,)
    assert tokenize("AC", 4)[-1] == 6


def test_tokenize_center_crops_long_guides():
    assert tokenize("AACCGG", 4).tolist() == [0, 1, 1, 2]


def test_genotype_coverage_uses_complementarity_threshold():
    assert genotype_coverage("AACCC", ["GGGTT", "GGGAT"], identity_threshold=0.8) == 1.0
    assert genotype_coverage("AACCC", ["GGGTT", "GGGAA"], identity_threshold=0.8) == 0.5


def test_genotype_coverage_aligns_21mer_guide_to_19mer_target_center():
    guide = "T" + "A" * 19 + "T"
    sirna = reverse_complement(guide)
    assert genotype_coverage(sirna, ["A" * 19]) == 1.0


def test_raw_guide_and_target_overhangs_are_aligned_before_trimming():
    guide = "ACGTTGCAACGATCGTACG"
    target = reverse_complement(guide)
    guide_window, target_window, identity = align_sirna_target(
        "T" + guide + "AA", "G" + target + "CC"
    )
    assert guide_window == guide
    assert target_window == target
    assert identity == 1.0

