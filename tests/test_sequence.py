import numpy as np

from hbv_sirna.sequence import conservation_profile, reverse_complement, tokenize


def test_reverse_complement():
    assert reverse_complement("ACGTT") == "AACGT"


def test_conservation_profile():
    values = conservation_profile(["ACGT", "ACGT", "ATGT"])
    np.testing.assert_allclose(values, [1.0, 2 / 3, 1.0, 1.0])


def test_tokenize_pads_to_length():
    assert tokenize("AC", 4).shape == (4,)
    assert tokenize("AC", 4)[-1] == 6

