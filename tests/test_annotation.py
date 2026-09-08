"""Tests for hbv_sirna.annotation, prepare integration, and train locus split."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hbv_sirna.annotation import (
    Region,
    build_msa_to_ref_map,
    compute_hbv_features,
    gc_content,
    msa_window_to_ref_interval,
    normalise_region_name,
    overlapping_classes,
    parse_bed,
)
from hbv_sirna.prepare import build_candidate_table
from hbv_sirna.train import _locus_group, _split_indices

DATA = Path(__file__).parent / "data"


# ---------------------------------------------------------------------------
# annotation unit tests
# ---------------------------------------------------------------------------

def test_normalise_region_name_aliases():
    assert normalise_region_name("PreS1") == "S"
    assert normalise_region_name("pol") == "P"
    assert normalise_region_name("Core") == "C"
    assert normalise_region_name("HBx") == "X"
    assert normalise_region_name("unknown_orfig") is None


def test_parse_bed(tmp_path):
    bed = tmp_path / "r.bed"
    bed.write_text("chr\t0\t20\tS\nchr\t15\t45\tP\n# comment\n", encoding="utf-8")
    regions = parse_bed(bed)
    assert regions == [Region("S", 0, 20), Region("P", 15, 45)]


def test_parse_bed_rejects_bad_line(tmp_path):
    bed = tmp_path / "bad.bed"
    bed.write_text("chr\t0\t20\n", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_bed(bed)


def test_build_msa_to_ref_map_with_gaps():
    m = build_msa_to_ref_map("A-C-T")
    np.testing.assert_array_equal(m, [0, -1, 1, -1, 2])


def test_msa_window_to_ref_interval():
    m = build_msa_to_ref_map("A-C-T")
    assert msa_window_to_ref_interval(0, 5, m) == (0, 3)
    assert msa_window_to_ref_interval(1, 2, m) is None  # all gaps


def test_overlapping_classes_single_and_overlap():
    regions = [Region("S", 0, 20), Region("P", 15, 45)]
    assert overlapping_classes(regions, 0, 10) == {"S"}
    assert overlapping_classes(regions, 16, 19) == {"S", "P"}
    assert overlapping_classes(regions, 30, 40) == {"P"}


def test_gc_content():
    assert gc_content("ACGT") == 0.5
    assert gc_content("AAAA") == 0.0
    assert gc_content("GGCC") == 1.0
    assert gc_content("A-T-N") == 0.0  # gaps and N ignored, no G/C bases


def test_compute_hbv_features_seven_dim():
    regions = [Region("S", 0, 20), Region("P", 15, 45)]
    conservation = np.array([1.0, 0.5, 1.0, 1.0])
    feat = compute_hbv_features("ACGT", regions, 16, 20, conservation)
    assert len(feat) == 7
    s, p, c, x, overlap, gc, avg_cons = feat
    assert s == 1.0 and p == 1.0  # window 16-20 overlaps S and P
    assert c == 0.0 and x == 0.0
    assert overlap == 1.0
    assert gc == pytest.approx(0.5)
    assert avg_cons == pytest.approx(0.875)


# ---------------------------------------------------------------------------
# prepare integration test
# ---------------------------------------------------------------------------

def test_prepare_without_annotation_yields_empty_features(tmp_path):
    out = tmp_path / "candidates.csv"
    frame = build_candidate_table(DATA / "mini_msa.fasta", out, "ref_A", window=19, stride=10)
    assert len(frame) > 0
    for feat_str in frame["hbv_features"]:
        assert json.loads(feat_str) == []


def test_prepare_with_annotation_yields_seven_dim_features(tmp_path):
    out = tmp_path / "candidates.csv"
    frame = build_candidate_table(
        DATA / "mini_msa.fasta",
        out,
        "ref_A",
        window=19,
        stride=10,
        annotation_bed=DATA / "mini_regions.bed",
    )
    assert len(frame) > 0
    for feat_str in frame["hbv_features"]:
        feat = json.loads(feat_str)
        assert len(feat) == 7
        # first four are binary S/P/C/X flags
        assert all(v in (0.0, 1.0) for v in feat[:4])
        assert feat[4] in (0.0, 1.0)  # overlap
        assert 0.0 <= feat[5] <= 1.0  # gc
        assert 0.0 <= feat[6] <= 1.0  # mean conservation


def test_prepare_overlap_window_flagged(tmp_path):
    out = tmp_path / "candidates.csv"
    frame = build_candidate_table(
        DATA / "mini_msa.fasta",
        out,
        "ref_A",
        window=19,
        stride=1,
        annotation_bed=DATA / "mini_regions.bed",
    )
    # window starting at 16 (1-based 17) covers ref positions 16-34 -> S(0-20)+P(15-45)
    overlap_row = frame[frame["start_1based"] == 17]
    assert len(overlap_row) == 1
    feat = json.loads(overlap_row.iloc[0]["hbv_features"])
    assert feat[0] == 1.0  # S
    assert feat[1] == 1.0  # P
    assert feat[4] == 1.0  # overlap


# ---------------------------------------------------------------------------
# train locus split test
# ---------------------------------------------------------------------------

def test_locus_group_extracts_start():
    assert _locus_group("ref_A:100-118") == "100"
    assert _locus_group("custom_id_no_colon") == "custom_id_no_colon"


def test_locus_split_keeps_same_locus_together():
    # Build a tiny in-memory dataset-like object with samples sharing loci
    class _S:
        def __init__(self, sid):
            self.sample_id = sid

    class _DS:
        def __init__(self, samples):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

    # 4 samples: locus 10 (x2), locus 20 (x1), locus 30 (x1)
    ds = _DS([_S("r:10-28"), _S("r:10-28"), _S("r:20-38"), _S("r:30-48")])
    train_idx, valid_idx = _split_indices(ds, seed=42, split_by="locus", val_ratio=0.5)
    # The two locus-10 samples must be in the same split
    locus10_indices = {0, 1}
    assert locus10_indices.issubset(set(train_idx)) or locus10_indices.issubset(set(valid_idx))
    assert len(set(train_idx) & set(valid_idx)) == 0


def test_random_split_does_not_group():
    class _S:
        def __init__(self, sid):
            self.sample_id = sid

    class _DS:
        def __init__(self, samples):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

    ds = _DS([_S(f"r:{i}-{i+18}") for i in range(20)])
    train_idx, valid_idx = _split_indices(ds, seed=42, split_by="random", val_ratio=0.2)
    assert len(train_idx) == 16
    assert len(valid_idx) == 4
