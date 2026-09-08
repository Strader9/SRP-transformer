"""HBV functional-region annotation and 7-dim feature computation.

Coordinate convention
---------------------
BED files use 0-based half-open intervals on the *gap-free* reference genome.
``prepare.py`` slices candidates from a gapped MSA, so its ``start``/``end``
are MSA coordinates.  Use :func:`build_msa_to_ref_map` to translate an MSA
window into reference coordinates before querying a BED file.

HBV is a circular genome (~3.2 kb).  A region that wraps around the origin must
be split into two BED records, e.g. ``chr 2800 3182 P`` and ``chr 0 1623 P``.

The 7-dim ``hbv_features`` vector (fixed order, per README):
    [S, P, C, X, is_overlap, gc_content, avg_conservation]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

REGION_CLASSES = ("S", "P", "C", "X")

# Free-form BED/GFF names -> canonical S/P/C/X class (lower-case keys).
_ALIASES: dict[str, str] = {
    "s": "S", "surface": "S", "pres1": "S", "pres2": "S", "pre-s1": "S",
    "pre-s2": "S", "l": "S", "m": "S", "hbs": "S", "hbsag": "S",
    "p": "P", "pol": "P", "polymerase": "P", "rt": "P", "reverse transcriptase": "P",
    "rh": "P", "rnaseh": "P", "tp": "P", "terminal protein": "P", "spacer": "P",
    "c": "C", "core": "C", "prec": "C", "pre-c": "C", "precore": "C",
    "hbc": "C", "hbcag": "C", "hbv core": "C",
    "x": "X", "hbx": "X", "x-protein": "X", "x protein": "X", "transactivator": "X",
}


@dataclass(frozen=True)
class Region:
    """A single BED interval, 0-based half-open, on the gap-free reference."""

    name: str
    start: int
    end: int


def normalise_region_name(name: str) -> str | None:
    """Map a free-form BED name to a canonical S/P/C/X class, or None."""
    return _ALIASES.get(name.strip().lower())


def parse_bed(path: str | Path) -> list[Region]:
    """Parse a BED file with at least 4 columns (chrom, start, end, name).

    Comment lines (``#``), ``track`` and ``browser`` lines are skipped.
    Circular-genome wrap-around regions must already be split into two records.
    """
    regions: list[Region] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("track") or line.startswith("browser"):
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            raise ValueError(f"BED line needs >=4 columns (chrom,start,end,name): {raw!r}")
        start, end = int(parts[1]), int(parts[2])
        if end <= start:
            raise ValueError(f"BED interval end<=start: {raw!r}")
        regions.append(Region(name=parts[3].strip(), start=start, end=end))
    return regions


def build_msa_to_ref_map(reference_with_gaps: str) -> np.ndarray:
    """Map MSA position -> gap-free reference position (0-based).

    Gap positions map to ``-1``.  The first non-gap base is reference pos 0.
    """
    ref_pos = 0
    mapping = np.full(len(reference_with_gaps), -1, dtype=np.int64)
    for i, base in enumerate(reference_with_gaps):
        if base == "-":
            continue
        mapping[i] = ref_pos
        ref_pos += 1
    return mapping


def msa_window_to_ref_interval(
    msa_start: int,
    msa_end: int,
    msa_to_ref: np.ndarray,
) -> tuple[int, int] | None:
    """Translate an MSA half-open window to a gap-free reference interval.

    Returns ``(ref_start, ref_end)`` (0-based half-open) or None if the window
    contains no valid reference bases.
    """
    valid = msa_to_ref[msa_start:msa_end]
    valid = valid[valid >= 0]
    if valid.size == 0:
        return None
    return int(valid.min()), int(valid.max()) + 1


def overlapping_classes(regions: list[Region], ref_start: int, ref_end: int) -> set[str]:
    """Return canonical S/P/C/X classes overlapping a reference interval."""
    classes: set[str] = set()
    for region in regions:
        if region.start < ref_end and region.end > ref_start:
            cls = normalise_region_name(region.name)
            if cls is not None:
                classes.add(cls)
    return classes


def gc_content(sequence: str) -> float:
    """Fraction of G/C among A/C/G/T bases (gaps and N ignored)."""
    seq = sequence.upper()
    gc = sum(1 for b in seq if b in "GC")
    total = sum(1 for b in seq if b in "ACGT")
    return 0.0 if total == 0 else gc / total


def compute_hbv_features(
    reference_target: str,
    regions: list[Region],
    ref_start: int,
    ref_end: int,
    conservation: np.ndarray | None = None,
) -> list[float]:
    """Compute the fixed 7-dim HBV feature vector.

    Parameters
    ----------
    reference_target:
        Target sequence on the reference genotype (gaps ignored for GC).
    regions:
        Parsed BED regions on the gap-free reference.
    ref_start, ref_end:
        Reference interval (0-based half-open) for the MSA window.
    conservation:
        Per-position conservation for the window; its mean is the 7th dim.
        Pass None to use 0.0 (the data loader also feeds the full 19-dim
        conservation vector to the model separately).
    """
    classes = overlapping_classes(regions, ref_start, ref_end)
    return [
        1.0 if "S" in classes else 0.0,
        1.0 if "P" in classes else 0.0,
        1.0 if "C" in classes else 0.0,
        1.0 if "X" in classes else 0.0,
        1.0 if len(classes) > 1 else 0.0,
        gc_content(reference_target),
        float(np.mean(conservation)) if conservation is not None else 0.0,
    ]
