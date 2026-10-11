from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

BASES = "ACGTUN-"
TOKEN_TO_ID = {base: i for i, base in enumerate(BASES)}
COMPLEMENT = str.maketrans("ACGTUN", "TGCAAN")


def normalize_sequence(sequence: str) -> str:
    seq = sequence.upper().replace(" ", "").replace("\n", "")
    seq = seq.replace("U", "T")
    invalid = set(seq) - set("ACGTN-")
    if invalid:
        raise ValueError(f"Unsupported bases: {sorted(invalid)}")
    return seq


def reverse_complement(sequence: str) -> str:
    return normalize_sequence(sequence).translate(COMPLEMENT)[::-1]


def align_sirna_target(sirna: str, target: str, length: int = 19) -> tuple[str, str, float]:
    """Select the most complementary aligned guide/target windows.

    Raw records can include overhangs and guide/target strings of different
    lengths. Selecting the first 19 bases independently can destroy their
    pairing. This searches all 19-mer windows and uses deterministic central
    tie-breaking. Returned guide orientation is unchanged.
    """
    if length < 1:
        raise ValueError("length must be positive")
    guide = normalize_sequence(sirna)
    target_seq = normalize_sequence(target)
    if len(guide) < length or len(target_seq) < length:
        raise ValueError(f"Both sequences must contain at least {length} bases")
    guide_windows = [(i, guide[i : i + length]) for i in range(len(guide) - length + 1)]
    target_windows = [(i, target_seq[i : i + length]) for i in range(len(target_seq) - length + 1)]
    center_guide = (len(guide) - length) / 2
    center_target = (len(target_seq) - length) / 2
    best = None
    for guide_start, guide_window in guide_windows:
        paired_guide = reverse_complement(guide_window)
        for target_start, target_window in target_windows:
            pairs = [(a, b) for a, b in zip(paired_guide, target_window)
                     if a in "ACGT" and b in "ACGT"]
            if not pairs:
                continue
            identity = sum(a == b for a, b in pairs) / len(pairs)
            center_penalty = abs(guide_start - center_guide) + abs(target_start - center_target)
            score = (identity, len(pairs), -center_penalty, -guide_start, -target_start)
            if best is None or score > best[0]:
                best = (score, guide_window, target_window)
    if best is None:
        raise ValueError("No unambiguous bases are available for guide-target alignment")
    return best[1], best[2], float(best[0][0])


def genotype_coverage(sirna: str, target_sequences: list[str], identity_threshold: float = 0.9) -> float:
    """Fraction of supplied genotype targets complementary at the threshold.

    This is a deterministic sequence-coverage statistic, not an efficacy
    prediction. Ambiguous and gap positions are excluded from the identity.
    """
    if not 0.0 <= identity_threshold <= 1.0:
        raise ValueError("identity_threshold must be between 0 and 1")
    guide = reverse_complement(sirna)
    covered = 0
    observed = 0
    for raw_target in target_sequences:
        target = normalize_sequence(raw_target)
        if len(guide) > len(target):
            left = (len(guide) - len(target)) // 2
            aligned_guide = guide[left : left + len(target)]
        elif len(target) > len(guide):
            left = (len(target) - len(guide)) // 2
            target = target[left : left + len(guide)]
            aligned_guide = guide
        else:
            aligned_guide = guide
        pairs = [(a, b) for a, b in zip(aligned_guide, target) if a in "ACGT" and b in "ACGT"]
        if not pairs:
            continue
        observed += 1
        identity = sum(a == b for a, b in pairs) / len(pairs)
        covered += identity >= identity_threshold
    return 0.0 if observed == 0 else covered / observed


def tokenize(sequence: str, length: int | None = None) -> np.ndarray:
    seq = normalize_sequence(sequence)
    if length is not None:
        if len(seq) > length:
            left = (len(seq) - length) // 2
            seq = seq[left : left + length]
        seq = seq[:length].ljust(length, "-")
    return np.asarray([TOKEN_TO_ID.get(base, TOKEN_TO_ID["N"]) for base in seq], dtype=np.int64)


def conservation_profile(sequences: list[str]) -> np.ndarray:
    """Return the non-gap majority-base frequency at each aligned position."""
    if not sequences:
        raise ValueError("At least one aligned sequence is required")
    normalized = [normalize_sequence(seq) for seq in sequences]
    lengths = {len(seq) for seq in normalized}
    if len(lengths) != 1:
        raise ValueError("Sequences must already be aligned and have equal length")
    values = []
    for column in zip(*normalized):
        observed = [base for base in column if base not in {"-", "N"}]
        values.append(0.0 if not observed else Counter(observed).most_common(1)[0][1] / len(observed))
    return np.asarray(values, dtype=np.float32)


def read_fasta(path: str | Path) -> dict[str, str]:
    records: dict[str, list[str]] = {}
    current: str | None = None
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            current = line[1:].split()[0]
            if not current or current in records:
                raise ValueError(f"Invalid or duplicate FASTA id: {current!r}")
            records[current] = []
        elif current is None:
            raise ValueError("FASTA sequence appears before the first header")
        else:
            records[current].append(line)
    return {name: normalize_sequence("".join(parts)) for name, parts in records.items()}

