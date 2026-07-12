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


def tokenize(sequence: str, length: int | None = None) -> np.ndarray:
    seq = normalize_sequence(sequence)
    if length is not None:
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

