"""Download HBV reference genomes for 8 genotypes from NCBI.

Accessions chosen as standard reference strains for each genotype.
All sequences are expected to be ~3.2 kb complete circular genomes.

After download, each sequence is linearised at its EcoRI site (GAATTC)
so that all genotypes share a common coordinate origin (EcoRI-based
numbering, consistent with the literature in hbv_experiments.csv).
"""

import re
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "data" / "raw" / "genomes"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# genotype -> (accession, description)
GENOMES = {
    "A": ("AM282986", "HBV genotype A (EcoRI-based, used in PMC3132485 review)"),
    "B": ("AB033551", "HBV genotype B1 reference"),
    "C": ("AB033554", "HBV genotype C1 reference"),
    "D": ("V01460", "HBV genotype D ayw (used in Ely/Morrissey)"),
    "E": ("X75657", "HBV genotype E reference"),
    "F": ("X69798", "HBV genotype F reference"),
    "G": ("AF160501", "HBV genotype G reference"),
    "H": ("AY090454", "HBV genotype H reference"),
}

ECORI = "GAATTC"


def fetch_fasta(accession: str) -> tuple[str, str]:
    """Fetch a FASTA record from NCBI efetch. Returns (header, sequence)."""
    url = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        f"?db=nucleotide&id={accession}&rettype=fasta&retmode=text"
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                text = resp.read().decode("utf-8")
            lines = text.strip().splitlines()
            header = lines[0].lstrip(">")
            seq = "".join(lines[1:]).upper()
            return header, seq
        except Exception as e:
            print(f"  attempt {attempt+1} failed: {e}", file=sys.stderr)
            time.sleep(2)
    raise RuntimeError(f"Failed to fetch {accession} after 3 attempts")


def linearise_at_ecori(seq: str) -> tuple[str, int]:
    """Rotate a circular genome so it starts at the first EcoRI site.

    Returns (rotated_sequence, ecori_position_in_original).
    If no EcoRI site found, returns sequence unchanged with position -1.
    """
    pos = seq.find(ECORI)
    if pos == -1:
        return seq, -1
    return seq[pos:] + seq[:pos], pos


def main() -> None:
    records = []
    for genotype, (accession, desc) in GENOMES.items():
        print(f"Downloading genotype {genotype} ({accession})...")
        header, seq = fetch_fasta(accession)
        print(f"  length: {len(seq)} bp")

        # Validate: HBV complete genome should be ~3100-3300 bp
        if not (3000 <= len(seq) <= 3400):
            print(f"  WARNING: unexpected length {len(seq)} for {accession}", file=sys.stderr)

        rotated, ecori_pos = linearise_at_ecori(seq)
        if ecori_pos >= 0:
            print(f"  EcoRI site at original pos {ecori_pos}, rotated to start")
        else:
            print(f"  WARNING: no EcoRI site found, using original orientation", file=sys.stderr)

        # Save individual FASTA
        fasta_path = OUT_DIR / f"hbv_genotype_{genotype}.fasta"
        fasta_path.write_text(
            f">HBV_{genotype}|{accession}|{desc}\n{rotated}\n",
            encoding="utf-8",
        )
        records.append((genotype, accession, rotated))

        # NCBI rate limit: 3 requests/sec without API key
        time.sleep(0.5)

    # Write combined FASTA for MAFFT
    combined = OUT_DIR / "hbv_genotypes_A-H_ecori.fasta"
    with open(combined, "w", encoding="utf-8") as f:
        for genotype, accession, seq in records:
            f.write(f">HBV_{genotype}|{accession}\n{seq}\n")
    print(f"\nWrote {len(records)} genomes to {combined}")
    print(f"Individual files in {OUT_DIR}")


if __name__ == "__main__":
    main()
