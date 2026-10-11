"""A2a: Download AM282986 (HBV genotype A) GenBank record, extract CDS
coordinates for S/P/C/X ORFs, generate BED file for annotation.

The BED file uses 0-based half-open intervals on the gap-free reference
genome (AM282986, which is already EcoRI-linearised — verified by exact
match of literature target positions).

Circular-genome ORFs that wrap around the origin are split into two BED
records, as required by annotation.parse_bed.
"""

import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from Bio import SeqIO

ACCESSION = "AM282986"
OUT_BED = REPO / "data" / "raw" / "hbv_regions_genotypeA.bed"
OUT_GB = REPO / "data" / "raw" / "genomes" / f"{ACCESSION}.gb"

# CDS product/name keywords -> canonical S/P/C/X class
CLASS_KEYWORDS = {
    "S": ["surface", "hbsag", "s protein", "pre-s", "pres"],
    "P": ["polymerase", "reverse transcriptase", "p protein", "dna polymerase", "pol "],
    "C": ["core", "hbcag", "c protein", "nucleocapsid", "precore", "hbeag"],
    "X": ["x protein", "hbx", "transactivator", "x antigen"],
}


def classify_cds(feature) -> str | None:
    """Map a GenBank CDS feature to canonical S/P/C/X, or None."""
    qualifiers = feature.qualifiers
    product = qualifiers.get("product", [""])[0].lower()
    gene = qualifiers.get("gene", [""])[0].lower()
    note = qualifiers.get("note", [""])[0].lower()
    text = f"{product} {gene} {note}"

    for cls, keywords in CLASS_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in text:
                return cls
    # Fallback: gene name single letter
    if gene in {"s", "p", "c", "x"}:
        return gene.upper()
    return None


def extract_intervals(feature, genome_length: int) -> list[tuple[int, int]]:
    """Extract 0-based half-open intervals from a CDS feature.

    Handles simple locations and join() (circular genome wrap-around).
    GenBank coordinates are 1-based inclusive -> BED: start-1, end.
    """
    loc = feature.location
    intervals = []

    if loc.strand == -1:
        # HBV CDS are all on the positive strand in GenBank; if we hit a
        # reverse-strand CDS, skip it (the complementary strand is not
        # protein-coding for HBV).
        return []

    parts = loc.parts if hasattr(loc, "parts") else [loc]
    for part in parts:
        start = int(part.start)  # already 0-based in BioPython
        end = int(part.end)      # already 0-based half-open in BioPython
        if end > genome_length:
            end = genome_length
        if start < end:
            intervals.append((start, end))

    return intervals


def main() -> None:
    # Download GenBank
    url = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        f"?db=nucleotide&id={ACCESSION}&rettype=gbwithparts&retmode=text"
    )
    print(f"Downloading {ACCESSION} GenBank...")
    with urllib.request.urlopen(url, timeout=30) as resp:
        gb_text = resp.read().decode("utf-8")
    OUT_GB.parent.mkdir(parents=True, exist_ok=True)
    OUT_GB.write_text(gb_text, encoding="utf-8")
    print(f"Saved to {OUT_GB} ({len(gb_text)} bytes)")

    # Parse
    record = SeqIO.read(OUT_GB, "genbank")
    genome_length = len(record.seq)
    print(f"Genome length: {genome_length} bp")

    bed_lines = []
    seen_classes = set()

    for feature in record.features:
        if feature.type != "CDS":
            continue
        cls = classify_cds(feature)
        if cls is None:
            product = feature.qualifiers.get("product", ["?"])[0]
            print(f"  SKIP unclassified CDS: product={product!r}")
            continue

        intervals = extract_intervals(feature, genome_length)
        for start, end in intervals:
            bed_lines.append(f"chrHBV_A\t{start}\t{end}\t{cls}")
            seen_classes.add(cls)
            product = feature.qualifiers.get("product", [""])[0]
            print(f"  {cls}: [{start}, {end}) ({end-start} bp) — {product}")

    # Write BED
    OUT_BED.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# HBV genotype A (AM282986) functional regions, 0-based half-open\n"
        "# Generated from GenBank CDS features; circular-genome ORFs split\n"
        f"# Genome length: {genome_length} bp\n"
    )
    OUT_BED.write_text(header + "\n".join(bed_lines) + "\n", encoding="utf-8")
    print(f"\nWrote BED with {len(bed_lines)} intervals to {OUT_BED}")
    print(f"Classes found: {sorted(seen_classes)}")

    if len(seen_classes) < 4:
        missing = {"S", "P", "C", "X"} - seen_classes
        print(f"WARNING: missing classes: {missing}")


if __name__ == "__main__":
    main()
