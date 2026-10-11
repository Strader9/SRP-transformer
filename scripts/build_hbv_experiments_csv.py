"""Build hbv_experiments.csv from manually extracted literature data.

Sources:
- PMC3132485 (review Table 1): consolidated HBV RNAi targets from Sun et al. [7],
  Wu et al. [10], Panjaworayan et al. [14], Zhang et al. [13], Gao et al. [52],
  Ely et al. [24], Sun et al. [40], Peng et al. [53]
- PMC2770508 (Ely et al. 2009, Mol Ther): 14 rationally designed HBV-miRs + HBVU6#2
- PMC3666629 (Morrissey et al. 2014): 4 lead siRNAs (siHBV-74/75/76/77)

Label convention: inhibition = 1 - residual_fraction; negative values are
preserved when treatment increases the measured signal above its control.
assay_type records what was measured (pgRNA, HBV_DNA, HBsAg, total_RNA, cccDNA).
Rows without a confirmed numeric label have label empty (usable for candidate ranking,
excluded from training by train.py).
"""

import csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "raw" / "hbv_experiments.csv"

FIELDS = [
    "experiment_id", "sirna", "target_sequence", "reference_accession",
    "genotype", "label", "assay_type", "dose_nm", "time_h", "cell_line", "source",
]

rows: list[dict] = []


def add(**kw):
    row = {f: kw.get(f, "") for f in FIELDS}
    rows.append(row)


# ---------------------------------------------------------------------------
# Source 1: PMC3132485 review Table 1
# Reference genome: genotype A (AM282986), EcoRI-based numbering
# ---------------------------------------------------------------------------
REF_A = "AM282986"
REF_D_AYW = "V01460"
SRC_REV = "PMC3132485"

# Group I: conservation-based shRNA, liposome delivery, human liver cell line
# Sun et al. [7]
add(experiment_id="Sun2009_Sh6", sirna="", target_sequence="CTGCTATGCCTCATCTTCT",
    reference_accession=REF_A, genotype="A", label=0.95, assay_type="pgRNA",
    cell_line="Huh7", source=f"{SRC_REV};ref[7];Sun2009")
add(experiment_id="Sun2009_Sh6_DNA", sirna="", target_sequence="CTGCTATGCCTCATCTTCT",
    reference_accession=REF_A, genotype="A", label=0.98, assay_type="HBV_DNA",
    cell_line="Huh7", source=f"{SRC_REV};ref[7];Sun2009")
add(experiment_id="Sun2009_1740SLAS", sirna="", target_sequence="GGTATGTTGCCCGTTTGTC",
    reference_accession=REF_A, genotype="A", label=0.92, assay_type="pgRNA",
    cell_line="Huh7", source=f"{SRC_REV};ref[7];Sun2009")
add(experiment_id="Sun2009_Sh10", sirna="", target_sequence="CTGTTTGGCTTTCAGTTAT",
    reference_accession=REF_A, genotype="A", label=0.95, assay_type="pgRNA",
    cell_line="Huh7", source=f"{SRC_REV};ref[7];Sun2009")
add(experiment_id="Sun2009_DR1", sirna="", target_sequence="TTTCACCTCTGCCTAATCA",
    reference_accession=REF_A, genotype="A", label=0.96, assay_type="HBV_DNA",
    cell_line="Huh7", source=f"{SRC_REV};ref[7];Sun2009")
add(experiment_id="Sun2009_Eps32", sirna="", target_sequence="TTCATGTCCCTACTGTTCAA",
    reference_accession=REF_A, genotype="A", label=0.96, assay_type="HBV_DNA",
    cell_line="Huh7", source=f"{SRC_REV};ref[7];Sun2009")

# Wu et al. [10]
add(experiment_id="Wu2005_Sh1", sirna="", target_sequence="GTCGCAGAAGATCTCAATC",
    reference_accession=REF_A, genotype="A", label=0.86, assay_type="pgRNA",
    cell_line="Huh7", source=f"{SRC_REV};ref[10];Wu2005")
add(experiment_id="Wu2005_S1", sirna="", target_sequence="GGTATGTTGCCCGTTTGTC",
    reference_accession=REF_A, genotype="A", label=0.90, assay_type="total_RNA",
    cell_line="Huh7", source=f"{SRC_REV};ref[10];Wu2005")

# Panjaworayan et al. [14]
add(experiment_id="Panjaworayan2010_PRE1317", sirna="", target_sequence="AAAGCTCATCGGAACTGACAA",
    reference_accession=REF_A, genotype="A", label=0.80, assay_type="cccDNA",
    cell_line="Huh7", source=f"{SRC_REV};ref[14];Panjaworayan2010")

# Zhang et al. [13] — BMC Microbiology 2010, full 40-shRNA screen (PMC2927532)
# Qualitative HBsAg knockdown in two cell lines (Y1021=genotype C1, N10=genotype Ae):
#   +++ = >90%, ++ = 50-90%, + = <50%, - = no significant inhibition
# NOTE: qualitative only — NO numeric label assigned (kept as candidate pool).
#       Mapping +/++/+++ to fractions would be subjective and is intentionally avoided.
_ZHANG_DATA = [
    ("B182",  "GGACCCCTGCTCGTGTTACAG", 182,  "S,P",       "++",  "+"),
    ("B183",  "GACCCCTGCTCGTGTTACAGG", 183,  "S,P",       "-",   "-"),
    ("B184",  "ACCCCTGCTCGTGTTACAGGC", 184,  "S,P",       "-",   "-"),
    ("B243",  "AGAGTCTAGACTCGTGGTGGA", 243,  "S,P",       "+",   "+"),
    ("B244",  "GAGTCTAGACTCGTGGTGGAC", 244,  "S,P",       "+++", "+++"),
    ("B245",  "AGTCTAGACTCGTGGTGGACT", 245,  "S,P",       "+++", "+++"),
    ("B246",  "GTCTAGACTCGTGGTGGACTT", 246,  "S,P",       "-",   "-"),
    ("B250",  "AGACTCGTGGTGGACTTCTCT", 250,  "S,P",       "-",   "+"),
    ("B251",  "GACTCGTGGTGGACTTCTCTC", 251,  "S,P",       "+",   "++"),
    ("B252",  "ACTCGTGGTGGACTTCTCTCA", 252,  "S,P",       "++",  "++"),
    ("B375",  "GGATGTGTCTGCGGCGTTTTA", 375,  "S,P",       "++",  "++"),
    ("B376",  "GATGTGTCTGCGGCGTTTTAT", 376,  "S,P",       "+++", "+++"),
    ("B377",  "ATGTGTCTGCGGCGTTTTATC", 377,  "S,P",       "+",   "++"),
    ("B379",  "GTGTCTGCGGCGTTTTATCAT", 379,  "S,P",       "+",   "+"),
    ("B410",  "ATCCTGCTGCTATGCCTCATC", 410,  "S,P",       "-",   "-"),
    ("B415",  "GCTGCTATGCCTCATCTTCTT", 415,  "S,P",       "+",   "++"),
    ("B456",  "AAGGTATGTTGCCCGTTTGTC", 456,  "S,P",       "++",  "++"),
    ("B457",  "AGGTATGTTGCCCGTTTGTCC", 457,  "S,P",       "-",   "+"),
    ("B458",  "GGTATGTTGCCCGTTTGTCCT", 458,  "S,P",       "++",  "++"),
    ("B459",  "GTATGTTGCCCGTTTGTCCTC", 459,  "S,P",       "++",  "++"),
    ("B461",  "ATGTTGCCCGTTTGTCCTCTA", 461,  "S,P",       "+",   "+"),
    ("B1260", "GCCGATCCATACTGCGGAACT", 1260, "EnhI,P",    "+",   "++"),
    ("B1577", "GTGTGCACTTCGCTTCACCTC", 1577, "X,P,DR1",   "+++", "++"),
    ("B1579", "GTGCACTTCGCTTCACCTCTG", 1579, "X,P,DR1",   "++",  "++"),
    ("B1581", "GCACTTCGCTTCACCTCTGCA", 1581, "X,P,DR1",   "+++", "+++"),
    ("B1583", "ACTTCGCTTCACCTCTGCACG", 1583, "X,P,DR1",   "++",  "++"),
    ("B1787", "GGAGGCTGTAGGCATAAATTG", 1787, "Pc,EnhII",  "++",  "++"),
    ("B1788", "GAGGCTGTAGGCATAAATTGG", 1788, "Pc,EnhII",  "++",  "+"),
    ("B1789", "AGGCTGTAGGCATAAATTGGT", 1789, "Pc,EnhII",  "+++", "+++"),
    ("B1880", "AAGCCTCCAAGCTGTGCCTTG", 1880, "Pc",        "+",   "-"),
    ("B1881", "AGCCTCCAAGCTGTGCCTTGG", 1881, "Pc",        "-",   "-"),
    ("B2389", "AGAAGAAGAACTCCCTCGCCT", 2389, "C,P",       "-",   "+"),
    ("B2390", "GAAGAAGAACTCCCTCGCCTC", 2390, "C,P",       "-",   "+"),
    ("B2391", "AAGAAGAACTCCCTCGCCTCG", 2391, "C,P",       "-",   "-"),
    ("B2392", "AGAAGAACTCCCTCGCCTCGC", 2392, "C,P",       "-",   "+"),
    ("B2393", "GAAGAAGAACTCCCTCGCCTC", 2393, "C,P",       "-",   "+"),
    ("B2394", "AAGAACTCCCTCGCCTCGCAG", 2394, "C,P",       "-",   "+"),
    ("B2395", "AGAACTCCCTCGCCTCGCAGA", 2395, "C,P",       "+",   "+"),
    ("B2396", "GAACTCCCTCGCCTCGCAGAC", 2396, "C,P",       "-",   "+"),
    ("B2397", "GATCCATACTGCGGAACTCCT", 2397, "C,P",       "-",   "-"),
]
for _id, _seq, _pos, _loc, _y, _n in _ZHANG_DATA:
    # Qualitative efficacy only (+/++/+++/-) — no numeric label, kept as candidate pool
    add(experiment_id=f"Zhang2010_{_id}", sirna="", target_sequence=_seq,
        reference_accession=REF_A, genotype="A", label="", assay_type="HBsAg",
        cell_line="Huh7", source=f"PMC2927532;Zhang2010;{_id};pos{_pos};{_loc};Y1021={_y},N10={_n};QUALITATIVE_ONLY")

# Wang et al. 2010 (PMC2778861) — 4 siRNAs targeting HBsAg, precise mRNA knockdown
# Reference: genotype D ayw (U95551, same subtype as V01460); HepG2.2.15 cells
# Label = HBsAg mRNA inhibition at 80nM (72h), precise values from real-time PCR
# S3 sequence not available (only S1/S2/S4 converted to shRNA in paper)
add(experiment_id="Wang2010_S1", sirna="", target_sequence="GTGGTGGACTTCTCTCAAT",
    reference_accession=REF_D_AYW, genotype="D", label=0.7177, assay_type="HBsAg_mRNA",
    cell_line="HepG2.2.15", source="PMC2778861;Wang2010;siRNA-S1;HBsAg_pos256;80nM_72h;mRNA")
add(experiment_id="Wang2010_S2", sirna="", target_sequence="AACCTCCAATCACTCACCAA",
    reference_accession=REF_D_AYW, genotype="D", label=0.5879, assay_type="HBsAg_mRNA",
    cell_line="HepG2.2.15", source="PMC2778861;Wang2010;siRNA-S2;HBsAg_pos322;80nM_72h;mRNA")
add(experiment_id="Wang2010_S4", sirna="", target_sequence="GGTATGTTGCCCGTTTGTCC",
    reference_accession=REF_D_AYW, genotype="D", label=0.8469, assay_type="HBsAg_mRNA",
    cell_line="HepG2.2.15", source="PMC2778861;Wang2010;siRNA-S4;HBsAg_pos458;80nM_72h;mRNA")

# Huang et al. 2026 — Nature Communications (DOI:10.1038/s41467-025-66876-5)
# 26 chemically-modified siRNAs (A1-A26) targeting HBV genotype D, HepG2.2.15
# Label = mean HBsAg inhibition (fraction) at 0.28 nM, Day 6, 3 biological reps
# Data from official Source Data 1 (Supplementary Fig. 1a); precise 4-decimal values
# Sequences converted from RNA (U) to DNA (T); some overlap with Zhang2010 target sites
# Negative values (A1, A12) = no significant inhibition / experimental noise, kept as-is
_HUANG2026_A = [
    ("A1",  "TCGTGTTACAGGCGGGGTTTT", -0.053),
    ("A10", "TGGTGGACTTCTCTCAATTTT", 0.7706),
    ("A11", "GTGGACTTCTCTCAATTTTCT", 0.9754),
    ("A12", "GTGTCTGCGGCGTTTTATCAT", -0.0916),
    ("A13", "TCCTGCTGCTATGCCTCATCT", 0.5087),
    ("A14", "TGCTGCTATGCCTCATCTTCT", 0.9422),
    ("A15", "ATGTTGCCCGTTTGTCCTCTA", 0.5083),
    ("A16", "CTGCTCAAGGAACCTCTATGT", 0.4946),
    ("A17", "TAGGGCTTTCCCCCACTGTTT", 0.4122),
    ("A18", "CAGTTATATGGATGATGTGGT", 0.7619),
    ("A19", "CTGCCGATCCATACTGCGGAA", 0.9502),
    ("A2",  "TACAGGCGGGGTTTTTCTTGT", 0.8973),
    ("A20", "ATCCATACTGCGGAACTCCTA", 0.5982),
    ("A21", "ACCACGGGGCGCACCTCTCTT", 0.2853),
    ("A22", "CACGGGGCGCACCTCTCTTTA", 0.1893),
    ("A23", "CTCCCCGTCTGTGCCTTCTCA", 0.9199),
    ("A24", "TCCCCGTCTGTGCCTTCTCAT", 0.9474),
    ("A25", "CCCGTCTGTGCCTTCTCATCT", 0.9638),
    ("A26", "CGGACCGTGTGCACTTCGCTT", 0.9402),
    ("A3",  "AGGCGGGGTTTTTCTTGTTGA", 0.9415),
    ("A4",  "GCGGGGTTTTTCTTGTTGACA", 0.9205),
    ("A5",  "CGGGGTTTTTCTTGTTGACAA", 0.988),
    ("A6",  "GTCTAGACTCGTGGTGGACTT", 0.7284),
    ("A7",  "CTCGTGGTGGACTTCTCTCAA", 0.7623),
    ("A8",  "TCGTGGTGGACTTCTCTCAAT", 0.7175),
    ("A9",  "GTGGTGGACTTCTCTCAATTT", 0.7141),
]
for _id, _seq, _lbl in _HUANG2026_A:
    add(experiment_id=f"Huang2026_{_id}", sirna="", target_sequence=_seq,
        reference_accession=REF_D_AYW, genotype="D", label=_lbl, assay_type="HBsAg",
        cell_line="HepG2.2.15",
        source=f"NatCommun2026;Huang2026;{_id};0.28nM_Day6;3reps_mean;chem_modified;SourceData1")

# Huang2026 A-series at 0.065 nM (26 records, 3 reps mean)
_HUANG2026_A_065 = [
    ("A1", "TCGTGTTACAGGCGGGGTTTT", -0.0628), ("A10", "TGGTGGACTTCTCTCAATTTT", 0.1731),
    ("A11", "GTGGACTTCTCTCAATTTTCT", 0.4522), ("A12", "GTGTCTGCGGCGTTTTATCAT", 0.1188),
    ("A13", "TCCTGCTGCTATGCCTCATCT", 0.1223), ("A14", "TGCTGCTATGCCTCATCTTCT", 0.2923),
    ("A15", "ATGTTGCCCGTTTGTCCTCTA", 0.0907), ("A16", "CTGCTCAAGGAACCTCTATGT", 0.2512),
    ("A17", "TAGGGCTTTCCCCCACTGTTT", -0.0417), ("A18", "CAGTTATATGGATGATGTGGT", 0.0618),
    ("A19", "CTGCCGATCCATACTGCGGAA", 0.158), ("A2", "TACAGGCGGGGTTTTTCTTGT", 0.1431),
    ("A20", "ATCCATACTGCGGAACTCCTA", 0.096), ("A21", "ACCACGGGGCGCACCTCTCTT", -0.0464),
    ("A22", "CACGGGGCGCACCTCTCTTTA", 0.0675), ("A23", "CTCCCCGTCTGTGCCTTCTCA", 0.2115),
    ("A24", "TCCCCGTCTGTGCCTTCTCAT", 0.2515), ("A25", "CCCGTCTGTGCCTTCTCATCT", 0.3877),
    ("A26", "CGGACCGTGTGCACTTCGCTT", 0.3084), ("A3", "AGGCGGGGTTTTTCTTGTTGA", 0.1659),
    ("A4", "GCGGGGTTTTTCTTGTTGACA", 0.2282), ("A5", "CGGGGTTTTTCTTGTTGACAA", 0.3927),
    ("A6", "GTCTAGACTCGTGGTGGACTT", 0.1391), ("A7", "CTCGTGGTGGACTTCTCTCAA", 0.0874),
    ("A8", "TCGTGGTGGACTTCTCTCAAT", 0.2089), ("A9", "GTGGTGGACTTCTCTCAATTT", 0.1715),
]
for _id, _seq, _lbl in _HUANG2026_A_065:
    add(experiment_id=f"Huang2026_{_id}_065nM", sirna="", target_sequence=_seq,
        reference_accession=REF_D_AYW, genotype="D", label=_lbl, assay_type="HBsAg",
        cell_line="HepG2.2.15",
        source=f"NatCommun2026;Huang2026;{_id};0.065nM_Day6;3reps_mean;chem_modified;SourceData1")

# Huang2026 A-series at 0.14 nM (26 records, 3 reps mean)
_HUANG2026_A_014 = [
    ("A1", "TCGTGTTACAGGCGGGGTTTT", 0.0225), ("A10", "TGGTGGACTTCTCTCAATTTT", 0.6261),
    ("A11", "GTGGACTTCTCTCAATTTTCT", 0.9501), ("A12", "GTGTCTGCGGCGTTTTATCAT", -0.043),
    ("A13", "TCCTGCTGCTATGCCTCATCT", 0.3725), ("A14", "TGCTGCTATGCCTCATCTTCT", 0.8858),
    ("A15", "ATGTTGCCCGTTTGTCCTCTA", 0.3719), ("A16", "CTGCTCAAGGAACCTCTATGT", 0.3196),
    ("A17", "TAGGGCTTTCCCCCACTGTTT", 0.2972), ("A18", "CAGTTATATGGATGATGTGGT", 0.6035),
    ("A19", "CTGCCGATCCATACTGCGGAA", 0.885), ("A2", "TACAGGCGGGGTTTTTCTTGT", 0.7877),
    ("A20", "ATCCATACTGCGGAACTCCTA", 0.4675), ("A21", "ACCACGGGGCGCACCTCTCTT", 0.239),
    ("A22", "CACGGGGCGCACCTCTCTTTA", -0.1627), ("A23", "CTCCCCGTCTGTGCCTTCTCA", 0.7677),
    ("A24", "TCCCCGTCTGTGCCTTCTCAT", 0.8505), ("A25", "CCCGTCTGTGCCTTCTCATCT", 0.9055),
    ("A26", "CGGACCGTGTGCACTTCGCTT", 0.9209), ("A3", "AGGCGGGGTTTTTCTTGTTGA", 0.9087),
    ("A4", "GCGGGGTTTTTCTTGTTGACA", 0.8416), ("A5", "CGGGGTTTTTCTTGTTGACAA", 0.9817),
    ("A6", "GTCTAGACTCGTGGTGGACTT", 0.5832), ("A7", "CTCGTGGTGGACTTCTCTCAA", 0.5286),
    ("A8", "TCGTGGTGGACTTCTCTCAAT", 0.5452), ("A9", "GTGGTGGACTTCTCTCAATTT", 0.557),
]
for _id, _seq, _lbl in _HUANG2026_A_014:
    add(experiment_id=f"Huang2026_{_id}_014nM", sirna="", target_sequence=_seq,
        reference_accession=REF_D_AYW, genotype="D", label=_lbl, assay_type="HBsAg",
        cell_line="HepG2.2.15",
        source=f"NatCommun2026;Huang2026;{_id};0.14nM_Day6;3reps_mean;chem_modified;SourceData1")

# Huang2026 KC13-M2G2 dose-response (9 doses, 3 reps mean)
_HUANG2026_KC13 = [
    (0.000153, "GGGGTTTTTCTTGTTGACAA", -0.0006), (0.000610, "GGGGTTTTTCTTGTTGACAA", 0.0631),
    (0.002441, "GGGGTTTTTCTTGTTGACAA", 0.0922), (0.009766, "GGGGTTTTTCTTGTTGACAA", 0.2544),
    (0.039062, "GGGGTTTTTCTTGTTGACAA", 0.5736), (0.156250, "GGGGTTTTTCTTGTTGACAA", 0.8686),
    (0.625000, "GGGGTTTTTCTTGTTGACAA", 0.9747), (2.500000, "GGGGTTTTTCTTGTTGACAA", 0.988),
    (10.000000, "GGGGTTTTTCTTGTTGACAA", 0.9894),
]
for _dose, _seq, _lbl in _HUANG2026_KC13:
    add(experiment_id=f"Huang2026_KC13_{_dose}nM", sirna="", target_sequence=_seq,
        reference_accession=REF_D_AYW, genotype="D", label=_lbl, assay_type="HBsAg",
        cell_line="HepG2.2.15",
        source=f"NatCommun2026;Huang2026;KC13-M2G2;{_dose}nM_Day6;3reps_mean;chem_modified;SourceData1")

# Huang2026 Elebsiran dose-response (9 doses, 3 reps mean)
_HUANG2026_ELEB = [
    (0.000153, "GTGTGCACTTCGCTTCACA", 0.0178), (0.000610, "GTGTGCACTTCGCTTCACA", 0.0391),
    (0.002441, "GTGTGCACTTCGCTTCACA", 0.0361), (0.009766, "GTGTGCACTTCGCTTCACA", 0.1614),
    (0.039062, "GTGTGCACTTCGCTTCACA", 0.3656), (0.156250, "GTGTGCACTTCGCTTCACA", 0.6942),
    (0.625000, "GTGTGCACTTCGCTTCACA", 0.9138), (2.500000, "GTGTGCACTTCGCTTCACA", 0.9663),
    (10.000000, "GTGTGCACTTCGCTTCACA", 0.9818),
]
for _dose, _seq, _lbl in _HUANG2026_ELEB:
    add(experiment_id=f"Huang2026_Eleb_{_dose}nM", sirna="", target_sequence=_seq,
        reference_accession=REF_D_AYW, genotype="D", label=_lbl, assay_type="HBsAg",
        cell_line="HepG2.2.15",
        source=f"NatCommun2026;Huang2026;Elebsiran;{_dose}nM_Day6;3reps_mean;chem_modified;SourceData1")

# Group II: conservation-based miRNA, liposome
# Gao et al. [52]
add(experiment_id="Gao2008_miR155_S608", sirna="", target_sequence="CCAAGTCTGTACAGCATCGTGA",
    reference_accession=REF_A, genotype="A", label=0.80, assay_type="HBs_mRNA",
    cell_line="Huh7", source=f"{SRC_REV};ref[52];Gao2008")
# Ely et al. [24] — Mir-51/5 and Mir-122/5 target same region
add(experiment_id="Ely2008_miR51_5", sirna="", target_sequence="CCGTGTGCACTTCGCTTCACCTCTG",
    reference_accession=REF_A, genotype="A", label=0.90, assay_type="HBsAg",
    cell_line="Huh7", source=f"{SRC_REV};ref[24];Ely2008")

# Group III: single genotype shRNA, viral vector
# Sun et al. [40]
add(experiment_id="Sun2009_S1_456", sirna="", target_sequence="AAGGTATGTTGCCCGTTTGTC",
    reference_accession=REF_A, genotype="D", label=0.90, assay_type="total_RNA",
    cell_line="Huh7", source=f"{SRC_REV};ref[40];Sun2009_groupIII")
add(experiment_id="Sun2009_X1", sirna="", target_sequence="AAGGTCTTACATAAGAGGACT",
    reference_accession=REF_A, genotype="D", label=0.90, assay_type="total_RNA",
    cell_line="Huh7", source=f"{SRC_REV};ref[40];Sun2009_groupIII")
# Peng et al. [53]
add(experiment_id="Peng2005_S1_322", sirna="", target_sequence="AACCTCCAATCACTCACCAAC",
    reference_accession=REF_A, genotype="D", label=0.80, assay_type="HBsAg",
    cell_line="HepG2.2.15", source=f"{SRC_REV};ref[53];Peng2005")
add(experiment_id="Peng2005_S2", sirna="", target_sequence="AAGGAACCTCTATGTATCCCT",
    reference_accession=REF_A, genotype="D", label=0.80, assay_type="HBsAg",
    cell_line="HepG2.2.15", source=f"{SRC_REV};ref[53];Peng2005")
add(experiment_id="Peng2005_S3", sirna="", target_sequence="AAATTGCACCTGTATTCCCAT",
    reference_accession=REF_A, genotype="D", label=0.80, assay_type="HBsAg",
    cell_line="HepG2.2.15", source=f"{SRC_REV};ref[53];Peng2005")

# ---------------------------------------------------------------------------
# Source 2: PMC2770508 (Ely et al. 2009, Mol Ther 17(3):538-547)
# 14 rationally designed HBV-miRs (miR-30 scaffold) + HBVU6#2 benchmark
# Reference: HBV ayw (genotype D), EcoRI-based numbering
# Knockdown of 3.5 kb pgRNA at stated plasmid dose in Huh-7 cells
# ---------------------------------------------------------------------------
SRC_ELY = "PMC2770508;Ely2009"

# HBVU6#2 benchmark: 2ug -> 69.1%, 0.5ug -> 48.8% pgRNA knockdown
add(experiment_id="Ely2009_HBVU6#2_2ug", sirna="", target_sequence="",
    reference_accession=REF_D_AYW, genotype="D", label=0.691, assay_type="pgRNA",
    dose_nm="2000", time_h="72", cell_line="Huh7", source=f"{SRC_ELY};HBVU6#2;benchmark")
add(experiment_id="Ely2009_HBVU6#2_0.5ug", sirna="", target_sequence="",
    reference_accession=REF_D_AYW, genotype="D", label=0.488, assay_type="pgRNA",
    dose_nm="500", time_h="72", cell_line="Huh7", source=f"{SRC_ELY};HBVU6#2;benchmark")

# HBV-miR-1: 0.5ug -> 81.2% pgRNA knockdown (significantly > HBVU6#2)
add(experiment_id="Ely2009_miR1_0.5ug", sirna="TGAAGTATGCCTCAAGGTCGGT",
    target_sequence="CGACCTTGAGGTATACTTCATA",
    reference_accession=REF_D_AYW, genotype="D", label=0.812, assay_type="pgRNA",
    dose_nm="500", time_h="72", cell_line="Huh7", source=f"{SRC_ELY};HBV-miR-1;pos1689")
# HBV-miR-4: 0.5ug -> 76.7%
add(experiment_id="Ely2009_miR4_0.5ug", sirna="TCTATAACTGTTTCTCTTCCAA",
    target_sequence="GGAAGAGAAACAGTTATAGATA",
    reference_accession=REF_D_AYW, genotype="D", label=0.767, assay_type="pgRNA",
    dose_nm="500", time_h="72", cell_line="Huh7", source=f"{SRC_ELY};HBV-miR-4;pos2229")
# HBV-miR-7: 0.5ug -> 74.8%
add(experiment_id="Ely2009_miR7_0.5ug", sirna="TACAAAGGCATCAACGCAGGAT",
    target_sequence="CTTGCGTTGATGTCTTTGTATA",
    reference_accession=REF_D_AYW, genotype="D", label=0.748, assay_type="pgRNA",
    dose_nm="500", time_h="72", cell_line="Huh7", source=f"{SRC_ELY};HBV-miR-7;pos1043")

# Remaining HBV-miRs: sequences confirmed, 2ug knockdown > HBVU6#2 (69.1%) for 8 of them,
# but exact numeric values only in Figure 5b (not in text). Label left empty pending
# chart extraction; these rows are usable for candidate ranking.
ely_mirs = [
    ("miR2", "TGACATTGCTGAGAGTCCAAGA", "TTGGACTCTTAGCAATGTTATA", "1664"),
    ("miR3", "TTACTGTTCCTGAACTGGAGCC", "CTCCAGTTCAGGAACAGTAATA", "65"),
    ("miR5", "TGTAGTATGCCCTGAGCCTGAG", "CAGGCTTAGGGTATACTACATA", "3050"),
    ("miR6", "TAGTGTTGACATAACTGACTAC", "AGTCAGTTATGTCAACATTATA", "2153"),
    ("miR8", "TAACTGACTACTAGGTCTCTAG", "AGAGACCTAGTAGTCAGTTATA", "2142"),
    ("miR9", "TACTAGGTCTCTAGACGCTGGA", "CAGCGTCTAGAGACTTAGTATA", "2134"),
    ("miR10", "TAGAGTGTGTAAATAGTGTCTA", "GACACTATTTACACACTTTATA", "2738"),
    ("miR11", "TTTGGTGGAAGGTTGTGGAATT", "TTCCACAACTTTCCACTAAATA", "3181"),
    ("miR12", "TTGGTGAGTGATTGGAGGTTGG", "AACCTCTAATCACTCACTAATA", "320"),
    ("miR13", "TGATAGTCCAGAAGAACCAACA", "TTGGTTCTTCTGGATTATCATA", "435"),
    ("miR14", "TGATTAACTAGATGTTCTGGAT", "CCAGAACATTTAGTTAATCATA", "2704"),
]
for name, guide, target, pos in ely_mirs:
    add(experiment_id=f"Ely2009_{name}", sirna=guide, target_sequence=target,
        reference_accession=REF_D_AYW, genotype="D", label="", assay_type="pgRNA",
        time_h="72", cell_line="Huh7", source=f"{SRC_ELY};HBV-{name};pos{pos};label_pending_fig5b")

# ---------------------------------------------------------------------------
# Source 3: PMC3666629 (Morrissey et al. 2014, Mol Ther Nucleic Acids)
# 4 lead siRNAs: siHBV-74/75/76/77, chemically modified (2'-OMe/2'-F)
# In vivo HBsAg knockdown in pHBV mouse model (hydrodynamic injection)
# Core 19-mer target extracted from modified antisense; position on genotype D (ayw)
# 0.25 mg/kg dose, day 8 maximal HBsAg reduction:
#   siHBV-74: 1.8 log10 (~98.4%), siHBV-77: 1.4 log10 (~96.0%),
#   siHBV-75: 1.1 log10 (~92.1%), siHBV-76: 0.9 log10 (~87.4%)
# ---------------------------------------------------------------------------
SRC_MOR = "PMC3666629;Morrissey2014"

# Core 19-mers extracted from modified antisense sequences (Methods section).
# Extraction method: keep all A/C/G/T/U letters (strip d/f/s/chol-/invdT modifiers),
# uppercase, take first 19 nt as the pairing region (last 2 nt are 3' dTdT overhang).
# Target = reverse_complement(guide_core_19).
morrissey_leads = [
    # name, guide_core_19 (DNA), target_core_19 (DNA), region
    ("siHBV-74", "TACCAATTTATGCCTACAG", "CTGTAGGCATAAATTGGTA", "S_region"),
    ("siHBV-75", "TATGATAAAACGCCGCAGA", "TCTGCGGCGTTTTATCATA", "S_region"),
    ("siHBV-76", "TACAATGGCACTAGTAAAT", "ATTTACTAGTGCCATTGTA", "overlap_S_P"),
    ("siHBV-77", "TAGATGATTAGGCAGAGGT", "ACCTCTGCCTAATCATCTA", "overlap_S_P"),
]
morrissey_labels = [0.984, 0.921, 0.874, 0.960]  # fractional HBsAg inhibition
for (name, guide, target, region), lbl in zip(morrissey_leads, morrissey_labels):
    add(experiment_id=f"Morrissey2014_{name}", sirna=guide, target_sequence=target,
        reference_accession=REF_D_AYW, genotype="D", label=lbl, assay_type="HBsAg",
        dose_nm="250", time_h="192", cell_line="pHBV_mouse",
        source=f"{SRC_MOR};{name};{region};0.25mg/kg_day8")

# ---------------------------------------------------------------------------
# Write CSV
# ---------------------------------------------------------------------------
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)

labelled = sum(1 for r in rows if r["label"] != "")
print(f"Wrote {len(rows)} rows to {OUT}")
print(f"  with label: {labelled}")
print(f"  without label (candidate pool): {len(rows) - labelled}")
print(f"  unique target sequences: {len(set(r['target_sequence'] for r in rows if r['target_sequence']))}")
