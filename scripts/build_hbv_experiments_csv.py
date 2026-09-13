"""Build hbv_experiments.csv from manually extracted literature data.

Sources:
- PMC3132485 (review Table 1): consolidated HBV RNAi targets from Sun et al. [7],
  Wu et al. [10], Panjaworayan et al. [14], Zhang et al. [13], Gao et al. [52],
  Ely et al. [24], Sun et al. [40], Peng et al. [53]
- PMC2770508 (Ely et al. 2009, Mol Ther): 14 rationally designed HBV-miRs + HBVU6#2
- PMC3666629 (Morrissey et al. 2014): 4 lead siRNAs (siHBV-74/75/76/77)

Label convention: fractional inhibition (0-1) = 1 - residual_fraction.
assay_type records what was measured (pgRNA, HBV_DNA, HBsAg, total_RNA, cccDNA).
Rows without a confirmed numeric label have label empty (usable for candidate ranking,
excluded from training by train.py).
"""

import csv
from pathlib import Path

OUT = Path(r"C:\Users\34359\Documents\SRP\SRP-transformer-HBV-v0.1\data\raw\hbv_experiments.csv")

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
# Map to fractional inhibition: +++ -> 0.95, ++ -> 0.70, + -> 0.30, - -> None
# Label = mean of two cell lines (if one is "-", use the other; both "-" -> no label)
_ZHANG_QUAL = {"+++": 0.95, "++": 0.70, "+": 0.30, "-": None}
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
    _ly, _ln = _ZHANG_QUAL[_y], _ZHANG_QUAL[_n]
    _vals = [v for v in (_ly, _ln) if v is not None]
    _label = sum(_vals) / len(_vals) if _vals else ""
    add(experiment_id=f"Zhang2010_{_id}", sirna="", target_sequence=_seq,
        reference_accession=REF_A, genotype="A", label=_label, assay_type="HBsAg",
        cell_line="Huh7", source=f"PMC2927532;Zhang2010;{_id};pos{_pos};{_loc};Y1021={_y},N10={_n}")

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
REF_D_AYW = "V01460"
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
