# 基于 OligoFormer 的 HBV 多基因型 siRNA 排序

这是 SRP Transformer 方向第二、三阶段的可运行工程骨架：引入 HBV 专属数据与特征，并让模型同时读取同一靶位点在多个 HBV 基因型中的对应序列。它不替代官方 OligoFormer，而是作为 HBV 扩展层；第一阶段的官方 Docker 推理仍可单独保留。

上游项目：[lulab/OligoFormer](https://github.com/lulab/OligoFormer)。使用上游代码、模型和数据前请阅读其许可证；本项目不得重新分发未获授权的权重或受限制数据。

## 一、最终要做成什么

每个候选 siRNA 不再只有一条参考靶序列，而是一个样本包：

```text
siRNA 19 nt
  + A/B/C/D/... 基因型对应靶区（先经过全基因组多序列比对）
  + 每个位点的保守性
  + HBV 功能区、重叠区及实验条件等结构化特征
  -> 序列内 Transformer
  -> 跨基因型 Transformer
  -> 融合层
  -> HBV efficacy / 抑制效果分数
```

`hbv_sirna/model.py` 使用两级编码：第一级学习每条 19 nt 序列内部的碱基关系，第二级学习多个基因型之间的差异；缺失基因型通过 mask 排除。保守性和 HBV 专属数值特征在输出头前融合。

## 二、第二阶段：HBV 数据与专属特征

### 2.1 先定义研究终点

不要把 HBsAg、HBeAg、HBV DNA、S mRNA、total RNA 当作普通输入列直接混在一起。它们优先是不同实验的输出标签。推荐第一版只选一个主终点，例如相对 HBsAg 抑制率，并统一为 0 到 1：

```text
label = (control - treated) / control
```

如果论文只给 fold change、百分比或残余表达量，先统一方向和范围，并在 `assay_type`、`time_h`、`dose_nm`、`cell_line` 中保留实验条件。不同终点混训时应改成多任务输出；不要把不可比较的值硬拼成一个 label。

### 2.2 推荐原始数据表

建立 `data/raw/hbv_experiments.csv`，一行对应一个“siRNA × 实验条件”：

| 字段 | 含义 |
|---|---|
| experiment_id | 文献/实验唯一编号 |
| sirna | antisense 序列，统一为 DNA 字母 A/C/G/T |
| target_sequence | 参考基因型上的 19 nt 靶序列 |
| reference_accession | 参考 HBV 序列编号 |
| genotype | 实验所用基因型 |
| label | 0–1 主终点 |
| assay_type | HBsAg、HBeAg、HBV_DNA 等 |
| dose_nm/time_h | 剂量与时间 |
| cell_line | HepG2、HepG2.2.15 等 |
| source | DOI、PMID 或内部实验编号 |

必须做的清洗：统一 5'→3' 方向；U 转 T；去除修饰符号；核对 antisense/sense；重复实验保留原始重复并另算均值；同一靶位点不能跨训练集和测试集泄漏。

### 2.3 HBV 序列与注释

1. 从可信数据库下载带明确基因型标签的 HBV 全基因组序列。
2. 每个基因型先做质量控制：长度异常、N 比例过高、重复序列和重组序列要记录或剔除。
3. 使用 MAFFT 对完整基因组做多序列比对，保留 gap：

```bash
mafft --auto --keeplength data/raw/hbv_genomes.fasta > data/processed/hbv_aligned.fasta
```

4. 功能区建议按参考基因组坐标建立 BED/GFF：S、P、C、X，以及重叠区域。HBV 是环状基因组，跨越坐标零点的区间必须拆成两段处理。
5. 第一版 `hbv_features` 推荐固定顺序：`S/P/C/X` 四个 one-hot、是否重叠区、全局保守性、GC 比例，共 7 维。顺序必须写入数据字典，不能训练时临时改变。

### 2.4 构造跨基因型候选表

输入必须是已经对齐且等长的 FASTA，header 是基因型或序列 ID：

```bash
pip install -e ".[dev]"
hbv-sirna prepare \
  --alignment data/processed/hbv_aligned.fasta \
  --reference-id genotype_C_reference \
  --output data/processed/candidates.csv
```

输出核心字段：

```csv
sample_id,sirna,target_sequences,genotype_ids,hbv_features,label
ref:100-118,AAC...,"[\"TTG...\",\"TCG...\"]","[\"A\",\"C\"]","[1,0,0,0,1,0.95,0.47]",0.82
```

`target_sequences` 必须来自同一个 MSA 坐标窗口。不能在各基因型中用字符串搜索“最像的 19 nt”，否则 indel 附近会配错同源位点。

## 三、第三阶段：让模型学习基因型差异

### 3.1 当前实现

- `SequenceEncoder`：共享参数地编码 siRNA 和每个基因型靶序列。
- `genotype_encoder`：把各基因型表示看成一个集合，学习哪些差异影响预测。
- `genotype_mask`：样本缺少某基因型时屏蔽填充序列。
- `conservation`：计算每个 MSA 位置的多数碱基频率。
- `hbv_features`：在预测头前与序列表征融合。
- 输出经过 sigmoid，范围为 0–1。

### 3.2 与原 OligoFormer 的正式融合方案

第一版先训练本仓库模型，得到可靠基线。第二版再把官方 OligoFormer 的倒数第二层表示记为 `oligo_embedding`，与本模型的 `pooled_genotypes` 拼接：

```python
fused = torch.cat([
    oligo_embedding,
    pooled_genotypes,
    conservation_embedding,
    hbv_features,
], dim=-1)
score = prediction_head(fused)
```

不要一开始就修改官方 RNA-FM 提取脚本。先在 OligoFormer `model.py` 的最终线性层之前增加一个可选返回值 `return_embedding=True`，并用 adapter 调用，这样官方推理接口仍然可复现。

### 3.3 训练

给 `candidates.csv` 补齐 label 和固定维度 `hbv_features` 后运行：

```bash
hbv-sirna train \
  --data data/processed/candidates_labelled.csv \
  --output-dir checkpoints/hbv_v1 \
  --epochs 50 \
  --batch-size 32 \
  --learning-rate 0.0001
```

输出：`checkpoints/hbv_v1/best.pt` 与 `history.json`。

数据切分应按“靶位点/基因”分组，而不是随机按行切分。当前命令提供的是最小可运行基线；正式实验应增加 GroupKFold，使相同或高度相似的靶区只进入一个 split。

### 3.4 推理排序

```bash
hbv-sirna predict \
  --data data/processed/candidates.csv \
  --checkpoint checkpoints/hbv_v1/best.pt \
  --output results/hbv_ranked.csv
```

最终不要只按模型分数筛选。建议加入硬过滤：跨基因型完全匹配比例、功能区、脱靶、GC、连续相同碱基、免疫刺激 motif 和可合成性，然后再产生候选 Top N。

## 四、必须完成的实验

至少报告以下对照，且所有模型使用同一数据划分：

1. 官方 OligoFormer 分数或其复现结果。
2. 单参考基因型模型。
3. 多基因型模型，但不输入保守性。
4. 多基因型 + 保守性。
5. 多基因型 + 保守性 + HBV 功能区特征。

回归指标：Spearman、Pearson、MAE、RMSE；若定义“高效 siRNA”阈值，再报告 AUROC、AUPRC、F1。额外报告 leave-one-genotype-out：每次留一个基因型只做测试，以证明模型真正学到了跨基因型泛化。

## 五、本地与 Docker 运行

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

Docker：

```bash
docker build -t hbv-oligoformer:0.1 .
docker run --rm -v "$PWD/data:/app/data" hbv-oligoformer:0.1 \
  prepare --alignment data/processed/hbv_aligned.fasta \
  --reference-id genotype_C_reference --output data/processed/candidates.csv
```

官方 OligoFormer 第一阶段仍可按原方式验证：

```bash
docker pull yilanbai/oligoformer:v1.0
docker run -it --name oligoformer-container -dt yilanbai/oligoformer:v1.0
docker exec -it oligoformer-container bash
oligoformer -i 1 -i1 data/example.fa -i2 data/example_siRNA.fa
```

## 六、推送到你的空 GitHub 仓库

在项目根目录执行：

```bash
git init
git branch -M main
git remote add origin https://github.com/Strader9/SRP-transformer.git
git add .
git commit -m "feat: add HBV multi-genotype OligoFormer extension"
git push -u origin main
```

GitHub Actions 会自动运行测试和 Docker 构建。原始生物数据、模型权重、患者/实验敏感信息不得提交；仓库只放数据字典、示例和下载/处理脚本。

## 七、建议时间表

- 第 1 周：确定主终点、数据字典和纳入排除标准。
- 第 2–3 周：收集 HBV siRNA 实验数据，完成方向、单位和重复清洗。
- 第 4 周：整理各基因型序列、MAFFT 对齐、功能区坐标映射。
- 第 5 周：生成候选表，检查 20 个随机位点的同源映射。
- 第 6 周：训练单基因型和多基因型基线。
- 第 7 周：接入 OligoFormer embedding，完成消融实验。
- 第 8 周：做 leave-one-genotype-out、误差分析和报告图表。

## 八、当前边界

仓库目前提供可测试的模型与工程流程，但不包含真实 HBV 实验数据、MAFFT 结果、RNA-FM 权重或训练好的 HBV checkpoint。没有带标签的数据，任何“训练完成”或生物学效果声明都不成立。下一步的核心不是继续加网络层，而是得到来源清楚、方向一致、不会泄漏的标注数据。

