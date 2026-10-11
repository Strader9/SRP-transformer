# HBV 多基因型 siRNA 效能建模

本项目实现了一个独立的 HBV siRNA Transformer，用于研究候选 guide 序列、HBV 靶区、基因型差异和实验抑制效应之间的关系。模型代码不依赖 OligoFormer，也不包含其代码或权重。项目当前是研究原型；现有数据不足以证明它具备稳定的效能排序能力或跨基因型泛化能力。

## 模型

`hbv_sirna/model.py` 包含以下组件：

- 共享序列 Transformer 编码 guide RNA 和各基因型靶序列，并加入核苷酸位置编码。
- 基因型 Transformer 对可用靶序列建模；缺失基因型由 mask 排除。
- 保守性同时以每位置 profile 和全局 summary 输入模型。
- HBV 特征包括 S/P/C/X 功能区、重叠区、GC 含量和平均保守性；`concat` 与 `gated` 两种融合可用于消融比较。
- assay 任务使用独立输出头。回归输出不限制在 0–1，因为交付数据中存在负抑制值；预测值使用训练标签的原始尺度。

推理还会给出 `genotype_coverage`：按 guide 与输入靶序列的互补度计算的序列覆盖统计，不是效能分数，也不能替代实验验证。

## 数据与解释边界

交付数据包含 175 条原始实验记录、122 条训练特征记录和 122 × 8 条基因型映射记录。原始数据中有 A、D 两种直接实验基因型；已标注的 HBsAg 样本为 D 型 103 条、A 型 1 条。B–H 的靶序列是参考基因组上的计算映射，**没有对应的直接实验标签**。原始表中 104 条 HBsAg 记录只有 4 条含有原始 siRNA guide；pgRNA 有 14 条记录提供了 guide，但其中 11 条没有效应标签，只有 3 条能用于监督训练。因此，所有终点合计只有 7 条同时具备实测 guide 和效应标签的记录；其余训练记录中的 guide 是从靶区反向互补推断的，不能确认与实验实际使用的分子一致。数据处理会将推断序列标成 `_guide_inferred`，常规训练默认拒绝此类记录。

最新交付包增加了原始实验详细表（175 条，与原始表按 `experiment_id` 一一对应；序列、标签、assay 和来源一致），明确标签方向为抑制值越高效应越强、0 为无抑制、1 为完全抑制、负值表示促进效应。该表为 57 条记录提供起止坐标，但链方向为空；用表中 accession 和仓库当前参考序列复核时，只有 26 条靶序列在记录的起点精确匹配，其余 31 条不匹配。因此这些坐标目前只作为待核验元数据，不能直接覆盖序列定位或作为已验证特征。新增表没有增加带实测 guide 和效应标签的记录数。

当前数据的 HBsAg 标签范围为 `-0.1627` 到 `0.9894`。大量记录缺少剂量和时间信息，且多个条件/来源混在一起；模型目前不会把剂量、时间或细胞系作为输入。因此训练分数不能解释为控制这些因素后的生物学效应。

`scripts/experiments_to_candidates.py` 对已提供的 guide 和靶区共同搜索最互补的 19-mer 窗口，避免把两条序列各自从第一个碱基截断造成错位；`_pair_identity` 记录互补度。缺少 guide 时生成的反向互补序列会写入 `_guide_inferred=True`，不会伪装成实测序列。

多基因型数据由全基因组滑窗最佳匹配构建，不等同于经过多序列比对验证的同源靶位点。错配距离较大的映射可能落在非同源区域。正式生物学结论前，应以有方向和坐标核验的全基因组 MSA 重建候选表，并人工复核一批靶区。

分组切分将相同或高度相似的 guide/多基因型靶序列束以及重叠参考坐标放在同一组，防止重复测量和近重复序列跨越训练/测试。随机行切分只用于调试，不应用作性能结论。外层分组交叉验证在每个训练折内另做分组验证来选择 epoch，并单独保留外层测试折。

## 训练和评估

创建 HBsAg 专用候选表：

```bash
python scripts/phase3/build_multitask_candidates.py --assay HBsAg \
  --output data/processed/candidates_hbsag.csv
```

训练小数据集优先使用 compact 模型：

```bash
hbv-sirna train --data data/processed/candidates_hbsag.csv \
  --output-dir checkpoints/hbv_hbsag --epochs 25 --batch-size 8 \
  --learning-rate 0.0005 --model-size compact --fusion gated \
  --genotype-dropout 0.25 --consistency-weight 0.1 --allow-inferred-guides
```

`--allow-inferred-guides` 表示用户明确接受代理序列假设，只适用于探索分析。当前 HBsAg 数据若仅使用 4 条实测 guide，pgRNA 数据若仅使用 3 条有标签的实测 guide，都不足以训练或验证模型；因此任何基于这 104 条 HBsAg 记录的模型结果都不是对真实实验 guide 的可靠效能评估。

运行按靶区分组的外层 5 折评估，并与每折训练集均值基线比较：

```bash
python scripts/phase3/evaluate_grouped_cross_validation.py \
  --data data/processed/candidates_hbsag.csv \
  --output-dir results/phase3_grouped_cv_repeated \
  --folds 5 --seeds 3 --model-size compact --allow-inferred-guides
```

脚本保存外层 out-of-fold 预测、逐折结果及汇总指标。以下为**探索性代理 guide** 5 折 × 3 次重复结果（每次 104 条 HBsAg、其中 100 条 guide 是推断值、17 个独立相似性组、compact、25 epochs；OOF 指标合计 312 个预测）：

| 方法 | MSE | RMSE | MAE | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|
| Transformer | 0.1494 | 0.3866 | 0.3405 | -0.0717 | -0.1078 |
| 每折训练集均值 | 0.1335 | 0.3654 | 0.3307 | -0.2125 | -0.1905 |

三次重复的 Transformer MSE 分别为 0.1345、0.1725、0.1342；每次均值基线分别为 0.1333、0.1299、0.1331。模型没有稳定优于均值基线，相关性接近零或为负。由于每次仍有 100 条 guide 是推断值，这些指标只能检查代码和评估流程，不能作为真实 siRNA 效能预测证据。当前不能声称取得预测性能提升或交付可靠排序器。

留一基因型评估：

```bash
python scripts/phase3/evaluate_leave_one_genotype_out.py \
  --data data/processed/candidates_hbsag.csv
```

脚本会在训练折屏蔽被留出的基因型序列，并在测试折只提供被留出基因型序列，避免输入层泄漏。当前 A/D 标签量为 1/103，两折都因样本不足而跳过；B–H 没有直接实验标签，无法进行该实验。计算出的 B–H 靶序列不能代替真实标签。

## 推理

```bash
hbv-sirna predict --data data/processed/candidates.csv \
  --checkpoint checkpoints/hbv_hbsag/best.pt \
  --output results/hbv_ranked.csv
```

多任务 checkpoint 必须明确选择排序终点，例如 `--rank-task HBsAg`。推理会核验输入特征维度，并使用 checkpoint 中的序列长度、融合、任务和基因型输入元数据。旧 checkpoint 可以加载；旧模型不含位置保守性分支时，该分支以零初始化兼容，旧 checkpoint 的 sigmoid 输出行为也会保留。预测文件会标出 `guide_inferred`。

## 检查

在项目 Conda 环境运行：

```bash
python -m pytest -q
python -m compileall -q hbv_sirna scripts
git diff --check
```

训练 CSV 至少需要 `sample_id`、`sirna`、JSON 数组 `target_sequences`、`genotype_ids`、`hbv_features` 和 `label`。若存在 `assay_type` 列，每行都必须有任务名；多任务训练要求每个 assay 至少 10 条有标签记录。缺失基因型数组要与靶序列一一对应。checkpoint、原始数据及实验结果默认由 `.gitignore` 排除，分享前需确认来源和许可。

## 第三阶段实验资产

`scripts/phase3/` 提供 genotype dropout/一致性正则搜索、HUVK 预训练数据构建、预训练微调、组合实验、集成不确定性和留一基因型评估入口。HUVK 配对数据当前有 2,816 条精确匹配记录。已有历史结果是早期模型和切分设置下的探索结果，不可与本 README 的外层交叉验证结果混为一谈。

## OligoFormer

OligoFormer 上游使用专有许可，限制修改、再分发和特定衍生开发用途；参见[上游许可证](https://github.com/lulab/OligoFormer/blob/main/LICENSE)。本项目不复制其代码/权重、不提取内部表示，也不接入其预测分数。若要与 OligoFormer 融合，需先获得权利方针对具体用途、接口和结果分发范围的书面授权。当前模型是独立实现，项目标题和包名不代表 OligoFormer 衍生模型。

## 后续工作优先级

1. 通过 MSA 和参考 accession 坐标核验所有 A–H 靶位点，记录唯一映射、错配、方向和歧义；过滤无法可靠映射的候选。
2. 补充 A–H 基因型的直接实验标签，并优先统一 assay、细胞系、剂量、时间和重复测量口径。
3. 在数据增加后开展重复分组交叉验证、独立外部测试和预先定义的消融；当前数据不支持有意义的留一基因型结论。
4. 仅在取得明确书面授权后评估 OligoFormer 互操作或融合方案。
