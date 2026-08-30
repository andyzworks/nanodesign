# NanoDesign v0 Specification

本文档是 NanoDesign v0 的冻结规格。实现、配置和实验不得在不升级规格版本的情况下改变
这里的任务定义。

## 1. Task

统一目标：**给定 molecular context，设计能够与 target interaction / bind 的新分子。**

### Protein Binder Design

- Input：target protein
- Output：protein binder sequence + structure
- Target fixed，binder generated

### Antibody CDR Design

- Input：antigen + fixed antibody framework
- Output：CDR sequence + structure
- Antigen 和 framework fixed，CDR generated

### RNA Aptamer Design

- Input：target protein
- Output：RNA aptamer sequence + structure
- Target fixed，RNA generated

三个任务使用同一个 baseline model architecture 和同一个 pipeline，不使用三个独立模型。

## 2. Data

### Protein Binder

数据源：PPIRef / PPIRef50K。

```text
Protein A → Target / Context
Protein B → Binder / Design Region
```

冻结为 PPIRef50K `ppiref_6A_filtered_clustered_04`。MMseqs2 30% identity/80%
coverage 的 target+binder connected components 做 80/10/10 split；较长 resolved chain
为 target，另一条为 binder。

### Antibody CDR

数据源：SAbDab2 antibody-antigen complexes。

```text
Antigen   → Fixed
Framework → Fixed
CDR       → Design Region
```

冻结为 H3-only。保留 SAbDab2 官方 antigen-aware test，从官方 train 的 `ab_ag` clusters
中固定 10% 为 validation。

### RNA Aptamer

Binding-design 数据：

- Ribocentre Aptamer
- PDB 中实验解析的 RNA-target complexes

辅助 prior 数据：RNAsolo2。

```text
Aptamer / RNA-target complex → RNA binding design
RNAsolo2 general RNA         → RNA sequence / structure prior only
```

在冻结 RNA data pool 之前，必须先统计两个 binding source 的候选 complex 数、可用数、
排除原因和筛选 protocol，并为 inventory 保存 SHA-256 指纹。

## 3. Baseline Model

Baseline：**Small RFD3NA-style unified diffusion model**。

```text
Known molecular context
+
Design / masked region
        ↓
Unified diffusion model
        ↓
Generated sequence + structure
```

v0 直接使用 RosettaCommons Foundry 的公开 RFD3NA 实现，固定 commit
`aad357b776e3c0d6b973080f8f8c4bcf3ed21e40`；不自行重写 geometry block。

保留的官方组件：

1. atom-level feature encoder；
2. atom-to-token cross attention；
3. token initializer、Pairformer 和 local diffusion transformer；
4. token-to-atom decoder；
5. coordinate diffusion head；
6. protein/RNA sequence head；
7. context/design mask conditioning。

第一版命名为 `NanoDesign-Tiny`，默认 6,849,538 参数。设计区使用官方 atom23 的
`UNK/X` sequence-independent slots，不能从 native side-chain/base atom names 泄漏序列。

## 4. Evaluation

三个任务采用各自领域的 protocol，不强行统一为一个 metric。

### Protein Binder Design

主指标：`In-silico Success Rate ↑`。

辅助指标：Interface confidence ↑、Self-consistency RMSD ↓、Rosetta interface ΔG ↓、
Shape Complementarity ↑、Clashes ↓、Diversity / Cluster-level Success ↑。

必须在 test evaluation 前冻结 independent structure predictor、success filters 和 threshold
来源。threshold 必须来自已有工作或预先冻结的 calibration，不得看 test 结果后手调。

### Antibody CDR Design

主指标：CDR AAR ↑、CDR RMSD ↓、DockQ ↑。

必须重点报告 CDR-H3 AAR 和 CDR-H3 RMSD。如果设计六个 CDR，必须逐个报告 H1、H2、
H3、L1、L2、L3。辅助指标为 Rosetta interface ΔG、clashes 和 geometry validity。

### RNA Aptamer Design

RNA structure quality：scTM ↑、scRMSD ↓、Structure confidence ↑。

当 test sample 存在真实 RNA-target complex 时，报告 RNA-target DockQ ↑。

`DockQ`、`scTM` 和 `scRMSD` 不代表 binding affinity。实验 Kd 是 affinity gold standard，
但不属于 NanoDesign v0 computational evaluation。

## 5. Frozen Decisions

| Component | Decision |
| --- | --- |
| Task 1 | Protein Binder Design |
| Data 1 | PPIRef50K filtered_clustered_04 |
| Evaluation 1 | In-silico Success Rate + interface/self-consistency metrics |
| Task 2 | Antibody CDR Design |
| Data 2 | SAbDab2 |
| Evaluation 2 | AAR + CDR RMSD + DockQ |
| Task 3 | RNA Aptamer Design conditioned on target protein |
| Data 3 | Ribocentre + PDB RNA-target complexes + RNAsolo2 auxiliary prior |
| Evaluation 3 | scTM + scRMSD + RNA-target DockQ |
| Baseline | NanoDesign-Tiny: official Foundry RFD3NA, 6,849,538 parameters |

## 6. Runtime requirements

数据、模型和 evaluator 的决定已写入 `configs/v0.yaml`。完整 benchmark 仍需要在执行机器
部署 ColabFold/RhoFold+ 权重与 licensed Rosetta，并记录 H100 peak memory 和 step time；
这些运行结果不能在没有实际执行时伪造。
