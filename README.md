# NanoDesign

NanoDesign v0 固定解决同一个问题：

> 给定 molecular context，设计能够与 target interaction / bind 的新分子。

v0 只有三个任务，不增加第四个任务：

| 任务 | 固定输入 | 生成输出 |
| --- | --- | --- |
| Protein Binder Design | Target protein | Binder sequence + structure |
| Antibody CDR Design | Antigen + antibody framework | CDR sequence + structure |
| RNA Aptamer Design | Target protein | RNA aptamer sequence + structure |

三个任务共用一个 `NanoDesign-Tiny` 模型和同一套训练/采样 pipeline，不是三个独立模型。

## 当前做到哪里

当前仓库完成的是 v0 infrastructure，不代表已经得到可用于科学结论的 trained model。

已经实现：

- 不可悄悄改动的三任务规格与输入/输出 contract
- 统一 token/atom 数据格式、NPZ 序列化和 batch collator
- 数据来源登记、manifest 指纹与 cluster-disjoint split 检查
- RNA binding data 与 RNAsolo2 structure prior 的硬隔离
- 冻结 RNA data pool 之前必须完成的 usable-complex inventory
- 一个共享的 `NanoDesign-Tiny` atom/token diffusion baseline
- 三个任务各自独立的 evaluation contract
- 训练 step、checkpoint provenance、CLI 和测试

尚未实现：真实数据下载与转换、正式 split、外部 structure predictor、DockQ/Rosetta
执行器、已校准 success thresholds，以及正式训练。这些依赖未决科学选择，代码不会擅自决定。

详细状态见 [Implementation Status](docs/IMPLEMENTATION_STATUS.md)。

## 数据边界

| 任务/用途 | 允许的数据源 |
| --- | --- |
| Protein binder binding design | PPIRef 或 PPIRef50K |
| Antibody CDR binding design | SAbDab2 antibody-antigen complexes |
| RNA aptamer binding design | Ribocentre Aptamer + experimental PDB RNA-target complexes |
| RNA sequence/structure prior | RNAsolo2 |

`RNAsolo2` 不能被标成 RNA-target binding 样本，也不能进入 RNA binding evaluation。
RNA aptamer binding 样本必须包含固定 target protein；没有 target 的 RNA 样本不能通过 contract。

正式处理数据前，还必须确定：

- PPIRef/PPIRef50K 选择、版本、split、redundancy filter、target/binder chain 规则
- SAbDab2 版本、H3-only/all-six、split、structure quality filter
- RNA 三个数据源版本、quality filter、split
- Ribocentre/PDB RNA-target 候选数、可用数和排除原因

## 模型

baseline 是独立实现的 small RFD3NA-style unified diffusion model：

```text
Known molecular context + noisy/masked design region
                         ↓
                 Atom-level encoder
                         ↓
              Atom → token downsampling
                         ↓
               Sparse token transformer
                         ↓
              Token → atom upsampling
                         ↓
           Coordinate noise + sequence logits
```

模型只支持本计划需要的 protein 和 RNA，不包含 DNA、small molecule 或其他任务。
task、polymer、role、chain、position、time 和 design mask 都进入同一个共享模型。

当前默认候选模型为 `12,161,949` 参数，处于要求的 5M–20M 范围内。它仍需在目标 GPU
上记录 peak memory 和训练速度，完成后才能冻结为正式容量。

该实现只采用公开 RFD3 的高层 atom/token 组织方式，不复制、不声称完整复现 RFD3。
参考：[RFdiffusion3 paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12458353/) 和
[RosettaCommons Foundry](https://github.com/RosettaCommons/foundry)。

## Evaluation

三个任务不合并成一个科学指标：

| 任务 | 主指标 | 其他要求 |
| --- | --- | --- |
| Protein Binder | In-silico Success Rate | interface confidence、scRMSD、Rosetta ΔG、shape complementarity、clashes、diversity/cluster success |
| Antibody CDR | AAR、CDR RMSD、DockQ | 必报 H3 AAR/RMSD；all-six 时逐个 CDR 报告 |
| RNA Aptamer | scTM、scRMSD、structure confidence | 有 native complex 时报告 RNA-target DockQ |

Binder 的 predictor、filters 和 thresholds 必须在 test evaluation 前冻结，并记录论文或
calibration 来源。`DockQ`、`scTM`、`scRMSD` 不等于 binding affinity；实验 Kd 不属于 v0。

## 快速检查

```bash
python -m pip install -e ".[dev]"

# 查看冻结规格
nanodesign-v0 spec

# 当前配置允许 TBD，但会完整列出 blockers
nanodesign-v0 validate-config --config configs/v0.yaml --allow-tbd

# 检查候选模型参数量
nanodesign-v0 model-summary --config configs/v0.yaml

# 三个任务通过同一个模型完成 forward/backward
nanodesign-v0 smoke --config configs/v0.yaml

pytest
```

去掉 `--allow-tbd` 后，任何尚未确定的数据版本、split、CDR 范围、RNA inventory、
design atom-slot schema、模型 capacity benchmark、外部 evaluator 或 threshold 都会阻止
正式配置通过。

完整冻结规格见 [NanoDesign v0 Specification](docs/V0_SPEC.md)，数据字段见
[Data Contract](docs/DATA_CONTRACT.md)。
