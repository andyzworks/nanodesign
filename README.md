# NanoDesign v0

NanoDesign v0 只做三件事：protein binder、antibody CDR-H3 和 RNA aptamer 的
sequence + structure design。三个任务共用同一个模型，不增加 Agent、RSI、leaderboard
或其他外围系统。

## 当前完成结果

| Task | Real Data Size（train / val / test） | Baseline | Real Evaluation |
| --- | ---: | --- | --- |
| Protein Binder | 40,883 / 5,110 / 5,110 | NanoDesign-Tiny | Success Rate + AF2/界面指标 |
| Antibody H3 | 3,878 / 438 / 984 | 同一个模型 | H3 AAR + framework-aligned RMSD + DockQ |
| RNA task | 2,117 / 83 / 88 | 同一个模型 | RhoFold+ scTM/scRMSD + RNA-target DockQ |

另有 RNAsolo2 structure-prior auxiliary data：419 / 234 / 229。它不属于 aptamer
binding ground truth，也不计入上表的 RNA Aptamer 数量。

这些是本仓库实际下载并通过结构过滤后的数量，不是 adapter 的理论数量。完整、带 SHA256
的 split 清单见 [data_v0_stats.json](docs/data_v0_stats.json)，各来源的过滤和拒绝统计见
[data_reports](docs/data_reports)。

## 真实数据

- Protein Binder：PPIRef50K `ppiref_6A_filtered_clustered_04`，官方 51,755 个候选；
  51,103 个可用。较长 resolved chain 固定为 target，另一条为 binder。
- Antibody H3：SAbDab2 ML dataset 0.1.0，15,641 个官方结构；5,300 个 holo、
  protein/peptide-antigen、完整 IMGT H3 样本可用。v0 只设计 H3。
- RNA task pool 明确保留三种数据语义：Ribocentre 的 33 个 contact components 是
  `true_aptamer`；RCSB PDB 的 2,255 个非重复实验结构是
  `general_rna_protein_interaction`，不能称为已验证 aptamer；RNAsolo2 的 882 个代表结构是
  `rna_structural_prior`。三者仍服务于同一个既定 RNA task，没有增加新 task。

Split 不是随机逐样本切分：PPIRef 使用 MMseqs2 30% identity/80% coverage，并对 target
和 binder 的 homology component 做 80/10/10；SAbDab2 保留官方 antigen-aware test，
从官方 train cluster 中固定 10% 作 validation；RNA 同时按 target protein 30% 和 RNA 80%
聚类，并把 RNAsolo2 纳入 leakage component 后再切分。

生成数据的命令：

```bash
python scripts/prepare_v0_data.py --repo-root . ppiref --workers 8
python scripts/prepare_v0_data.py --repo-root . sabdab2 --workers 8
python scripts/prepare_v0_data.py --repo-root . ribocentre \
  --structures-json data/raw/ribocentre/structures_merged.json --workers 8
python scripts/prepare_v0_data.py --repo-root . rnasolo2 --workers 8
python scripts/prepare_pdb_rna.py --repo-root . \
  --ribocentre-ids data/raw/ribocentre/protein_target_pdb_ids.txt --download-workers 8
python scripts/split_v0_data.py --repo-root .
```

## 可信 baseline model

`NanoDesignTiny` 直接实例化 RosettaCommons Foundry 的公开
`rfd3na.model.RFD3.RFD3`，固定 commit
`aad357b776e3c0d6b973080f8f8c4bcf3ed21e40`。仓库没有重新发明 Cartesian
Transformer，也没有复制一套“类似 RFD3NA”的 geometry block。

缩小的只有 channel、Pairformer/diffusion transformer block、atom attention depth、
recycling 和 EDM sampling steps。默认模型实测为 **6,849,538 parameters**。它保留官方
atom attention、token initializer、Pairformer、local diffusion transformer、atom decoder、
EDM loss/sampler，以及 RFD3NA 对 3D geometry 的处理。

训练输入使用官方 atom23 layout。未知设计序列使用 `UNK` protein scheme 或 `X` RNA
scheme，只暴露 sequence-independent backbone/CB 或 phosphate-ribose slots；不会根据 native
residue 提前创建 residue-specific side-chain/base atoms。完整复合物保留在 catalog，训练时
围绕 design region 选最近的固定 context tokens。

模型依赖要求 Python 3.12：

```bash
python3.12 -m pip install -e '.[model,data]'
```

GPU 训练环境必须另外安装 CUDA-enabled PyTorch；普通 PyPI 在无 CUDA 的安装节点上可能
解析成 CPU-only wheel。本次冻结环境使用与 Foundry 兼容的 PyTorch 2.7.1 + CUDA 12.8：

```bash
python3.12 -m pip install --force-reinstall 'torch==2.7.1' \
  --index-url https://download.pytorch.org/whl/cu128
python3.12 -c "import torch; assert torch.backends.cuda.is_built()"
```

正式 Slurm 命令还会在训练前强制检查 `torch.cuda.is_available()`，避免 CPU wheel 静默占用
GPU allocation。

## 真实 evaluation

- Binder：ColabFold `alphafold2_multimer_v3` 作独立结构验证；Rosetta
  `InterfaceAnalyzer` 计算 ΔG、shape complementarity、dSASA 和界面 H-bond；success filters
  固定自 BindCraft public defaults。预算固定为每 target 1,000 backbones × 2 sequences。
- Antibody H3：代码强制先按 heavy-chain 非 H3 framework（加可用 light framework）Kabsch
  对齐，再算 H3 backbone RMSD；AAR 与 DockQ v2 直接从结构运行，不允许 caller 改 alignment。
- RNA：RhoFold+ 独立 refold，US-align RNA 计算 scTM/scRMSD，DockQ v2 计算 native
  RNA–target interface。

本地已实际运行 DockQ 2.1.3、US-align v20260527、RhoFold+、官方 quarterly PyRosetta
InterfaceAnalyzer，以及官方 RFD3NA forward/backward/50-step EDM generation。ColabFold 使用
集群的 1.5.5 module；代码不会在任何外部工具缺失时接受手填 metric 冒充真实结果。PyRosetta
受 Rosetta non-commercial license 约束，不作为 CI 的自动下载依赖。

第一次真实三任务 pilot（12 steps、D=4、seed 7）已完成，三个 task 的固定 validation
coordinate/sequence/total loss 均下降，三条 generation 均写出有限 PDB；精确运行记录见
[baseline_pilot.json](docs/baseline_pilot.json)。这只是训练链路和 learning-signal 验证，不冒充
正式 baseline result。

`scTM`、`scRMSD` 和 `DockQ` 是 computational structure/interface evaluation，**不等于
真实 binding affinity**。NanoDesign v0 不声称得到实验 Kd。

## 测试

```bash
python -m pip install -e '.[data,evaluation,dev]'
pytest
```
