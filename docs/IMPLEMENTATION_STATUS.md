# NanoDesign v0 Implementation Status

## 已完成：infra

| 模块 | 状态 | 当前能力 |
| --- | --- | --- |
| Frozen spec | 完成 | 只允许三个任务，并固定各自 context/design region |
| Unified contract | 完成 | Protein/RNA 独立 token、atom/token 映射、role 与 mask 校验 |
| Data registry | 完成 contract | 固定六个允许来源及其 task/purpose，不包含真实 converter |
| Manifest | 完成 | 来源版本、split、cluster leakage、SHA-256 audit |
| RNA inventory | 完成 | 强制统计两个 binding source 的候选数、可用数和排除原因 |
| Dataset IO | 完成 | 校验后的 NPZ round-trip、manifest-backed dataset、collation |
| NanoDesign-Tiny | 完成 baseline | 一个共享的 atom/token transformer；默认 12,161,949 参数 |
| Unified diffusion | 完成 baseline | 只对 design region 做坐标/序列加噪，context 保持固定 |
| Evaluation contract | 完成 | 三套独立指标、H3/all-six 与 RNA DockQ 条件校验 |
| Training plumbing | 完成 | 共享 train step、task macro loss、带 provenance 的 checkpoint |
| Tests | 完成 | contract、data、model、diffusion、evaluation、checkpoint 门禁 |

“完成 baseline”表示代码路径可运行，不表示模型已经训练，也不表示科学效果已经验证。

## 尚未开始：需要先做决定

1. 选定 PPIRef/PPIRef50K 与版本，冻结 chain assignment、去重和 split。
2. 选定 SAbDab2 版本、H3-only/all-six、quality filter 和 split。
3. 对 Ribocentre/PDB RNA-target 数据做 inventory，得到真正可用的 complex 数量。
4. 根据 inventory 冻结 RNA data pool，同时把 RNAsolo2 保持为 prior-only。
5. 在目标 GPU 上测试默认候选模型的 step time 与 peak memory，再冻结模型容量。
6. 冻结不会泄露 native sequence 的 protein/RNA design atom-slot schema。
7. 冻结三个任务的外部 evaluator、版本、参数、success filters 和 threshold 来源。

这些决定完成后，才实现 source-specific converters、正式训练脚本和外部 evaluation runners。

## 推荐执行顺序

```text
现在：infra contract 已搭好
  ↓
Data decision + RNA inventory
  ↓
Source-specific converters + frozen manifests
  ↓
GPU capacity benchmark
  ↓
Formal training
  ↓
Frozen task-specific evaluation
```

任何一步如果需要改变三任务定义、数据边界、共享模型原则或指标集合，都必须升级 spec，
不能在同一个 `nanodesign.v0` 配置下静默修改。
