# NanoDesign v0 当前实验结果

**状态时间：2026-09-01 21:21 CDT**  
**实验阶段：早期 baseline validation / training-budget calibration**

> 本报告总结 3K 与 9K `samples seen` 的已完成结果，以及 18K 实验的进行中结果。
> 当前数字主要是固定 validation set 上的 loss、sequence recovery 和固定样本 generation
> diagnostic，并不是最终的 Binder / H3 / RNA 外部 benchmark。

## 1. 结论摘要

- 3K 和 9K budget sweep 已全部完成；9K 包含 seed 17 与 seed 23 两个独立训练重复。
- 当前最稳定的候选是 **`lr5e4-d16-c10`**：9K 时两个 seed 的 H3 recovery
  均约为 21%，H3 已出现明确且可复现的 learning signal。
- Binder 在 9K 的 recovery 约为 8%–10%，只有较弱的 signal。
- RNA 在 9K 的 recovery 为 17%–24%，跨 seed 不稳定，固定生成仍可能出现单碱基偏置。
- 9K 因此可以称为“可运行并有部分 learning signal 的工程 baseline”，但还不能称为
  “三个任务均稳定、非塌缩且通过真实 evaluation 的合格 scientific baseline”。
- 18K 已从 9K checkpoint 原地继续。两个 D4 配置已完成 18K training；D16 配置仍在运行。

## 2. 冻结的实验设置

本轮实验没有改变 task、dataset、split、model architecture、loss 或 evaluation protocol。

| 项目 | 当前设置 |
| --- | --- |
| Model | NanoDesign-Tiny，pinned RFD3NA / RFD3 architecture |
| Parameter count | **6,849,538** |
| Tasks | Protein Binder / Antibody H3 / RNA binding |
| Task sampling | 1:1:1，允许重复采样 |
| Budget unit | Global `samples seen` |
| Per-GPU complex batch | 1 |
| Validation | 每个 task 固定 16 个 samples |
| Optimizer | AdamW，betas `(0.9, 0.95)`，weight decay `1e-4` |
| Gradient clipping | 10.0 |
| EMA | 0.999 |
| Hardware | 每个实验单张 H100；最多 8 个实验并行 |
| Seeds | 3K screening 使用 17；9K/18K 使用 17 和 23 |

当前 sweep 使用的冻结 task split 数量如下：

| Task | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| Protein Binder | 40,883 | 5,110 | 5,110 |
| Antibody H3 | 3,878 | 438 | 984 |
| RNA binding | 2,117 | 83 | 88 |

RNAsolo2 是 RNA structural prior，不在这里作为 RNA-target binding ground truth 统计。

## 3. 3K screening 结果

以下是 seed 17、3,000 samples seen 的固定 validation 结果：

| Experiment | Wall time (s) | Binder loss | Binder recovery | H3 loss | H3 recovery | RNA loss | RNA recovery |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lr2e4-d4-c10` | 2,319.1 | 2.0449 | 3.99% | 0.6727 | 2.11% | 1.2486 | 22.87% |
| `lr5e4-d4-c10` | 2,325.8 | 1.9038 | 5.49% | 0.6371 | 9.53% | 1.1679 | 24.30% |
| `lr1e3-d4-c10` | 2,318.7 | 1.9916 | 0.00% | 0.7141 | 0.00% | 1.1660 | 22.61% |
| `af3-d4-c10` | 2,341.3 | 1.8539 | 3.69% | 0.6473 | 4.71% | 1.0648 | 22.24% |
| `lr2e4-d16-c10` | 4,132.0 | 1.9903 | 9.38% | 0.6527 | 7.30% | 1.2498 | 22.14% |
| `lr5e4-d16-c10` | 4,116.5 | 1.9015 | 9.37% | 0.6520 | 7.17% | 1.1776 | 17.74% |
| `lr1e3-d16-c10` | 4,113.9 | 1.8868 | 7.97% | 0.6531 | 8.89% | 1.1695 | 17.87% |
| `af3-d16-c10` | 4,117.9 | 1.8949 | 5.56% | 0.6655 | 2.70% | 1.1103 | 18.17% |

3K 相比 900 samples 已出现更明确的 loss 改善，但 generation 仍有显著残基偏置。因此保留
四个候选进入 9K：`lr5e4-d4-c10`、`af3-d4-c10`、`lr5e4-d16-c10`
和 `lr1e3-d16-c10`。

更早的 3K 运行记录见
[`data/runs/signal-maskfix-matrix-3000-seed17/RESULTS.md`](../data/runs/signal-maskfix-matrix-3000-seed17/RESULTS.md)。

## 4. 9K 双 seed 结果

### 4.1 每个 seed 的 validation 结果

表中每个 task 以 `loss / recovery` 表示。

| Seed | Experiment | Binder | H3 | RNA | Optimization wall time |
| ---: | --- | ---: | ---: | ---: | ---: |
| 17 | `lr5e4-d4-c10` | 1.7023 / 7.14% | 0.6253 / 11.07% | 1.0339 / 20.40% | 2.07 h |
| 17 | `af3-d4-c10` | 1.6204 / 7.64% | 0.6195 / 9.03% | 0.8938 / 22.63% | 2.05 h |
| 17 | `lr5e4-d16-c10` | 1.6807 / 7.88% | 0.6070 / 20.77% | 1.0365 / 17.24% | 3.69 h |
| 17 | `lr1e3-d16-c10` | 1.6385 / 6.02% | 0.6062 / 12.88% | 0.9270 / 20.28% | 3.67 h |
| 23 | `lr5e4-d4-c10` | 1.7377 / 8.80% | 0.6835 / 8.24% | 0.8618 / 24.46% | 2.10 h |
| 23 | `af3-d4-c10` | 1.5021 / 9.59% | 0.6508 / 4.02% | 0.7440 / 23.59% | 2.10 h |
| 23 | `lr5e4-d16-c10` | 1.6084 / 9.63% | 0.6437 / 21.09% | 0.8078 / 24.34% | 3.77 h |
| 23 | `lr1e3-d16-c10` | 1.6411 / 10.69% | 0.6650 / 18.17% | 0.7596 / 20.57% | 3.75 h |

### 4.2 跨 seed recovery 均值

| Experiment | Binder recovery | H3 recovery | RNA recovery |
| --- | ---: | ---: | ---: |
| `lr5e4-d4-c10` | 7.97% | 9.66% | 22.43% |
| `af3-d4-c10` | 8.62% | 6.53% | **23.11%** |
| `lr5e4-d16-c10` | **8.76%** | **20.93%** | 20.79% |
| `lr1e3-d16-c10` | 8.36% | 15.53% | 20.43% |

`lr5e4-d16-c10` 是当前最稳定候选，主要原因是 H3 recovery 在两个 seed 上都约为 21%。
但它的 RNA recovery 和 generation 稳定性仍不够。

### 4.3 固定样本 generation diagnostic

下面只展示当前候选 `lr5e4-d16-c10`。`Max residue fraction` 是生成设计区域中出现最多的
单一残基比例；它用于发现塌缩，不是新增的正式 evaluation metric。

| Seed | Weights | Binder rec / max | H3 rec / max | RNA rec / max |
| ---: | --- | ---: | ---: | ---: |
| 17 | EMA | 15.4% / 88.5% | 36.4% / 81.8% | 33.3% / 42.9% |
| 17 | Online | 15.4% / 73.1% | 27.3% / 54.5% | 19.0% / **95.2%** |
| 23 | EMA | 15.4% / 73.1% | 36.4% / 63.6% | 28.6% / 76.2% |
| 23 | Online | 15.4% / 50.0% | 27.3% / 45.5% | 23.8% / 52.4% |

seed 17 online RNA 中单一碱基占 95.2%，说明 9K 仍存在明显的 seed-dependent collapse。
因此不能仅凭 validation loss 下降把它称为合格 baseline。

## 5. 18K 进行中结果

18K 不是从头重跑，而是从完全相同配置的 9K checkpoint 原地继续。

### 5.1 已完成的 D4 validation

| Seed | Experiment | Binder loss / rec | H3 loss / rec | RNA loss / rec |
| ---: | --- | ---: | ---: | ---: |
| 17 | `lr5e4-d4-c10` | 1.5784 / 8.71% | 0.5967 / 12.28% | 0.9068 / 22.34% |
| 17 | `af3-d4-c10` | 1.4959 / 8.21% | 0.5980 / 13.17% | 0.8310 / 26.07% |
| 23 | `lr5e4-d4-c10` | 1.7249 / 9.21% | 0.6773 / 16.05% | 0.8253 / 26.11% |
| 23 | `af3-d4-c10` | **1.3833 / 8.42%** | 0.6247 / 11.07% | **0.7283 / 26.89%** |

18K D4 validation 中 RNA recovery 已接近或略高于四字母均匀随机的 25%，但固定 generation
仍可能塌缩。例如 seed 23 的 `af3-d4-c10` EMA RNA 是单碱基序列。因此这仍是部分结果，
不能视为 RNA task 已通过。

### 5.2 当前运行进度

截至本报告状态时间：

- seed 17 D4：18K training 完成，generation 进行中。
- seed 23 D4：18K training 与 generation 完成。
- seed 17 D16：约 14,456–14,459 / 18,000 samples。
- seed 23 D16：约 14,075 / 18,000 samples。
- 所有 18K resume 日志均无 traceback、OOM 或配置错误。

## 6. 为什么当前仍不是最终合格 baseline

当前结果已经证明训练链路可运行，而且 H3 有可复现 learning signal；但按照 NanoDesign v0
的目标，还缺少以下条件：

1. **三个 task 都要有明确 signal。** H3 已满足；Binder 较弱；RNA 跨 seed 不稳定。
2. **生成不能明显塌缩。** 9K 和部分 18K generation 仍出现 90%–100% 单残基/单碱基占比。
3. **需要跨 seed 可复现。** 同一配置在 seed 17 与 23 的 RNA generation 差异仍很大。
4. **需要真实 task evaluation。** 当前尚未对最终候选完整运行 Binder In-silico Success
   Rate、H3 RMSD/DockQ、RNA scTM/scRMSD/DockQ。
5. **固定样本 generation 不是正式 benchmark。** 当前 generation diagnostic 每个 task 仅使用
   一个固定 test sample，不能替代完整 test evaluation。

## 7. 训练可靠性记录

- 3K D4 实验曾在 sample 1,803 因 CUDA allocator fragmentation OOM；四组都有 step-1,750
  checkpoint，随后启用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 精确恢复并完成。
- D16 在同一位置前主动从 step 1,750 恢复，避免重复 OOM。
- 该修复不改变数据、模型、loss 或训练顺序。
- 9K → 18K 使用安全 milestone extension：旧 milestone 必须是新列表的严格前缀、checkpoint
  必须位于旧预算终点、其余 training config 必须完全一致。
- 完整测试结果：**54 passed, 1 skipped**。
- 对应代码提交：`9d5d837` (`allow safe training budget extensions`)。

## 8. 原始产物

- 3K matrix：[`data/runs/signal-maskfix-matrix-3000-seed17`](../data/runs/signal-maskfix-matrix-3000-seed17/)
- 9K/18K matrix：[`data/runs/signal-maskfix-matrix-9000-seeds17-23`](../data/runs/signal-maskfix-matrix-9000-seeds17-23/)
- Frozen data statistics：[`docs/data_v0_stats.json`](data_v0_stats.json)
- Model smoke report：[`docs/model_smoke.json`](model_smoke.json)

## 9. 当前下一步

1. 等待 18K D16 双 seed training 和 generation 完成。
2. 比较 9K → 18K 是否真正改善三个 task，而不是只降低 loss。
3. 只有非塌缩候选才进入完整 Binder / H3 / RNA 外部 evaluator。
4. 根据三个 task 的 learning signal 与 GPU-hour 成本，选择最小有效固定 budget。
