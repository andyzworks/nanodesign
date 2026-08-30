# NanoDesign v0 Implementation Status

## P0 status

| P0 | 状态 | 证据 |
| --- | --- | --- |
| Real data | 完成 | 真实 catalog、MMseqs2/SAbDab2 cluster-disjoint splits、SHA256 与实际数量见 `data_v0_stats.json` |
| Credible baseline | 完成代码与 smoke | 官方 Foundry RFD3NA commit 固定；6,849,538 参数；真实 atom23 forward/backward/EDM generation 已运行 |
| Real evaluation | 完成执行器；重量级依赖需部署 | DockQ/US-align/H3 protocol 已实跑；ColabFold、RhoFold+、Rosetta runner 拒绝手填 metric |

## 尚未声称完成的事情

- 尚未完成正式多 GPU training，也没有 trained checkpoint 或科学性能结论。
- 尚未在本机跑完整 1,000 × 2 binder benchmark；需要 ColabFold weights、Rosetta
  licensed installation 和 GPU。
- 尚未得到实验 Kd；scTM、scRMSD、DockQ 不等于 binding affinity。

本轮没有新增 Agent、RSI、leaderboard 或第四个 task。
