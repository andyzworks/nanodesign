# NanoDesign v0 Data Contract

## Unified example

每个样本保存两层信息：

- token 层：sequence token、polymer、role、chain、residue index、design mask
- atom 层：坐标、element、有效 mask，以及 atom 属于哪个 token

三类 binding-design 样本只允许以下 role：

| Task | Fixed role | Design role |
| --- | --- | --- |
| Protein binder | `target` | `binder` |
| Antibody CDR | `antigen` + `antibody_framework` | `cdr` |
| RNA aptamer | protein `target` | `rna_aptamer` |

RNAsolo2 样本使用 `rna_structure_prior` purpose，只含 RNA，不得伪造 target。

## Manifest

manifest 是 JSONL；每一行对应一个 NPZ 样本。Binding-design record 必须包含
`complex_cluster_id`、`target_cluster_id` 和 `design_cluster_id`。所有 cluster 都必须跨
train/validation/test 隔离。

```json
{"schema_version":"nanodesign.v0","sample_id":"example","task":"rna_aptamer","source":"pdb_rna_target_complex","source_version":"frozen-version","purpose":"binding_design","split":"train","path":"samples/example.npz","design_cluster_id":"rna-1","complex_cluster_id":"complex-1","target_cluster_id":"protein-1","structure_cluster_id":null}
```

RNAsolo2 prior record 不允许 `complex_cluster_id` 或 `target_cluster_id`，但必须提供
`structure_cluster_id`。

## RNA usable-complex inventory

冻结 RNA pool 前，inventory JSON 必须恰好包含 Ribocentre Aptamer 和 PDB RNA-target
两个来源。每个来源记录：

- frozen source version
- 候选 complex 数
- 可用 complex 数
- 互斥的排除原因与数量
- 筛选 protocol

`usable + exclusive rejections` 必须严格等于 candidate count。CLI 会输出规范化内容的
SHA-256；该指纹和路径随后写入 `configs/v0.yaml`。

```bash
nanodesign-v0 validate-rna-inventory --inventory path/to/inventory.json
nanodesign-v0 validate-manifest --manifest path/to/manifest.jsonl
```

当前 source adapters 是 contract-only。真实 converter 必须等相应版本、filter、split 和
chain/CDR 规则冻结后再实现。
