# NanoDesign

## What is NanoDesign?

**NanoDesign is a nano-scale testbed for unified biomolecular design. A single small
model is trained to solve three design tasks: protein binder design, antibody CDR-H3
design, and RNA binding/aptamer design.**

Biomolecular design is often studied separately for proteins, antibodies, and RNA.
NanoDesign provides a unified, compact setting that can be trained frequently from
scratch. NanoDesign v0 fixes the tasks, data pools, baseline model, and evaluation
protocols to establish a clear and reproducible reference baseline.

## Tasks

| Task | Input | Output |
| --- | --- | --- |
| Protein Binder Design | target protein | binder sequence + structure |
| Antibody CDR-H3 Design | antigen + fixed antibody framework | CDR-H3 sequence + structure |
| RNA Binding / Aptamer Design | target protein | RNA sequence + structure |

All three tasks use the same formulation:

```text
known molecular context → design sequence + structure
```

## NanoDesign v0 at a Glance

| Task | Data | Baseline Model | Evaluation |
| --- | --- | --- | --- |
| Protein Binder | PPIRef / PPIRef50K | NanoDesign-Tiny | In-silico Success Rate + frozen binder metrics |
| Antibody H3 | SAbDab2 | same model | H3 AAR + H3 RMSD + DockQ |
| RNA | Ribocentre + frozen PDB RNA–protein pool + RNAsolo2 structural prior | same model | scTM + scRMSD + RNA-target DockQ |

The complete frozen v0 definition is in the [v0 specification](docs/V0_SPEC.md).

## Current Validation Evidence

NanoDesign-Tiny has completed controlled 3K, 9K, and 18K `samples seen` sweeps. The
9K and 18K experiments use two independent seeds. The current best unified candidate
is **`lr5e4-d16-c10 @ 18K`**: a constant `5e-4` learning rate, 16 diffusion
realizations per complex, gradient clipping at 10, and the unchanged `1 : 1 : 1`
task mixture.

| Task | 18K validation recovery (two-seed mean) | Training-set majority baseline | Evidence |
| --- | ---: | ---: | --- |
| Protein Binder | **10.29%** | 8.96% | modest learning signal |
| Antibody H3 | **18.55%** | 14.93% | clearest reproducible learning signal |
| RNA binding | **22.58%** | 27.55% | not yet above the simple baseline |

For the same candidate, fixed-sample online generations at 18K contain multiple valid
residue types for every task under both seeds; their largest single-residue fractions
range from 26.9% to 52.4%, with no homopolymer output. This is a substantial improvement
over the collapsed generations observed at earlier budgets and verifies that the shared
model can train and generate across all three task paths. It does **not** yet establish
biological design success: sequence recovery and one fixed generation per task do not
replace the frozen external evaluation protocols below.

The complete per-seed results, generation diagnostics, limitations, and original run
locations are recorded in the
[current NanoDesign v0 results report](docs/NANODESIGN_V0_RESULTS_2026-09-01.md).

## Data

NanoDesign v0 uses release `nanodesign-v0-data-2026-08-30`. The counts below are
samples that passed preprocessing and structural filtering, not source-dataset totals.

| Pool | Train | Validation | Test | Total |
| --- | ---: | ---: | ---: | ---: |
| Protein Binder | 40,883 | 5,110 | 5,110 | 51,103 |
| Antibody H3 | 3,878 | 438 | 984 | 5,300 |
| RNA binding | 2,117 | 83 | 88 | 2,288 |
| RNAsolo2 structural prior (auxiliary) | 419 | 234 | 229 | 882 |

- **Protein Binder** uses the frozen PPIRef50K
  `ppiref_6A_filtered_clustered_04` pool. The longer resolved chain is the target and
  the other chain is the binder. Protein-homology components are split across
  train/validation/test.
- **Antibody H3** uses the SAbDab2 ML dataset 0.1.0. The antigen and antibody framework
  are fixed; NanoDesign v0 designs CDR-H3 only. The split preserves the official
  antigen-aware test partition.
- **RNA binding** combines 33 Ribocentre `true_aptamer` examples with 2,255 experimental
  PDB `general_rna_protein_interaction` examples. Ordinary PDB RNA–protein complexes are
  not described as validated aptamers. The 882 RNAsolo2 structures are
  `rna_structural_prior` auxiliary data and are not aptamer binding ground truth.

Exact manifest SHA-256 values and split protocols are recorded in
[frozen data statistics](docs/data_v0_stats.json). Source-specific filtering and
rejection counts are available for [PPIRef](docs/data_reports/ppiref50k.json),
[SAbDab2](docs/data_reports/sabdab2.json),
[Ribocentre](docs/data_reports/ribocentre.json),
[PDB RNA–protein](docs/data_reports/pdb_rna_target.json), and
[RNAsolo2](docs/data_reports/rnasolo2.json).

## Baseline Model

**NanoDesign-Tiny** directly instantiates the pinned RosettaCommons Foundry
`rfd3na.model.RFD3.RFD3` implementation at commit
`aad357b776e3c0d6b973080f8f8c4bcf3ed21e40`. It is a reduced RFD3NA/RFD3 configuration,
not a separately invented Cartesian Transformer.

The current configuration has **6,849,538 parameters** (approximately 6.85M). The same
architecture and input pipeline support Protein Binder, Antibody H3, and RNA design.
Its purpose is to be a small, credible reference model that can be retrained often,
not to maximize state-of-the-art performance. Model configuration is frozen in
[`configs/v0.yaml`](configs/v0.yaml), and the three-task model smoke report is in
[`docs/model_smoke.json`](docs/model_smoke.json).

## Evaluation

NanoDesign does not force the three tasks into one metric. Each task uses its frozen,
domain-specific computational evaluation protocol.

### Protein Binder

The primary metric is **In-silico Success Rate**. Generated binders are independently
predicted with ColabFold `alphafold2_multimer_v3`, then evaluated using the frozen
BindCraft-style filters and Rosetta InterfaceAnalyzer. Reported auxiliary metrics are
interface confidence, self-consistency RMSD, Rosetta interface ΔG, shape
complementarity, clashes, diversity, and cluster-level success. Thresholds and the
generation budget remain fixed by the v0 protocol.

### Antibody H3

- H3 AAR
- framework-aligned H3 RMSD
- DockQ

The evaluator aligns the fixed antibody framework before computing H3 backbone RMSD;
alignment is not delegated to the caller.

### RNA

- scTM
- scRMSD
- RNA-target DockQ

RNA sequences are independently refolded with RhoFold+, compared with US-align, and
the RNA–target interface is evaluated with DockQ.

**Computational structural/interface metrics are not equivalent to experimentally
measured binding affinity.** NanoDesign v0 does not claim experimental affinity or Kd.

## Training Protocol and Budget

The model architecture and data pools are fixed. Multi-task sampling uses the frozen
Protein Binder : Antibody H3 : RNA ratio of `1 : 1 : 1`, and training budget is measured
by global **samples seen**, not by epochs.

The completed calibration compared checkpoints at 3K, 9K, and 18K samples seen.
**18K is the current early-baseline budget candidate**: it gives the strongest overall
learning evidence while remaining practical for repeated experiments. It is not yet
the official scientific benchmark budget because full multi-sample generation and
external evaluation remain outstanding. NanoDesign does not claim a 36K result.

## Current Status

- [x] **Data pools and manifests:** frozen, cluster-disjoint, counted, and checksummed.
- [x] **NanoDesign-Tiny:** implemented with the pinned RFD3 class; all three tasks have
  passed forward, backward, generation, and checkpoint save/load smoke tests.
- [x] **Evaluation components:** Binder, H3, and RNA end-to-end runners are implemented
  and computationally smoke-tested. A complete formal benchmark has not been reported.
- [x] **Learning-signal calibration:** 3K, 9K, and 18K sweeps are complete. At 18K,
  Binder and H3 recovery exceed their training-set majority baselines; RNA does not.
- [x] **Training performance:** feature caching, asynchronous loading, standard/chunked
  execution, DDP, preflight checks, and checkpoint/resume are implemented and tested.
- [x] **Early budget candidate:** 18K samples seen is selected for the current unified
  candidate based on two-seed validation and fixed-sample generation diagnostics.
- [ ] **Scientific baseline evaluation:** full multi-sample generation and the frozen
  Binder/H3/RNA external evaluations have not yet been reported.

See the [training profiler](docs/training_profile_h100_batch4.md),
[standard/chunked equivalence report](docs/training_mode_equivalence_h100.md), and
[pilot record](docs/baseline_pilot.json), and
[18K results report](docs/NANODESIGN_V0_RESULTS_2026-09-01.md) for the current evidence.

## Quickstart

### 1. Install

NanoDesign requires Python `>=3.12,<3.13`. The RFD3NA dependency is installed from the
pinned Foundry commit.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[data,evaluation,model,dev]'
```

For the frozen H100 environment, install CUDA-enabled PyTorch 2.7.1 with CUDA 12.8 and
verify CUDA before training:

```bash
python -m pip install --force-reinstall 'torch==2.7.1' \
  --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; assert torch.cuda.is_available()"
```

ColabFold, Rosetta/PyRosetta, RhoFold+, US-align, and DockQ are external evaluation
runtimes. Rosetta/PyRosetta must be installed under its applicable license.

### 2. Prepare and split data

Stage the source archives and structures under their default `data/raw/...` paths, then
run the frozen converters and splitter:

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

Build integrity-checked model-ready feature caches for training and validation:

```bash
python scripts/build_v0_feature_cache.py \
  --catalog data/processed/v0/splits/protein_binder/train.jsonl \
  --catalog data/processed/v0/splits/protein_binder/validation.jsonl \
  --catalog data/processed/v0/splits/antibody_h3/train.jsonl \
  --catalog data/processed/v0/splits/antibody_h3/validation.jsonl \
  --catalog data/processed/v0/splits/rna_binding/train.jsonl \
  --catalog data/processed/v0/splits/rna_binding/validation.jsonl \
  --cache-root data/cache/v0 \
  --manifest docs/data_v0_stats.json \
  --max-context-tokens 384 \
  --diffusion-batch-size 4 \
  --report runs/cache-report.json
```

### 3. Test the three-task path

```bash
pytest
python scripts/smoke_rfd3na_real.py --sampling-steps 2
```

The smoke test exercises forward, backward, generation, and checkpoint save/load for
all three tasks using real processed samples.

### 4. Train

An example four-GPU samples-seen training command is:

```bash
torchrun --standalone --nproc_per_node=4 scripts/train_v0.py \
  --milestone-samples 3000,9000,18000 \
  --seed 7 \
  --validation-samples-per-task 16 \
  --diffusion-batch-size 4 \
  --feature-cache-root data/cache/v0 \
  --data-workers 4 \
  --data-prefetch-factor 4 \
  --output-dir runs/budget-sweep-seed7
```

Use `--resume PATH` to resume the same run from a saved checkpoint. The command is an
example of the supported four-GPU path; the reported candidate above was screened as
independent single-GPU runs under two seeds, not produced by this exact command.

### 5. Generate and evaluate

Generate one evaluator-ready sample per task from a saved milestone:

```bash
python scripts/generate_milestone.py \
  --checkpoint runs/budget-sweep-seed7/milestones/samples-00003000.pt \
  --samples-seen 3000 \
  --device cuda \
  --output-root runs/budget-sweep-seed7/generations
```

The task-specific evaluation entry points are:

```text
nanodesign-v0 evaluate-protein-binder --help
python scripts/evaluate_antibody_h3.py --help
python scripts/evaluate_rna.py --help
```

They consume generated structures and milestone metadata and invoke the frozen external
evaluation tools; they do not accept manually supplied metrics as evaluation results.
