# NanoDesign Progress — 2026-09-05

Last updated: 2026-09-05 16:51 CDT

## Executive Summary

NanoDesign has completed and passed Stage 0 (baseline freeze), Stage 1 (evaluator
audit), and Stage 2 (32-example learnability). The current 6,849,538-parameter
RFD3NA model can jointly optimize sequence and structure for Protein Binder,
Antibody CDR-H3, and RNA Binding Design. On the larger frozen 128-example panels,
Binder and H3 have passed their task-level Stage-3 gates. RNA shows a strong,
continuous loss/recovery learning curve. Its 12K checkpoint resolves the residual
generation collapse seen at 6K, but target-sequence-shuffle sensitivity remains too
weak for the stated gate; Stage 3 therefore remains **RUNNING**, not PASS.

| Stage | Status | Current evidence |
| --- | --- | --- |
| Stage 0 — Freeze Baseline | **PASS** | Model, recipe, initialization, 9K and 18K references frozen |
| Stage 1 — Evaluator Audit | **PASS** | Deterministic evaluator, semantic audit and sanity ordering complete |
| Stage 2 — 32-Sample Learnability | **PASS** | All three tasks memorize, use context and generate without collapse at selected budgets |
| Stage 3 — 128-Sample Validation | **RUNNING** | Binder and H3 pass; RNA continues beyond 6K |
| Stages 4–8 | **NOT STARTED** | Strictly gated on Stage 3 PASS |

The compact authoritative state remains
[`docs/PROGRESS.md`](../docs/PROGRESS.md). This dated file is a progress snapshot,
not a replacement for the stage gate documents.

## Frozen NanoDesign Setup

- Architecture: pinned public RosettaCommons RFD3NA `RFD3`, commit
  `aad357b776e3c0d6b973080f8f8c4bcf3ed21e40`.
- Parameter count: **6,849,538**.
- One architecture supports all three declared tasks.
- Design sequence is fully masked; fixed molecular context remains visible.
- Sequence supervision is design-only and normalized per active design token.
- Loss is joint: coordinate weight `4.0`, sequence weight `0.1`.
- Optimizer: AdamW, betas `(0.9, 0.95)`, weight decay `1e-4`.
- Learning rate: constant `5e-4`; gradient clipping: global norm `10`.
- Coordinate diffusion uses the public EDM distribution with `sigma_data=16` and
  coordinate augmentation enabled.
- EMA: `0.999`; validation and generation use EMA weights.
- Controlled-stage seed: `17`; official-family generation uses 50 diffusion steps.

The complete comparison with the public recipe is in
[`docs/RFD3NA_RECIPE_AUDIT.md`](../docs/RFD3NA_RECIPE_AUDIT.md).

## Frozen Data Pools

Release: `nanodesign-v0-data-2026-08-30`.

| Pool | Train | Validation | Test | Total |
| --- | ---: | ---: | ---: | ---: |
| Protein Binder / PPIRef50K | 40,883 | 5,110 | 5,110 | 51,103 |
| Antibody H3 / SAbDab2 | 3,878 | 438 | 984 | 5,300 |
| RNA binding | 2,117 | 83 | 88 | 2,288 |
| RNAsolo2 structural prior | 419 | 234 | 229 | 882 |

RNA binding contains 33 Ribocentre `true_aptamer` examples and 2,255 PDB
`general_rna_protein_interaction` examples. RNAsolo2 remains an
`rna_structural_prior`; neither ordinary PDB complexes nor RNAsolo2 are mislabeled
as experimentally validated aptamers. Exact hashes and filtering statistics are in
[`docs/data_v0_stats.json`](../docs/data_v0_stats.json).

## Evaluator Result

The evaluator audit is **PASS**:

- fixed panels, seeds, diffusion timesteps, coordinate noise and corruption;
- deterministic `model.eval()` execution;
- continuous total loss, raw sequence CE, recovery and coordinate loss;
- learnability evaluation separated from true-generation evaluation;
- three identical reruns of the same checkpoint had maximum metric difference
  exactly `0.0`;
- Perfect > Perturbed > Broken sanity ordering passed for Binder, H3 and RNA.

The original evaluator used an inconsistent deposited-PDB coordinate frame. After
matching the deterministic training frame, the same Binder 12K checkpoint changed
from 10.14% recovery to 44.86%, identifying an evaluator false negative rather than
a model-capacity failure. Details are in
[`docs/EVALUATOR_AUDIT.md`](../docs/EVALUATOR_AUDIT.md).

## Earlier Unified Baseline Evidence

The frozen historical unified model uses Binder : H3 : RNA = `1 : 1 : 1`.
Deterministic re-evaluation of the existing checkpoints gives:

| Task | 0K recovery | Unified 9K | Unified 18K |
| --- | ---: | ---: | ---: |
| Protein Binder | 3.07% | 8.67% | **11.23%** |
| Antibody H3 | 2.98% | **21.84%** | 20.98% |
| RNA | 1.26% | 18.89% | **23.35%** |

This proves detectable unified learning, but H3 plateaus/slightly regresses from 9K
to 18K. These values are learnability diagnostics rather than final biological
design metrics, so they do not yet constitute the final NanoDesign reference
baseline.

## Stage 2 — 32-Example Learnability

All values use frozen training panels and measure memorization, not held-out
generalization.

| Task | Untrained recovery | Selected budget | Selected recovery | Shuffled context | Detached context |
| --- | ---: | ---: | ---: | ---: | ---: |
| Binder | 1.29% | 12K | **44.81%** | 40.47% | 25.79% |
| H3 | 1.79% | 6K | **55.69%** | 43.27% | 36.60% |
| RNA | 5.86% | 24K | **70.13%** | 69.34% | 67.39% |

All selected checkpoints generated 8/8 distinct sequences with finite coordinates.
Maximum dominant-token fractions were 50.00% for Binder, 45.83% for H3, and 43.02%
for RNA. Binder and H3 show strong sequence/context dependence. For RNA at the
official training distribution's median noise, coordinate RMSD worsens from
1.168 Å with correct context to 1.212 Å after target-sequence shuffle and 1.278 Å
after spatial detachment. Stage 2 is therefore **PASS**.

## Stage 3 — Frozen 128-Example Learning Curves

The current Stage-3 experiment changes only panel size and exposure; architecture,
recipe, seed and evaluator remain frozen.

### Protein Binder

| Samples seen | Validation loss ↓ | Coordinate loss ↓ | Recovery ↑ |
| ---: | ---: | ---: | ---: |
| 0 | 2.5278 | 2.1615 | 1.64% |
| 300 | 1.9803 | 1.6592 | 6.14% |
| 900 | 1.8796 | 1.5852 | 6.84% |
| 3K | 1.0666 | 0.7992 | 15.53% |
| 6K | 0.8349 | 0.5742 | 17.66% |
| 12K | 0.6974 | 0.4499 | 22.64% |
| 24K | **0.5949** | **0.3663** | **28.31%** |

The full 24K audit reports 29.38% correct-context recovery, 28.71% shuffled, and
22.24% detached. Coordinate RMSD worsens from 0.296 Å to 0.321/0.477 Å, and
21.37%/53.83% of sequence predictions change under the two controls. Generation is
finite and non-collapsed: 8/8 distinct sequences, 34.20% mean and 47.06% maximum
dominant-token fraction. **Binder task-level Stage-3 gate: PASS.**

### Antibody CDR-H3

| Samples seen | Validation loss ↓ | Coordinate loss ↓ | Recovery ↑ |
| ---: | ---: | ---: | ---: |
| 0 | 0.7188 | 0.3620 | 1.98% |
| 300 | 0.6098 | 0.3097 | 12.25% |
| 900 | 0.5705 | 0.2968 | 14.85% |
| 3K | 0.4527 | 0.2258 | 32.59% |
| 6K | **0.4045** | **0.2125** | **43.89%** |

The full 6K audit reports 42.79% correct-context recovery, 35.03% shuffled, and
28.99% detached. Coordinate RMSD worsens from 0.321 Å to 0.342/0.694 Å, and
21.25%/35.05% of sequence predictions change. Generation is finite and
non-collapsed: 8/8 distinct sequences, 38.94% mean and 53.33% maximum dominant-token
fraction. **H3 task-level Stage-3 gate: PASS.**

### RNA Binding Design

| Samples seen | Validation loss ↓ | Coordinate loss ↓ | Recovery ↑ |
| ---: | ---: | ---: | ---: |
| 0 | 1.8746 | 1.5446 | 5.90% |
| 300 | 1.2408 | 1.0387 | 25.91% |
| 900 | 1.1320 | 0.9896 | 26.19% |
| 3K | 0.7010 | 0.5626 | 27.83% |
| 6K | 0.5852 | 0.4486 | 31.16% |
| 12K | **0.5259** | **0.3945** | **37.02%** |

The 6K audit confirms continuous learning but does not pass the complete gate:

- near-clean correct/shuffled/detached recovery is 32.21%/32.14%/32.14%;
- near-clean coordinate RMSD is 0.4892/0.4893/0.4911 Å;
- at median training noise, coordinate RMSD is 1.9023 Å correct, 1.9043 Å
  shuffled, and 1.9320 Å detached;
- generation improves substantially from the 3K all-homopolymer failure to a 73.80%
  mean dominant-token fraction, but the maximum remains 100% and one of eight
  generated sequences still contains only one token type.

Thus RNA 6K has a clear optimization signal but retains residual collapse and only a
weak context advantage. It is **not** recorded as a Stage-3 PASS.

At 12K, near-clean correct/shuffled/detached recovery is
35.85%/35.89%/35.67%, and coordinate RMSD is 0.4342/0.4343/0.4369 Å. At median
training noise, correct-context coordinate RMSD is 1.6435 Å, versus 1.6464 Å after
target-sequence shuffle and 1.7020 Å after spatial detachment; detachment changes
predicted design coordinates by 0.4441 Å. Generation collapse is now resolved:
all 8 sequences are distinct and finite, every generated RNA uses at least three
base types, and mean/maximum dominant-token fractions fall to 67.24%/87.50%.

The 12K checkpoint therefore passes the loss, recovery, sample-specific prediction,
finite-generation and non-collapse checks. Spatial detachment is clearly adverse,
but the target-sequence-shuffle RMSD change is only 0.0029 Å (0.18%) and sequence
recovery does not worsen. That is too weak to call the required shuffle control
"markedly worse." RNA 12K remains an interim improvement rather than a forced PASS;
the unchanged run continues to 24K.

## What Has Been Proven

1. The pinned 6.85M RFD3NA architecture has enough capacity to learn all three tasks
   on 32-example panels without changing architecture.
2. The corrected deterministic evaluator is sensitive to checkpoint improvement and
   rejects broken predictions in the expected direction.
3. Binder and H3 learning extends from 32 to 128 examples with stable joint
   sequence/structure optimization, context dependence and non-collapsed generation.
4. RNA loss and recovery improve continuously through 12K and its generation collapse
   is solved by 12K, so the RNA path is not disconnected; however, the complete
   128-example context gate is still open.
5. The historical unified model has a measurable signal on all three tasks, but it is
   not yet a strong reproducible reference baseline.

## What Is Not Yet Proven

- Stage 3 as a whole has not passed because RNA's 128-example gate remains open.
- Formal full-pool single-task baselines have not started under the staged protocol.
- Whether 6.85M is the best minimum model size has not been calibrated.
- Single-task versus unified transfer has not been established with the corrected
  recipe/evaluator.
- The final unified mixture and fixed `samples seen` budget are not selected.
- Multi-seed full true-generation evaluation and external Binder/H3/RNA metrics are
  not complete.
- NanoDesign is therefore **not yet declared ready for AutoResearch**.

## Live Execution State

At this snapshot, QGPU3021 is running or retaining the following independent
single-GPU work:

- GPU 0: waiting for the RNA 24K checkpoint, then runs its full context +
  eight-generation audit;
- GPU 1: waiting for the RNA 24K checkpoint, then runs its median-noise context audit;
- GPU 2: unchanged RNA continuation from 12K toward 24K.

The former QGPU3006 allocation completed the RNA 6K checkpoint; the durable
continuation and audits now run inside the unlimited QGPU3021 allocation. No Stage-4
experiment is started before Stage 3 passes and `docs/STAGE_3_128_SAMPLE.md` is
written.

## Next Gate

Complete RNA 24K and evaluate it with the already-frozen context and generation
controls. If all Stage-3 criteria pass, write `docs/STAGE_3_128_SAMPLE.md`, update
`docs/PROGRESS.md`, and only then enter Stage 4. If target-sequence-shuffle
sensitivity remains weak, stay in Stage 3 and diagnose that single remaining gate
without changing the evaluator or advancing prematurely.
