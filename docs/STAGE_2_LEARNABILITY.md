# Stage 2 — 32-Sample Learnability

## Status

**PASS**

All three frozen 32-example tasks show strong memorization, context-sensitive
predictions, finite non-collapsed generation, and stable joint sequence/structure
optimization. Stage 2 uses learnability diagnostics only; it is not a held-out or
biological design benchmark.

## Experiments Run

The following controlled experiments were actually run with the frozen sample IDs and
seed 17. Machine-readable results are in `9-3-experiment/`.

1. Untrained controls for Binder, H3, and RNA.
2. Reference-recipe continuations:
   - Binder: 3K and 12K selected checkpoints.
   - H3: 3K and 6K selected checkpoints.
   - RNA: 3K, 6K, 12K, and 24K selected checkpoints.
3. Supervision controls: design-only versus full-valid sequence supervision.
4. Loss-isolation controls: sequence-only, joint sequence/structure, and sequence
   weight 1.0.
5. Optimizer/schedule controls: Adam, reference AdamW, and the AF3 learning-rate
   schedule.
6. Diffusion controls: 16 versus 32 realizations and fixed near-clean noise.
7. EMA controls: online, EMA 0.999, and EMA 0.99 weights.
8. Coordinate controls: official-style augmentation versus no augmentation.
9. One-example Binder/H3/RNA capacity controls.
10. Deterministic context controls:
    - cyclic permutation of fixed-context residue identity;
    - spatial detachment of the fixed context from the design region;
    - design-coordinate response at the analytical median of the official EDM
      training distribution, `t = 16 * exp(-1.2) = 4.8191073906`.
11. Eight-sample official EDM generation checks on the selected checkpoints.

The final selected checkpoints are:

| Task | Samples seen | Checkpoint |
| --- | ---: | --- |
| Binder | 12,000 | `data/runs/nanodesign-v1/stage2-binder-long/binder-reference-48k/milestones/samples-00012000.pt` |
| H3 | 6,000 | `data/runs/nanodesign-v1/stage2-other-tasks-long/h3-reference-48000/milestones/samples-00006000.pt` |
| RNA | 24,000 | `data/runs/nanodesign-v1/stage2-other-tasks-long/rna-reference-24000/milestones/samples-00024000.pt` |

## Key Results

All recovery values below use the exact frozen overfit-32 panel in the corrected v2
training frame. They measure memorization, not generalization.

| Task | Untrained recovery | Selected recovery | Sequence CE | Sequence-shuffled recovery | Spatially detached recovery |
| --- | ---: | ---: | ---: | ---: | ---: |
| Binder | 1.29% | **44.81%** | 1.6664 | 40.47% | 25.79% |
| H3 | 1.79% | **55.69%** | 1.4122 | 43.27% | 36.60% |
| RNA | 5.86% | **70.13%** | 0.7159 | 69.34% | 67.39% |

Training-log window means also move in the correct direction:

| Task | Sequence loss, first 300 → last 300 | Coordinate loss, first 300 → last 300 |
| --- | ---: | ---: |
| Binder | 0.2400 → 0.1320 | 0.5103 → 0.3295 |
| H3 | 0.1740 → 0.1247 | 0.2110 → 0.2055 |
| RNA | 0.1230 → 0.0746 | 0.3685 → 0.2904 |

### Binder

The selected 12K EMA checkpoint reaches 44.81% recovery. Shuffling target residue
identity changes 28.05% of design-token predictions and spatially detaching the target
changes 66.30%. Design-coordinate RMSD worsens from 0.286 Å to 0.336 Å after sequence
shuffle and to 0.585 Å after detachment.

### H3

The selected 6K EMA checkpoint reaches 55.69% recovery. Sequence shuffle changes
38.05% of predictions and lowers recovery by 12.42 percentage points. Spatial
detachment changes 45.89% and lowers recovery by 19.09 points.

### RNA

The selected 24K EMA checkpoint reaches 70.13% recovery. At the frozen near-clean
`t=0.5`, sequence-only context shuffle has a modest effect because the noisy native RNA
backbone itself contains strong base/secondary-structure information. The
structure-conditioned control at the official training distribution's median noise is
therefore the decisive test:

| RNA 24K, `t=4.8191` | Correct context | Target sequence shuffled | Target spatially detached |
| --- | ---: | ---: | ---: |
| Design-coordinate RMSD ↓ | **1.168 Å** | 1.212 Å | 1.278 Å |
| Coordinate prediction change from correct | — | 0.201 Å | 0.454 Å |
| Sequence recovery | **50.97%** | 51.28% | 49.18% |
| Sequence CE ↓ | **1.1636** | 1.1575 | 1.1990 |

Target sequence shuffle worsens structure RMSD by 3.7%; breaking the target–RNA
spatial relationship worsens it by 9.4%. The sequence recovery under identity shuffle
does not decrease, which is expected to be a weaker RNA diagnostic because RNA design
is one-to-many. The coordinate response demonstrates that the joint model does use the
molecular context rather than emitting a context-independent structure prior.

## Root Cause

There were two separate causes behind the original apparent failure:

1. **Evaluator coordinate-frame mismatch.** The original `learnability.v1` panel fed
   arbitrary deposited PDB frames to a model trained with official RFD3NA centering and
   rigid augmentation. On the same Binder 12K online checkpoint, recovery was 10.14%
   in v1 but 44.86% in the corrected deterministic training frame. This was an
   evaluator false negative, not a model-capacity result.
2. **Insufficient exposure for stable RNA generation.** RNA-32 at 3K memorized only
   32.77% and generated severe dominant-base collapse. The unchanged joint reference
   recipe reaches 39.12% at 6K and 70.13% at 24K; collapse disappears by 6K and remains
   absent at 24K.

The batch audit found no design-mask, target, atom/token association, alphabet, noise,
or active-token-normalization corruption. The one-RNA isolation run reached 81.44%
recovery, independently demonstrating that the representation, loss, and sampler have
the required capacity.

## Changes That Worked

- Normalize the masked sequence supervision over active design tokens, preventing
  fixed context length from diluting H3/RNA gradients.
- Use the deterministic v2 evaluation frame that reproduces official training
  centering/augmentation while keeping panel IDs and seeds frozen.
- Preserve joint sequence and coordinate supervision.
- Continue the same reference recipe until each task's generation is stable; the
  smallest observed non-collapsed checkpoints were Binder 3K, H3 3K, and RNA 6K,
  while the selected context/memorization checkpoints are 12K, 6K, and 24K.
- Evaluate RNA context dependence through both sequence and 3D geometry at a
  representative official-training noise level.

## Changes That Did Not Work

The following branches were rejected and must not be repeated in Stage 3:

- sequence-only training: it improved neither shared context dependence nor joint
  structure learning;
- near-clean plus sequence-only RNA training: 27.43% recovery at 3K and essentially no
  context response;
- full-valid sequence supervision: it optimizes fixed context instead of the declared
  design region and did not improve the shared task;
- increasing sequence-loss weight to 1.0: Binder improved locally, but H3 and RNA did
  not show a consistent shared-recipe gain;
- online weights instead of EMA: RNA-3K generation remained collapsed;
- changing only EMA decay, diffusion realization count, Adam/AdamW, or the AF3
  schedule: none explained the cross-task failure;
- disabling coordinate augmentation: it moves training away from the pinned official
  recipe and did not provide a consistent solution.

## Final Training Recipe

The Stage-3 recipe is the retained reference recipe, not any rejected ablation:

- model: pinned public RFD3NA `RFD3`, 6,849,538 parameters;
- design sequence: fully masked; fixed-context sequence visible;
- sequence supervision: design region only, normalized per design token;
- coordinate diffusion: design atoms only, official EDM `sigma_data=16` distribution;
- coordinate augmentation: enabled with the pinned official settings;
- diffusion realizations per complex: 16;
- loss: official coordinate loss weight 4.0 plus sequence loss weight 0.1;
- optimizer: AdamW, betas `(0.9, 0.95)`, weight decay `1e-4`;
- learning rate: constant `5e-4`;
- gradient clipping: global norm 10;
- EMA: 0.999, used for validation and generation;
- generation: official EDM sampler, 50 steps, fixed motif centered and native design
  coordinates removed before sampling;
- seed: 17 for the first controlled validation run.

Stage 2 determines recipe correctness only. Its different per-task overfit budgets are
not frozen as the production training budget.

## Generation Collapse

**Solved on all three tasks.** All selected checkpoints produced 8/8 distinct
sequences with finite coordinates:

| Task | Distinct / generated | Mean dominant-token fraction | Maximum dominant-token fraction |
| --- | ---: | ---: | ---: |
| Binder | 8 / 8 | 31.04% | 50.00% |
| H3 | 8 / 8 | 33.87% | 45.83% |
| RNA | 8 / 8 | 37.28% | 43.02% |

RNA outputs use all four bases in every audited 24K sample. These are collapse
diagnostics, not task-quality scores.

## Evidence of Context Usage

- Binder: 44.81% correct versus 40.47% sequence-shuffled and 25.79% spatially
  detached; 28.05%/66.30% of predictions change.
- H3: 55.69% correct versus 43.27% sequence-shuffled and 36.60% spatially detached;
  38.05%/45.89% change.
- RNA: the sequence head is relatively invariant to target residue relabeling, but at
  representative training noise its design-coordinate RMSD worsens from 1.168 Å to
  1.212 Å under target sequence shuffle and 1.278 Å under spatial detachment. The
  predicted design coordinates move by 0.201 Å and 0.454 Å, respectively.

Thus every task produces sample-specific predictions and responds adversely to a
broken molecular context. RNA's principal context evidence is structural, consistent
with the task's one-to-many sequence semantics.

## Decision

**PASS.** The original low scores were traced to a reproducible evaluator-frame bug;
the corrected frozen panel shows clear memorization. Every selected task checkpoint
has lower sequence/coordinate losses, strong recovery above initialization,
context-sensitive output, finite diverse generation, and stable joint structure loss.
No Stage-2 evidence requires changing the RFD3NA architecture.

## Next Stage

Enter **Stage 3 — 128-Sample Validation** using exactly the retained recipe above.
Before launching any Stage-3 run, read this file and `docs/PROGRESS.md`; do not repeat
the rejected Stage-2 branches.
