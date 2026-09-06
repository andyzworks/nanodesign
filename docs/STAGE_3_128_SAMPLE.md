# Stage 3 — 128-Sample Validation

## Status

**PASS**

The retained Stage-2 recipe learns all three fixed 128-example panels. Binder,
H3, and RNA each show lower loss, substantially higher recovery than the
untrained model, sample-specific non-collapsed generation, and an adverse
response when the fixed molecular context is spatially detached. These are
training-panel learnability diagnostics, not held-out biological benchmarks.

## Recipe Used

Stage 3 used the recipe frozen by `docs/STAGE_2_LEARNABILITY.md` without a
training change:

- pinned public RFD3NA `RFD3`, 6,849,538 parameters;
- fully masked design sequence and visible fixed-context sequence;
- design-only sequence supervision normalized over active design tokens;
- joint coordinate and sequence training with weights 4.0 and 0.1;
- official EDM noise distribution with `sigma_data=16` and coordinate
  augmentation enabled;
- 16 diffusion realizations per complex;
- AdamW, learning rate `5e-4`, betas `(0.9, 0.95)`, weight decay `1e-4`;
- gradient clipping at global norm 10 and EMA 0.999;
- seed 17 and the frozen `overfit128_v2` panel;
- official 50-step EDM generation sampler.

## Binder

Binder required 24K samples seen. Its frozen 128-example recovery rose from
1.54% at initialization to 18.59% at 6K, 22.65% at 12K, and **29.38% at 24K**.
At 24K, raw sequence CE is 2.2753 and near-clean design-coordinate RMSD is
0.2964 Angstrom. The checkpoint training record reports validation loss
0.5949, comprising coordinate loss 0.3663 and sequence loss 0.2286, with
validation recovery 28.31%.

Spatially detaching the target lowers recovery from 29.38% to 22.24%, worsens
coordinate RMSD from 0.2964 to 0.4767 Angstrom, and changes 53.83% of design
predictions. Generation is finite and distinct for 8/8 samples; mean/maximum
dominant-token fractions are 34.20%/47.06%.

## H3

H3 passes at 6K samples seen. Recovery rises from 1.83% at initialization to
**42.79%**, raw sequence CE falls to 1.9290, and near-clean H3-coordinate RMSD
falls from 1.0477 to 0.3213 Angstrom. The checkpoint record reports validation
loss 0.4045, coordinate loss 0.2125, sequence loss 0.1920, and validation
recovery 43.89%.

Sequence-shuffled context lowers recovery to 35.03%. Spatial detachment lowers
it to 28.99%, worsens coordinate RMSD to 0.6940 Angstrom, and changes 35.05% of
design predictions. Generation is finite and distinct for 8/8 samples;
mean/maximum dominant-token fractions are 38.94%/53.33%.

## RNA

RNA was continued with the unchanged recipe to 24K samples seen; the existing
checkpoint was evaluated directly and was not retrained. The learnability curve
is continuous:

| Samples seen | Recovery | Raw sequence CE | Near-clean coordinate RMSD |
| ---: | ---: | ---: | ---: |
| 0 | 5.27% | 3.2899 | 1.1599 A |
| 3K | 27.99% | 1.3833 | 0.5860 A |
| 6K | 32.21% | 1.3665 | 0.4892 A |
| 12K | 35.85% | 1.3144 | 0.4342 A |
| 24K | **41.18%** | **1.2576** | **0.3795 A** |

The 24K checkpoint record independently reports validation loss 0.4757,
coordinate loss 0.3501, sequence loss 0.1256, and validation recovery 41.02%.

At the analytical median of the official EDM training distribution,
`t=4.8191073906`, RNA responds to the fixed 3D context as follows:

| Context | Recovery | Raw sequence CE | Design-coordinate RMSD |
| --- | ---: | ---: | ---: |
| Correct | 33.06% | 1.3522 | **1.4187 A** |
| Target sequence shuffled | 33.01% | 1.3525 | 1.4203 A |
| Target spatially detached | 31.95% | 1.3556 | **1.4911 A** |

Spatial detachment moves predicted design coordinates by 0.4388 Angstrom and
changes 10.10% of sequence predictions. Target-identity shuffling is weak, but
the spatial control is clearly adverse. Per the frozen decision rule, this is
sufficient evidence for an all-atom model whose RNA output may rely primarily
on 3D context.

The 24K 8-sample generation audit is non-collapsed: 8/8 sequences are distinct,
all coordinates are finite, and every sequence uses three or four RNA bases.
Mean/maximum dominant-token fractions are **60.90%/78.79%**, improving from
67.24%/87.50% at 12K. Machine-readable results are in
`9-3-experiment/stage3_128_rna_24k_v2_audit.json` and
`9-3-experiment/stage3_128_rna_24k_median_noise_context_audit.json`.

## Comparison with 32 Samples

The Stage-2 32-example panels reached 44.81% Binder recovery at 12K, 55.69% H3
at 6K, and 70.13% RNA at 24K. The lower 128-example recoveries are expected from
the fourfold increase in unique examples. Crucially, all tasks retain monotonic
or clearly positive learning curves, context-dependent outputs, stable joint
coordinate learning, and non-collapsed generation. The Stage-2 learning behavior
therefore scales to 128 examples rather than being limited to memorizing 32.

## Problems Found

- RNA 6K still had one homopolymer generation despite improved denoising scores.
- RNA 12K removed the hard collapse but retained a high 87.5% maximum
  dominant-base fraction.
- The first RNA 24K audit attempt ran on GPUs already occupied by an unrelated
  workload and failed with CUDA OOM; the checkpoint and training were intact.
- A single large RNA generation sample uses the frozen chunked path and is much
  slower than standard-path samples.

## Fixes Applied

- No model or training-recipe change was made.
- RNA was evaluated at 24K exactly as required by the Stage-3 gate.
- The OOM was resolved by moving evaluation to empty QGPU3011 GPUs.
- The same fixed eight generation examples were split into deterministic 3/3/2
  shards. A `--generation-start-index` audit CLI option was added so evaluation
  can be distributed without changing panel membership, seeds, sampler, or
  metrics. The default remains zero.

## Final Decision

**PASS.** Binder, H3, and RNA all show stable loss reduction, large recovery
improvements over initialization, finite sample-specific generation, and no
generation collapse at their selected checkpoints. All three respond adversely
to spatially detached context. RNA target-sequence shuffle remains weak, but its
median-noise spatial detachment worsens coordinate RMSD by 0.0724 Angstrom
(5.10%) and lowers recovery by 1.11 percentage points, satisfying the explicitly
stated all-atom context criterion.

## Next Stage

Enter **Stage 4 — Formal Single-Task Baselines**. Before any Stage-4 execution,
read this file, `docs/STAGE_2_LEARNABILITY.md`, and `docs/PROGRESS.md`. Reuse the
retained recipe and existing checkpoints where valid; begin with 6K samples seen
per task and evaluate both learnability and the already-frozen task-specific
metrics. Do not repeat Stage-3 training.
